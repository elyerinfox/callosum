//! DeepSeek-V2/V3 (`deepseek2`, incl. Kimi-K2) forward pass on wgpu —
//! MLA attention in the decompressed formulation plus DeepSeek MoE,
//! mirroring the CUDA backend op-for-op:
//!
//! - q (optionally LoRA'd) split into no-PE + PE parts per head;
//!   latent KV from `attn_kv_a_mqa` split into the compressed vector
//!   and a single rope'd K head; `attn_kv_b` expands to per-head
//!   K-noPE + V (K and V widths differ).
//! - Interleaved RoPE on the PE dims with YaRN-corrected frequencies
//!   (delivered to the kernel as per-dim divisors); the YaRN mscale²
//!   correction rides the attention scale.
//! - MoE: leading dense layers, softmax/sigmoid routing with optional
//!   selection bias + grouped top-k + renormalisation + routed scale
//!   (all on-GPU in moe_topk_opt), fused shared experts added on top.

use callosum::quantized::{gguf_file, GgmlDType};

use crate::llama::{
    host_moe_forward, split_expert_qmatmuls_host, HostExperts, HostRowTable, KvStore, StageInput,
    StageOutput,
};
use crate::{GpuBuffer, QuantBuffer, QuantDtype, Result, WgpuDevice, WgpuError};

pub struct DsConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub layer_start: usize,
    pub layer_end: usize,
    pub n_heads: usize,
    pub vocab: usize,
    pub rope_dim: usize,
    pub nope_dim: usize,
    pub v_dim: usize,
    pub kv_lora_rank: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub softmax_scale: f32,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub sigmoid_gating: bool,
    pub weights_norm: bool,
    pub weights_scale: f32,
    pub n_group: usize,
    pub topk_group: usize,
}

enum Weight {
    F32 { buf: GpuBuffer, n: usize, k: usize },
    Quant(QuantBuffer),
}

impl Weight {
    fn matmul_t(&self, dev: &WgpuDevice, x: &GpuBuffer, m: usize) -> Result<GpuBuffer> {
        match self {
            Weight::F32 { buf, n, k } => dev.matmul_t(x, buf, m, *k, *n),
            Weight::Quant(q) => dev.matmul_t_quant(x, q, m, q.k),
        }
    }

    fn out_features(&self) -> usize {
        match self {
            Weight::F32 { n, .. } => *n,
            Weight::Quant(q) => q.n,
        }
    }
}

fn quant_dtype(d: GgmlDType) -> Option<QuantDtype> {
    match d {
        GgmlDType::Q4_0 => Some(QuantDtype::Q4_0),
        GgmlDType::Q4_1 => Some(QuantDtype::Q4_1),
        GgmlDType::Q5_0 => Some(QuantDtype::Q5_0),
        GgmlDType::Q5_1 => Some(QuantDtype::Q5_1),
        GgmlDType::Q8_0 => Some(QuantDtype::Q8_0),
        GgmlDType::Q2K => Some(QuantDtype::Q2K),
        GgmlDType::Q3K => Some(QuantDtype::Q3K),
        GgmlDType::Q4K => Some(QuantDtype::Q4K),
        GgmlDType::Q5K => Some(QuantDtype::Q5K),
        GgmlDType::Q6K => Some(QuantDtype::Q6K),
        GgmlDType::F16 => Some(QuantDtype::F16),
        GgmlDType::BF16 => Some(QuantDtype::Bf16),
        _ => None,
    }
}

struct Dense {
    gate: Weight,
    up: Weight,
    down: Weight,
}

struct Moe {
    router: Weight,
    router_bias: Option<GpuBuffer>,
    experts: ExpertStore,
    ffn_dim: usize,
    shared: Option<Dense>,
}

/// Routed expert weights: VRAM-resident, or host RAM under `cpu_moe`
/// expert offload (CPU expert math, activations-only bus traffic).
enum ExpertStore {
    Resident(Box<ResidentExperts>),
    Host(Box<HostExperts>),
}

/// VRAM-resident fused expert weights ([n_experts * rows, cols],
/// quantized as stored on disk).
struct ResidentExperts {
    gates: Weight,
    ups: Weight,
    downs: Weight,
}

enum Ffn {
    Dense(Box<Dense>),
    Moe(Box<Moe>),
}

struct Block {
    attn_norm: GpuBuffer,
    q: Weight,
    q_a_norm: Option<GpuBuffer>,
    q_b: Option<Weight>,
    kv_a: Weight,
    kv_a_norm: GpuBuffer,
    kv_b: Weight,
    o: Weight,
    ffn_norm: GpuBuffer,
    ffn: Ffn,
}

pub struct WgpuDeepSeek {
    dev: WgpuDevice,
    pub cfg: DsConfig,
    embed: Option<HostRowTable>,
    /// Per-dim inv-freq divisors turning the kernel's base frequency
    /// into the YaRN-corrected one (rope kernel divides by these).
    yarn_divisors: GpuBuffer,
    blocks: Vec<Block>,
    out_norm: Option<GpuBuffer>,
    lm_head: Option<Weight>,
    pub weight_bytes: u64,
    /// Expert bytes held in host RAM under `cpu_moe` offload —
    /// excluded from the GPU-resident `weight_bytes`.
    pub host_expert_bytes: u64,
}

pub struct Session {
    k: Vec<KvStore>,
    v: Vec<KvStore>,
    pub len: usize,
    max_seq: usize,
}

fn meta_u32(c: &gguf_file::Content, key: &str) -> Option<u32> {
    c.metadata.get(key).and_then(|v| v.to_u32().ok())
}

fn meta_f32(c: &gguf_file::Content, key: &str) -> Option<f32> {
    c.metadata.get(key).and_then(|v| v.to_f32().ok())
}

impl WgpuDeepSeek {
    pub fn from_gguf(path: &std::path::Path, dev: &WgpuDevice) -> Result<Self> {
        Self::from_gguf_stage(path, dev, 0, usize::MAX, true, true)
    }

    /// `from_gguf` with `cpu_moe` expert offload.
    pub fn from_gguf_cpu_moe(path: &std::path::Path, dev: &WgpuDevice) -> Result<Self> {
        Self::from_gguf_stage_opts(path, dev, 0, usize::MAX, true, true, true)
    }

    pub fn from_gguf_stage(
        path: &std::path::Path,
        dev: &WgpuDevice,
        layer_start: usize,
        layer_end: usize,
        owns_input: bool,
        owns_output: bool,
    ) -> Result<Self> {
        Self::from_gguf_stage_opts(
            path,
            dev,
            layer_start,
            layer_end,
            owns_input,
            owns_output,
            false,
        )
    }

    /// [`Self::from_gguf_stage`] plus `cpu_moe` expert offload: routed
    /// expert tensors are byte-sliced into host RAM instead of being
    /// uploaded, and the expert FFN runs on the CPU per forward.
    #[allow(clippy::too_many_arguments)]
    pub fn from_gguf_stage_opts(
        path: &std::path::Path,
        dev: &WgpuDevice,
        layer_start: usize,
        layer_end: usize,
        owns_input: bool,
        owns_output: bool,
        cpu_moe: bool,
    ) -> Result<Self> {
        let mut file =
            std::fs::File::open(path).map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| WgpuError::Device(format!("gguf parse: {e}")))?;
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        if arch != "deepseek2" {
            return Err(WgpuError::Device(format!(
                "callosum-wgpu deepseek loader supports deepseek2 (got {arch:?})"
            )));
        }
        let key = |s: &str| format!("deepseek2.{s}");
        let need = |k: &str| {
            meta_u32(&content, k).ok_or_else(|| WgpuError::Device(format!("missing {k}")))
        };

        let hidden = need(&key("embedding_length"))? as usize;
        let n_layers = need(&key("block_count"))? as usize;
        let n_heads = need(&key("attention.head_count"))? as usize;
        let q_head = need(&key("attention.key_length"))? as usize;
        let rope_dim = need(&key("rope.dimension_count"))? as usize;
        let nope_dim = q_head - rope_dim;
        let v_dim = need(&key("attention.value_length"))? as usize;
        let kv_lora_rank = need(&key("attention.kv_lora_rank"))? as usize;
        let rms_eps = meta_f32(&content, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        let rope_theta = meta_f32(&content, &key("rope.freq_base")).unwrap_or(10_000.0);
        let n_experts = meta_u32(&content, &key("expert_count")).unwrap_or(0) as usize;
        let n_experts_used = meta_u32(&content, &key("expert_used_count")).unwrap_or(0) as usize;
        let sigmoid_gating = meta_u32(&content, &key("expert_gating_func")) == Some(2);
        let weights_norm = content
            .metadata
            .get(&key("expert_weights_norm"))
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(false);
        let weights_scale = meta_f32(&content, &key("expert_weights_scale")).unwrap_or(1.0);
        let n_group = meta_u32(&content, &key("expert_group_count"))
            .unwrap_or(1)
            .max(1) as usize;
        let topk_group = meta_u32(&content, &key("expert_group_used_count"))
            .unwrap_or(1)
            .max(1) as usize;

        // YaRN: corrected inv-freqs for the PE dims + mscale² on the
        // attention scale — same math as the CUDA backend.
        let factor = meta_f32(&content, &key("rope.scaling.factor")).filter(|&f| f > 1.0);
        let original_ctx =
            meta_u32(&content, &key("rope.scaling.original_context_length")).unwrap_or(4096) as f32;
        let log_mult = meta_f32(&content, &key("rope.scaling.yarn_log_multiplier"));
        let mut softmax_scale = 1.0 / (q_head as f32).sqrt();
        if let (Some(f), Some(lm)) = (factor, log_mult) {
            let mscale = lm * f.ln() + 1.0;
            softmax_scale *= mscale * mscale;
        }
        let half = rope_dim / 2;
        let base_freqs: Vec<f32> = (0..rope_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / rope_dim as f32))
            .collect();
        let yarn_freqs: Vec<f32> = match factor {
            None => base_freqs.clone(),
            Some(factor) => {
                let beta_fast = 32f32;
                let beta_slow = 1f32;
                let corr = |n_rot: f32| -> f32 {
                    (rope_dim as f32 * (original_ctx / (n_rot * 2.0 * std::f32::consts::PI)).ln())
                        / (2.0 * rope_theta.ln())
                };
                let low = corr(beta_fast).floor().max(0.0);
                let high = corr(beta_slow).ceil().min((rope_dim - 1) as f32);
                (0..half)
                    .map(|i| {
                        let r = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
                        let mask = 1.0 - r;
                        base_freqs[i] / factor * (1.0 - mask) + base_freqs[i] * mask
                    })
                    .collect()
            }
        };
        // Kernel computes pow(theta, -2d/rope_dim) / divisor[d].
        let divisors: Vec<f32> = base_freqs
            .iter()
            .zip(&yarn_freqs)
            .map(|(b, y)| b / y)
            .collect();

        let cpu = callosum::Device::Cpu;
        let weight_bytes = std::cell::Cell::new(0u64);
        let host_expert_bytes = std::cell::Cell::new(0u64);
        let mut file2 =
            std::fs::File::open(path).map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
        let mut load_f32 = |name: &str| -> Result<GpuBuffer> {
            let qt = content
                .tensor(&mut file2, name, &cpu)
                .map_err(|e| WgpuError::Device(format!("load {name}: {e}")))?;
            let t = qt
                .dequantize(&cpu)
                .and_then(|t| t.to_dtype(callosum::DType::F32))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| WgpuError::Device(format!("dequantize {name}: {e}")))?;
            weight_bytes.set(weight_bytes.get() + (t.len() * 4) as u64);
            Ok(dev.upload(&t))
        };
        let mut file3 =
            std::fs::File::open(path).map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
        let mut load_weight = |name: &str| -> Result<Weight> {
            let qt = content
                .tensor(&mut file3, name, &cpu)
                .map_err(|e| WgpuError::Device(format!("load {name}: {e}")))?;
            let dims = qt.shape().dims().to_vec();
            let (n, k) = match dims.len() {
                2 => (dims[0], dims[1]),
                3 => (dims[0] * dims[1], dims[2]),
                _ => return Err(WgpuError::Shape(format!("{name}: bad rank"))),
            };
            match quant_dtype(qt.dtype()) {
                Some(fmt) if k % fmt.block_elems() == 0 => {
                    let raw = qt
                        .data()
                        .map_err(|e| WgpuError::Device(format!("{name} bytes: {e}")))?;
                    weight_bytes.set(weight_bytes.get() + raw.len() as u64);
                    dev.upload_quantized(&raw, n, k, fmt).map(Weight::Quant)
                }
                _ => {
                    let t = qt
                        .dequantize(&cpu)
                        .and_then(|t| t.to_dtype(callosum::DType::F32))
                        .and_then(|t| t.flatten_all())
                        .and_then(|t| t.to_vec1::<f32>())
                        .map_err(|e| WgpuError::Device(format!("dequantize {name}: {e}")))?;
                    weight_bytes.set(weight_bytes.get() + (t.len() * 4) as u64);
                    Ok(Weight::F32 {
                        buf: dev.upload(&t),
                        n,
                        k,
                    })
                }
            }
        };
        let has = |name: &str| content.tensor_infos.contains_key(name);

        let layer_end = layer_end.min(n_layers);
        if layer_start >= layer_end {
            return Err(WgpuError::Shape(format!(
                "empty layer range [{layer_start},{layer_end}) of {n_layers}"
            )));
        }

        let mut embed = None;
        let mut vocab = 0usize;
        if owns_input {
            let mut file_e = std::fs::File::open(path)
                .map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
            let qt = content
                .tensor(&mut file_e, "token_embd.weight", &cpu)
                .map_err(|e| WgpuError::Device(format!("load token_embd: {e}")))?;
            let table = HostRowTable::from_qtensor(&qt)?;
            vocab = table.rows_total;
            embed = Some(table);
        }

        let mut blocks = Vec::with_capacity(layer_end - layer_start);
        for b in layer_start..layer_end {
            let q_lora = has(&format!("blk.{b}.attn_q_a.weight"));
            let (q, q_a_norm, q_b) = if q_lora {
                (
                    load_weight(&format!("blk.{b}.attn_q_a.weight"))?,
                    Some(load_f32(&format!("blk.{b}.attn_q_a_norm.weight"))?),
                    Some(load_weight(&format!("blk.{b}.attn_q_b.weight"))?),
                )
            } else {
                (load_weight(&format!("blk.{b}.attn_q.weight"))?, None, None)
            };
            let ffn = if has(&format!("blk.{b}.ffn_gate_inp.weight")) {
                let (experts, ffn_dim) = if cpu_moe {
                    let mut file_h = std::fs::File::open(path)
                        .map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
                    let mut load_host =
                        |name: String| -> Result<(Vec<callosum::quantized::QMatMul>, usize)> {
                            let qt = content
                                .tensor(&mut file_h, &name, &cpu)
                                .map_err(|e| WgpuError::Device(format!("load {name}: {e}")))?;
                            let dims = qt.shape().dims().to_vec();
                            let bytes = qt
                                .data()
                                .map_err(|e| WgpuError::Device(format!("{name} bytes: {e}")))?
                                .len() as u64;
                            host_expert_bytes.set(host_expert_bytes.get() + bytes);
                            Ok((split_expert_qmatmuls_host(&qt, n_experts)?, dims[1]))
                        };
                    let (gates, ffn_dim) = load_host(format!("blk.{b}.ffn_gate_exps.weight"))?;
                    let (ups, _) = load_host(format!("blk.{b}.ffn_up_exps.weight"))?;
                    let (downs, _) = load_host(format!("blk.{b}.ffn_down_exps.weight"))?;
                    (
                        ExpertStore::Host(Box::new(HostExperts::new(gates, ups, downs))),
                        ffn_dim,
                    )
                } else {
                    let gates = load_weight(&format!("blk.{b}.ffn_gate_exps.weight"))?;
                    let ffn_dim = gates.out_features() / n_experts;
                    (
                        ExpertStore::Resident(Box::new(ResidentExperts {
                            gates,
                            ups: load_weight(&format!("blk.{b}.ffn_up_exps.weight"))?,
                            downs: load_weight(&format!("blk.{b}.ffn_down_exps.weight"))?,
                        })),
                        ffn_dim,
                    )
                };
                Ffn::Moe(Box::new(Moe {
                    router: load_weight(&format!("blk.{b}.ffn_gate_inp.weight"))?,
                    router_bias: if has(&format!("blk.{b}.exp_probs_b.bias")) {
                        Some(load_f32(&format!("blk.{b}.exp_probs_b.bias"))?)
                    } else {
                        None
                    },
                    experts,
                    ffn_dim,
                    shared: if has(&format!("blk.{b}.ffn_gate_shexp.weight")) {
                        Some(Dense {
                            gate: load_weight(&format!("blk.{b}.ffn_gate_shexp.weight"))?,
                            up: load_weight(&format!("blk.{b}.ffn_up_shexp.weight"))?,
                            down: load_weight(&format!("blk.{b}.ffn_down_shexp.weight"))?,
                        })
                    } else {
                        None
                    },
                }))
            } else {
                Ffn::Dense(Box::new(Dense {
                    gate: load_weight(&format!("blk.{b}.ffn_gate.weight"))?,
                    up: load_weight(&format!("blk.{b}.ffn_up.weight"))?,
                    down: load_weight(&format!("blk.{b}.ffn_down.weight"))?,
                }))
            };
            // Recycle upload staging so weights aren't resident twice.
            dev.reclaim_staging();
            blocks.push(Block {
                attn_norm: load_f32(&format!("blk.{b}.attn_norm.weight"))?,
                q,
                q_a_norm,
                q_b,
                kv_a: load_weight(&format!("blk.{b}.attn_kv_a_mqa.weight"))?,
                kv_a_norm: load_f32(&format!("blk.{b}.attn_kv_a_norm.weight"))?,
                kv_b: load_weight(&format!("blk.{b}.attn_kv_b.weight"))?,
                o: load_weight(&format!("blk.{b}.attn_output.weight"))?,
                ffn_norm: load_f32(&format!("blk.{b}.ffn_norm.weight"))?,
                ffn,
            });
        }

        let mut out_norm = None;
        let mut lm_head = None;
        if owns_output {
            out_norm = Some(load_f32("output_norm.weight")?);
            let head = if has("output.weight") {
                load_weight("output.weight")?
            } else {
                load_weight("token_embd.weight")?
            };
            if vocab == 0 {
                vocab = head.out_features();
            }
            lm_head = Some(head);
        }

        Ok(Self {
            dev: dev.clone(),
            cfg: DsConfig {
                hidden,
                n_layers,
                layer_start,
                layer_end,
                n_heads,
                vocab,
                rope_dim,
                nope_dim,
                v_dim,
                kv_lora_rank,
                rms_eps,
                rope_theta,
                softmax_scale,
                n_experts,
                n_experts_used,
                sigmoid_gating,
                weights_norm,
                weights_scale,
                n_group,
                topk_group,
            },
            embed,
            yarn_divisors: dev.upload(&divisors),
            blocks,
            out_norm,
            lm_head,
            weight_bytes: weight_bytes.get(),
            host_expert_bytes: host_expert_bytes.get(),
        })
    }

    pub fn new_session(&self, max_seq: usize) -> Session {
        self.new_session_opts(max_seq, false)
    }

    /// [`Self::new_session`] with an optional int8 KV cache. The K and
    /// V per-head widths differ under MLA (nope+rope vs v_dim); both
    /// get per-(token, head) scales.
    pub fn new_session_opts(&self, max_seq: usize, kv_int8: bool) -> Session {
        let k_head = self.cfg.nope_dim + self.cfg.rope_dim;
        let int8 = kv_int8 && k_head.is_multiple_of(4) && self.cfg.v_dim.is_multiple_of(4);
        Session {
            k: self
                .blocks
                .iter()
                .map(|_| KvStore::alloc(&self.dev, max_seq, self.cfg.n_heads, k_head, int8))
                .collect(),
            v: self
                .blocks
                .iter()
                .map(|_| KvStore::alloc(&self.dev, max_seq, self.cfg.n_heads, self.cfg.v_dim, int8))
                .collect(),
            len: 0,
            max_seq,
        }
    }

    pub fn n_logits(&self) -> usize {
        self.lm_head.as_ref().map(Weight::out_features).unwrap_or(0)
    }

    pub fn forward(&self, session: &mut Session, tokens: &[u32]) -> Result<Vec<f32>> {
        match self.forward_stage(session, StageInput::Tokens(tokens), 1)? {
            StageOutput::Logits(l) => Ok(l),
            StageOutput::Hidden(_) => Err(WgpuError::Shape(
                "forward on a shard without output globals".into(),
            )),
        }
    }

    pub fn forward_stage(
        &self,
        session: &mut Session,
        input: StageInput<'_>,
        last_n: usize,
    ) -> Result<StageOutput> {
        let cfg = &self.cfg;
        let seq = match &input {
            StageInput::Tokens(t) => t.len(),
            StageInput::Hidden { seq, .. } => *seq,
        };
        if seq == 0 {
            return Err(WgpuError::Shape("empty input batch".into()));
        }
        if session.len + seq > session.max_seq {
            return Err(WgpuError::Shape(format!(
                "session overflow: {} + {seq} > {}",
                session.len, session.max_seq
            )));
        }
        let pos0 = session.len;
        let kv_len = pos0 + seq;
        let q_head = cfg.nope_dim + cfg.rope_dim;
        let k_row = cfg.n_heads * q_head;

        self.dev.begin_batch();
        let mut x = match input {
            StageInput::Tokens(tokens) => {
                let embed = self.embed.as_ref().ok_or_else(|| {
                    WgpuError::Shape("token input on a shard without input globals".into())
                })?;
                self.dev.upload(&embed.rows(tokens)?)
            }
            StageInput::Hidden { data, seq: s } => {
                if data.len() != s * cfg.hidden {
                    return Err(WgpuError::Shape("hidden input mismatch".into()));
                }
                self.dev.upload(data)
            }
        };

        for (li, blk) in self.blocks.iter().enumerate() {
            let h = self
                .dev
                .rms_norm(&x, &blk.attn_norm, seq, cfg.hidden, cfg.rms_eps)?;

            // Q path.
            let q = match (&blk.q_a_norm, &blk.q_b) {
                (Some(norm), Some(q_b)) => {
                    let a = blk.q.matmul_t(&self.dev, &h, seq)?;
                    let rank = blk.q.out_features();
                    let a = self.dev.rms_norm(&a, norm, seq, rank, cfg.rms_eps)?;
                    q_b.matmul_t(&self.dev, &a, seq)?
                }
                _ => blk.q.matmul_t(&self.dev, &h, seq)?,
            }; // [seq, heads*q_head]
               // Per-head splits ([seq*heads] rows of q_head).
            let q_nope = self
                .dev
                .slice_cols(&q, seq * cfg.n_heads, q_head, 0, cfg.nope_dim)?;
            let q_pe =
                self.dev
                    .slice_cols(&q, seq * cfg.n_heads, q_head, cfg.nope_dim, cfg.rope_dim)?;

            // Latent KV + single rope'd K head.
            let ckv_full = blk.kv_a.matmul_t(&self.dev, &h, seq)?; // [seq, lora+rope]
            let lora_rope = cfg.kv_lora_rank + cfg.rope_dim;
            let ckv = self
                .dev
                .slice_cols(&ckv_full, seq, lora_rope, 0, cfg.kv_lora_rank)?;
            let k_pe =
                self.dev
                    .slice_cols(&ckv_full, seq, lora_rope, cfg.kv_lora_rank, cfg.rope_dim)?;
            let ckv =
                self.dev
                    .rms_norm(&ckv, &blk.kv_a_norm, seq, cfg.kv_lora_rank, cfg.rms_eps)?;
            let kv = blk.kv_b.matmul_t(&self.dev, &ckv, seq)?; // [seq, heads*(nope+v)]
            let per_head = cfg.nope_dim + cfg.v_dim;
            let k_nope = self
                .dev
                .slice_cols(&kv, seq * cfg.n_heads, per_head, 0, cfg.nope_dim)?;
            let v =
                self.dev
                    .slice_cols(&kv, seq * cfg.n_heads, per_head, cfg.nope_dim, cfg.v_dim)?;

            // Interleaved YaRN rope on the PE parts (q: heads, k: 1 head).
            let q_pe = self.dev.rope_scaled(
                &q_pe,
                seq,
                cfg.n_heads,
                cfg.rope_dim,
                pos0,
                cfg.rope_theta,
                true,
                1.0,
                Some(&self.yarn_divisors),
                cfg.rope_dim,
            )?;
            let k_pe = self.dev.rope_scaled(
                &k_pe,
                seq,
                1,
                cfg.rope_dim,
                pos0,
                cfg.rope_theta,
                true,
                1.0,
                Some(&self.yarn_divisors),
                cfg.rope_dim,
            )?;

            // Assemble per-head K/Q = [noPE | PE] (K's PE head broadcast
            // to every head) and append this step's K/V rows.
            let q_full = self.dev.alloc(seq * k_row);
            self.dev.scatter_cols(
                &q_nope,
                &q_full,
                seq * cfg.n_heads,
                q_head,
                0,
                cfg.nope_dim,
                1,
            )?;
            self.dev.scatter_cols(
                &q_pe,
                &q_full,
                seq * cfg.n_heads,
                q_head,
                cfg.nope_dim,
                cfg.rope_dim,
                1,
            )?;
            let k_full = self.dev.alloc(seq * k_row);
            self.dev.scatter_cols(
                &k_nope,
                &k_full,
                seq * cfg.n_heads,
                q_head,
                0,
                cfg.nope_dim,
                1,
            )?;
            self.dev.scatter_cols(
                &k_pe,
                &k_full,
                seq * cfg.n_heads,
                q_head,
                cfg.nope_dim,
                cfg.rope_dim,
                cfg.n_heads,
            )?;
            session.k[li].append(&self.dev, &k_full, pos0, seq, cfg.n_heads, q_head)?;
            session.v[li].append(&self.dev, &v, pos0, seq, cfg.n_heads, cfg.v_dim)?;

            let scores = session.k[li].attn_scores(
                &self.dev,
                &q_full,
                seq,
                kv_len,
                cfg.n_heads,
                cfg.n_heads,
                q_head,
                pos0,
                cfg.softmax_scale,
                0,
            )?;
            let probs = self.dev.softmax(&scores, cfg.n_heads * seq, kv_len)?;
            let att = session.v[li].attn_out(
                &self.dev,
                &probs,
                seq,
                kv_len,
                cfg.n_heads,
                cfg.n_heads,
                cfg.v_dim,
            )?;
            let o = blk.o.matmul_t(&self.dev, &att, seq)?;
            x = self.dev.add(&x, &o)?;

            // FFN.
            let h2 = self
                .dev
                .rms_norm(&x, &blk.ffn_norm, seq, cfg.hidden, cfg.rms_eps)?;
            let d = match &blk.ffn {
                Ffn::Dense(dn) => {
                    let g = dn.gate.matmul_t(&self.dev, &h2, seq)?;
                    let u = dn.up.matmul_t(&self.dev, &h2, seq)?;
                    let gu = self.dev.silu_mul(&g, &u)?;
                    dn.down.matmul_t(&self.dev, &gu, seq)?
                }
                Ffn::Moe(m) => {
                    let logits = m.router.matmul_t(&self.dev, &h2, seq)?;
                    let routing = self.dev.moe_topk_opt(
                        &logits,
                        seq,
                        cfg.n_experts,
                        cfg.n_experts_used,
                        cfg.sigmoid_gating,
                        m.router_bias.as_ref(),
                        cfg.weights_norm,
                        cfg.n_group,
                        cfg.topk_group,
                        cfg.weights_scale,
                    )?;
                    let slots = cfg.n_experts_used;
                    let mut routed = match &m.experts {
                        ExpertStore::Resident(rx) => {
                            let ResidentExperts { gates, ups, downs } = rx.as_ref();
                            let g = expert_matmul(
                                &self.dev, gates, &h2, &routing, seq, slots, m.ffn_dim, cfg.hidden,
                                false,
                            )?;
                            let u = expert_matmul(
                                &self.dev, ups, &h2, &routing, seq, slots, m.ffn_dim, cfg.hidden,
                                false,
                            )?;
                            let gu = self.dev.silu_mul(&g, &u)?;
                            let dd = expert_matmul(
                                &self.dev, downs, &gu, &routing, seq, slots, cfg.hidden, m.ffn_dim,
                                true,
                            )?;
                            self.dev
                                .moe_combine(&dd, &routing, seq, slots, cfg.hidden)?
                        }
                        ExpertStore::Host(hx) => {
                            let table = self.dev.download(&routing)?;
                            let h2v = self.dev.download(&h2)?;
                            let out = host_moe_forward(hx, &table, &h2v, seq, cfg.hidden, slots)
                                .map_err(|e| WgpuError::Device(format!("cpu_moe forward: {e}")))?;
                            self.dev.upload(&out)
                        }
                    };
                    if let Some(sh) = &m.shared {
                        let g = sh.gate.matmul_t(&self.dev, &h2, seq)?;
                        let u = sh.up.matmul_t(&self.dev, &h2, seq)?;
                        let gu = self.dev.silu_mul(&g, &u)?;
                        let sd = sh.down.matmul_t(&self.dev, &gu, seq)?;
                        routed = self.dev.add(&routed, &sd)?;
                    }
                    routed
                }
            };
            x = self.dev.add(&x, &d)?;
        }
        session.len = kv_len;

        let (Some(out_norm), Some(lm_head)) = (&self.out_norm, &self.lm_head) else {
            return Ok(StageOutput::Hidden(self.dev.download(&x)?));
        };
        let h = self
            .dev
            .rms_norm(&x, out_norm, seq, cfg.hidden, cfg.rms_eps)?;
        let logits = lm_head.matmul_t(&self.dev, &h, seq)?;
        if last_n == 0 || last_n > seq {
            return Err(WgpuError::Shape("last_n out of range".into()));
        }
        let all = self.dev.download(&logits)?;
        let n_out = lm_head.out_features();
        Ok(StageOutput::Logits(
            all[(seq - last_n) * n_out..seq * n_out].to_vec(),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn expert_matmul(
    dev: &WgpuDevice,
    w: &Weight,
    x: &GpuBuffer,
    routing: &GpuBuffer,
    m: usize,
    slots: usize,
    rows_per_expert: usize,
    k: usize,
    x_per_slot: bool,
) -> Result<GpuBuffer> {
    match w {
        Weight::Quant(q) => {
            dev.matmul_expert(x, q, routing, m, slots, rows_per_expert, k, x_per_slot)
        }
        Weight::F32 { buf, .. } => {
            dev.matmul_expert_f32(x, buf, routing, m, slots, rows_per_expert, k, x_per_slot)
        }
    }
}
