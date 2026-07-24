//! Gemma-family forward pass on wgpu — gemma 1/2/3/4, mirroring the
//! CUDA backend's `backend_callosum/gemma.rs` op-for-op so the two
//! backends produce identical greedy tokens:
//!
//! - embed scaled by sqrt(hidden); rotate-half RoPE.
//! - Up to four norms per block (pre/post attention, pre/post FFN) —
//!   all plain-mul RMSNorm (the GGUF weights carry the `1 + w` form).
//! - GELU (tanh approximation) FFN.
//! - gemma 2: attention + final logit soft-capping (tanh(x/c)·c).
//! - gemma 3/4: per-head Q/K RMSNorm; two RoPE bases selected by a
//!   block's head_dim matching `attention.key_length_swa`; optional
//!   `rope_freqs.weight` divisors on global layers.
//! - gemma 3n/4: AltUp per-layer input pipeline + per-block branch,
//!   shared-KV tail layers (reuse an earlier layer's cache), attention
//!   scale 1.0, weightless V RMSNorm (eps 1e-6).
//!
//! The per-layer token-embedding table stays **quantized in host RAM**
//! and is row-dequantized per forward (fully dequantized it is ~9 GB —
//! larger than most adapters' max binding size, and pure waste for an
//! input-side lookup).
//!
//! Deliberately mirrored CUDA-backend approximations: no sliding-window
//! score masking, and SWA-vs-global classification only where head_dim
//! differs (gemma 4). Both backends share these, so cross-backend token
//! parity holds; both inherit the same long-context caveat.

use callosum::quantized::{gguf_file, GgmlDType};

use crate::llama::{HostRowTable, StageInput, StageOutput};
use crate::{GpuBuffer, QuantBuffer, QuantDtype, Result, WgpuDevice, WgpuError};

pub struct GemmaConfig {
    pub arch: String,
    pub hidden: usize,
    pub n_layers: usize,
    pub layer_start: usize,
    pub layer_end: usize,
    pub n_heads: usize,
    pub vocab: usize,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub key_length_swa: Option<usize>,
    pub final_softcap: Option<f32>,
    pub attn_softcap: Option<f32>,
    pub rms_eps: f32,
    /// gemma 3n/4: layers >= this index reuse an earlier layer's K/V.
    pub n_layer_kv_from_start: Option<usize>,
    /// gemma 2/3 sliding-window size.
    pub sliding_window: Option<usize>,
    /// Layer il is local (sliding) when (il+1) % pattern > 0.
    pub sliding_window_pattern: u32,
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
        GgmlDType::Q8_0 => Some(QuantDtype::Q8_0),
        GgmlDType::Q4K => Some(QuantDtype::Q4K),
        GgmlDType::Q5K => Some(QuantDtype::Q5K),
        GgmlDType::Q6K => Some(QuantDtype::Q6K),
        _ => None,
    }
}

struct AltUp {
    /// [hidden_per_layer, hidden] gate projection.
    inp_gate: Weight,
    /// [hidden, hidden_per_layer] output projection.
    proj: Weight,
    /// Plain RMSNorm weight, len hidden.
    post_norm: GpuBuffer,
    /// Per-channel output scale, len hidden.
    out_scale: GpuBuffer,
}

struct Block {
    /// (head_dim, n_kv_heads) — per block: gemma 4 mixes head dims.
    head_dim: usize,
    n_kv_heads: usize,
    theta: f32,
    /// Divide RoPE inv-freqs by the global freq table (non-SWA layers).
    use_freqs: bool,
    /// Sliding-window size on local layers (0 = full attention).
    window: usize,
    pre_attn_norm: GpuBuffer,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    q_norm: Option<GpuBuffer>,
    k_norm: Option<GpuBuffer>,
    post_attn_norm: Option<GpuBuffer>,
    pre_ffn_norm: GpuBuffer,
    gate: Weight,
    up: Weight,
    down: Weight,
    post_ffn_norm: Option<GpuBuffer>,
    altup: Option<AltUp>,
}

struct AltUpGlobals {
    table: HostRowTable,
    /// [n_layers * hidden_per_layer, hidden].
    model_proj: Weight,
    /// Plain RMSNorm weight, len hidden_per_layer.
    proj_norm: GpuBuffer,
    hidden_per_layer: usize,
}

pub struct WgpuGemma {
    dev: WgpuDevice,
    pub cfg: GemmaConfig,
    embed: Option<HostRowTable>,
    /// Scalar constants as 1-element buffers (mul_bias with a length-1
    /// bias is a broadcast scalar multiply).
    embed_scale: GpuBuffer,
    inv_sqrt2: GpuBuffer,
    proj_scale: GpuBuffer,
    per_layer_scale: GpuBuffer,
    /// Ones per distinct head_dim — weight for the weightless V
    /// RMSNorm (gemma 4 mixes head dims across blocks).
    ones_head: std::collections::HashMap<usize, GpuBuffer>,
    rope_freqs: Option<GpuBuffer>,
    altup_globals: Option<AltUpGlobals>,
    blocks: Vec<Block>,
    out_norm: Option<GpuBuffer>,
    lm_head: Option<Weight>,
    pub weight_bytes: u64,
}

pub struct Session {
    k: Vec<GpuBuffer>,
    v: Vec<GpuBuffer>,
    pub len: usize,
    max_seq: usize,
}

fn meta_u32(c: &gguf_file::Content, key: &str) -> Option<u32> {
    c.metadata.get(key).and_then(|v| v.to_u32().ok())
}

fn meta_f32(c: &gguf_file::Content, key: &str) -> Option<f32> {
    c.metadata.get(key).and_then(|v| v.to_f32().ok())
}

impl WgpuGemma {
    pub fn from_gguf(path: &std::path::Path, dev: &WgpuDevice) -> Result<Self> {
        Self::from_gguf_stage(path, dev, 0, usize::MAX, true, true)
    }

    pub fn from_gguf_stage(
        path: &std::path::Path,
        dev: &WgpuDevice,
        layer_start: usize,
        layer_end: usize,
        owns_input: bool,
        owns_output: bool,
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
        if !matches!(arch.as_str(), "gemma" | "gemma2" | "gemma3" | "gemma4") {
            return Err(WgpuError::Device(format!(
                "callosum-wgpu gemma loader supports gemma/gemma2/gemma3/gemma4 (got {arch:?})"
            )));
        }
        let key = |s: &str| format!("{arch}.{s}");

        let hidden = meta_u32(&content, &key("embedding_length"))
            .ok_or_else(|| WgpuError::Device("missing embedding_length".into()))?
            as usize;
        let n_layers = meta_u32(&content, &key("block_count"))
            .ok_or_else(|| WgpuError::Device("missing block_count".into()))?
            as usize;
        let n_heads = meta_u32(&content, &key("attention.head_count"))
            .ok_or_else(|| WgpuError::Device("missing head_count".into()))?
            as usize;
        let rope_base = meta_f32(&content, &key("rope.freq_base")).unwrap_or(10_000.0);
        // Two GGUF spellings for the local-layer base: gemma 4 uses
        // freq_base_swa, gemma 3 local_freq_base.
        // Local-layer base: freq_base_swa (gemma 4) or local_freq_base
        // (gemma 3); when absent the convention (llama.cpp and the CPU
        // reference alike) is 10k for local layers, NOT the global base.
        let rope_base_swa = meta_f32(&content, &key("rope.freq_base_swa"))
            .or_else(|| meta_f32(&content, &key("rope.local_freq_base")))
            .unwrap_or(10_000.0);
        let sliding_window =
            meta_u32(&content, &key("attention.sliding_window")).map(|v| v as usize);
        let sliding_window_pattern = meta_u32(&content, &key("attention.sliding_window_type"))
            .unwrap_or(if arch == "gemma2" { 2 } else { 6 });
        let key_length_swa =
            meta_u32(&content, &key("attention.key_length_swa")).map(|v| v as usize);
        let final_softcap = meta_f32(&content, &key("final_logit_softcapping"));
        let attn_softcap = meta_f32(&content, &key("attn_logit_softcapping"));
        let rms_eps = meta_f32(&content, &key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        let n_layer_kv_from_start = meta_u32(&content, &key("attention.shared_kv_layers"))
            .map(|shared| n_layers - shared as usize);

        let layer_end = layer_end.min(n_layers);
        if layer_start >= layer_end {
            return Err(WgpuError::Shape(format!(
                "empty layer range [{layer_start},{layer_end}) of {n_layers}"
            )));
        }
        // Shared-KV tail layers must be co-resident with their source
        // layers; the planner keeps gemma 4 single-stage (same rule as
        // the CUDA backend).
        if let Some(nk) = n_layer_kv_from_start {
            if layer_end > nk && layer_start > nk.saturating_sub(2) {
                return Err(WgpuError::Shape(
                    "shared-KV tail layers split from their source layers — gemma 4 \
                     requires a single-stage placement"
                        .into(),
                ));
            }
        }

        let cpu = callosum::Device::Cpu;
        let weight_bytes = std::cell::Cell::new(0u64);
        let mut load_f32 = |name: &str| -> Result<(GpuBuffer, Vec<usize>)> {
            let qt = content
                .tensor(&mut file, name, &cpu)
                .map_err(|e| WgpuError::Device(format!("load {name}: {e}")))?;
            let dims = qt.shape().dims().to_vec();
            let t = qt
                .dequantize(&cpu)
                .and_then(|t| t.to_dtype(callosum::DType::F32))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| WgpuError::Device(format!("dequantize {name}: {e}")))?;
            weight_bytes.set(weight_bytes.get() + (t.len() * 4) as u64);
            Ok((dev.upload(&t), dims))
        };

        let mut file2 =
            std::fs::File::open(path).map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
        let mut load_weight = |name: &str| -> Result<Weight> {
            let qt = content
                .tensor(&mut file2, name, &cpu)
                .map_err(|e| WgpuError::Device(format!("load {name}: {e}")))?;
            let dims = qt.shape().dims().to_vec();
            if dims.len() != 2 {
                return Err(WgpuError::Shape(format!("{name}: expected rank-2")));
            }
            let (n, k) = (dims[0], dims[1]);
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
        let shape_of = |name: &str| -> Option<Vec<usize>> {
            content
                .tensor_infos
                .get(name)
                .map(|i| i.shape.dims().to_vec())
        };

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

        let rope_freqs = if has("rope_freqs.weight") {
            Some(load_f32("rope_freqs.weight")?.0)
        } else {
            None
        };

        // AltUp globals — input-stage only (gemma 3n/4).
        let altup_globals = if owns_input
            && has("per_layer_token_embd.weight")
            && has("per_layer_proj_norm.weight")
            && has("per_layer_model_proj.weight")
        {
            let mut file3 = std::fs::File::open(path)
                .map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
            let qt = content
                .tensor(&mut file3, "per_layer_token_embd.weight", &cpu)
                .map_err(|e| WgpuError::Device(format!("load per_layer_token_embd: {e}")))?;
            let table = HostRowTable::from_qtensor(&qt)?;
            if table.cols % n_layers != 0 {
                return Err(WgpuError::Shape(format!(
                    "per_layer_token_embd width {} not divisible by n_layers {n_layers}",
                    table.cols
                )));
            }
            let hidden_per_layer = table.cols / n_layers;
            let (proj_norm, _) = load_f32("per_layer_proj_norm.weight")?;
            let model_proj = load_weight("per_layer_model_proj.weight")?;
            Some(AltUpGlobals {
                table,
                model_proj,
                proj_norm,
                hidden_per_layer,
            })
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(layer_end - layer_start);
        let mut head_dims: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for b in layer_start..layer_end {
            // Per-block dims from the Q/K weight shapes (gemma 4 mixes
            // head dims across layers).
            let q_shape = shape_of(&format!("blk.{b}.attn_q.weight"))
                .ok_or_else(|| WgpuError::Device(format!("blk.{b}.attn_q.weight missing")))?;
            let k_shape = shape_of(&format!("blk.{b}.attn_k.weight"))
                .ok_or_else(|| WgpuError::Device(format!("blk.{b}.attn_k.weight missing")))?;
            let head_dim = q_shape[0] / n_heads;
            let n_kv_heads = k_shape[0] / head_dim;
            head_dims.insert(head_dim);
            let is_local = if let Some(swa_hd) = key_length_swa {
                head_dim == swa_hd
            } else {
                sliding_window.is_some() && (b + 1) as u32 % sliding_window_pattern > 0
            };
            let theta = if is_local { rope_base_swa } else { rope_base };
            // Freq table length is head_dim_global/2 — only attach where
            // it fits (gemma 4 mixes head dims across layers).
            let use_freqs = !is_local && rope_freqs.as_ref().is_some_and(|f| f.len == head_dim / 2);
            let window = if is_local {
                sliding_window.unwrap_or(0)
            } else {
                0
            };

            let mut opt_f32 = |name: String| -> Result<Option<GpuBuffer>> {
                if has(&name) {
                    load_f32(&name).map(|(buf, _)| Some(buf))
                } else {
                    Ok(None)
                }
            };
            let q_norm = opt_f32(format!("blk.{b}.attn_q_norm.weight"))?;
            let k_norm = opt_f32(format!("blk.{b}.attn_k_norm.weight"))?;
            let post_attn_norm = opt_f32(format!("blk.{b}.post_attention_norm.weight"))?;
            let post_ffn_norm = opt_f32(format!("blk.{b}.post_ffw_norm.weight"))?;

            let altup = if has(&format!("blk.{b}.inp_gate.weight"))
                && has(&format!("blk.{b}.proj.weight"))
                && has(&format!("blk.{b}.post_norm.weight"))
                && has(&format!("blk.{b}.layer_output_scale.weight"))
            {
                Some(AltUp {
                    inp_gate: load_weight(&format!("blk.{b}.inp_gate.weight"))?,
                    proj: load_weight(&format!("blk.{b}.proj.weight"))?,
                    post_norm: load_f32(&format!("blk.{b}.post_norm.weight"))?.0,
                    out_scale: load_f32(&format!("blk.{b}.layer_output_scale.weight"))?.0,
                })
            } else {
                None
            };

            blocks.push(Block {
                head_dim,
                n_kv_heads,
                theta,
                use_freqs,
                window,
                pre_attn_norm: load_f32(&format!("blk.{b}.attn_norm.weight"))?.0,
                wq: load_weight(&format!("blk.{b}.attn_q.weight"))?,
                wk: load_weight(&format!("blk.{b}.attn_k.weight"))?,
                wv: load_weight(&format!("blk.{b}.attn_v.weight"))?,
                wo: load_weight(&format!("blk.{b}.attn_output.weight"))?,
                q_norm,
                k_norm,
                post_attn_norm,
                pre_ffn_norm: load_f32(&format!("blk.{b}.ffn_norm.weight"))?.0,
                gate: load_weight(&format!("blk.{b}.ffn_gate.weight"))?,
                up: load_weight(&format!("blk.{b}.ffn_up.weight"))?,
                down: load_weight(&format!("blk.{b}.ffn_down.weight"))?,
                post_ffn_norm,
                altup,
            });
        }

        let mut out_norm = None;
        let mut lm_head = None;
        if owns_output {
            out_norm = Some(load_f32("output_norm.weight")?.0);
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

        let hpl = altup_globals
            .as_ref()
            .map(|g| g.hidden_per_layer)
            .unwrap_or(1);
        Ok(Self {
            dev: dev.clone(),
            cfg: GemmaConfig {
                arch,
                hidden,
                n_layers,
                layer_start,
                layer_end,
                n_heads,
                vocab,
                rope_base,
                rope_base_swa,
                key_length_swa,
                final_softcap,
                attn_softcap,
                rms_eps,
                n_layer_kv_from_start,
                sliding_window,
                sliding_window_pattern,
            },
            embed,
            embed_scale: dev.upload(&[(hidden as f32).sqrt()]),
            inv_sqrt2: dev.upload(&[1.0 / 2f32.sqrt()]),
            proj_scale: dev.upload(&[1.0 / (hidden as f32).sqrt()]),
            per_layer_scale: dev.upload(&[(hpl as f32).sqrt()]),
            ones_head: head_dims
                .into_iter()
                .map(|hd| (hd, dev.upload(&vec![1.0f32; hd])))
                .collect(),
            rope_freqs,
            altup_globals,
            blocks,
            out_norm,
            lm_head,
            weight_bytes: weight_bytes.get(),
        })
    }

    pub fn new_session(&self, max_seq: usize) -> Session {
        Session {
            k: self
                .blocks
                .iter()
                .map(|b| self.dev.alloc(max_seq * b.n_kv_heads * b.head_dim))
                .collect(),
            v: self
                .blocks
                .iter()
                .map(|b| self.dev.alloc(max_seq * b.n_kv_heads * b.head_dim))
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
                "forward on a shard without output globals; use forward_stage".into(),
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

        // Host-side AltUp table rows must be gathered before the batch
        // opens (host work interleaved with recording is fine, but do
        // it up front for clarity).
        let altup_rows: Option<Vec<f32>> = match (&input, &self.altup_globals) {
            (StageInput::Tokens(t), Some(g)) => Some(g.table.rows(t)?),
            _ => None,
        };

        self.dev.begin_batch();
        let (mut x, per_layer_all) = match input {
            StageInput::Tokens(tokens) => {
                let embed = self.embed.as_ref().ok_or_else(|| {
                    WgpuError::Shape("token input on a shard without input globals".into())
                })?;
                let h = self.dev.upload(&embed.rows(tokens)?);
                // Gemma scales the embedding by sqrt(hidden).
                let h = self.dev.mul_bias(&h, &self.embed_scale)?;

                // AltUp per-layer inputs (gemma 3n/4):
                //   mixed = (RMSNorm(model_proj·h · 1/sqrt(hidden)) +
                //            table[ids]·sqrt(hpl)) · 1/sqrt(2)
                let per_layer_all = if let Some(g) = &self.altup_globals {
                    let hpl = g.hidden_per_layer;
                    let rows = self.dev.upload(altup_rows.as_ref().unwrap());
                    let emb_slice = self.dev.mul_bias(&rows, &self.per_layer_scale)?;
                    let proj = g.model_proj.matmul_t(&self.dev, &h, seq)?;
                    let proj = self.dev.mul_bias(&proj, &self.proj_scale)?;
                    let proj = self.dev.rms_norm(
                        &proj,
                        &g.proj_norm,
                        seq * cfg.n_layers,
                        hpl,
                        cfg.rms_eps,
                    )?;
                    let sum = self.dev.add(&proj, &emb_slice)?;
                    Some(self.dev.mul_bias(&sum, &self.inv_sqrt2)?)
                } else {
                    None
                };
                (h, per_layer_all)
            }
            StageInput::Hidden { data, seq: s } => {
                if data.len() != s * cfg.hidden {
                    return Err(WgpuError::Shape(format!(
                        "hidden input {} != seq {s} × hidden {}",
                        data.len(),
                        cfg.hidden
                    )));
                }
                (self.dev.upload(data), None)
            }
        };

        // Shared-KV source mapping (gemma 3n/4): SWA tail layers reuse
        // layer nk-2, global tail layers reuse nk-1. Sources always
        // precede the tail in the same stage (checked at load).
        let shared_swa_src = cfg.n_layer_kv_from_start.map(|n| n.saturating_sub(2));
        let shared_global_src = cfg.n_layer_kv_from_start.map(|n| n.saturating_sub(1));
        // gemma 3n/4 pin attention scale to 1.0; gemma 1/2/3 use the
        // standard rsqrt(head_dim). Same discriminator as CUDA.
        let scale_is_one = cfg.n_layer_kv_from_start.is_some();

        for (li, blk) in self.blocks.iter().enumerate() {
            let b_abs = cfg.layer_start + li;
            let hd = blk.head_dim;
            let n_kv = blk.n_kv_heads;
            let kv_row = n_kv * hd;
            let freqs = if blk.use_freqs {
                self.rope_freqs.as_ref()
            } else {
                None
            };

            let h = self
                .dev
                .rms_norm(&x, &blk.pre_attn_norm, seq, cfg.hidden, cfg.rms_eps)?;
            let q = blk.wq.matmul_t(&self.dev, &h, seq)?;
            let q = match &blk.q_norm {
                Some(w) => self
                    .dev
                    .rms_norm(&q, w, seq * cfg.n_heads, hd, cfg.rms_eps)?,
                None => q,
            };
            let q = self.dev.rope_scaled(
                &q,
                seq,
                cfg.n_heads,
                hd,
                pos0,
                blk.theta,
                false,
                1.0,
                freqs,
            )?;

            let reuses = matches!(cfg.n_layer_kv_from_start, Some(nk) if b_abs >= nk);
            let (k_buf, v_buf): (&GpuBuffer, &GpuBuffer) = if reuses {
                let is_swa = cfg.key_length_swa == Some(hd);
                let src_abs = if is_swa {
                    shared_swa_src.unwrap()
                } else {
                    shared_global_src.unwrap()
                };
                if src_abs < cfg.layer_start {
                    return Err(WgpuError::Shape(format!(
                        "block {b_abs}: shared-KV source layer {src_abs} not in this shard"
                    )));
                }
                let src_local = src_abs - cfg.layer_start;
                (&session.k[src_local], &session.v[src_local])
            } else {
                let k = blk.wk.matmul_t(&self.dev, &h, seq)?;
                let v = blk.wv.matmul_t(&self.dev, &h, seq)?;
                let k = match &blk.k_norm {
                    Some(w) => self.dev.rms_norm(&k, w, seq * n_kv, hd, cfg.rms_eps)?,
                    None => k,
                };
                // Weightless V RMSNorm when Q/K norm is active
                // (gemma 3n/4), eps 1e-6 — llama.cpp's extra
                // ggml_rms_norm(Vcur) line.
                let v = if blk.q_norm.is_some() && scale_is_one {
                    let ones = self.ones_head.get(&hd).ok_or_else(|| {
                        WgpuError::Shape(format!("no ones buffer for head_dim {hd}"))
                    })?;
                    self.dev.rms_norm(&v, ones, seq * n_kv, hd, 1e-6)?
                } else {
                    v
                };
                let k = self
                    .dev
                    .rope_scaled(&k, seq, n_kv, hd, pos0, blk.theta, false, 1.0, freqs)?;
                self.dev.copy_rows(&k, &session.k[li], pos0, seq, kv_row)?;
                self.dev.copy_rows(&v, &session.v[li], pos0, seq, kv_row)?;
                (&session.k[li], &session.v[li])
            };

            let scale = if scale_is_one {
                1.0
            } else {
                1.0 / (hd as f32).sqrt()
            };
            let scores = self.dev.attn_scores_opt(
                &q,
                k_buf,
                seq,
                kv_len,
                cfg.n_heads,
                n_kv,
                hd,
                pos0,
                scale,
                blk.window,
            )?;
            // gemma 2 soft-caps attention scores. The kernel's causal
            // mask writes -3e38 which tanh squashes to -cap — still the
            // row minimum, and exp(-cap - max) vanishes, so masking
            // survives the cap (same net behavior as CUDA's mask-after-
            // cap ordering).
            let scores = match cfg.attn_softcap {
                Some(cap) => self.dev.softcap(&scores, cap)?,
                None => scores,
            };
            let probs = self.dev.softmax(&scores, cfg.n_heads * seq, kv_len)?;
            let att = self
                .dev
                .attn_out(&probs, v_buf, seq, kv_len, cfg.n_heads, n_kv, hd)?;
            let o = blk.wo.matmul_t(&self.dev, &att, seq)?;
            let o = match &blk.post_attn_norm {
                Some(w) => self.dev.rms_norm(&o, w, seq, cfg.hidden, cfg.rms_eps)?,
                None => o,
            };
            x = self.dev.add(&x, &o)?;

            let h2 = self
                .dev
                .rms_norm(&x, &blk.pre_ffn_norm, seq, cfg.hidden, cfg.rms_eps)?;
            let g = blk.gate.matmul_t(&self.dev, &h2, seq)?;
            let u = blk.up.matmul_t(&self.dev, &h2, seq)?;
            let gu = self.dev.gelu_mul(&g, &u)?;
            let d = blk.down.matmul_t(&self.dev, &gu, seq)?;
            let d = match &blk.post_ffn_norm {
                Some(w) => self.dev.rms_norm(&d, w, seq, cfg.hidden, cfg.rms_eps)?,
                None => d,
            };
            let h_after_ffn = self.dev.add(&x, &d)?;

            // AltUp per-block branch (gemma 3n/4).
            x = match (&blk.altup, &per_layer_all) {
                (Some(alt), Some(all)) => {
                    let hpl = self
                        .altup_globals
                        .as_ref()
                        .map(|g| g.hidden_per_layer)
                        .unwrap_or(0);
                    let slice =
                        self.dev
                            .slice_cols(all, seq, cfg.n_layers * hpl, b_abs * hpl, hpl)?;
                    let cur = alt.inp_gate.matmul_t(&self.dev, &h_after_ffn, seq)?;
                    let cur = self.dev.gelu(&cur)?;
                    let cur = self.dev.mul(&cur, &slice)?;
                    let cur = alt.proj.matmul_t(&self.dev, &cur, seq)?;
                    let cur =
                        self.dev
                            .rms_norm(&cur, &alt.post_norm, seq, cfg.hidden, cfg.rms_eps)?;
                    let cur = self.dev.add(&h_after_ffn, &cur)?;
                    self.dev.mul_bias(&cur, &alt.out_scale)?
                }
                _ => h_after_ffn,
            };
        }
        session.len = kv_len;

        let (Some(out_norm), Some(lm_head)) = (&self.out_norm, &self.lm_head) else {
            return Ok(StageOutput::Hidden(self.dev.download(&x)?));
        };
        let h = self
            .dev
            .rms_norm(&x, out_norm, seq, cfg.hidden, cfg.rms_eps)?;
        let logits = lm_head.matmul_t(&self.dev, &h, seq)?;
        let logits = match cfg.final_softcap {
            Some(cap) => self.dev.softcap(&logits, cap)?,
            None => logits,
        };
        if last_n == 0 || last_n > seq {
            return Err(WgpuError::Shape(format!(
                "last_n {last_n} out of range for batch of {seq}"
            )));
        }
        let all = self.dev.download(&logits)?;
        let n_out = lm_head.out_features();
        Ok(StageOutput::Logits(
            all[(seq - last_n) * n_out..seq * n_out].to_vec(),
        ))
    }
}
