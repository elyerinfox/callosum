//! Llama-family forward pass running entirely on wgpu — the "model
//! loads on an Arc card" milestone.
//!
//! Loads a GGUF via callosum's reader, keeps supported quant formats
//! **quantized on the GPU** (dequant-in-shader matmul), dequantizes any
//! other dtype to f32 as a correctness fallback, and runs the full
//! pipeline — embed → [rmsnorm → QKV (+bias) → optional per-head Q/K
//! rmsnorm → RoPE → causal GQA attention (KV-cached) → O → residual →
//! rmsnorm → SwiGLU → residual] × L → final norm → logits — with every
//! FLOP on the wgpu adapter. Vulkan/DX12/Metal, so Intel Arc, AMD,
//! NVIDIA and Apple run this identical code path.
//!
//! Supported architectures: `llama`/`mistral` (interleaved RoPE) and
//! `qwen2`/`qwen3` (rotate-half RoPE; qwen2 adds QKV biases, qwen3 adds
//! per-head Q/K RMSNorm) — the same family the CUDA backend's
//! llama-family loader accepts.
//!
//! Pipeline-parallel shards load a **layer range** via
//! [`WgpuLlama::from_gguf_stage`]: a stage that owns the input globals
//! embeds tokens, any other stage consumes a hidden-state matrix, and
//! only the stage owning the output globals holds the final norm +
//! lm_head. [`WgpuLlama::forward_stage`] mirrors this: tokens or hidden
//! in, logits or hidden out.
//!
//! Correctness anchor: `tests/llama_parity.rs` compares generated
//! tokens against callosum-models' CPU `quantized_llama` on a real
//! GGUF; `stage_parity` there splits the same model in two and demands
//! the identical token stream.

use callosum::quantized::{gguf_file, GgmlDType};

use crate::{GpuBuffer, QuantBuffer, QuantDtype, Result, WgpuDevice, WgpuError};

pub struct LlamaConfig {
    pub arch: String,
    pub hidden: usize,
    /// Total layer count of the model (GGUF metadata), not the local
    /// slice — see `layer_start`/`layer_end` for what this shard runs.
    pub n_layers: usize,
    pub layer_start: usize,
    pub layer_end: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// Llama/Mistral/SmolLM2 use interleaved RoPE pairs; qwen-style
    /// GGUFs set false (rotate-half).
    pub rope_interleaved: bool,
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

/// GGML dtypes with in-shader kernels; anything else dequantizes to
/// f32 at load (correct, memory-expensive).
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

struct Block {
    attn_norm: GpuBuffer,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    /// qwen2-style QKV biases (row-broadcast after the matmul).
    bq: Option<GpuBuffer>,
    bk: Option<GpuBuffer>,
    bv: Option<GpuBuffer>,
    /// qwen3-style per-head Q/K RMSNorm weights (len = head_dim),
    /// applied between the projection and RoPE.
    q_norm: Option<GpuBuffer>,
    k_norm: Option<GpuBuffer>,
    ffn_norm: GpuBuffer,
    gate: Weight,
    up: Weight,
    down: Weight,
}

pub struct WgpuLlama {
    dev: WgpuDevice,
    pub cfg: LlamaConfig,
    /// f32 [vocab, hidden] gather source — present only on shards that
    /// own the input globals.
    embed: Option<GpuBuffer>,
    blocks: Vec<Block>,
    /// Final norm + lm_head — present only on shards that own the
    /// output globals.
    out_norm: Option<GpuBuffer>,
    lm_head: Option<Weight>,
    /// Total bytes uploaded for weights (quantized at on-disk density,
    /// f32 where a format fell back). What a serving layer should
    /// report as resident.
    pub weight_bytes: u64,
}

/// Per-conversation KV state: one K and one V buffer per **local**
/// layer, laid out [max_seq, n_kv_heads, head_dim], appended in place.
pub struct Session {
    k: Vec<GpuBuffer>,
    v: Vec<GpuBuffer>,
    pub len: usize,
    max_seq: usize,
}

/// What a pipeline stage receives: tokens on the stage owning the
/// input globals, the previous stage's hidden matrix everywhere else.
pub enum StageInput<'a> {
    Tokens(&'a [u32]),
    /// Row-major [seq, hidden] activations.
    Hidden {
        data: &'a [f32],
        seq: usize,
    },
}

/// What a stage produces: logits rows on the stage owning the output
/// globals, the residual-stream hidden matrix [seq, hidden] otherwise.
pub enum StageOutput {
    Logits(Vec<f32>),
    Hidden(Vec<f32>),
}

fn meta_u32(c: &gguf_file::Content, keys: &[&str]) -> Option<u32> {
    for k in keys {
        if let Some(v) = c.metadata.get(*k) {
            if let Ok(x) = v.to_u32() {
                return Some(x);
            }
        }
    }
    None
}

fn meta_f32(c: &gguf_file::Content, keys: &[&str]) -> Option<f32> {
    for k in keys {
        if let Some(v) = c.metadata.get(*k) {
            if let Ok(x) = v.to_f32() {
                return Some(x);
            }
        }
    }
    None
}

impl WgpuLlama {
    /// Load the whole model: input embedding through lm_head.
    pub fn from_gguf(path: &std::path::Path, dev: &WgpuDevice) -> Result<Self> {
        Self::from_gguf_stage(path, dev, 0, usize::MAX, true, true)
    }

    /// Load layers `[layer_start, layer_end)` (end clamps to the model's
    /// layer count; `usize::MAX` means "through the last layer").
    /// `owns_input` additionally loads the token embedding, `owns_output`
    /// the final norm + lm_head — matching splitbrain's shard-assignment
    /// globals flags.
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
            .unwrap_or_else(|| "llama".to_string());
        if !matches!(arch.as_str(), "llama" | "mistral" | "qwen2" | "qwen3") {
            return Err(WgpuError::Device(format!(
                "callosum-wgpu llama loader supports llama/mistral/qwen2/qwen3 (got {arch:?})"
            )));
        }
        // Interleaved pairs (2i, 2i+1) for llama-lineage GGUFs,
        // rotate-half (i, i+d/2) for the qwen family.
        let rope_interleaved = matches!(arch.as_str(), "llama" | "mistral");
        let key = |suffix: &str| format!("{arch}.{suffix}");

        let hidden = meta_u32(&content, &[&key("embedding_length")])
            .ok_or_else(|| WgpuError::Device("missing embedding_length".into()))?
            as usize;
        let n_layers = meta_u32(&content, &[&key("block_count")])
            .ok_or_else(|| WgpuError::Device("missing block_count".into()))?
            as usize;
        let n_heads = meta_u32(&content, &[&key("attention.head_count")])
            .ok_or_else(|| WgpuError::Device("missing head_count".into()))?
            as usize;
        let n_kv_heads = meta_u32(&content, &[&key("attention.head_count_kv")])
            .map(|v| v as usize)
            .unwrap_or(n_heads);
        let head_dim = meta_u32(&content, &[&key("attention.key_length")])
            .map(|v| v as usize)
            .unwrap_or(hidden / n_heads);
        let rope_theta = meta_f32(&content, &[&key("rope.freq_base")]).unwrap_or(10_000.0);
        let rms_eps =
            meta_f32(&content, &[&key("attention.layer_norm_rms_epsilon")]).unwrap_or(1e-5);

        let layer_end = layer_end.min(n_layers);
        if layer_start >= layer_end {
            return Err(WgpuError::Shape(format!(
                "empty layer range [{layer_start},{layer_end}) of {n_layers}"
            )));
        }

        // Cheap CPU device for dequantizing non-quant-kernel tensors.
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

        let mut embed = None;
        let mut vocab = 0usize;
        if owns_input {
            let (e, dims) = load_f32("token_embd.weight")?;
            vocab = dims[0];
            embed = Some(e);
        }

        // Weight loader: supported quant formats stay quantized on the
        // GPU, everything else falls back to dequantized f32.
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
        let mut blocks = Vec::with_capacity(layer_end - layer_start);
        for b in layer_start..layer_end {
            let (attn_norm, _) = load_f32(&format!("blk.{b}.attn_norm.weight"))?;
            let (ffn_norm, _) = load_f32(&format!("blk.{b}.ffn_norm.weight"))?;
            let mut opt_f32 = |name: String| -> Result<Option<GpuBuffer>> {
                if has(&name) {
                    load_f32(&name).map(|(buf, _)| Some(buf))
                } else {
                    Ok(None)
                }
            };
            let bq = opt_f32(format!("blk.{b}.attn_q.bias"))?;
            let bk = opt_f32(format!("blk.{b}.attn_k.bias"))?;
            let bv = opt_f32(format!("blk.{b}.attn_v.bias"))?;
            let q_norm = opt_f32(format!("blk.{b}.attn_q_norm.weight"))?;
            let k_norm = opt_f32(format!("blk.{b}.attn_k_norm.weight"))?;
            blocks.push(Block {
                attn_norm,
                wq: load_weight(&format!("blk.{b}.attn_q.weight"))?,
                wk: load_weight(&format!("blk.{b}.attn_k.weight"))?,
                wv: load_weight(&format!("blk.{b}.attn_v.weight"))?,
                wo: load_weight(&format!("blk.{b}.attn_output.weight"))?,
                bq,
                bk,
                bv,
                q_norm,
                k_norm,
                ffn_norm,
                gate: load_weight(&format!("blk.{b}.ffn_gate.weight"))?,
                up: load_weight(&format!("blk.{b}.ffn_up.weight"))?,
                down: load_weight(&format!("blk.{b}.ffn_down.weight"))?,
            });
        }

        let mut out_norm = None;
        let mut lm_head = None;
        if owns_output {
            let (on, _) = load_f32("output_norm.weight")?;
            out_norm = Some(on);
            let head = if has("output.weight") {
                load_weight("output.weight")?
            } else {
                // Tied embeddings: the lm_head is token_embd used as a
                // [vocab, hidden] projection.
                load_weight("token_embd.weight")?
            };
            if vocab == 0 {
                vocab = head.out_features();
            }
            lm_head = Some(head);
        }

        Ok(Self {
            dev: dev.clone(),
            cfg: LlamaConfig {
                arch,
                hidden,
                n_layers,
                layer_start,
                layer_end,
                n_heads,
                n_kv_heads,
                head_dim,
                vocab,
                rope_theta,
                rms_eps,
                rope_interleaved,
            },
            embed,
            blocks,
            out_norm,
            lm_head,
            weight_bytes: weight_bytes.get(),
        })
    }

    pub fn new_session(&self, max_seq: usize) -> Session {
        let kv_row = self.cfg.n_kv_heads * self.cfg.head_dim;
        Session {
            k: (0..self.blocks.len())
                .map(|_| self.dev.alloc(max_seq * kv_row))
                .collect(),
            v: (0..self.blocks.len())
                .map(|_| self.dev.alloc(max_seq * kv_row))
                .collect(),
            len: 0,
            max_seq,
        }
    }

    /// Run `tokens` through the model at the session's current
    /// position; returns the last position's logits.
    pub fn forward(&self, session: &mut Session, tokens: &[u32]) -> Result<Vec<f32>> {
        self.forward_logits(session, tokens, 1)
    }

    /// Vocab size of the logits rows this model produces (0 when this
    /// shard doesn't own the output globals).
    pub fn n_logits(&self) -> usize {
        self.lm_head.as_ref().map(Weight::out_features).unwrap_or(0)
    }

    /// The wgpu device this model lives on.
    pub fn device(&self) -> &WgpuDevice {
        &self.dev
    }

    /// Run `tokens` at the session's current position and return the
    /// logits of the **last `last_n` positions**, flattened row-major
    /// — speculative verification needs the k+1 tail rows, plain decode
    /// needs one. Requires a shard owning both input and output globals.
    pub fn forward_logits(
        &self,
        session: &mut Session,
        tokens: &[u32],
        last_n: usize,
    ) -> Result<Vec<f32>> {
        match self.forward_stage(session, StageInput::Tokens(tokens), last_n)? {
            StageOutput::Logits(l) => Ok(l),
            StageOutput::Hidden(_) => Err(WgpuError::Shape(
                "forward_logits on a shard without output globals; use forward_stage".into(),
            )),
        }
    }

    /// Run one pipeline-stage forward at the session's current position.
    /// Tokens are only valid on shards owning the input globals; hidden
    /// input everywhere else. Shards owning the output globals return
    /// the last `last_n` logits rows, all others the [seq, hidden]
    /// residual stream for the next stage.
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
        let kv_row = cfg.n_kv_heads * cfg.head_dim;

        // One command buffer for the whole forward — the readback at
        // the end flushes it.
        self.dev.begin_batch();
        let mut x = match input {
            StageInput::Tokens(tokens) => {
                let embed = self.embed.as_ref().ok_or_else(|| {
                    WgpuError::Shape("token input on a shard without input globals".into())
                })?;
                let ids: Vec<f32> = tokens.iter().map(|&t| t as f32).collect();
                let ids = self.dev.upload(&ids);
                self.dev.embed_gather(&ids, embed, seq, cfg.hidden)?
            }
            StageInput::Hidden { data, seq } => {
                if data.len() != seq * cfg.hidden {
                    return Err(WgpuError::Shape(format!(
                        "hidden input {} != seq {seq} × hidden {}",
                        data.len(),
                        cfg.hidden
                    )));
                }
                self.dev.upload(data)
            }
        };

        for (li, blk) in self.blocks.iter().enumerate() {
            let h = self
                .dev
                .rms_norm(&x, &blk.attn_norm, seq, cfg.hidden, cfg.rms_eps)?;
            let mut q = blk.wq.matmul_t(&self.dev, &h, seq)?;
            let mut k = blk.wk.matmul_t(&self.dev, &h, seq)?;
            let mut v = blk.wv.matmul_t(&self.dev, &h, seq)?;
            if let Some(b) = &blk.bq {
                q = self.dev.add_bias(&q, b)?;
            }
            if let Some(b) = &blk.bk {
                k = self.dev.add_bias(&k, b)?;
            }
            if let Some(b) = &blk.bv {
                v = self.dev.add_bias(&v, b)?;
            }
            // qwen3 per-head norms: the [seq, heads*head_dim] projection
            // is exactly [seq*heads, head_dim] rows for the row-norm.
            if let Some(w) = &blk.q_norm {
                q = self
                    .dev
                    .rms_norm(&q, w, seq * cfg.n_heads, cfg.head_dim, cfg.rms_eps)?;
            }
            if let Some(w) = &blk.k_norm {
                k = self
                    .dev
                    .rms_norm(&k, w, seq * cfg.n_kv_heads, cfg.head_dim, cfg.rms_eps)?;
            }
            let q = self.dev.rope(
                &q,
                seq,
                cfg.n_heads,
                cfg.head_dim,
                pos0,
                cfg.rope_theta,
                cfg.rope_interleaved,
            )?;
            let k = self.dev.rope(
                &k,
                seq,
                cfg.n_kv_heads,
                cfg.head_dim,
                pos0,
                cfg.rope_theta,
                cfg.rope_interleaved,
            )?;
            self.dev.copy_rows(&k, &session.k[li], pos0, seq, kv_row)?;
            self.dev.copy_rows(&v, &session.v[li], pos0, seq, kv_row)?;

            let scores = self.dev.attn_scores(
                &q,
                &session.k[li],
                seq,
                kv_len,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                pos0,
            )?;
            let probs = self.dev.softmax(&scores, cfg.n_heads * seq, kv_len)?;
            let att = self.dev.attn_out(
                &probs,
                &session.v[li],
                seq,
                kv_len,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
            )?;
            let o = blk.wo.matmul_t(&self.dev, &att, seq)?;
            x = self.dev.add(&x, &o)?;

            let h2 = self
                .dev
                .rms_norm(&x, &blk.ffn_norm, seq, cfg.hidden, cfg.rms_eps)?;
            let g = blk.gate.matmul_t(&self.dev, &h2, seq)?;
            let u = blk.up.matmul_t(&self.dev, &h2, seq)?;
            let gu = self.dev.silu_mul(&g, &u)?;
            let d = blk.down.matmul_t(&self.dev, &gu, seq)?;
            x = self.dev.add(&x, &d)?;
        }
        session.len = kv_len;

        let (Some(out_norm), Some(lm_head)) = (&self.out_norm, &self.lm_head) else {
            // Mid-pipeline stage: hand the residual stream to the next
            // stage (the download flushes the batch).
            return Ok(StageOutput::Hidden(self.dev.download(&x)?));
        };
        let h = self
            .dev
            .rms_norm(&x, out_norm, seq, cfg.hidden, cfg.rms_eps)?;
        let logits = lm_head.matmul_t(&self.dev, &h, seq)?;
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
