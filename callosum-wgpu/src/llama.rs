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
    /// MoE expert counts (qwen3moe). 0 for dense models.
    pub n_experts: usize,
    pub n_experts_used: usize,
    /// Partial-rotary width (GLM-4 rotates only the first half of each
    /// head). Equal to head_dim when full-width.
    pub rot_dim: usize,
    /// MoE routing policy (glm4moe): sigmoid mixture scores instead of
    /// softmax, weight renormalisation over the selected set, and a
    /// final scale on the mixture weights.
    pub moe_sigmoid: bool,
    pub moe_weights_norm: bool,
    pub moe_weights_scale: f32,
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

/// Host-resident quantized row-lookup table. Embedding tables live
/// here instead of VRAM: dequantized to f32 they can exceed
/// max_storage_buffer_binding_size (gemma-2's 256k x 2304 table is
/// 2.36 GB), and an input-side gather only ever needs `seq` rows per
/// forward — dequantized on the CPU in microseconds.
pub(crate) struct HostRowTable {
    raw: Vec<u8>,
    dtype: GgmlDType,
    row_bytes: usize,
    pub(crate) rows_total: usize,
    pub(crate) cols: usize,
}

impl HostRowTable {
    pub(crate) fn from_qtensor(qt: &callosum::quantized::QTensor) -> Result<Self> {
        let dims = qt.shape().dims().to_vec();
        if dims.len() != 2 {
            return Err(WgpuError::Shape(format!(
                "row table must be rank-2, got {dims:?}"
            )));
        }
        let (rows_total, cols) = (dims[0], dims[1]);
        let dtype = qt.dtype();
        if cols % dtype.block_size() != 0 {
            return Err(WgpuError::Shape(format!(
                "row table row of {cols} not block-aligned for {dtype:?}"
            )));
        }
        let row_bytes = cols / dtype.block_size() * dtype.type_size();
        let raw = qt
            .data()
            .map_err(|e| WgpuError::Device(format!("row table bytes: {e}")))?
            .into_owned();
        Ok(Self {
            raw,
            dtype,
            row_bytes,
            rows_total,
            cols,
        })
    }

    /// Dequantize the given rows on the CPU, concatenated row-major.
    pub(crate) fn rows(&self, ids: &[u32]) -> Result<Vec<f32>> {
        use callosum::quantized::{QStorage, QTensor};
        let cpu = callosum::Device::Cpu;
        let mut out: Vec<f32> = Vec::with_capacity(ids.len() * self.cols);
        for &id in ids {
            let off = id as usize * self.row_bytes;
            let slice = self
                .raw
                .get(off..off + self.row_bytes)
                .ok_or_else(|| WgpuError::Shape(format!("row id {id} out of table range")))?;
            let storage = QStorage::from_data(std::borrow::Cow::Borrowed(slice), &cpu, self.dtype)
                .map_err(|e| WgpuError::Device(format!("row {id} storage: {e}")))?;
            let row = QTensor::new(storage, (1, self.cols))
                .and_then(|t| t.dequantize(&cpu))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| WgpuError::Device(format!("row {id} dequant: {e}")))?;
            out.extend(row);
        }
        Ok(out)
    }
}

/// Dispatch an expert-indexed matmul on either weight representation.
#[allow(clippy::too_many_arguments)]
fn expert_matmul(
    dev: &WgpuDevice,
    w: &Weight,
    x: &crate::GpuBuffer,
    routing: &crate::GpuBuffer,
    m: usize,
    slots: usize,
    rows_per_expert: usize,
    k: usize,
    x_per_slot: bool,
) -> Result<crate::GpuBuffer> {
    match w {
        Weight::Quant(q) => {
            dev.matmul_expert(x, q, routing, m, slots, rows_per_expert, k, x_per_slot)
        }
        Weight::F32 { buf, .. } => {
            dev.matmul_expert_f32(x, buf, routing, m, slots, rows_per_expert, k, x_per_slot)
        }
    }
}

/// GGML dtypes with in-shader kernels; anything else dequantizes to
/// f32 at load (correct, memory-expensive).
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

/// Tensor-parallel hook: the engine computes per-rank partials and
/// calls `all_reduce` (in-place sum across ranks) at the two reduce
/// points per block (attn output, FFN down). The comm mechanism —
/// coordinator relay, NCCL-alike, anything — is the caller's; the
/// engine only needs rank/world for slicing and this callback.
#[derive(Clone)]
pub struct TpHook {
    pub rank: usize,
    pub world: usize,
    /// (layer_idx, op_kind, data) -> summed data in place. op_kind
    /// folds in the token position so back-to-back steps on one layer
    /// pair up at distinct barriers.
    #[allow(clippy::type_complexity)]
    pub all_reduce: std::sync::Arc<
        dyn Fn(u32, &str, &mut [f32]) -> std::result::Result<(), String> + Send + Sync,
    >,
}

/// Engine load options beyond the layer-range/globals basics.
#[derive(Default, Clone)]
pub struct LoadOpts {
    /// Routed MoE experts stay quantized in host RAM; expert FFN math
    /// runs on the CPU (activations-only bus traffic).
    pub cpu_moe: bool,
    /// Tensor parallelism: this rank holds a contiguous head subset of
    /// every attention and a row/col slice of every dense FFN.
    pub tp: Option<TpHook>,
}

/// Where routed expert weights live. Resident: fused and quantized on
/// the GPU exactly as stored on disk, dequantized in-shader by the
/// expert-indexed matmul kernels. Host (`cpu_moe` expert offload):
/// per-expert quantized matmuls in host RAM — the routed expert math
/// runs on the CPU and only the [seq, hidden] activations cross the
/// bus, never the weights. Routing itself stays on the GPU either way.
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

/// Byte-sliced per-expert quantized weights on the CPU device — the
/// same on-disk encoding, so host residency costs GGUF-size RAM.
pub(crate) struct HostExperts {
    gates: Vec<callosum::quantized::QMatMul>,
    ups: Vec<callosum::quantized::QMatMul>,
    downs: Vec<callosum::quantized::QMatMul>,
}

impl HostExperts {
    pub(crate) fn new(
        gates: Vec<callosum::quantized::QMatMul>,
        ups: Vec<callosum::quantized::QMatMul>,
        downs: Vec<callosum::quantized::QMatMul>,
    ) -> Self {
        Self { gates, ups, downs }
    }
}

/// Slice a fused rank-3 [n_experts, rows, cols] quantized tensor into
/// per-expert CPU `QMatMul`s over the raw quantized bytes — same
/// blocks, same scales, no re-encoding.
pub(crate) fn split_expert_qmatmuls_host(
    fused: &callosum::quantized::QTensor,
    n_experts: usize,
) -> Result<Vec<callosum::quantized::QMatMul>> {
    split_expert_qmatmuls_host_range(fused, n_experts, 0, n_experts)
}

/// [`split_expert_qmatmuls_host`] for the expert subrange
/// [start, end) — expert-parallel TP ranks each hold their own slice.
pub(crate) fn split_expert_qmatmuls_host_range(
    fused: &callosum::quantized::QTensor,
    n_experts: usize,
    start: usize,
    end: usize,
) -> Result<Vec<callosum::quantized::QMatMul>> {
    use callosum::quantized::{QMatMul, QStorage, QTensor};
    let cpu = callosum::Device::Cpu;
    let dims = fused.shape().dims().to_vec();
    if dims.len() != 3 || dims[0] != n_experts || start >= end || end > n_experts {
        return Err(WgpuError::Shape(format!(
            "fused expert tensor must be [{n_experts}, rows, cols] with a valid range, got {dims:?} [{start},{end})"
        )));
    }
    let (rows, cols) = (dims[1], dims[2]);
    let dtype = fused.dtype();
    let per_expert_elems = rows * cols;
    if per_expert_elems % dtype.block_size() != 0 {
        return Err(WgpuError::Shape(format!(
            "expert slice of {per_expert_elems} elems not block-aligned for {dtype:?}"
        )));
    }
    let per_expert_bytes = per_expert_elems / dtype.block_size() * dtype.type_size();
    let data = fused
        .data()
        .map_err(|e| WgpuError::Device(format!("fused expert bytes: {e}")))?;
    if data.len() < n_experts * per_expert_bytes {
        return Err(WgpuError::Shape(format!(
            "fused expert tensor has {} bytes, expected at least {}",
            data.len(),
            n_experts * per_expert_bytes
        )));
    }
    let mut out = Vec::with_capacity(end - start);
    for e in start..end {
        let slice = &data[e * per_expert_bytes..(e + 1) * per_expert_bytes];
        let storage = QStorage::from_data(std::borrow::Cow::Borrowed(slice), &cpu, dtype)
            .map_err(|e| WgpuError::Device(format!("expert {e} storage: {e}")))?;
        let qt = QTensor::new(storage, (rows, cols))
            .map_err(|e| WgpuError::Device(format!("expert {e} tensor: {e}")))?;
        out.push(
            QMatMul::from_qtensor(qt)
                .map_err(|e| WgpuError::Device(format!("expert {e} matmul: {e}"))),
        );
    }
    out.into_iter().collect()
}

/// Routed expert FFN on the CPU from a downloaded routing table
/// ([seq, slots, 2] of (expert_id, weight) — weights already
/// renormalised and scaled by the GPU top-k kernel) and hidden rows.
/// Mirrors the CUDA backend's run_moe expert-major batching.
pub(crate) fn host_moe_forward(
    hx: &HostExperts,
    table: &[f32],
    h2: &[f32],
    seq: usize,
    hidden: usize,
    slots: usize,
) -> callosum::Result<Vec<f32>> {
    use callosum::{Device, Module, Tensor};
    let cpu = Device::Cpu;
    let x = Tensor::from_vec(h2.to_vec(), (seq, hidden), &cpu)?;
    let mut by_expert: std::collections::BTreeMap<usize, (Vec<u32>, Vec<f32>)> =
        std::collections::BTreeMap::new();
    for t in 0..seq {
        for s in 0..slots {
            let e = table[(t * slots + s) * 2] as usize;
            let w = table[(t * slots + s) * 2 + 1];
            // Zero-weight slots are experts owned by other TP ranks
            // (moe_localize) -- skip the wasted matmuls.
            if w == 0.0 {
                continue;
            }
            let entry = by_expert.entry(e).or_default();
            entry.0.push(t as u32);
            entry.1.push(w);
        }
    }
    let mut acc = vec![0f32; seq * hidden];
    for (e, (positions, weights)) in by_expert {
        let ids = Tensor::new(positions.as_slice(), &cpu)?;
        let xs = x.index_select(&ids, 0)?; // [n_pos, hidden]
        let g = hx.gates[e].forward(&xs)?;
        let u = hx.ups[e].forward(&xs)?;
        let gated = (g.silu()? * u)?;
        let out: Vec<f32> = hx.downs[e]
            .forward(&gated)?
            .to_dtype(callosum::DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        for (i, &pos) in positions.iter().enumerate() {
            let w = weights[i];
            let dst = &mut acc[pos as usize * hidden..(pos as usize + 1) * hidden];
            for (d, o) in dst.iter_mut().zip(&out[i * hidden..(i + 1) * hidden]) {
                *d += w * o;
            }
        }
    }
    Ok(acc)
}

/// Dense SwiGLU or a mixture-of-experts FFN (qwen3moe). Expert weights
/// stay fused and quantized on the GPU exactly as stored on disk; the
/// expert-indexed matmul kernels dequantize the routed expert's rows
/// in-shader.
enum Ffn {
    Dense {
        /// None = GLM-4 fused gate+up: `up` is [2*intermediate, hidden]
        /// and SWIGLU splits its output halves.
        gate: Option<Weight>,
        up: Weight,
        down: Weight,
    },
    Moe {
        /// hidden → n_experts router.
        router: Weight,
        /// V3-style selection bias (`exp_probs_b`) added to the scores
        /// before top-k; mixture weights stay unbiased.
        router_bias: Option<GpuBuffer>,
        /// Routed expert weights — VRAM-resident or host-offloaded.
        experts: ExpertStore,
        ffn_dim: usize,
        /// Expert-parallel TP: the global expert range this rank owns
        /// (`None` = all experts local, no reduce needed).
        expert_range: Option<(usize, usize)>,
        /// Fused shared experts (glm4moe/deepseek `shexp`), run dense
        /// on the same input and added to the routed mixture.
        shared: Option<Box<(Weight, Weight, Weight)>>,
    },
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
    ffn: Ffn,
    /// GLM-4 sandwich norms (output-normalised before residual adds).
    post_attn_norm: Option<GpuBuffer>,
    post_ffn_norm: Option<GpuBuffer>,
}

pub struct WgpuLlama {
    dev: WgpuDevice,
    pub cfg: LlamaConfig,
    /// Host-side quantized [vocab, hidden] gather source — present only
    /// on shards that own the input globals.
    embed: Option<HostRowTable>,
    blocks: Vec<Block>,
    /// Final norm + lm_head — present only on shards that own the
    /// output globals.
    out_norm: Option<GpuBuffer>,
    lm_head: Option<Weight>,
    /// Tensor-parallel hook; None when this shard owns all heads.
    tp: Option<TpHook>,
    /// Total bytes uploaded for weights (quantized at on-disk density,
    /// f32 where a format fell back). What a serving layer should
    /// report as resident.
    pub weight_bytes: u64,
    /// Bytes of expert weights held in host RAM under `cpu_moe`
    /// offload — excluded from `weight_bytes` (the GPU-resident
    /// figure) so free-VRAM accounting stays honest.
    pub host_expert_bytes: u64,
}

/// One layer's K or V history: dense f32, or int8 with per-(token,
/// head) scales (`SPLITBRAIN_KV_DTYPE=int8` — same encoding as the
/// CUDA backend's QuantizedKv, ~4x smaller than f32).
pub(crate) enum KvStore {
    F32(GpuBuffer),
    Q8 { data: GpuBuffer, scale: GpuBuffer },
}

impl KvStore {
    pub(crate) fn alloc(
        dev: &WgpuDevice,
        max_seq: usize,
        kv_heads: usize,
        head_dim: usize,
        int8: bool,
    ) -> Self {
        if int8 {
            Self::Q8 {
                data: dev.alloc(max_seq * kv_heads * head_dim / 4),
                scale: dev.alloc(max_seq * kv_heads),
            }
        } else {
            Self::F32(dev.alloc(max_seq * kv_heads * head_dim))
        }
    }

    /// Append `seq` rows of [kv_heads, head_dim] at token `pos0`.
    pub(crate) fn append(
        &self,
        dev: &WgpuDevice,
        src: &GpuBuffer,
        pos0: usize,
        seq: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        match self {
            Self::F32(buf) => dev.copy_rows(src, buf, pos0, seq, kv_heads * head_dim),
            Self::Q8 { data, scale } => {
                dev.kv_quant_append(src, data, scale, pos0, seq, kv_heads, head_dim)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_scores(
        &self,
        dev: &WgpuDevice,
        q: &GpuBuffer,
        seq_q: usize,
        kv_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos0: usize,
        scale: f32,
        window: usize,
    ) -> Result<GpuBuffer> {
        match self {
            Self::F32(buf) => dev.attn_scores_opt(
                q, buf, seq_q, kv_len, heads, kv_heads, head_dim, pos0, scale, window,
            ),
            Self::Q8 { data, scale: sc } => dev.attn_scores_q8_opt(
                q, data, sc, seq_q, kv_len, heads, kv_heads, head_dim, pos0, scale, window,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_out(
        &self,
        dev: &WgpuDevice,
        probs: &GpuBuffer,
        seq_q: usize,
        kv_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<GpuBuffer> {
        match self {
            Self::F32(buf) => dev.attn_out(probs, buf, seq_q, kv_len, heads, kv_heads, head_dim),
            Self::Q8 { data, scale } => {
                dev.attn_out_q8(probs, data, scale, seq_q, kv_len, heads, kv_heads, head_dim)
            }
        }
    }
}

/// Per-conversation KV state: one K and one V store per **local**
/// layer, laid out [max_seq, n_kv_heads, head_dim], appended in place.
pub struct Session {
    k: Vec<KvStore>,
    v: Vec<KvStore>,
    pub len: usize,
    max_seq: usize,
}

impl Session {
    /// True when this session stores its KV history as int8.
    pub fn kv_int8(&self) -> bool {
        matches!(self.k.first(), Some(KvStore::Q8 { .. }))
    }
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

    /// `from_gguf` with `cpu_moe` expert offload: routed expert
    /// weights stay quantized in host RAM and the expert FFN math
    /// runs on the CPU; routing and everything else stays on the GPU.
    pub fn from_gguf_cpu_moe(path: &std::path::Path, dev: &WgpuDevice) -> Result<Self> {
        Self::from_gguf_stage_opts(path, dev, 0, usize::MAX, true, true, true)
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

    /// [`Self::from_gguf_stage`] plus `cpu_moe`: when set, routed
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
        Self::from_gguf_stage_with(
            path,
            dev,
            layer_start,
            layer_end,
            owns_input,
            owns_output,
            LoadOpts {
                cpu_moe,
                ..Default::default()
            },
        )
    }

    /// [`Self::from_gguf_stage`] with full [`LoadOpts`] (cpu_moe
    /// expert offload, tensor parallelism).
    #[allow(clippy::too_many_arguments)]
    pub fn from_gguf_stage_with(
        path: &std::path::Path,
        dev: &WgpuDevice,
        layer_start: usize,
        layer_end: usize,
        owns_input: bool,
        owns_output: bool,
        opts: LoadOpts,
    ) -> Result<Self> {
        let cpu_moe = opts.cpu_moe;
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
        if !matches!(
            arch.as_str(),
            "llama" | "mistral" | "qwen2" | "qwen3" | "qwen3moe" | "qwen35moe" | "glm4" | "glm4moe"
        ) {
            return Err(WgpuError::Device(format!(
                "callosum-wgpu llama loader supports llama/mistral/qwen2/qwen3/qwen3moe/glm4/glm4moe (got {arch:?})"
            )));
        }
        // Interleaved pairs (2i, 2i+1) for llama-lineage GGUFs,
        // rotate-half (i, i+d/2) for the qwen family.
        let rope_interleaved = matches!(arch.as_str(), "llama" | "mistral" | "glm4");
        let is_moe = arch.ends_with("moe");
        let key = |suffix: &str| format!("{arch}.{suffix}");

        let hidden = meta_u32(&content, &[&key("embedding_length")])
            .ok_or_else(|| WgpuError::Device("missing embedding_length".into()))?
            as usize;
        // glm4moe appends `nextn_predict_layers` MTP blocks (nextn.*
        // tensors) after the real blocks — excluded from the forward
        // pass, exactly as llama.cpp does.
        let nextn = meta_u32(&content, &[&key("nextn_predict_layers")]).unwrap_or(0) as usize;
        let n_layers = meta_u32(&content, &[&key("block_count")])
            .ok_or_else(|| WgpuError::Device("missing block_count".into()))?
            as usize;
        let n_layers = n_layers.saturating_sub(nextn);
        let n_heads = meta_u32(&content, &[&key("attention.head_count")])
            .ok_or_else(|| WgpuError::Device("missing head_count".into()))?
            as usize;
        let n_kv_heads = meta_u32(&content, &[&key("attention.head_count_kv")])
            .map(|v| v as usize)
            .unwrap_or(n_heads);
        let head_dim = meta_u32(&content, &[&key("attention.key_length")])
            .map(|v| v as usize)
            .unwrap_or(hidden / n_heads);
        let rot_dim = meta_u32(&content, &[&key("rope.dimension_count")])
            .map(|v| v as usize)
            .filter(|&d| d < head_dim)
            .unwrap_or(head_dim);
        let rope_theta = meta_f32(&content, &[&key("rope.freq_base")]).unwrap_or(10_000.0);
        let n_experts = meta_u32(&content, &[&key("expert_count")]).unwrap_or(0) as usize;
        let n_experts_used = meta_u32(&content, &[&key("expert_used_count")]).unwrap_or(0) as usize;
        if is_moe && (n_experts == 0 || n_experts_used == 0) {
            return Err(WgpuError::Device(
                "MoE arch without expert_count/expert_used_count metadata".into(),
            ));
        }
        let rms_eps =
            meta_f32(&content, &[&key("attention.layer_norm_rms_epsilon")]).unwrap_or(1e-5);
        // glm4moe defaults to sigmoid gating when the gating-func key is
        // absent (llama.cpp load_arch_hparams); 2 = sigmoid explicitly.
        let moe_sigmoid = meta_u32(&content, &[&key("expert_gating_func")])
            .map(|g| g == 2)
            .unwrap_or(arch == "glm4moe");
        let moe_weights_norm = content
            .metadata
            .get(&key("expert_weights_norm"))
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(!moe_sigmoid);
        let moe_weights_scale = meta_f32(&content, &[&key("expert_weights_scale")]).unwrap_or(1.0);

        let layer_end = layer_end.min(n_layers);
        if layer_start >= layer_end {
            return Err(WgpuError::Shape(format!(
                "empty layer range [{layer_start},{layer_end}) of {n_layers}"
            )));
        }

        // Tensor parallelism: this rank keeps heads
        // [rank*local, (rank+1)*local) of every attention and the
        // matching row/col slice of every dense FFN. Same divisibility
        // contract as the CUDA backend.
        let tp_world = opts.tp.as_ref().map(|t| t.world).unwrap_or(1);
        let tp_rank = opts.tp.as_ref().map(|t| t.rank).unwrap_or(0);
        if tp_world > 1 {
            // qwen-family MoEs run expert-parallel (each rank owns a
            // contiguous expert range; partials summed by the
            // all-reduce). glm4moe's shared experts + sigmoid routing
            // are refused, matching the CUDA backend.
            if is_moe && !matches!(arch.as_str(), "qwen3moe" | "qwen35moe") {
                return Err(WgpuError::Device(format!(
                    "wgpu TP supports dense archs and qwen-family MoEs (got {arch:?})"
                )));
            }
            if is_moe && !n_experts.is_multiple_of(tp_world) {
                return Err(WgpuError::Shape(format!(
                    "n_experts {n_experts} not divisible by tp world {tp_world}"
                )));
            }
            if !n_heads.is_multiple_of(tp_world) {
                return Err(WgpuError::Shape(format!(
                    "n_heads {n_heads} not divisible by tp world {tp_world}"
                )));
            }
            if !n_kv_heads.is_multiple_of(tp_world) {
                return Err(WgpuError::Shape(format!(
                    "n_kv_heads {n_kv_heads} not divisible by tp world {tp_world}"
                )));
            }
        }
        let n_heads_local = n_heads / tp_world.max(1);
        let n_kv_local = n_kv_heads / tp_world.max(1);
        // Expert-parallel range for MoE blocks (full range without TP).
        let e_local = if is_moe {
            n_experts / tp_world.max(1)
        } else {
            0
        };
        let (e_start, e_end) = (tp_rank * e_local, (tp_rank + 1) * e_local);

        // Cheap CPU device for dequantizing non-quant-kernel tensors.
        let cpu = callosum::Device::Cpu;
        let weight_bytes = std::cell::Cell::new(0u64);
        let host_expert_bytes = std::cell::Cell::new(0u64);
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
            let mut file_e = std::fs::File::open(path)
                .map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
            let qt = content
                .tensor(&mut file_e, "token_embd.weight", &cpu)
                .map_err(|e| WgpuError::Device(format!("load token_embd: {e}")))?;
            let table = HostRowTable::from_qtensor(&qt)?;
            vocab = table.rows_total;
            embed = Some(table);
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

        // TP loader: slice a rank-2 weight along rows (dim 0) or
        // columns (dim 1) BEFORE upload, keeping the on-disk quant
        // density whenever the slice is block-aligned (rows always
        // are; columns when the span is a whole number of blocks).
        // Falls back to a dequantized f32 slice otherwise.
        let mut file4 =
            std::fs::File::open(path).map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
        let mut load_weight_slice =
            |name: &str, dim: usize, start: usize, count: usize| -> Result<Weight> {
                let qt = content
                    .tensor(&mut file4, name, &cpu)
                    .map_err(|e| WgpuError::Device(format!("load {name}: {e}")))?;
                let dims = qt.shape().dims().to_vec();
                if dims.len() != 2 {
                    return Err(WgpuError::Shape(format!("{name}: expected rank-2")));
                }
                let (n, k) = (dims[0], dims[1]);
                if (dim == 0 && start + count > n) || (dim == 1 && start + count > k) {
                    return Err(WgpuError::Shape(format!("{name}: TP slice out of bounds")));
                }
                let fmt = quant_dtype(qt.dtype());
                let aligned = match (dim, fmt) {
                    (0, Some(f)) => k.is_multiple_of(f.block_elems()),
                    (1, Some(f)) => {
                        start.is_multiple_of(f.block_elems())
                            && count.is_multiple_of(f.block_elems())
                    }
                    _ => false,
                };
                if let (Some(f), true) = (fmt, aligned) {
                    let raw = qt
                        .data()
                        .map_err(|e| WgpuError::Device(format!("{name} bytes: {e}")))?;
                    let row_bytes = k / f.block_elems() * f.block_bytes();
                    let sliced: Vec<u8> = if dim == 0 {
                        raw[start * row_bytes..(start + count) * row_bytes].to_vec()
                    } else {
                        let off = start / f.block_elems() * f.block_bytes();
                        let span = count / f.block_elems() * f.block_bytes();
                        let mut out = Vec::with_capacity(n * span);
                        for r in 0..n {
                            out.extend_from_slice(
                                &raw[r * row_bytes + off..r * row_bytes + off + span],
                            );
                        }
                        out
                    };
                    weight_bytes.set(weight_bytes.get() + sliced.len() as u64);
                    let (sn, sk) = if dim == 0 { (count, k) } else { (n, count) };
                    return dev.upload_quantized(&sliced, sn, sk, f).map(Weight::Quant);
                }
                let t = qt
                    .dequantize(&cpu)
                    .and_then(|t| t.to_dtype(callosum::DType::F32))
                    .and_then(|t| t.narrow(dim, start, count))
                    .and_then(|t| t.contiguous())
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| WgpuError::Device(format!("dequantize {name}: {e}")))?;
                weight_bytes.set(weight_bytes.get() + (t.len() * 4) as u64);
                let (sn, sk) = if dim == 0 { (count, k) } else { (n, count) };
                Ok(Weight::F32 {
                    buf: dev.upload(&t),
                    n: sn,
                    k: sk,
                })
            };

        let has = |name: &str| content.tensor_infos.contains_key(name);
        // Fused MoE expert tensors are rank-3 [n_experts, rows, cols];
        // flatten to [n_experts * rows, cols] so one QuantBuffer holds
        // every expert at on-disk density. The expert-indexed kernels
        // offset by expert_id * rows.
        let mut file3 =
            std::fs::File::open(path).map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
        let mut load_expert_weight = |name: &str| -> Result<Weight> {
            let qt = content
                .tensor(&mut file3, name, &cpu)
                .map_err(|e| WgpuError::Device(format!("load {name}: {e}")))?;
            let dims = qt.shape().dims().to_vec();
            if dims.len() != 3 {
                return Err(WgpuError::Shape(format!(
                    "{name}: expected rank-3 fused experts"
                )));
            }
            let (n_e, rows, cols) = (dims[0], dims[1], dims[2]);
            // Expert-parallel TP: keep only this rank's expert range.
            // Experts are contiguous in the fused payload, so the
            // slice stays at on-disk quant density.
            let (e_lo, e_hi) = if tp_world > 1 {
                (e_start.min(n_e), e_end.min(n_e))
            } else {
                (0, n_e)
            };
            if (e_lo, e_hi) != (0, n_e) {
                let dtype = qt.dtype();
                let per_expert_elems = rows * cols;
                if per_expert_elems.is_multiple_of(dtype.block_size()) {
                    if let Some(fmt) = quant_dtype(dtype) {
                        if cols.is_multiple_of(fmt.block_elems()) {
                            let per_expert_bytes =
                                per_expert_elems / dtype.block_size() * dtype.type_size();
                            let raw = qt
                                .data()
                                .map_err(|e| WgpuError::Device(format!("{name} bytes: {e}")))?;
                            let sliced = &raw[e_lo * per_expert_bytes..e_hi * per_expert_bytes];
                            weight_bytes.set(weight_bytes.get() + sliced.len() as u64);
                            return dev
                                .upload_quantized(sliced, (e_hi - e_lo) * rows, cols, fmt)
                                .map(Weight::Quant);
                        }
                    }
                }
                let t = qt
                    .dequantize(&cpu)
                    .and_then(|t| t.to_dtype(callosum::DType::F32))
                    .and_then(|t| t.narrow(0, e_lo, e_hi - e_lo))
                    .and_then(|t| t.contiguous())
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| WgpuError::Device(format!("dequantize {name}: {e}")))?;
                weight_bytes.set(weight_bytes.get() + (t.len() * 4) as u64);
                return Ok(Weight::F32 {
                    buf: dev.upload(&t),
                    n: (e_hi - e_lo) * rows,
                    k: cols,
                });
            }
            match quant_dtype(qt.dtype()) {
                Some(fmt) if cols % fmt.block_elems() == 0 => {
                    let raw = qt
                        .data()
                        .map_err(|e| WgpuError::Device(format!("{name} bytes: {e}")))?;
                    weight_bytes.set(weight_bytes.get() + raw.len() as u64);
                    dev.upload_quantized(&raw, n_e * rows, cols, fmt)
                        .map(Weight::Quant)
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
                        n: n_e * rows,
                        k: cols,
                    })
                }
            }
        };
        let mut blocks = Vec::with_capacity(layer_end - layer_start);
        for b in layer_start..layer_end {
            let (attn_norm, _) = load_f32(&format!("blk.{b}.attn_norm.weight"))?;
            let mut opt_f32 = |name: String| -> Result<Option<GpuBuffer>> {
                if has(&name) {
                    load_f32(&name).map(|(buf, _)| Some(buf))
                } else {
                    Ok(None)
                }
            };
            // Under TP, biases follow the head slice of their matrix.
            // Self-contained (own file handle) so it doesn't contend
            // with the other loader closures for `load_f32`.
            let opt_bias = |name: String, span: usize| -> Result<Option<GpuBuffer>> {
                if !has(&name) {
                    return Ok(None);
                }
                let mut file_b = std::fs::File::open(path)
                    .map_err(|e| WgpuError::Device(format!("open gguf: {e}")))?;
                let t = content
                    .tensor(&mut file_b, &name, &cpu)
                    .map_err(|e| WgpuError::Device(format!("load {name}: {e}")))?
                    .dequantize(&cpu)
                    .and_then(|t| t.to_dtype(callosum::DType::F32))
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| WgpuError::Device(format!("dequantize {name}: {e}")))?;
                let (lo, hi) = if tp_world > 1 {
                    (tp_rank * span, (tp_rank + 1) * span)
                } else {
                    (0, t.len())
                };
                weight_bytes.set(weight_bytes.get() + ((hi - lo) * 4) as u64);
                Ok(Some(dev.upload(&t[lo..hi])))
            };
            let bq = opt_bias(format!("blk.{b}.attn_q.bias"), n_heads_local * head_dim)?;
            let bk = opt_bias(format!("blk.{b}.attn_k.bias"), n_kv_local * head_dim)?;
            let bv = opt_bias(format!("blk.{b}.attn_v.bias"), n_kv_local * head_dim)?;
            let q_norm = opt_f32(format!("blk.{b}.attn_q_norm.weight"))?;
            let k_norm = opt_f32(format!("blk.{b}.attn_k_norm.weight"))?;
            let mut post_attn_norm = opt_f32(format!("blk.{b}.post_attention_norm.weight"))?;
            let post_ffn_norm = opt_f32(format!("blk.{b}.post_ffw_norm.weight"))?;
            let router_bias = opt_f32(format!("blk.{b}.exp_probs_b.bias"))?;
            // glm4moe ships no ffn_norm: post_attention_norm IS the
            // pre-FFN norm (standard residual structure — llama.cpp
            // build_glm4_moe), not a sandwich norm.
            let ffn_norm = if has(&format!("blk.{b}.ffn_norm.weight")) {
                load_f32(&format!("blk.{b}.ffn_norm.weight"))?.0
            } else {
                post_attn_norm.take().ok_or_else(|| {
                    WgpuError::Device(format!(
                        "blk.{b}: neither ffn_norm nor post_attention_norm present"
                    ))
                })?
            };
            let ffn = if has(&format!("blk.{b}.ffn_gate_inp.weight")) {
                let (experts, ffn_dim) = if cpu_moe {
                    // Expert offload: fused tensors load on the CPU and
                    // get byte-sliced per expert — never touching VRAM,
                    // which is the whole point for models whose experts
                    // don't fit.
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
                            let (s, e) = if tp_world > 1 {
                                (e_start, e_end)
                            } else {
                                (0, n_experts)
                            };
                            Ok((
                                split_expert_qmatmuls_host_range(&qt, n_experts, s, e)?,
                                dims[1],
                            ))
                        };
                    let (gates, ffn_dim) = load_host(format!("blk.{b}.ffn_gate_exps.weight"))?;
                    let (ups, _) = load_host(format!("blk.{b}.ffn_up_exps.weight"))?;
                    let (downs, _) = load_host(format!("blk.{b}.ffn_down_exps.weight"))?;
                    (
                        ExpertStore::Host(Box::new(HostExperts { gates, ups, downs })),
                        ffn_dim,
                    )
                } else {
                    let n_local = if tp_world > 1 {
                        e_end - e_start
                    } else {
                        n_experts
                    };
                    let gates = load_expert_weight(&format!("blk.{b}.ffn_gate_exps.weight"))?;
                    let ffn_dim = match &gates {
                        // Fused rows = n_local * ffn_dim (expert-range
                        // sliced under TP).
                        Weight::F32 { n, .. } => *n / n_local,
                        Weight::Quant(q) => q.n / n_local,
                    };
                    (
                        ExpertStore::Resident(Box::new(ResidentExperts {
                            gates,
                            ups: load_expert_weight(&format!("blk.{b}.ffn_up_exps.weight"))?,
                            downs: load_expert_weight(&format!("blk.{b}.ffn_down_exps.weight"))?,
                        })),
                        ffn_dim,
                    )
                };
                let shared = if has(&format!("blk.{b}.ffn_gate_shexp.weight")) {
                    Some(Box::new((
                        load_weight(&format!("blk.{b}.ffn_gate_shexp.weight"))?,
                        load_weight(&format!("blk.{b}.ffn_up_shexp.weight"))?,
                        load_weight(&format!("blk.{b}.ffn_down_shexp.weight"))?,
                    )))
                } else {
                    None
                };
                Ffn::Moe {
                    router: load_weight(&format!("blk.{b}.ffn_gate_inp.weight"))?,
                    router_bias,
                    experts,
                    ffn_dim,
                    expert_range: (tp_world > 1).then_some((e_start, e_end)),
                    shared,
                }
            } else if tp_world > 1 {
                // Row-parallel gate/up, column-parallel down; the down
                // matmul yields a per-rank partial over the full
                // hidden dim, summed by the all-reduce hook. Fused
                // gate+up (GLM-4) can't row-split cleanly — refused,
                // matching the CUDA backend.
                if !has(&format!("blk.{b}.ffn_gate.weight")) {
                    return Err(WgpuError::Device(
                        "TP with fused gate+up FFN (glm4) is not supported".into(),
                    ));
                }
                let ffn_dim = content
                    .tensor_infos
                    .get(&format!("blk.{b}.ffn_gate.weight"))
                    .map(|i| i.shape.dims()[0])
                    .unwrap_or(0);
                if !ffn_dim.is_multiple_of(tp_world) {
                    return Err(WgpuError::Shape(format!(
                        "ffn_dim {ffn_dim} not divisible by tp world {tp_world}"
                    )));
                }
                let local = ffn_dim / tp_world;
                Ffn::Dense {
                    gate: Some(load_weight_slice(
                        &format!("blk.{b}.ffn_gate.weight"),
                        0,
                        tp_rank * local,
                        local,
                    )?),
                    up: load_weight_slice(
                        &format!("blk.{b}.ffn_up.weight"),
                        0,
                        tp_rank * local,
                        local,
                    )?,
                    down: load_weight_slice(
                        &format!("blk.{b}.ffn_down.weight"),
                        1,
                        tp_rank * local,
                        local,
                    )?,
                }
            } else {
                Ffn::Dense {
                    gate: if has(&format!("blk.{b}.ffn_gate.weight")) {
                        Some(load_weight(&format!("blk.{b}.ffn_gate.weight"))?)
                    } else {
                        None
                    },
                    up: load_weight(&format!("blk.{b}.ffn_up.weight"))?,
                    down: load_weight(&format!("blk.{b}.ffn_down.weight"))?,
                }
            };
            // Recycle upload staging so weights aren't resident twice.
            dev.reclaim_staging();
            let (wq, wk, wv, wo) = if tp_world > 1 {
                let qr = n_heads_local * head_dim;
                let kr = n_kv_local * head_dim;
                (
                    load_weight_slice(&format!("blk.{b}.attn_q.weight"), 0, tp_rank * qr, qr)?,
                    load_weight_slice(&format!("blk.{b}.attn_k.weight"), 0, tp_rank * kr, kr)?,
                    load_weight_slice(&format!("blk.{b}.attn_v.weight"), 0, tp_rank * kr, kr)?,
                    load_weight_slice(&format!("blk.{b}.attn_output.weight"), 1, tp_rank * qr, qr)?,
                )
            } else {
                (
                    load_weight(&format!("blk.{b}.attn_q.weight"))?,
                    load_weight(&format!("blk.{b}.attn_k.weight"))?,
                    load_weight(&format!("blk.{b}.attn_v.weight"))?,
                    load_weight(&format!("blk.{b}.attn_output.weight"))?,
                )
            };
            blocks.push(Block {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                bq,
                bk,
                bv,
                q_norm,
                k_norm,
                ffn_norm,
                ffn,
                post_attn_norm,
                post_ffn_norm,
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
                // LOCAL head counts under TP — the kernels, sessions,
                // and rope all operate on this rank's slice.
                n_heads: n_heads_local,
                n_kv_heads: n_kv_local,
                head_dim,
                vocab,
                rope_theta,
                rms_eps,
                rope_interleaved,
                n_experts,
                n_experts_used,
                rot_dim,
                moe_sigmoid,
                moe_weights_norm,
                moe_weights_scale,
            },
            embed,
            blocks,
            out_norm,
            lm_head,
            tp: opts.tp,
            weight_bytes: weight_bytes.get(),
            host_expert_bytes: host_expert_bytes.get(),
        })
    }

    pub fn new_session(&self, max_seq: usize) -> Session {
        self.new_session_opts(max_seq, false)
    }

    /// [`Self::new_session`] with an optional int8 KV cache (~4x
    /// smaller). Falls back to f32 when head_dim isn't word-aligned
    /// (packing needs head_dim % 4 == 0 — true for every real arch).
    pub fn new_session_opts(&self, max_seq: usize, kv_int8: bool) -> Session {
        let int8 = kv_int8 && self.cfg.head_dim.is_multiple_of(4);
        let mk = |_: usize| {
            KvStore::alloc(
                &self.dev,
                max_seq,
                self.cfg.n_kv_heads,
                self.cfg.head_dim,
                int8,
            )
        };
        Session {
            k: (0..self.blocks.len()).map(mk).collect(),
            v: (0..self.blocks.len()).map(mk).collect(),
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

    /// All-reduce a per-rank partial across the TP group (identity
    /// without TP): download, sum via the hook, re-upload.
    fn tp_reduce(&self, buf: GpuBuffer, layer: u32, op_kind: &str) -> Result<GpuBuffer> {
        let Some(tp) = &self.tp else { return Ok(buf) };
        let mut host = self.dev.download(&buf)?;
        (tp.all_reduce)(layer, op_kind, &mut host).map_err(WgpuError::Device)?;
        Ok(self.dev.upload(&host))
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

        // One command buffer for the whole forward — the readback at
        // the end flushes it.
        self.dev.begin_batch();
        let mut x = match input {
            StageInput::Tokens(tokens) => {
                let embed = self.embed.as_ref().ok_or_else(|| {
                    WgpuError::Shape("token input on a shard without input globals".into())
                })?;
                self.dev.upload(&embed.rows(tokens)?)
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
            let q = self.dev.rope_scaled(
                &q,
                seq,
                cfg.n_heads,
                cfg.head_dim,
                pos0,
                cfg.rope_theta,
                cfg.rope_interleaved,
                1.0,
                None,
                cfg.rot_dim,
            )?;
            let k = self.dev.rope_scaled(
                &k,
                seq,
                cfg.n_kv_heads,
                cfg.head_dim,
                pos0,
                cfg.rope_theta,
                cfg.rope_interleaved,
                1.0,
                None,
                cfg.rot_dim,
            )?;
            session.k[li].append(&self.dev, &k, pos0, seq, cfg.n_kv_heads, cfg.head_dim)?;
            session.v[li].append(&self.dev, &v, pos0, seq, cfg.n_kv_heads, cfg.head_dim)?;

            let scores = session.k[li].attn_scores(
                &self.dev,
                &q,
                seq,
                kv_len,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                pos0,
                1.0 / (cfg.head_dim as f32).sqrt(),
                0,
            )?;
            let probs = self.dev.softmax(&scores, cfg.n_heads * seq, kv_len)?;
            let att = session.v[li].attn_out(
                &self.dev,
                &probs,
                seq,
                kv_len,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
            )?;
            let o = blk.wo.matmul_t(&self.dev, &att, seq)?;
            let o = self.tp_reduce(
                o,
                (cfg.layer_start + li) as u32,
                &format!("attn_out:{pos0}"),
            )?;
            let o = match &blk.post_attn_norm {
                Some(w) => self.dev.rms_norm(&o, w, seq, cfg.hidden, cfg.rms_eps)?,
                None => o,
            };
            x = self.dev.add(&x, &o)?;

            let h2 = self
                .dev
                .rms_norm(&x, &blk.ffn_norm, seq, cfg.hidden, cfg.rms_eps)?;
            let d = match &blk.ffn {
                Ffn::Dense { gate, up, down } => {
                    let gu = match gate {
                        Some(gate) => {
                            let g = gate.matmul_t(&self.dev, &h2, seq)?;
                            let u = up.matmul_t(&self.dev, &h2, seq)?;
                            self.dev.silu_mul(&g, &u)?
                        }
                        None => {
                            // Fused gate+up (GLM-4): one matmul to
                            // [seq, 2*intermediate], split halves,
                            // silu(first) * second.
                            let w = up.matmul_t(&self.dev, &h2, seq)?;
                            let two_f = up.out_features();
                            let f = two_f / 2;
                            let g = self.dev.slice_cols(&w, seq, two_f, 0, f)?;
                            let u = self.dev.slice_cols(&w, seq, two_f, f, f)?;
                            self.dev.silu_mul(&g, &u)?
                        }
                    };
                    let d = down.matmul_t(&self.dev, &gu, seq)?;
                    self.tp_reduce(
                        d,
                        (cfg.layer_start + li) as u32,
                        &format!("ffn_down:{pos0}"),
                    )?
                }
                Ffn::Moe {
                    router,
                    router_bias,
                    experts,
                    ffn_dim,
                    expert_range,
                    shared,
                } => {
                    // Same routing rule as the CUDA backend's
                    // run_moe_opts: softmax (qwen/deepseek-v2) or
                    // sigmoid (glm4moe/v3) mixture scores, optional
                    // selection bias, top-k, optional renormalisation
                    // over the selected set, SwiGLU per expert,
                    // weighted sum, plus dense shared experts.
                    let logits = router.matmul_t(&self.dev, &h2, seq)?;
                    let routing = self.dev.moe_topk_opt(
                        &logits,
                        seq,
                        cfg.n_experts,
                        cfg.n_experts_used,
                        cfg.moe_sigmoid,
                        router_bias.as_ref(),
                        cfg.moe_weights_norm,
                        1,
                        1,
                        cfg.moe_weights_scale,
                    )?;
                    let slots = cfg.n_experts_used;
                    // Expert-parallel TP: zero foreign slots and rebase
                    // local ids before the expert-indexed matmuls; the
                    // cross-rank all-reduce below restores the full sum.
                    let routing = match expert_range {
                        Some((es, ee)) => self.dev.moe_localize(&routing, seq * slots, *es, *ee)?,
                        None => routing,
                    };
                    let routed = match experts {
                        ExpertStore::Resident(rx) => {
                            let ResidentExperts { gates, ups, downs } = rx.as_ref();
                            let g = expert_matmul(
                                &self.dev, gates, &h2, &routing, seq, slots, *ffn_dim, cfg.hidden,
                                false,
                            )?;
                            let u = expert_matmul(
                                &self.dev, ups, &h2, &routing, seq, slots, *ffn_dim, cfg.hidden,
                                false,
                            )?;
                            let gu = self.dev.silu_mul(&g, &u)?;
                            let d = expert_matmul(
                                &self.dev, downs, &gu, &routing, seq, slots, cfg.hidden, *ffn_dim,
                                true,
                            )?;
                            self.dev.moe_combine(&d, &routing, seq, slots, cfg.hidden)?
                        }
                        ExpertStore::Host(hx) => {
                            // cpu_moe: routing table + hidden rows hop
                            // to the host, the expert FFN runs on the
                            // CPU over byte-sliced quantized weights,
                            // and only the [seq, hidden] result comes
                            // back — weights never cross the bus.
                            let table = self.dev.download(&routing)?;
                            let h2v = self.dev.download(&h2)?;
                            let out = host_moe_forward(hx, &table, &h2v, seq, cfg.hidden, slots)
                                .map_err(|e| WgpuError::Device(format!("cpu_moe forward: {e}")))?;
                            self.dev.upload(&out)
                        }
                    };
                    let routed = if expert_range.is_some() {
                        self.tp_reduce(
                            routed,
                            (cfg.layer_start + li) as u32,
                            &format!("moe_out:{pos0}"),
                        )?
                    } else {
                        routed
                    };
                    match shared.as_deref() {
                        Some((sg, su, sd)) => {
                            let g = sg.matmul_t(&self.dev, &h2, seq)?;
                            let u = su.matmul_t(&self.dev, &h2, seq)?;
                            let gu = self.dev.silu_mul(&g, &u)?;
                            let s = sd.matmul_t(&self.dev, &gu, seq)?;
                            self.dev.add(&routed, &s)?
                        }
                        None => routed,
                    }
                }
            };
            let d = match &blk.post_ffn_norm {
                Some(w) => self.dev.rms_norm(&d, w, seq, cfg.hidden, cfg.rms_eps)?,
                None => d,
            };
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
