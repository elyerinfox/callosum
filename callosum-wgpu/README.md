# callosum-wgpu

Pure-Rust multi-vendor GPU compute for the splitbrain fork of callosum,
built on [wgpu](https://github.com/gfx-rs/wgpu). One build reaches
every vendor wgpu reaches — **Intel Arc**, **AMD**, **NVIDIA** (Vulkan
/ DX12), **Apple** (Metal) — with no C bindings anywhere in the chain.

## Status

**Runs real models.** `llama::WgpuLlama` loads a llama-family GGUF and
executes the full forward — embed → RoPE → causal GQA attention with a
KV cache → SwiGLU → logits — entirely on the wgpu adapter, with q8_0
weights kept **quantized in VRAM** (dequant-in-shader matmul, ~1.06
bytes/param). Other quant dtypes load via a dequantize-to-f32 fallback.

Verified with token-for-token greedy parity against callosum-models'
CPU `quantized_llama` on SmolLM2-135M-Q8_0, on **two vendors on one
machine**: NVIDIA RTX 3090 (Vulkan) and AMD Radeon iGPU (Vulkan) —
identical outputs. Intel Arc executes this same Vulkan path; there is
nothing vendor-specific anywhere in the chain.

**Quant coverage** — dequant-in-shader matmul + decode matvec for
`q4_0`, `q8_0`, `q4_K`, `q5_K`, `q6_K`: the formats real GGUFs ship
(Q4_K_M/S, Q5_K_M/S, Q6_K, Q8_0, Q4_0 models all load with weights at
on-disk density in VRAM). Every format is parity-tested against
callosum's own quantize→dequantize on both the matmul grid and the
matvec path, and a Q4_K_M model generates token-identically to the CPU
reference on NVIDIA and AMD alike.

**Perf architecture** — one command submission per forward (batched
encoder), one long-lived compute pass (KV appends are dispatches, not
encoder copies), pooled output buffers + a uniform ring (zero
per-dispatch allocations at steady state), bind-group caching, and
word-wise dequant inner loops. Decode routes matmuls to
workgroup-reduction matvec kernels. Measured on SmolLM2-135M / RTX
3090 / Vulkan: decode 50.7 → **153 tok/s** (3.0×), prefill 680 →
**1136 tok/s**, identical outputs throughout; Q4_K_M decodes at
~160 tok/s. (Small-model caveat: at hidden=576 the matvec workgroups
run mostly idle lanes — utilization improves inherently on
2048+-hidden models.)

Kernel set (each with a CPU-parity test, auto-skip headless):
`matmul`, `matmul_t`, `matmul_t_<quant>` ×5, `matvec_<quant>` ×5,
`matvec_f32`, `add`, `mul`, `silu`, `silu_mul`, `rms_norm`, `softmax`,
`embed_gather`, `rope_interleaved`, `rope_half`, `attn_scores`
(causal + GQA), `attn_out`, `copy_to`. Pick the adapter under test
with `CALLOSUM_WGPU_ADAPTER=<index>`; end-to-end tests take
`SMOLLM2_GGUF=<path to any llama-family gguf>`; `tests/bench.rs`
prints prefill/decode tok/s.

## Roadmap

1. **splitbrain `backend_wgpu`** — full-model replica lanes on any
   wgpu adapter via `WgpuLlama` (per-request sessions, truncate for
   speculative rollback).
2. **`callosum_core::Device::Wgpu`** — first-class backend via the
   `dummy_cuda_backend` pattern, removing per-backend model code.
3. **Perf, round 2** — tiled/shared-memory prefill matmul, f16
   activations, multi-column matvec workgroups for small hidden dims,
   i-quants (IQ4_XS etc.) if models demand them.
