# callosum

Pure-Rust tensor and inference engine. The corpus callosum is the bridge
between brain hemispheres — callosum is the compute layer that
[splitbrain](https://git.foxcorp.lol/dumbfox/splitbrain)'s distributed
pipeline stages run on.

**No C bindings anywhere.** Every backend is reached from Rust:

| crate | what it is |
|---|---|
| `callosum-core` | tensors, ops, autograd, quantized (GGUF/GGML) types; CPU + CUDA backends |
| `callosum-nn` | layers + fused ops (softmax-last-dim, rms-norm, rotary embeddings) |
| `callosum-kernels` | CUDA kernels (compiled with nvcc via `bindgen_cuda`, linked from Rust) |
| `callosum-metal-kernels` | Metal kernels for the macOS backend |
| `callosum-wgpu` | WGSL compute kernels via wgpu — one build serves NVIDIA, AMD, Intel Arc, and Apple through Vulkan/DX12/Metal |
| `callosum-models` | reference model implementations (used as CPU parity anchors in tests) |
| `callosum-ug` | micro-kernel codegen glue |

## callosum-wgpu

The multi-vendor engine: quantized llama-family inference (llama /
mistral / qwen2 / qwen3) with q4_0, q8_0, q4_K, q5_K, q6_K weights kept
at on-disk density in VRAM and dequantized in-shader. Supports
full-model and pipeline-stage (layer-range) loading. Single-submission
forwards inside one long-lived compute pass, pooled buffers, bind-group
caching, workgroup-reduction matvecs.

Every kernel has a CPU-parity test; end-to-end tests prove
token-for-token greedy parity against the `callosum-models` CPU
references on real GGUFs (`SMOLLM2_GGUF`, `QWEN2_GGUF`, `QWEN3_GGUF`
env vars; `CALLOSUM_WGPU_ADAPTER` picks the adapter):

```bash
cargo test -p callosum-wgpu --release
```

## License

MIT or Apache-2.0, at your option. See `NOTICE.md` for provenance.
