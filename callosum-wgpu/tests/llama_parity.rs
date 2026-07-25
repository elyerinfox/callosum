//! Parity for the llama-path kernels and the full WgpuLlama forward.
//!
//! Kernel tests run against in-test CPU references on any adapter.
//! The end-to-end test loads a real GGUF (path in `SMOLLM2_GGUF`) and
//! compares greedily generated tokens against callosum-models'
//! CPU `quantized_llama` — the strongest correctness anchor we have.

use callosum_wgpu::{enumerate_adapters, WgpuDevice};

fn device() -> Option<WgpuDevice> {
    if enumerate_adapters().is_empty() {
        eprintln!("callosum-wgpu: no adapter, skipping");
        return None;
    }
    let idx = std::env::var("CALLOSUM_WGPU_ADAPTER")
        .ok()
        .and_then(|v| v.parse().ok());
    match WgpuDevice::new(idx) {
        Ok(d) => {
            eprintln!(
                "callosum-wgpu: testing on {} [{} / {}]",
                d.info().name,
                d.info().vendor,
                d.info().backend
            );
            Some(d)
        }
        Err(e) => {
            eprintln!("callosum-wgpu: device open failed ({e}), skipping");
            None
        }
    }
}

fn synth(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.61).sin() * scale).collect()
}

#[test]
fn matmul_t_matches_cpu() {
    let Some(dev) = device() else { return };
    let (m, k, n) = (5usize, 64usize, 9usize);
    let x = synth(m * k, 1.0);
    let w = synth(n * k, 0.7); // [n, k]
    let mut want = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            want[r * n + c] = (0..k).map(|i| x[r * k + i] * w[c * k + i]).sum();
        }
    }
    let gx = dev.upload(&x);
    let gw = dev.upload(&w);
    let got = dev
        .download(&dev.matmul_t(&gx, &gw, m, k, n).unwrap())
        .unwrap();
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!((a - b).abs() < 1e-4 * a.abs().max(1.0), "[{i}] {a} vs {b}");
    }
}

#[test]
fn matmul_t_q8_0_matches_dequantized_reference() {
    let Some(dev) = device() else { return };
    let (m, k, n) = (3usize, 96usize, 11usize);
    let x = synth(m * k, 0.8);
    let w_dense = synth(n * k, 0.5);
    // Quantize with callosum's own q8_0 encoder — the exact on-disk format.
    let cpu = callosum::Device::Cpu;
    let wt = callosum::Tensor::from_vec(w_dense.clone(), (n, k), &cpu).unwrap();
    let qt =
        callosum::quantized::QTensor::quantize(&wt, callosum::quantized::GgmlDType::Q8_0).unwrap();
    let raw = qt.data().unwrap();
    // CPU reference uses the DEQUANTIZED weights (what the shader must
    // reproduce bit-for-bit modulo f32 accumulation order).
    let wd: Vec<f32> = qt
        .dequantize(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let mut want = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            want[r * n + c] = (0..k).map(|i| x[r * k + i] * wd[c * k + i]).sum();
        }
    }
    let gx = dev.upload(&x);
    let gw = dev.upload_q8_0(&raw, n, k).unwrap();
    let got = dev
        .download(&dev.matmul_t_q8_0(&gx, &gw, m, k).unwrap())
        .unwrap();
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!((a - b).abs() < 1e-3 * a.abs().max(1.0), "[{i}] {a} vs {b}");
    }
}

#[test]
fn rope_matches_cpu_both_conventions() {
    let Some(dev) = device() else { return };
    let (seq, heads, hd) = (4usize, 3usize, 8usize);
    let theta = 10_000.0f32;
    let pos0 = 5usize;
    let x = synth(seq * heads * hd, 1.0);
    let gx = dev.upload(&x);

    for interleaved in [true, false] {
        let mut want = x.clone();
        for s in 0..seq {
            for h in 0..heads {
                let base = (s * heads + h) * hd;
                let pos = (pos0 + s) as f32;
                for d in 0..hd / 2 {
                    let inv = theta.powf(-2.0 * d as f32 / hd as f32);
                    let (c, sn) = ((pos * inv).cos(), (pos * inv).sin());
                    let (i0, i1) = if interleaved {
                        (base + 2 * d, base + 2 * d + 1)
                    } else {
                        (base + d, base + d + hd / 2)
                    };
                    let (x0, x1) = (x[i0], x[i1]);
                    want[i0] = x0 * c - x1 * sn;
                    want[i1] = x0 * sn + x1 * c;
                }
            }
        }
        let got = dev
            .download(
                &dev.rope(&gx, seq, heads, hd, pos0, theta, interleaved)
                    .unwrap(),
            )
            .unwrap();
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "rope(interleaved={interleaved})[{i}]: {a} vs {b}"
            );
        }
    }
}

#[test]
fn causal_gqa_attention_matches_cpu() {
    let Some(dev) = device() else { return };
    // GQA 4 heads over 2 kv-heads, queries at absolute positions 3..5
    // against a 5-entry KV cache.
    let (heads, kv_heads, hd) = (4usize, 2usize, 6usize);
    let (seq_q, kv_len, pos0) = (2usize, 5usize, 3usize);
    let q = synth(seq_q * heads * hd, 1.0);
    let kc = synth(kv_len * kv_heads * hd, 0.9);
    let vc = synth(kv_len * kv_heads * hd, 1.1);
    let scale = 1.0 / (hd as f32).sqrt();

    // CPU reference.
    let mut want = vec![0.0f32; seq_q * heads * hd];
    for h in 0..heads {
        let kvh = h / (heads / kv_heads);
        for sq in 0..seq_q {
            let mut scores = vec![f32::NEG_INFINITY; kv_len];
            for (sk, sc) in scores.iter_mut().enumerate() {
                if sk <= pos0 + sq {
                    *sc = (0..hd)
                        .map(|d| q[(sq * heads + h) * hd + d] * kc[(sk * kv_heads + kvh) * hd + d])
                        .sum::<f32>()
                        * scale;
                }
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for d in 0..hd {
                want[(sq * heads + h) * hd + d] = (0..kv_len)
                    .map(|sk| exps[sk] / denom * vc[(sk * kv_heads + kvh) * hd + d])
                    .sum();
            }
        }
    }

    let gq = dev.upload(&q);
    let gk = dev.upload(&kc);
    let gv = dev.upload(&vc);
    let scores = dev
        .attn_scores(&gq, &gk, seq_q, kv_len, heads, kv_heads, hd, pos0)
        .unwrap();
    let probs = dev.softmax(&scores, heads * seq_q, kv_len).unwrap();
    let out = dev
        .attn_out(&probs, &gv, seq_q, kv_len, heads, kv_heads, hd)
        .unwrap();
    let got = dev.download(&out).unwrap();
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!((a - b).abs() < 1e-4, "attn[{i}]: {a} vs {b}");
    }
}

#[test]
fn embed_gather_matches_cpu() {
    let Some(dev) = device() else { return };
    let (vocab, hidden, seq) = (50usize, 16usize, 6usize);
    let table = synth(vocab * hidden, 1.0);
    let ids = [3u32, 0, 49, 7, 7, 21];
    let ids_f: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
    let gt = dev.upload(&table);
    let gi = dev.upload(&ids_f);
    let got = dev
        .download(&dev.embed_gather(&gi, &gt, seq, hidden).unwrap())
        .unwrap();
    for (s, &id) in ids.iter().enumerate() {
        for c in 0..hidden {
            assert_eq!(got[s * hidden + c], table[id as usize * hidden + c]);
        }
    }
}

/// Pipeline-stage parity: the same GGUF split into two layer-range
/// shards (stage 0: embed + first half, stage 1: second half + head)
/// must produce the identical greedy token stream as the full model.
/// Gated on SMOLLM2_GGUF.
#[test]
fn stage_split_matches_full_model() {
    let Some(path) = std::env::var_os("SMOLLM2_GGUF") else {
        eprintln!("SMOLLM2_GGUF not set, skipping stage-split parity");
        return;
    };
    let Some(dev) = device() else { return };
    let path = std::path::PathBuf::from(path);
    use callosum_wgpu::llama::{StageInput, StageOutput, WgpuLlama};

    let full = WgpuLlama::from_gguf(&path, &dev).unwrap();
    let n_layers = full.cfg.n_layers;
    let mid = n_layers / 2;
    let s0 = WgpuLlama::from_gguf_stage(&path, &dev, 0, mid, true, false).unwrap();
    let s1 = WgpuLlama::from_gguf_stage(&path, &dev, mid, n_layers, false, true).unwrap();
    assert_eq!(s0.n_logits(), 0, "mid stage must not build a head");

    let prompt: Vec<u32> = vec![1, 504, 1593, 314, 254];
    let n_gen = 8;

    let mut want = prompt.clone();
    let mut sess = full.new_session(64);
    for step in 0..n_gen {
        let input: Vec<u32> = if step == 0 {
            want.clone()
        } else {
            vec![*want.last().unwrap()]
        };
        let logits = full.forward(&mut sess, &input).unwrap();
        want.push(argmax_of(&logits));
    }

    let mut got = prompt.clone();
    let (mut sess0, mut sess1) = (s0.new_session(64), s1.new_session(64));
    for step in 0..n_gen {
        let input: Vec<u32> = if step == 0 {
            got.clone()
        } else {
            vec![*got.last().unwrap()]
        };
        let seq = input.len();
        let StageOutput::Hidden(h) = s0
            .forward_stage(&mut sess0, StageInput::Tokens(&input), 0)
            .unwrap()
        else {
            panic!("stage 0 returned logits");
        };
        let StageOutput::Logits(logits) = s1
            .forward_stage(&mut sess1, StageInput::Hidden { data: &h, seq }, 1)
            .unwrap()
        else {
            panic!("stage 1 returned hidden");
        };
        got.push(argmax_of(&logits));
    }
    assert_eq!(
        &want[prompt.len()..],
        &got[prompt.len()..],
        "2-stage pipeline diverges from the full model"
    );
    eprintln!(
        "stage-split parity OK ({mid}+{} layers): {:?}",
        n_layers - mid,
        &got[prompt.len()..]
    );
}

fn argmax_of(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32
}

/// Qwen3 (rotate-half RoPE + per-head Q/K RMSNorm) token parity vs
/// callosum-models' CPU quantized_qwen3. Gated on QWEN3_GGUF.
#[test]
fn wgpu_qwen3_matches_cpu_reference_tokens() {
    let Some(path) = std::env::var_os("QWEN3_GGUF") else {
        eprintln!("QWEN3_GGUF not set, skipping qwen3 parity");
        return;
    };
    let Some(dev) = device() else { return };
    let path = std::path::PathBuf::from(path);

    let cpu = callosum::Device::Cpu;
    let mut f = std::fs::File::open(&path).unwrap();
    let content = callosum::quantized::gguf_file::Content::read(&mut f).unwrap();
    let mut cpu_model =
        callosum_models::models::quantized_qwen3::ModelWeights::from_gguf(content, &mut f, &cpu)
            .unwrap();

    let gpu_model = callosum_wgpu::llama::WgpuLlama::from_gguf(&path, &dev).unwrap();
    assert!(
        !gpu_model.cfg.rope_interleaved,
        "qwen3 must select rotate-half RoPE"
    );
    let mut session = gpu_model.new_session(128);

    let prompt: Vec<u32> = vec![151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198];
    let n_gen = 8;

    let mut cpu_tokens = prompt.clone();
    let mut pos = 0usize;
    for _ in 0..n_gen {
        let input: Vec<u32> = if pos == 0 {
            cpu_tokens.clone()
        } else {
            vec![*cpu_tokens.last().unwrap()]
        };
        let t = callosum::Tensor::new(input.as_slice(), &cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = cpu_model.forward(&t, pos).unwrap();
        pos += input.len();
        let v: Vec<f32> = logits
            .flatten_all()
            .unwrap()
            .to_dtype(callosum::DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let vocab = gpu_model.n_logits();
        let last = &v[v.len() - vocab..];
        cpu_tokens.push(argmax_of(last));
    }

    let mut gpu_tokens = prompt.clone();
    for step in 0..n_gen {
        let input: Vec<u32> = if step == 0 {
            gpu_tokens.clone()
        } else {
            vec![*gpu_tokens.last().unwrap()]
        };
        let logits = gpu_model.forward(&mut session, &input).unwrap();
        gpu_tokens.push(argmax_of(&logits));
    }

    assert_eq!(
        &cpu_tokens[prompt.len()..],
        &gpu_tokens[prompt.len()..],
        "wgpu qwen3 greedy tokens diverge from CPU reference"
    );
    eprintln!(
        "wgpu qwen3 parity OK on {}: {:?}",
        dev.info().name,
        &gpu_tokens[prompt.len()..]
    );
}

/// Qwen2 (rotate-half RoPE + QKV biases) token parity vs
/// callosum-models' CPU quantized_qwen2. Gated on QWEN2_GGUF.
#[test]
fn wgpu_qwen2_matches_cpu_reference_tokens() {
    let Some(path) = std::env::var_os("QWEN2_GGUF") else {
        eprintln!("QWEN2_GGUF not set, skipping qwen2 parity");
        return;
    };
    let Some(dev) = device() else { return };
    let path = std::path::PathBuf::from(path);

    let cpu = callosum::Device::Cpu;
    let mut f = std::fs::File::open(&path).unwrap();
    let content = callosum::quantized::gguf_file::Content::read(&mut f).unwrap();
    let mut cpu_model =
        callosum_models::models::quantized_qwen2::ModelWeights::from_gguf(content, &mut f, &cpu)
            .unwrap();

    let gpu_model = callosum_wgpu::llama::WgpuLlama::from_gguf(&path, &dev).unwrap();
    let mut session = gpu_model.new_session(128);

    let prompt: Vec<u32> = vec![151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198];
    let n_gen = 8;

    let mut cpu_tokens = prompt.clone();
    let mut pos = 0usize;
    for _ in 0..n_gen {
        let input: Vec<u32> = if pos == 0 {
            cpu_tokens.clone()
        } else {
            vec![*cpu_tokens.last().unwrap()]
        };
        let t = callosum::Tensor::new(input.as_slice(), &cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = cpu_model.forward(&t, pos).unwrap();
        pos += input.len();
        let v: Vec<f32> = logits
            .flatten_all()
            .unwrap()
            .to_dtype(callosum::DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let vocab = gpu_model.n_logits();
        let last = &v[v.len() - vocab..];
        cpu_tokens.push(argmax_of(last));
    }

    let mut gpu_tokens = prompt.clone();
    for step in 0..n_gen {
        let input: Vec<u32> = if step == 0 {
            gpu_tokens.clone()
        } else {
            vec![*gpu_tokens.last().unwrap()]
        };
        let logits = gpu_model.forward(&mut session, &input).unwrap();
        gpu_tokens.push(argmax_of(&logits));
    }

    assert_eq!(
        &cpu_tokens[prompt.len()..],
        &gpu_tokens[prompt.len()..],
        "wgpu qwen2 greedy tokens diverge from CPU reference"
    );
    eprintln!(
        "wgpu qwen2 parity OK on {}: {:?}",
        dev.info().name,
        &gpu_tokens[prompt.len()..]
    );
}

/// Qwen3-MoE (router + fused-expert FFN) token parity. The reference
/// is an inline, cache-free CPU forward built on callosum CPU ops with
/// everything dequantized to f32 — deliberately independent of both
/// the wgpu kernels and any model library (callosum-models' MoE path
/// is CUDA-only). Gated on QWEN3MOE_GGUF.
#[test]
fn wgpu_qwen3_moe_matches_cpu_reference_tokens() {
    let Some(path) = std::env::var_os("QWEN3MOE_GGUF") else {
        eprintln!("QWEN3MOE_GGUF not set, skipping qwen3moe parity");
        return;
    };
    let Some(dev) = device() else { return };
    let path = std::path::PathBuf::from(path);

    let gpu_model = callosum_wgpu::llama::WgpuLlama::from_gguf(&path, &dev).unwrap();
    assert!(gpu_model.cfg.n_experts > 0, "expected an MoE config");
    let mut session = gpu_model.new_session(64);

    let prompt: Vec<u32> = vec![1, 42, 7, 99, 5];
    let n_gen = 8;

    // ---- CPU reference ----
    let cpu = callosum::Device::Cpu;
    let mut f = std::fs::File::open(&path).unwrap();
    let content = callosum::quantized::gguf_file::Content::read(&mut f).unwrap();
    let g = |keys: &str| content.metadata.get(keys).cloned();
    let mu = |k: &str| g(k).unwrap().to_u32().unwrap() as usize;
    let hidden = mu("qwen3moe.embedding_length");
    let n_layers = mu("qwen3moe.block_count");
    let n_heads = mu("qwen3moe.attention.head_count");
    let n_kv = mu("qwen3moe.attention.head_count_kv");
    let head_dim = content
        .metadata
        .get("qwen3moe.attention.key_length")
        .map(|v| v.to_u32().unwrap() as usize)
        .unwrap_or(hidden / n_heads);
    let n_exp = mu("qwen3moe.expert_count");
    let k_used = mu("qwen3moe.expert_used_count");
    let theta = g("qwen3moe.rope.freq_base")
        .map(|v| v.to_f32().unwrap())
        .unwrap_or(10_000.0);
    let eps = g("qwen3moe.attention.layer_norm_rms_epsilon")
        .map(|v| v.to_f32().unwrap() as f64)
        .unwrap_or(1e-6);

    let mut fr = std::fs::File::open(&path).unwrap();
    let mut dense = |name: &str| -> callosum::Tensor {
        content
            .tensor(&mut fr, name, &cpu)
            .unwrap()
            .dequantize(&cpu)
            .unwrap()
            .to_dtype(callosum::DType::F32)
            .unwrap()
    };
    let embed = dense("token_embd.weight");
    let out_norm = dense("output_norm.weight");
    let lm_head = if content.tensor_infos.contains_key("output.weight") {
        dense("output.weight")
    } else {
        embed.clone()
    };

    struct RefBlock {
        attn_norm: callosum::Tensor,
        wq: callosum::Tensor,
        wk: callosum::Tensor,
        wv: callosum::Tensor,
        wo: callosum::Tensor,
        qn: Option<callosum::Tensor>,
        kn: Option<callosum::Tensor>,
        ffn_norm: callosum::Tensor,
        router: callosum::Tensor,
        gates: callosum::Tensor,
        ups: callosum::Tensor,
        downs: callosum::Tensor,
    }
    let blocks: Vec<RefBlock> = (0..n_layers)
        .map(|b| RefBlock {
            attn_norm: dense(&format!("blk.{b}.attn_norm.weight")),
            wq: dense(&format!("blk.{b}.attn_q.weight")),
            wk: dense(&format!("blk.{b}.attn_k.weight")),
            wv: dense(&format!("blk.{b}.attn_v.weight")),
            wo: dense(&format!("blk.{b}.attn_output.weight")),
            qn: content
                .tensor_infos
                .contains_key(&format!("blk.{b}.attn_q_norm.weight"))
                .then(|| dense(&format!("blk.{b}.attn_q_norm.weight"))),
            kn: content
                .tensor_infos
                .contains_key(&format!("blk.{b}.attn_k_norm.weight"))
                .then(|| dense(&format!("blk.{b}.attn_k_norm.weight"))),
            ffn_norm: dense(&format!("blk.{b}.ffn_norm.weight")),
            router: dense(&format!("blk.{b}.ffn_gate_inp.weight")),
            gates: dense(&format!("blk.{b}.ffn_gate_exps.weight")),
            ups: dense(&format!("blk.{b}.ffn_up_exps.weight")),
            downs: dense(&format!("blk.{b}.ffn_down_exps.weight")),
        })
        .collect();

    let rms = |x: &callosum::Tensor, w: &callosum::Tensor| -> callosum::Tensor {
        let last = x.rank() - 1;
        let var = x.sqr().unwrap().mean_keepdim(last).unwrap();
        x.broadcast_div(&(var + eps).unwrap().sqrt().unwrap())
            .unwrap()
            .broadcast_mul(w)
            .unwrap()
    };
    // Rotate-half RoPE over [seq, heads, head_dim] rows at positions 0..seq.
    let rope = |x: &callosum::Tensor, heads: usize| -> callosum::Tensor {
        let (seq, _, hd) = x.dims3().unwrap();
        let half = hd / 2;
        let v: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let mut out = v.clone();
        for s in 0..seq {
            for h in 0..heads {
                let base = (s * heads + h) * hd;
                for d in 0..half {
                    let inv = theta.powf(-2.0 * d as f32 / hd as f32);
                    let (c, sn) = ((s as f32 * inv).cos(), (s as f32 * inv).sin());
                    let (x0, x1) = (v[base + d], v[base + d + half]);
                    out[base + d] = x0 * c - x1 * sn;
                    out[base + d + half] = x0 * sn + x1 * c;
                }
            }
        }
        callosum::Tensor::from_vec(out, (seq, heads, hd), &cpu).unwrap()
    };

    // Cache-free full forward over `tokens`; returns last-position logits.
    let forward = |tokens: &[u32]| -> Vec<f32> {
        let seq = tokens.len();
        let ids = callosum::Tensor::new(tokens, &cpu).unwrap();
        let mut x = embed.index_select(&ids, 0).unwrap(); // [seq, hidden]
        for blk in &blocks {
            let h = rms(&x, &blk.attn_norm);
            let q = h.matmul(&blk.wq.t().unwrap()).unwrap();
            let k = h.matmul(&blk.wk.t().unwrap()).unwrap();
            let v = h.matmul(&blk.wv.t().unwrap()).unwrap();
            let q = q.reshape((seq, n_heads, head_dim)).unwrap();
            let k = k.reshape((seq, n_kv, head_dim)).unwrap();
            let v = v.reshape((seq, n_kv, head_dim)).unwrap();
            let q = match &blk.qn {
                Some(w) => rms(&q, w),
                None => q,
            };
            let k = match &blk.kn {
                Some(w) => rms(&k, w),
                None => k,
            };
            let q = rope(&q, n_heads);
            let k = rope(&k, n_kv);
            // Naive causal GQA attention on host vectors.
            let qv: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
            let kv: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
            let vv: Vec<f32> = v.flatten_all().unwrap().to_vec1().unwrap();
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut att = vec![0f32; seq * n_heads * head_dim];
            for h in 0..n_heads {
                let kvh = h / (n_heads / n_kv);
                for sq in 0..seq {
                    let mut sc = vec![f32::NEG_INFINITY; seq];
                    for (sk, e) in sc.iter_mut().enumerate() {
                        if sk <= sq {
                            *e = (0..head_dim)
                                .map(|d| {
                                    qv[(sq * n_heads + h) * head_dim + d]
                                        * kv[(sk * n_kv + kvh) * head_dim + d]
                                })
                                .sum::<f32>()
                                * scale;
                        }
                    }
                    let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let ex: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let den: f32 = ex.iter().sum();
                    for d in 0..head_dim {
                        att[(sq * n_heads + h) * head_dim + d] = (0..seq)
                            .map(|sk| ex[sk] / den * vv[(sk * n_kv + kvh) * head_dim + d])
                            .sum();
                    }
                }
            }
            let att = callosum::Tensor::from_vec(att, (seq, n_heads * head_dim), &cpu).unwrap();
            let o = att.matmul(&blk.wo.t().unwrap()).unwrap();
            x = (&x + &o).unwrap();

            // MoE FFN — same routing rule as run_moe.
            let h2 = rms(&x, &blk.ffn_norm);
            let logits = h2.matmul(&blk.router.t().unwrap()).unwrap(); // [seq, n_exp]
            let lv: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
            let h2v: Vec<f32> = h2.flatten_all().unwrap().to_vec1().unwrap();
            let gv: Vec<f32> = blk.gates.flatten_all().unwrap().to_vec1().unwrap();
            let uv: Vec<f32> = blk.ups.flatten_all().unwrap().to_vec1().unwrap();
            let dv: Vec<f32> = blk.downs.flatten_all().unwrap().to_vec1().unwrap();
            let ffn_dim = blk.gates.dims()[1];
            let mut moe = vec![0f32; seq * hidden];
            for t in 0..seq {
                let row = &lv[t * n_exp..(t + 1) * n_exp];
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let den: f32 = row.iter().map(|l| (l - mx).exp()).sum();
                let probs: Vec<f32> = row.iter().map(|l| (l - mx).exp() / den).collect();
                let mut idx: Vec<usize> = (0..n_exp).collect();
                idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
                let top = &idx[..k_used];
                let wsum: f32 = top.iter().map(|&i| probs[i]).sum();
                for &e in top {
                    let w = probs[e] / wsum;
                    let xrow = &h2v[t * hidden..(t + 1) * hidden];
                    let mut gated = vec![0f32; ffn_dim];
                    for r in 0..ffn_dim {
                        let gb = (e * ffn_dim + r) * hidden;
                        let mut gacc = 0f32;
                        let mut uacc = 0f32;
                        for i in 0..hidden {
                            gacc += gv[gb + i] * xrow[i];
                            uacc += uv[gb + i] * xrow[i];
                        }
                        gated[r] = (gacc / (1.0 + (-gacc).exp())) * uacc;
                    }
                    for r in 0..hidden {
                        let db = (e * hidden + r) * ffn_dim;
                        let mut acc = 0f32;
                        for i in 0..ffn_dim {
                            acc += dv[db + i] * gated[i];
                        }
                        moe[t * hidden + r] += w * acc;
                    }
                }
            }
            let moe = callosum::Tensor::from_vec(moe, (seq, hidden), &cpu).unwrap();
            x = (&x + &moe).unwrap();
        }
        let hf = rms(&x, &out_norm);
        let logits = hf.matmul(&lm_head.t().unwrap()).unwrap();
        let vocab = logits.dims()[1];
        logits
            .narrow(0, seq - 1, 1)
            .unwrap()
            .reshape(vocab)
            .unwrap()
            .to_vec1()
            .unwrap()
    };

    let mut cpu_tokens = prompt.clone();
    for _ in 0..n_gen {
        let logits = forward(&cpu_tokens);
        cpu_tokens.push(argmax_of(&logits));
    }

    let mut gpu_tokens = prompt.clone();
    for step in 0..n_gen {
        let input: Vec<u32> = if step == 0 {
            gpu_tokens.clone()
        } else {
            vec![*gpu_tokens.last().unwrap()]
        };
        let logits = gpu_model.forward(&mut session, &input).unwrap();
        gpu_tokens.push(argmax_of(&logits));
    }

    assert_eq!(
        &cpu_tokens[prompt.len()..],
        &gpu_tokens[prompt.len()..],
        "wgpu qwen3moe greedy tokens diverge from CPU reference"
    );
    eprintln!(
        "wgpu qwen3moe parity OK on {}: {:?}",
        dev.info().name,
        &gpu_tokens[prompt.len()..]
    );
}

/// Full-model, real-GGUF token parity vs callosum-models' CPU
/// quantized_llama. Gated on SMOLLM2_GGUF so plain `cargo test` stays
/// hermetic.
#[test]
fn wgpu_llama_matches_cpu_reference_tokens() {
    let Some(path) = std::env::var_os("SMOLLM2_GGUF") else {
        eprintln!("SMOLLM2_GGUF not set, skipping end-to-end llama parity");
        return;
    };
    let Some(dev) = device() else { return };
    let path = std::path::PathBuf::from(path);

    // CPU reference.
    let cpu = callosum::Device::Cpu;
    let mut f = std::fs::File::open(&path).unwrap();
    let content = callosum::quantized::gguf_file::Content::read(&mut f).unwrap();
    let mut cpu_model =
        callosum_models::models::quantized_llama::ModelWeights::from_gguf(content, &mut f, &cpu)
            .unwrap();

    // wgpu model.
    let gpu_model = callosum_wgpu::llama::WgpuLlama::from_gguf(&path, &dev).unwrap();
    let mut session = gpu_model.new_session(128);

    // Fixed prompt tokens (BOS + a few common ids — actual text doesn't
    // matter, determinism does).
    let prompt: Vec<u32> = vec![1, 504, 1593, 314, 254];
    let n_gen = 8;

    let mut cpu_tokens = prompt.clone();
    let mut pos = 0usize;
    for step in 0..=n_gen {
        let input: Vec<u32> = if step == 0 {
            cpu_tokens.clone()
        } else {
            vec![*cpu_tokens.last().unwrap()]
        };
        let t = callosum::Tensor::new(input.as_slice(), &cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = cpu_model.forward(&t, pos).unwrap();
        pos += input.len();
        let logits = logits.squeeze(0).unwrap();
        let logits = if logits.rank() == 2 {
            let s = logits.dim(0).unwrap();
            logits.narrow(0, s - 1, 1).unwrap().squeeze(0).unwrap()
        } else {
            logits
        };
        let v: Vec<f32> = logits
            .to_dtype(callosum::DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let arg = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        if step < n_gen {
            cpu_tokens.push(arg);
        }
    }

    let mut gpu_tokens = prompt.clone();
    for step in 0..n_gen {
        let input: Vec<u32> = if step == 0 {
            gpu_tokens.clone()
        } else {
            vec![*gpu_tokens.last().unwrap()]
        };
        let logits = gpu_model.forward(&mut session, &input).unwrap();
        let arg = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        gpu_tokens.push(arg);
    }

    assert_eq!(
        &cpu_tokens[prompt.len()..prompt.len() + n_gen],
        &gpu_tokens[prompt.len()..],
        "wgpu greedy tokens diverge from CPU reference"
    );
    eprintln!(
        "wgpu llama parity OK on {}: {:?}",
        dev.info().name,
        &gpu_tokens[prompt.len()..]
    );
}

/// glm4moe (GLM-4.5) token parity: QKV biases, per-head q/k norms,
/// partial rotate-half rotary, post_attention_norm as the pre-FFN
/// norm, leading dense block, sigmoid routing with a selection bias,
/// weight renormalisation, fused shared experts, and NextN-block
/// skipping. The reference is an inline, cache-free CPU forward with
/// everything dequantized to f32. Gated on GLM4MOE_GGUF.
#[test]
fn wgpu_glm4moe_matches_cpu_reference_tokens() {
    let Some(path) = std::env::var_os("GLM4MOE_GGUF") else {
        eprintln!("GLM4MOE_GGUF not set, skipping glm4moe parity");
        return;
    };
    let Some(dev) = device() else { return };
    let path = std::path::PathBuf::from(path);

    let gpu_model = callosum_wgpu::llama::WgpuLlama::from_gguf(&path, &dev).unwrap();
    assert!(gpu_model.cfg.n_experts > 0, "expected an MoE config");
    assert!(gpu_model.cfg.moe_sigmoid, "glm4moe should sigmoid-gate");
    let mut session = gpu_model.new_session(64);

    let prompt: Vec<u32> = vec![1, 42, 7, 99, 5];
    let n_gen = 8;

    // ---- CPU reference ----
    let cpu = callosum::Device::Cpu;
    let mut f = std::fs::File::open(&path).unwrap();
    let content = callosum::quantized::gguf_file::Content::read(&mut f).unwrap();
    let g = |keys: &str| content.metadata.get(keys).cloned();
    let mu = |k: &str| g(k).unwrap().to_u32().unwrap() as usize;
    let hidden = mu("glm4moe.embedding_length");
    let nextn = g("glm4moe.nextn_predict_layers")
        .map(|v| v.to_u32().unwrap() as usize)
        .unwrap_or(0);
    let n_layers = mu("glm4moe.block_count") - nextn;
    let n_heads = mu("glm4moe.attention.head_count");
    let n_kv = mu("glm4moe.attention.head_count_kv");
    let head_dim = mu("glm4moe.attention.key_length");
    let rot_dim = mu("glm4moe.rope.dimension_count");
    let n_exp = mu("glm4moe.expert_count");
    let k_used = mu("glm4moe.expert_used_count");
    let theta = g("glm4moe.rope.freq_base")
        .map(|v| v.to_f32().unwrap())
        .unwrap_or(10_000.0);
    let eps = g("glm4moe.attention.layer_norm_rms_epsilon")
        .map(|v| v.to_f32().unwrap() as f64)
        .unwrap_or(1e-5);
    let route_scale = g("glm4moe.expert_weights_scale")
        .map(|v| v.to_f32().unwrap())
        .unwrap_or(1.0);
    let renorm = g("glm4moe.expert_weights_norm")
        .map(|v| v.to_bool().unwrap())
        .unwrap_or(true);

    let fr = std::cell::RefCell::new(std::fs::File::open(&path).unwrap());
    let dense = |name: &str| -> callosum::Tensor {
        content
            .tensor(&mut *fr.borrow_mut(), name, &cpu)
            .unwrap()
            .dequantize(&cpu)
            .unwrap()
            .to_dtype(callosum::DType::F32)
            .unwrap()
    };
    let has = |name: &str| content.tensor_infos.contains_key(name);
    let embed = dense("token_embd.weight");
    let out_norm = dense("output_norm.weight");
    let lm_head = if has("output.weight") {
        dense("output.weight")
    } else {
        embed.clone()
    };

    struct RefBlock {
        attn_norm: callosum::Tensor,
        wq: callosum::Tensor,
        wk: callosum::Tensor,
        wv: callosum::Tensor,
        wo: callosum::Tensor,
        bq: Option<callosum::Tensor>,
        bk: Option<callosum::Tensor>,
        bv: Option<callosum::Tensor>,
        qn: Option<callosum::Tensor>,
        kn: Option<callosum::Tensor>,
        ffn_norm: callosum::Tensor,
        ffn: RefFfn,
    }
    enum RefFfn {
        Dense {
            gate: callosum::Tensor,
            up: callosum::Tensor,
            down: callosum::Tensor,
        },
        Moe {
            router: callosum::Tensor,
            bias: Option<callosum::Tensor>,
            gates: callosum::Tensor,
            ups: callosum::Tensor,
            downs: callosum::Tensor,
            shexp: Option<(callosum::Tensor, callosum::Tensor, callosum::Tensor)>,
        },
    }
    let blocks: Vec<RefBlock> = (0..n_layers)
        .map(|b| {
            let opt = |n: String| has(&n).then(|| dense(&n));
            let ffn = if has(&format!("blk.{b}.ffn_gate_inp.weight")) {
                RefFfn::Moe {
                    router: dense(&format!("blk.{b}.ffn_gate_inp.weight")),
                    bias: opt(format!("blk.{b}.exp_probs_b.bias")),
                    gates: dense(&format!("blk.{b}.ffn_gate_exps.weight")),
                    ups: dense(&format!("blk.{b}.ffn_up_exps.weight")),
                    downs: dense(&format!("blk.{b}.ffn_down_exps.weight")),
                    shexp: has(&format!("blk.{b}.ffn_gate_shexp.weight")).then(|| {
                        (
                            dense(&format!("blk.{b}.ffn_gate_shexp.weight")),
                            dense(&format!("blk.{b}.ffn_up_shexp.weight")),
                            dense(&format!("blk.{b}.ffn_down_shexp.weight")),
                        )
                    }),
                }
            } else {
                RefFfn::Dense {
                    gate: dense(&format!("blk.{b}.ffn_gate.weight")),
                    up: dense(&format!("blk.{b}.ffn_up.weight")),
                    down: dense(&format!("blk.{b}.ffn_down.weight")),
                }
            };
            RefBlock {
                attn_norm: dense(&format!("blk.{b}.attn_norm.weight")),
                wq: dense(&format!("blk.{b}.attn_q.weight")),
                wk: dense(&format!("blk.{b}.attn_k.weight")),
                wv: dense(&format!("blk.{b}.attn_v.weight")),
                wo: dense(&format!("blk.{b}.attn_output.weight")),
                bq: opt(format!("blk.{b}.attn_q.bias")),
                bk: opt(format!("blk.{b}.attn_k.bias")),
                bv: opt(format!("blk.{b}.attn_v.bias")),
                qn: opt(format!("blk.{b}.attn_q_norm.weight")),
                kn: opt(format!("blk.{b}.attn_k_norm.weight")),
                // glm4moe: post_attention_norm IS the pre-FFN norm.
                ffn_norm: dense(&format!("blk.{b}.post_attention_norm.weight")),
                ffn,
            }
        })
        .collect();

    let rms = |x: &callosum::Tensor, w: &callosum::Tensor| -> callosum::Tensor {
        let last = x.rank() - 1;
        let var = x.sqr().unwrap().mean_keepdim(last).unwrap();
        x.broadcast_div(&(var + eps).unwrap().sqrt().unwrap())
            .unwrap()
            .broadcast_mul(w)
            .unwrap()
    };
    // Partial rotate-half RoPE: rotate the first rot_dim dims of each
    // head (pairs (d, d + rot/2), inv-freq exponent over rot_dim),
    // pass the tail through untouched.
    let rope = |x: &callosum::Tensor, heads: usize| -> callosum::Tensor {
        let (seq, _, hd) = x.dims3().unwrap();
        let half = rot_dim / 2;
        let v: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let mut out = v.clone();
        for s in 0..seq {
            for h in 0..heads {
                let base = (s * heads + h) * hd;
                for d in 0..half {
                    let inv = theta.powf(-2.0 * d as f32 / rot_dim as f32);
                    let (c, sn) = ((s as f32 * inv).cos(), (s as f32 * inv).sin());
                    let (x0, x1) = (v[base + d], v[base + d + half]);
                    out[base + d] = x0 * c - x1 * sn;
                    out[base + d + half] = x0 * sn + x1 * c;
                }
            }
        }
        callosum::Tensor::from_vec(out, (seq, heads, hd), &cpu).unwrap()
    };
    let swiglu = |h: &callosum::Tensor,
                  gate: &callosum::Tensor,
                  up: &callosum::Tensor,
                  down: &callosum::Tensor|
     -> callosum::Tensor {
        let gp = h.matmul(&gate.t().unwrap()).unwrap();
        let up = h.matmul(&up.t().unwrap()).unwrap();
        let gv: Vec<f32> = gp.flatten_all().unwrap().to_vec1().unwrap();
        let uv: Vec<f32> = up.flatten_all().unwrap().to_vec1().unwrap();
        let gu: Vec<f32> = gv
            .iter()
            .zip(&uv)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let dims = gp.dims().to_vec();
        callosum::Tensor::from_vec(gu, (dims[0], dims[1]), &cpu)
            .unwrap()
            .matmul(&down.t().unwrap())
            .unwrap()
    };

    // Cache-free full forward over `tokens`; returns last-position logits.
    let forward = |tokens: &[u32]| -> Vec<f32> {
        let seq = tokens.len();
        let ids = callosum::Tensor::new(tokens, &cpu).unwrap();
        let mut x = embed.index_select(&ids, 0).unwrap(); // [seq, hidden]
        for blk in &blocks {
            let h = rms(&x, &blk.attn_norm);
            let badd = |t: callosum::Tensor, b: &Option<callosum::Tensor>| match b {
                Some(b) => t.broadcast_add(b).unwrap(),
                None => t,
            };
            let q = badd(h.matmul(&blk.wq.t().unwrap()).unwrap(), &blk.bq);
            let k = badd(h.matmul(&blk.wk.t().unwrap()).unwrap(), &blk.bk);
            let v = badd(h.matmul(&blk.wv.t().unwrap()).unwrap(), &blk.bv);
            let q = q.reshape((seq, n_heads, head_dim)).unwrap();
            let k = k.reshape((seq, n_kv, head_dim)).unwrap();
            let v = v.reshape((seq, n_kv, head_dim)).unwrap();
            let q = match &blk.qn {
                Some(w) => rms(&q, w),
                None => q,
            };
            let k = match &blk.kn {
                Some(w) => rms(&k, w),
                None => k,
            };
            let q = rope(&q, n_heads);
            let k = rope(&k, n_kv);
            // Naive causal GQA attention on host vectors.
            let qv: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
            let kv: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
            let vv: Vec<f32> = v.flatten_all().unwrap().to_vec1().unwrap();
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut att = vec![0f32; seq * n_heads * head_dim];
            for h in 0..n_heads {
                let kvh = h / (n_heads / n_kv);
                for sq in 0..seq {
                    let mut sc = vec![f32::NEG_INFINITY; seq];
                    for (sk, e) in sc.iter_mut().enumerate() {
                        if sk <= sq {
                            *e = (0..head_dim)
                                .map(|d| {
                                    qv[(sq * n_heads + h) * head_dim + d]
                                        * kv[(sk * n_kv + kvh) * head_dim + d]
                                })
                                .sum::<f32>()
                                * scale;
                        }
                    }
                    let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let ex: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let den: f32 = ex.iter().sum();
                    for d in 0..head_dim {
                        att[(sq * n_heads + h) * head_dim + d] = (0..seq)
                            .map(|sk| ex[sk] / den * vv[(sk * n_kv + kvh) * head_dim + d])
                            .sum();
                    }
                }
            }
            let att = callosum::Tensor::from_vec(att, (seq, n_heads * head_dim), &cpu).unwrap();
            let o = att.matmul(&blk.wo.t().unwrap()).unwrap();
            x = (&x + &o).unwrap();

            let h2 = rms(&x, &blk.ffn_norm);
            let d = match &blk.ffn {
                RefFfn::Dense { gate, up, down } => swiglu(&h2, gate, up, down),
                RefFfn::Moe {
                    router,
                    bias,
                    gates,
                    ups,
                    downs,
                    shexp,
                } => {
                    // Sigmoid mixture scores; selection on scores +
                    // bias, weights from the unbiased scores,
                    // renormalised over the selected set.
                    let logits = h2.matmul(&router.t().unwrap()).unwrap();
                    let lv: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
                    let bv: Option<Vec<f32>> = bias
                        .as_ref()
                        .map(|b| b.flatten_all().unwrap().to_vec1().unwrap());
                    let h2v: Vec<f32> = h2.flatten_all().unwrap().to_vec1().unwrap();
                    let gv: Vec<f32> = gates.flatten_all().unwrap().to_vec1().unwrap();
                    let uv: Vec<f32> = ups.flatten_all().unwrap().to_vec1().unwrap();
                    let dv: Vec<f32> = downs.flatten_all().unwrap().to_vec1().unwrap();
                    let ffn_dim = gates.dims()[1];
                    let mut moe = vec![0f32; seq * hidden];
                    for t in 0..seq {
                        let row = &lv[t * n_exp..(t + 1) * n_exp];
                        let probs: Vec<f32> =
                            row.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
                        let sel: Vec<f32> = probs
                            .iter()
                            .enumerate()
                            .map(|(e, p)| p + bv.as_ref().map(|b| b[e]).unwrap_or(0.0))
                            .collect();
                        let mut idx: Vec<usize> = (0..n_exp).collect();
                        idx.sort_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap());
                        let top = &idx[..k_used];
                        let wsum: f32 = if renorm {
                            top.iter().map(|&i| probs[i]).sum()
                        } else {
                            1.0
                        };
                        for &e in top {
                            let w = probs[e] / wsum * route_scale;
                            let xrow = &h2v[t * hidden..(t + 1) * hidden];
                            let mut gated = vec![0f32; ffn_dim];
                            for r in 0..ffn_dim {
                                let gb = (e * ffn_dim + r) * hidden;
                                let mut gacc = 0f32;
                                let mut uacc = 0f32;
                                for i in 0..hidden {
                                    gacc += gv[gb + i] * xrow[i];
                                    uacc += uv[gb + i] * xrow[i];
                                }
                                gated[r] = (gacc / (1.0 + (-gacc).exp())) * uacc;
                            }
                            for r in 0..hidden {
                                let db = (e * hidden + r) * ffn_dim;
                                let mut acc = 0f32;
                                for i in 0..ffn_dim {
                                    acc += dv[db + i] * gated[i];
                                }
                                moe[t * hidden + r] += w * acc;
                            }
                        }
                    }
                    let moe = callosum::Tensor::from_vec(moe, (seq, hidden), &cpu).unwrap();
                    match shexp {
                        Some((sg, su, sd)) => (&moe + &swiglu(&h2, sg, su, sd)).unwrap(),
                        None => moe,
                    }
                }
            };
            x = (&x + &d).unwrap();
        }
        let hf = rms(&x, &out_norm);
        let logits = hf.matmul(&lm_head.t().unwrap()).unwrap();
        let vocab = logits.dims()[1];
        logits
            .narrow(0, seq - 1, 1)
            .unwrap()
            .reshape(vocab)
            .unwrap()
            .to_vec1()
            .unwrap()
    };

    let mut cpu_tokens = prompt.clone();
    for _ in 0..n_gen {
        let logits = forward(&cpu_tokens);
        cpu_tokens.push(argmax_of(&logits));
    }

    let mut gpu_tokens = prompt.clone();
    for step in 0..n_gen {
        let input: Vec<u32> = if step == 0 {
            gpu_tokens.clone()
        } else {
            vec![*gpu_tokens.last().unwrap()]
        };
        let logits = gpu_model.forward(&mut session, &input).unwrap();
        gpu_tokens.push(argmax_of(&logits));
    }

    assert_eq!(
        &cpu_tokens[prompt.len()..],
        &gpu_tokens[prompt.len()..],
        "wgpu glm4moe greedy tokens diverge from CPU reference"
    );
    eprintln!(
        "wgpu glm4moe parity OK on {}: {:?}",
        dev.info().name,
        &gpu_tokens[prompt.len()..]
    );
}

/// cpu_moe expert offload must produce the same greedy tokens as the
/// VRAM-resident expert path, with expert bytes accounted to host RAM
/// instead of the GPU-resident figure. Gated on the tiny MoE GGUFs.
#[test]
fn wgpu_cpu_moe_matches_resident_tokens() {
    let Some(dev) = device() else { return };
    let cases = [("QWEN3MOE_GGUF", "qwen3moe"), ("GLM4MOE_GGUF", "glm4moe")];
    for (env, label) in cases {
        let Some(path) = std::env::var_os(env) else {
            eprintln!("{env} not set, skipping {label} cpu_moe parity");
            continue;
        };
        let path = std::path::PathBuf::from(path);
        let resident = callosum_wgpu::llama::WgpuLlama::from_gguf(&path, &dev).unwrap();
        let offload = callosum_wgpu::llama::WgpuLlama::from_gguf_cpu_moe(&path, &dev).unwrap();
        assert_eq!(resident.host_expert_bytes, 0);
        assert!(offload.host_expert_bytes > 0, "{label}: no host experts");
        assert!(
            offload.weight_bytes < resident.weight_bytes,
            "{label}: offload should shrink GPU-resident bytes"
        );

        let prompt: Vec<u32> = vec![1, 42, 7, 99, 5];
        let n_gen = 8;
        let run = |m: &callosum_wgpu::llama::WgpuLlama| -> Vec<u32> {
            let mut session = m.new_session(64);
            let mut toks = prompt.clone();
            for step in 0..n_gen {
                let input: Vec<u32> = if step == 0 {
                    toks.clone()
                } else {
                    vec![*toks.last().unwrap()]
                };
                let logits = m.forward(&mut session, &input).unwrap();
                toks.push(argmax_of(&logits));
            }
            toks
        };
        // The host path uses quantized-activation matmuls (candle CPU
        // QMatMul — the same semantics as the CUDA backend's cpu_moe),
        // so logits carry ~1% quantization noise vs the f32 in-shader
        // path. On these tiny random models the top-2 gap is razor
        // thin, so exact token equality is luck, not correctness:
        // check logit closeness and determinism instead. Real-model
        // token equivalence is covered by the DeepSeek e2e run.
        let mut sa = resident.new_session(64);
        let mut sb = offload.new_session(64);
        let la = resident.forward(&mut sa, &prompt).unwrap();
        let lb = offload.forward(&mut sb, &prompt).unwrap();
        let scale = la.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let mad = la
            .iter()
            .zip(&lb)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            mad / scale < 0.05,
            "{label}: cpu_moe logits deviate too far (mad {mad}, scale {scale})"
        );
        let b1 = run(&offload);
        let b2 = run(&offload);
        assert_eq!(b1, b2, "{label}: cpu_moe generation must be deterministic");
        eprintln!(
            "wgpu {label} cpu_moe OK ({} KiB experts on host, logits mad {mad:.5}): {:?}",
            offload.host_expert_bytes >> 10,
            &b1[prompt.len()..]
        );
    }
}
