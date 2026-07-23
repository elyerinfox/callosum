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
    let qt = callosum::quantized::QTensor::quantize(&wt, callosum::quantized::GgmlDType::Q8_0).unwrap();
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
    let mut cpu_model = callosum_models::models::quantized_qwen3::ModelWeights::from_gguf(
        content, &mut f, &cpu,
    )
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
    let mut cpu_model = callosum_models::models::quantized_qwen2::ModelWeights::from_gguf(
        content, &mut f, &cpu,
    )
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
    let mut cpu_model = callosum_models::models::quantized_llama::ModelWeights::from_gguf(
        content, &mut f, &cpu,
    )
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
