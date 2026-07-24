//! Gemma-family parity for the wgpu engine.
//!
//! gemma3 has an independent CPU reference in callosum-models
//! (`quantized_gemma3`) — greedy tokens must match on a real GGUF
//! (`GEMMA3_GGUF`). gemma 1/2/4 have no CPU reference here; their
//! anchor is cross-backend equality against the CUDA worker, exercised
//! by splitbrain's e2e flow.

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

fn argmax_of(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32
}

#[test]
fn wgpu_gemma3_matches_cpu_reference_tokens() {
    let Some(path) = std::env::var_os("GEMMA3_GGUF") else {
        eprintln!("GEMMA3_GGUF not set, skipping gemma3 parity");
        return;
    };
    let Some(dev) = device() else { return };
    let path = std::path::PathBuf::from(path);

    let cpu = callosum::Device::Cpu;
    let mut f = std::fs::File::open(&path).unwrap();
    let content = callosum::quantized::gguf_file::Content::read(&mut f).unwrap();
    let mut cpu_model =
        callosum_models::models::quantized_gemma3::ModelWeights::from_gguf(content, &mut f, &cpu)
            .unwrap();

    let gpu_model = callosum_wgpu::gemma::WgpuGemma::from_gguf(&path, &dev).unwrap();
    assert_eq!(gpu_model.cfg.arch, "gemma3");
    let mut session = gpu_model.new_session(128);

    // <bos>The capital of France is (gemma tokenizer ids don't matter —
    // determinism does; 2 is <bos> for gemma).
    let prompt: Vec<u32> = vec![2, 818, 5279, 529, 8161, 563];
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
        cpu_tokens.push(argmax_of(&v[v.len() - vocab..]));
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
        "wgpu gemma3 greedy tokens diverge from CPU reference"
    );
    eprintln!(
        "wgpu gemma3 parity OK on {}: {:?}",
        dev.info().name,
        &gpu_tokens[prompt.len()..]
    );
}

/// CPU-parity for the gemma-path kernels — separated so a vendor
/// driver quirk in one kernel is directly attributable.
#[test]
fn gemma_kernels_match_cpu() {
    let Some(dev) = device() else { return };
    let n = 37usize;
    // Large magnitudes on purpose: gemma activations reach 1e4+, and
    // driver tanh implementations NaN on overflow without clamping.
    let x: Vec<f32> = (0..n * 4)
        .map(|i| ((i as f32) * 0.37).sin() * if i % 5 == 0 { 30000.0 } else { 2.0 })
        .collect();
    let gx = dev.upload(&x);

    // mul_bias with a length-1 bias (scalar broadcast).
    let s = dev.upload(&[3.25f32]);
    let got = dev.download(&dev.mul_bias(&gx, &s).unwrap()).unwrap();
    for (i, (a, b)) in x.iter().zip(&got).enumerate() {
        assert!(
            (a * 3.25 - b).abs() < 1e-5,
            "mul_bias scalar [{i}]: {} vs {b}",
            a * 3.25
        );
    }

    // mul_bias with a full-width bias.
    let bias: Vec<f32> = (0..n * 4).map(|i| (i as f32) * 0.01 + 0.5).collect();
    let gb = dev.upload(&bias);
    let got = dev.download(&dev.mul_bias(&gx, &gb).unwrap()).unwrap();
    for i in 0..x.len() {
        assert!((x[i] * bias[i] - got[i]).abs() < 1e-5, "mul_bias [{i}]");
    }

    // gelu + gelu_mul.
    fn gelu_ref(v: f32) -> f32 {
        0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v * v * v)).tanh())
    }
    let got = dev.download(&dev.gelu(&gx).unwrap()).unwrap();
    for i in 0..x.len() {
        assert!(
            (gelu_ref(x[i]) - got[i]).abs() < 1e-4,
            "gelu [{i}]: {} vs {}",
            gelu_ref(x[i]),
            got[i]
        );
    }
    let got = dev.download(&dev.gelu_mul(&gx, &gb).unwrap()).unwrap();
    for i in 0..x.len() {
        assert!(
            (gelu_ref(x[i]) * bias[i] - got[i]).abs() < 1e-4,
            "gelu_mul [{i}]"
        );
    }

    // softcap.
    let got = dev.download(&dev.softcap(&gx, 50.0).unwrap()).unwrap();
    for i in 0..x.len() {
        let want = 50.0 * (x[i] / 50.0).tanh();
        assert!((want - got[i]).abs() < 1e-4, "softcap [{i}]");
    }

    // slice_cols.
    let rows = 4usize;
    let stride = n;
    let m = dev.upload(&x[..rows * stride]);
    let got = dev
        .download(&dev.slice_cols(&m, rows, stride, 5, 7).unwrap())
        .unwrap();
    for r in 0..rows {
        for j in 0..7 {
            assert_eq!(
                got[r * 7 + j],
                x[r * stride + 5 + j],
                "slice_cols [{r},{j}]"
            );
        }
    }
    eprintln!("gemma kernel parity OK on {}", dev.info().name);
}

/// Layer-bisection probe for vendor-specific numerical issues: loads
/// GEMMA_PROBE_GGUF as a growing input-stage prefix and reports the
/// first layer whose hidden output goes non-finite.
#[test]
fn gemma_probe_layers() {
    let Some(path) = std::env::var_os("GEMMA_PROBE_GGUF") else {
        return;
    };
    let Some(dev) = device() else { return };
    let path = std::path::PathBuf::from(path);
    use callosum_wgpu::llama::{StageInput, StageOutput};
    for le in [0usize, 1, 2, 4, 8, 13, 18, 26] {
        let m = if le == 0 {
            // Embed-only probe via layer range [0,1) but inspecting after upload isn't
            // exposed; skip 0 and start at 1.
            continue;
        } else {
            callosum_wgpu::gemma::WgpuGemma::from_gguf_stage(&path, &dev, 0, le, true, false)
                .unwrap()
        };
        let mut s = m.new_session(16);
        let out = m
            .forward_stage(&mut s, StageInput::Tokens(&[2, 100, 200]), 0)
            .unwrap();
        let StageOutput::Hidden(h) = out else {
            panic!("expected hidden")
        };
        let bad = h.iter().filter(|v| !v.is_finite()).count();
        let mx = h.iter().cloned().fold(0f32, |a, b| a.max(b.abs()));
        eprintln!(
            "layers 0..{le}: {} elems, non-finite {bad}, max|x| {mx:.3}",
            h.len()
        );
        if bad > 0 {
            break;
        }
    }
}

/// GLM-4 (partial interleaved rotary, fused gate+up SWIGLU, sandwich
/// norms, QKV biases) token parity vs callosum-models' CPU
/// quantized_glm4. Gated on GLM4_GGUF.
#[test]
fn wgpu_glm4_matches_cpu_reference_tokens() {
    let Some(path) = std::env::var_os("GLM4_GGUF") else {
        eprintln!("GLM4_GGUF not set, skipping glm4 parity");
        return;
    };
    let Some(dev) = device() else { return };
    let path = std::path::PathBuf::from(path);

    let cpu = callosum::Device::Cpu;
    let mut f = std::fs::File::open(&path).unwrap();
    let content = callosum::quantized::gguf_file::Content::read(&mut f).unwrap();
    let mut cpu_model = callosum_models::models::quantized_glm4::ModelWeights::from_gguf(
        content,
        &mut f,
        &cpu,
        callosum::DType::F32,
    )
    .unwrap();

    let gpu_model = callosum_wgpu::llama::WgpuLlama::from_gguf(&path, &dev).unwrap();
    assert!(
        gpu_model.cfg.rot_dim < gpu_model.cfg.head_dim,
        "partial rotary expected"
    );
    let mut session = gpu_model.new_session(64);

    // [gMASK]<sop><|user|>\nHello<|assistant|> token ids don't matter —
    // determinism does. Use small in-vocab ids.
    let prompt: Vec<u32> = vec![151331, 151333, 151336, 198, 9707, 151337];
    let n_gen = 6;

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
        cpu_tokens.push(argmax_of(&v[v.len() - vocab..]));
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
        "wgpu glm4 greedy tokens diverge from CPU reference"
    );
    eprintln!(
        "wgpu glm4 parity OK on {}: {:?}",
        dev.info().name,
        &gpu_tokens[prompt.len()..]
    );
}
