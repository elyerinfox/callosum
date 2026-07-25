//! Decode-throughput measurement, not a pass/fail test. Gated on
//! SMOLLM2_GGUF (any llama-family GGUF works); prints tok/s for a
//! prefill + N greedy decode steps so kernel work has an honest
//! before/after number. Run with `--nocapture`.

use callosum_wgpu::{enumerate_adapters, llama::WgpuLlama, WgpuDevice};

#[test]
fn decode_throughput() {
    let Some(path) = std::env::var_os("SMOLLM2_GGUF") else {
        eprintln!("SMOLLM2_GGUF not set, skipping bench");
        return;
    };
    if enumerate_adapters().is_empty() {
        eprintln!("no adapter, skipping bench");
        return;
    }
    let idx = std::env::var("CALLOSUM_WGPU_ADAPTER")
        .ok()
        .and_then(|v| v.parse().ok());
    let dev = WgpuDevice::new(idx).unwrap();
    eprintln!(
        "bench on {} [{} / {}]",
        dev.info().name,
        dev.info().vendor,
        dev.info().backend
    );
    let model = WgpuLlama::from_gguf(std::path::Path::new(&path), &dev).unwrap();
    let mut session = model.new_session(512);

    let prompt: Vec<u32> = (0..32u32).map(|i| 100 + i * 7).collect();
    let t0 = std::time::Instant::now();
    let logits = model.forward(&mut session, &prompt).unwrap();
    let prefill = t0.elapsed();

    let mut tok = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;

    // Warm-up decode step (pipeline caches etc.), then timed steps.
    let l = model.forward(&mut session, &[tok]).unwrap();
    tok = l
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;

    let n = 64usize;
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        let l = model.forward(&mut session, &[tok]).unwrap();
        tok = l
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
    }
    let decode = t1.elapsed();
    eprintln!(
        "prefill: {} tokens in {:.1} ms ({:.1} tok/s) | decode: {} tokens in {:.1} ms ({:.1} tok/s)",
        prompt.len(),
        prefill.as_secs_f64() * 1e3,
        prompt.len() as f64 / prefill.as_secs_f64(),
        n,
        decode.as_secs_f64() * 1e3,
        n as f64 / decode.as_secs_f64(),
    );
}

/// Per-format matvec microbenchmark at big-model dims — effective
/// GB/s so kernel work has an honest number per quant format.
#[test]
fn matvec_format_throughput() {
    use callosum_wgpu::QuantDtype;
    if enumerate_adapters().is_empty() {
        return;
    }
    let dev = WgpuDevice::new(None).unwrap();
    eprintln!("matvec bench on {}", dev.info().name);
    let cpu = callosum::Device::Cpu;
    for (n, k) in [(4096usize, 4096usize), (27392, 4096), (4096, 14336)] {
        eprintln!(" shape [{n} x {k}]");
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.31).cos()).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.83).sin()).collect();
        for fmt in [
            QuantDtype::Q8_0,
            QuantDtype::Q4_0,
            QuantDtype::Q4K,
            QuantDtype::Q5K,
            QuantDtype::Q6K,
            QuantDtype::Q2K,
            QuantDtype::Q3K,
            QuantDtype::F16,
        ] {
            let g = |f: QuantDtype| -> callosum::quantized::GgmlDType {
                use callosum::quantized::GgmlDType as G;
                match f {
                    QuantDtype::Q8_0 => G::Q8_0,
                    QuantDtype::Q4_0 => G::Q4_0,
                    QuantDtype::Q4K => G::Q4K,
                    QuantDtype::Q5K => G::Q5K,
                    QuantDtype::Q6K => G::Q6K,
                    QuantDtype::Q2K => G::Q2K,
                    QuantDtype::Q3K => G::Q3K,
                    _ => G::F16,
                }
            };
            let wt = callosum::Tensor::from_vec(w.clone(), (n, k), &cpu).unwrap();
            let qt = callosum::quantized::QTensor::quantize(&wt, g(fmt)).unwrap();
            let raw = qt.data().unwrap();
            let bytes = raw.len();
            let gw = dev.upload_quantized(&raw, n, k, fmt).unwrap();
            let gx = dev.upload(&x);
            // warmup
            for _ in 0..3 {
                let o = dev.matmul_t_quant(&gx, &gw, 1, k).unwrap();
                dev.download(&o).unwrap();
            }
            // Enqueue many dispatches, sync once — decode forwards batch
            // ~hundreds of dispatches per submission, so per-call sync
            // would only measure submit latency. Two rounds: the first
            // warms the uniform pool + bind cache (one-time creations),
            // the second is the steady-state number.
            let iters = 200;
            for round in 0..2 {
                let t = std::time::Instant::now();
                dev.begin_batch();
                let mut outs = Vec::with_capacity(iters);
                for _ in 0..iters {
                    outs.push(dev.matmul_t_quant(&gx, &gw, 1, k).unwrap());
                }
                let t_enqueue = t.elapsed().as_secs_f64();
                std::hint::black_box(dev.download(outs.last().unwrap()).unwrap());
                drop(outs);
                if round == 0 {
                    continue;
                }
                let dt = t.elapsed().as_secs_f64() / iters as f64;
                eprintln!(
                    "  {fmt:?}: {:.3} ms/matvec ({:.3} ms enqueue)  {:.0} GB/s effective",
                    dt * 1e3,
                    t_enqueue / iters as f64 * 1e3,
                    bytes as f64 / dt / 1e9
                );
            }
        }
    }
}

/// Fixed cost of a small chained dispatch (rms-norm-sized): the
/// per-token overhead floor is dispatch_count x this number.
#[test]
fn small_dispatch_overhead() {
    if enumerate_adapters().is_empty() {
        return;
    }
    let dev = WgpuDevice::new(None).unwrap();
    let a = dev.upload(&vec![1.0f32; 4096]);
    let b = dev.upload(&vec![2.0f32; 4096]);
    // warm
    dev.begin_batch();
    let mut o = dev.add(&a, &b).unwrap();
    for _ in 0..999 {
        o = dev.add(&o, &b).unwrap();
    }
    dev.download(&o).unwrap();
    let t = std::time::Instant::now();
    dev.begin_batch();
    let mut o = dev.add(&a, &b).unwrap();
    for _ in 0..999 {
        o = dev.add(&o, &b).unwrap();
    }
    std::hint::black_box(dev.download(&o).unwrap());
    let dt = t.elapsed().as_secs_f64();
    eprintln!("chained small dispatch: {:.1} us each", dt / 1000.0 * 1e6);
}

/// DeepSeek engine decode throughput (MLA + MoE path).
#[test]
fn deepseek_decode_throughput() {
    let Some(path) = std::env::var_os("DEEPSEEK_GGUF") else {
        return;
    };
    if enumerate_adapters().is_empty() {
        return;
    }
    let dev = WgpuDevice::new(None).unwrap();
    let model = callosum_wgpu::deepseek::WgpuDeepSeek::from_gguf(std::path::Path::new(&path), &dev)
        .unwrap();
    let mut session = model.new_session(512);
    let prompt: Vec<u32> = (0..32u32).map(|i| 100 + i * 7).collect();
    let t0 = std::time::Instant::now();
    let logits = model.forward(&mut session, &prompt).unwrap();
    let prefill = t0.elapsed();
    let mut tok = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;
    let _ = model.forward(&mut session, &[tok]).unwrap();
    let n = 32usize;
    let t1 = std::time::Instant::now();
    for _ in 0..n {
        let l = model.forward(&mut session, &[tok]).unwrap();
        tok = l
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
    }
    let dt = t1.elapsed();
    eprintln!(
        "deepseek prefill {:.1} ms ({:.1} tok/s) | decode {:.1} ms/tok ({:.1} tok/s)",
        prefill.as_secs_f64() * 1e3,
        32.0 / prefill.as_secs_f64(),
        dt.as_secs_f64() * 1e3 / n as f64,
        n as f64 / dt.as_secs_f64()
    );
}

/// cpu_moe decode throughput at engine level (MOE30B_GGUF).
#[test]
fn cpu_moe_decode_throughput() {
    let Some(path) = std::env::var_os("MOE30B_GGUF") else {
        return;
    };
    if enumerate_adapters().is_empty() {
        return;
    }
    let dev = WgpuDevice::new(None).unwrap();
    let model = WgpuLlama::from_gguf_cpu_moe(std::path::Path::new(&path), &dev).unwrap();
    let mut session = model.new_session(256);
    let prompt: Vec<u32> = (0..16u32).map(|i| 100 + i * 7).collect();
    let l = model.forward(&mut session, &prompt).unwrap();
    let mut tok = l
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;
    let n = 16usize;
    let t = std::time::Instant::now();
    for _ in 0..n {
        let l = model.forward(&mut session, &[tok]).unwrap();
        tok = l
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
    }
    eprintln!(
        "cpu_moe decode: {:.0} ms/tok ({:.1} tok/s)",
        t.elapsed().as_secs_f64() * 1e3 / n as f64,
        n as f64 / t.elapsed().as_secs_f64()
    );
}
