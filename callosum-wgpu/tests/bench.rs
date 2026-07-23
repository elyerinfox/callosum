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
