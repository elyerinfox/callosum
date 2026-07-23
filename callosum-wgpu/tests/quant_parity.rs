//! Per-format quantized-matmul parity: every supported GGML format,
//! both the matmul grid (m > 1) and the matvec reduction (m == 1),
//! against a CPU reference computed from callosum's own
//! quantize→dequantize round-trip — the exact bytes and the exact
//! dequant semantics the shaders must reproduce.

use callosum_wgpu::{enumerate_adapters, QuantDtype, WgpuDevice};

fn device() -> Option<WgpuDevice> {
    if enumerate_adapters().is_empty() {
        eprintln!("callosum-wgpu: no adapter, skipping");
        return None;
    }
    let idx = std::env::var("CALLOSUM_WGPU_ADAPTER")
        .ok()
        .and_then(|v| v.parse().ok());
    WgpuDevice::new(idx).ok()
}

fn ggml_of(fmt: QuantDtype) -> callosum::quantized::GgmlDType {
    use callosum::quantized::GgmlDType as G;
    match fmt {
        QuantDtype::Q4_0 => G::Q4_0,
        QuantDtype::Q8_0 => G::Q8_0,
        QuantDtype::Q4K => G::Q4K,
        QuantDtype::Q5K => G::Q5K,
        QuantDtype::Q6K => G::Q6K,
    }
}

#[test]
fn all_quant_formats_match_dequantized_reference() {
    let Some(dev) = device() else { return };
    eprintln!(
        "quant parity on {} [{} / {}]",
        dev.info().name,
        dev.info().vendor,
        dev.info().backend
    );
    let cpu = callosum::Device::Cpu;
    // k covers several K-quant super-blocks; n is deliberately odd.
    let (m, k, n) = (3usize, 512usize, 13usize);
    let x = (0..m * k)
        .map(|i| ((i as f32) * 0.83).sin())
        .collect::<Vec<f32>>();
    // Weights with per-block dynamic range to stress scale handling.
    let w_dense: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.31).cos() * (1.0 + ((i / 96) % 5) as f32))
        .collect();

    for fmt in QuantDtype::ALL {
        let wt = callosum::Tensor::from_vec(w_dense.clone(), (n, k), &cpu).unwrap();
        let qt = callosum::quantized::QTensor::quantize(&wt, ggml_of(fmt)).unwrap();
        let raw = qt.data().unwrap();
        let wd: Vec<f32> = qt
            .dequantize(&cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let gw = dev.upload_quantized(&raw, n, k, fmt).unwrap();

        // m > 1: matmul grid.
        let gx = dev.upload(&x);
        let got = dev
            .download(&dev.matmul_t_quant(&gx, &gw, m, k).unwrap())
            .unwrap();
        for r in 0..m {
            for c in 0..n {
                let want: f32 = (0..k).map(|i| x[r * k + i] * wd[c * k + i]).sum();
                let g = got[r * n + c];
                assert!(
                    (want - g).abs() < 2e-3 * want.abs().max(1.0),
                    "{fmt:?} matmul [{r},{c}]: want {want}, got {g}"
                );
            }
        }

        // m == 1: matvec reduction path.
        let gx1 = dev.upload(&x[..k]);
        let got = dev
            .download(&dev.matmul_t_quant(&gx1, &gw, 1, k).unwrap())
            .unwrap();
        for c in 0..n {
            let want: f32 = (0..k).map(|i| x[i] * wd[c * k + i]).sum();
            let g = got[c];
            assert!(
                (want - g).abs() < 2e-3 * want.abs().max(1.0),
                "{fmt:?} matvec [{c}]: want {want}, got {g}"
            );
        }
        eprintln!("  {fmt:?}: matmul + matvec OK");
    }
}
