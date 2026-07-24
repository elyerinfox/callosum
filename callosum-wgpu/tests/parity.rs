//! CPU-parity tests for every callosum-wgpu kernel. Skipped (pass with a
//! note) on machines without any compute adapter so headless CI stays
//! green.

use callosum_wgpu::{enumerate_adapters, WgpuDevice};

fn device() -> Option<WgpuDevice> {
    if enumerate_adapters().is_empty() {
        eprintln!("callosum-wgpu: no adapter, skipping GPU parity tests");
        return None;
    }
    match WgpuDevice::new(None) {
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
    (0..n).map(|i| ((i as f32) * 0.37).sin() * scale).collect()
}

fn assert_close(want: &[f32], got: &[f32], tol: f32, what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert!(
            (w - g).abs() <= tol * w.abs().max(1.0),
            "{what}[{i}]: want {w}, got {g}"
        );
    }
}

#[test]
fn matmul_matches_cpu() {
    let Some(dev) = device() else { return };
    let (m, k, n) = (33usize, 29usize, 47usize);
    let a = synth(m * k, 1.0);
    let b = synth(k * n, 0.5);
    let mut want = vec![0.0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0;
            for i in 0..k {
                acc += a[r * k + i] * b[i * n + c];
            }
            want[r * n + c] = acc;
        }
    }
    let ga = dev.upload(&a);
    let gb = dev.upload(&b);
    let out = dev.matmul(&ga, &gb, m, k, n).unwrap();
    let got = dev.download(&out).unwrap();
    assert_close(&want, &got, 1e-4, "matmul");
}

#[test]
fn elementwise_matches_cpu() {
    let Some(dev) = device() else { return };
    let a = synth(1000, 2.0);
    let b = synth(1000, 3.0);
    let ga = dev.upload(&a);
    let gb = dev.upload(&b);

    let got = dev.download(&dev.add(&ga, &gb).unwrap()).unwrap();
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_close(&want, &got, 1e-6, "add");

    let got = dev.download(&dev.mul(&ga, &gb).unwrap()).unwrap();
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    assert_close(&want, &got, 1e-6, "mul");

    let got = dev.download(&dev.silu(&ga).unwrap()).unwrap();
    let want: Vec<f32> = a.iter().map(|x| x / (1.0 + (-x).exp())).collect();
    assert_close(&want, &got, 1e-5, "silu");
}

#[test]
fn rms_norm_matches_cpu() {
    let Some(dev) = device() else { return };
    let (rows, k) = (7usize, 300usize);
    let eps = 1e-6f32;
    let a = synth(rows * k, 1.5);
    let w = synth(k, 1.0);
    let mut want = vec![0.0f32; rows * k];
    for r in 0..rows {
        let row = &a[r * k..(r + 1) * k];
        let ms: f32 = row.iter().map(|x| x * x).sum::<f32>() / k as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..k {
            want[r * k + i] = row[i] * inv * w[i];
        }
    }
    let ga = dev.upload(&a);
    let gw = dev.upload(&w);
    let out = dev.rms_norm(&ga, &gw, rows, k, eps).unwrap();
    let got = dev.download(&out).unwrap();
    assert_close(&want, &got, 1e-4, "rms_norm");
}

#[test]
fn softmax_matches_cpu() {
    let Some(dev) = device() else { return };
    let (rows, k) = (5usize, 411usize);
    let a = synth(rows * k, 4.0);
    let mut want = vec![0.0f32; rows * k];
    for r in 0..rows {
        let row = &a[r * k..(r + 1) * k];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|x| (x - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for i in 0..k {
            want[r * k + i] = exps[i] / sum;
        }
    }
    let ga = dev.upload(&a);
    let out = dev.softmax(&ga, rows, k).unwrap();
    let got = dev.download(&out).unwrap();
    assert_close(&want, &got, 1e-5, "softmax");
    // Rows sum to 1.
    for r in 0..rows {
        let s: f32 = got[r * k..(r + 1) * k].iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "row {r} sums to {s}");
    }
}

#[test]
fn dedup_keeps_identical_cards_apart() {
    use callosum_wgpu::{dedup_adapter_indices, AdapterDesc};
    let mk = |index, name: &str, backend: &str, device_type: &str| AdapterDesc {
        index,
        name: name.into(),
        vendor: "Intel".into(),
        backend: backend.into(),
        device_type: device_type.into(),
    };
    // 3 identical Arcs, each listed under Vulkan AND DX12, plus a
    // software rasterizer: dedup must keep exactly the 3 Vulkan
    // entries and drop everything else.
    let adapters = vec![
        mk(0, "Intel(R) Arc(TM) A770", "Vulkan", "DiscreteGpu"),
        mk(1, "Intel(R) Arc(TM) A770", "Vulkan", "DiscreteGpu"),
        mk(2, "Intel(R) Arc(TM) A770", "Vulkan", "DiscreteGpu"),
        mk(3, "Intel(R) Arc(TM) A770", "Dx12", "DiscreteGpu"),
        mk(4, "Intel(R) Arc(TM) A770", "Dx12", "DiscreteGpu"),
        mk(5, "Intel(R) Arc(TM) A770", "Dx12", "DiscreteGpu"),
        mk(6, "Microsoft Basic Render Driver", "Dx12", "Cpu"),
    ];
    assert_eq!(dedup_adapter_indices(&adapters), vec![0, 1, 2]);

    // Mixed box: NVIDIA (Vulkan+DX12) + AMD iGPU (Vulkan+DX12) keeps
    // one Vulkan entry per card.
    let adapters = vec![
        mk(0, "AMD Radeon(TM) Graphics", "Vulkan", "IntegratedGpu"),
        mk(1, "NVIDIA GeForce RTX 3090", "Vulkan", "DiscreteGpu"),
        mk(2, "AMD Radeon(TM) Graphics", "Dx12", "IntegratedGpu"),
        mk(3, "NVIDIA GeForce RTX 3090", "Dx12", "DiscreteGpu"),
        mk(4, "Microsoft Basic Render Driver", "Dx12", "Cpu"),
        // GL mangles the device name, so it can't be name-grouped with
        // its Vulkan twin — it must be excluded outright.
        mk(5, "NVIDIA GeForce RTX 3090/PCIe/SSE2", "Gl", "DiscreteGpu"),
    ];
    assert_eq!(dedup_adapter_indices(&adapters), vec![0, 1]);
}
