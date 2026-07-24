//! Pure-Rust multi-vendor GPU compute for callosum, built on wgpu.
//!
//! One build reaches every vendor wgpu reaches: **Intel Arc** (Vulkan /
//! DX12), **AMD** (Vulkan / DX12 / Metal), **NVIDIA** (Vulkan / DX12),
//! **Apple** (Metal) — with no C bindings anywhere in the chain; wgpu,
//! naga, and this crate are Rust throughout.
//!
//! This crate is the compute engine for the splitbrain fork of callosum.
//! It starts life standalone — devices, buffers, and the core LLM
//! forward-pass kernels (matmul, elementwise, RMSNorm, softmax) with
//! CPU-parity tests — and grows toward full `callosum_core::Device`
//! integration (see README roadmap). Keeping it standalone first means
//! every op lands with a parity test before it's reachable from model
//! code.

pub mod gemma;
pub mod llama;

use std::sync::Arc;

use wgpu::util::DeviceExt;

#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("no compatible GPU adapter found")]
    NoAdapter,
    #[error("device request failed: {0}")]
    Device(String),
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("buffer readback failed: {0}")]
    Readback(String),
}

pub type Result<T> = std::result::Result<T, WgpuError>;

/// A visible compute adapter, in wgpu's preference order.
#[derive(Debug, Clone)]
pub struct AdapterDesc {
    pub index: usize,
    pub name: String,
    pub vendor: String,
    pub backend: String,
    pub device_type: String,
}

fn vendor_name(id: u32) -> String {
    match id {
        0x1002 => "AMD".to_string(),
        0x10DE => "NVIDIA".to_string(),
        0x8086 => "Intel".to_string(),
        0x106B => "Apple".to_string(),
        0x5143 => "Qualcomm".to_string(),
        0x13B5 => "ARM".to_string(),
        other => format!("vendor:{other:#06x}"),
    }
}

/// Enumerate every compute-capable adapter on the machine. This is the
/// multi-vendor discovery surface splitbrain workers will advertise
/// from: an Arc A770 and an RTX 3090 both show up here, through the
/// same code path.
pub fn enumerate_adapters() -> Vec<AdapterDesc> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .enumerate()
        .map(|(index, a)| {
            let info = a.get_info();
            AdapterDesc {
                index,
                name: info.name,
                vendor: vendor_name(info.vendor),
                backend: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
            }
        })
        .collect()
}

struct Pipelines {
    matmul: wgpu::ComputePipeline,
    matmul_t: wgpu::ComputePipeline,
    matvec_f32: wgpu::ComputePipeline,
    /// (format, is_matvec) -> pipeline, one pair per quant format.
    quant: std::collections::HashMap<(QuantDtype, bool), wgpu::ComputePipeline>,
    add: wgpu::ComputePipeline,
    add_bias: wgpu::ComputePipeline,
    mul: wgpu::ComputePipeline,
    mul_bias: wgpu::ComputePipeline,
    gelu: wgpu::ComputePipeline,
    gelu_mul: wgpu::ComputePipeline,
    softcap: wgpu::ComputePipeline,
    slice_cols: wgpu::ComputePipeline,
    moe_topk: wgpu::ComputePipeline,
    moe_combine: wgpu::ComputePipeline,
    /// Per-format expert-indexed matmul (MoE), plus an f32 fallback.
    expert: std::collections::HashMap<QuantDtype, wgpu::ComputePipeline>,
    expert_f32: wgpu::ComputePipeline,
    silu: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    embed_gather: wgpu::ComputePipeline,
    rope_interleaved: wgpu::ComputePipeline,
    rope_half: wgpu::ComputePipeline,
    attn_scores: wgpu::ComputePipeline,
    attn_out: wgpu::ComputePipeline,
    copy_to: wgpu::ComputePipeline,
    silu_mul: wgpu::ComputePipeline,
}

/// A GPU device + queue + compiled kernel set. Cheap to clone.
#[derive(Clone)]
pub struct WgpuDevice {
    inner: Arc<DeviceInner>,
}

struct DeviceInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    layout: wgpu::BindGroupLayout,
    /// When batching is active, every dispatch/copy records into this
    /// encoder and nothing hits the queue until [`WgpuDevice::flush`]
    /// (or a readback forces one). One submission per forward pass
    /// instead of one per op is the single largest decode-latency win:
    /// a llama layer is ~12 dispatches, so per-op submits cost hundreds
    /// of queue round-trips per token.
    batch: std::sync::Mutex<Option<Batch>>,
    /// Reusable output/scratch buffers, keyed by element count.
    /// `GpuBuffer::drop` returns storage here; `alloc_out` reuses it.
    /// Without this every op allocates a fresh device buffer — ~400
    /// creations per decode token, which dwarfs the GPU work itself.
    out_pool: std::sync::Mutex<std::collections::HashMap<usize, Vec<(wgpu::Buffer, u64)>>>,
    /// Ring of small uniform buffers for per-dispatch Params. Each
    /// dispatch takes the next slot and writes it via
    /// `queue.write_buffer` (ordered before the batch submission, and
    /// each slot is written at most once per batch, so batched
    /// dispatches all see their own values). The cursor resets on
    /// `begin_batch`.
    uniform_pool: std::sync::Mutex<(Vec<wgpu::Buffer>, usize)>,
    /// Bind groups keyed by the four bound buffers' global ids. With
    /// pooled buffers the same combinations recur every token, so this
    /// converges to a full hit rate after the first forward.
    bind_cache: std::sync::Mutex<
        std::collections::HashMap<(u64, u64, u64, u64, u64), std::sync::Arc<wgpu::BindGroup>>,
    >,
    pub info: AdapterDesc,
}

/// An open batch: one command encoder and (lazily) one long-lived
/// compute pass that successive dispatches share. Ending the pass only
/// around buffer copies (and at flush) removes the per-op pass
/// begin/end + full-barrier cost, which dominates decode at small
/// model sizes. SAFETY: `pass` borrows `enc`; we keep both in this
/// struct and always drop the pass before finishing the encoder.
struct Batch {
    enc: wgpu::CommandEncoder,
    pass: Option<wgpu::ComputePass<'static>>,
}

impl Batch {
    fn pass(&mut self) -> &mut wgpu::ComputePass<'static> {
        if self.pass.is_none() {
            let pass = self
                .enc
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default())
                .forget_lifetime();
            self.pass = Some(pass);
        }
        self.pass.as_mut().expect("just opened")
    }

    fn end_pass(&mut self) {
        self.pass = None;
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct Params {
    m: u32,
    n: u32,
    k: u32,
    len: u32,
    eps: f32,
    pos0: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    theta: f32,
    scale: f32,
    window: u32,
    fscale: f32,
    cap: f32,
    flags: u32,
    _pad: u32,
}

/// The adapter index [`WgpuDevice::new(None)`] would pick: the first
/// discrete GPU, else the first integrated one, else whatever exists.
/// Exposed so serving layers can advertise the same adapter the device
/// will actually open.
pub fn default_adapter_index() -> Option<usize> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let rank = |a: &wgpu::Adapter| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::Cpu => 3,
    };
    adapters
        .iter()
        .enumerate()
        .min_by_key(|(i, a)| (rank(a), *i))
        .map(|(i, _)| i)
}

/// Rank a backend string for dedup preference (lower = better).
fn backend_rank(b: &str) -> usize {
    match b {
        "Vulkan" => 0,
        "Metal" => 1,
        "Dx12" => 2,
        _ => 3,
    }
}

/// Pick one enumeration entry per *physical* adapter. The same card
/// shows up once per API backend (Vulkan + DX12 on Windows); identical
/// cards (3× the same Arc) are distinct entries within one backend's
/// enumeration. Grouping by (name, device_type, backend) and keeping
/// every entry of the best-ranked backend present for that name keeps
/// k physical cards as k entries. Software rasterizers are dropped.
pub fn dedup_adapter_indices(adapters: &[AdapterDesc]) -> Vec<usize> {
    use std::collections::HashMap;
    // name/type -> best backend rank seen.
    let mut best: HashMap<(&str, &str), usize> = HashMap::new();
    // GL entries are always redundant listings of a device the Vulkan
    // or DX12 path already covers — and they mangle the name
    // ("RTX 3090/PCIe/SSE2"), so name-grouping can't pair them with
    // their twin. Drop them outright, like software rasterizers.
    let skip = |a: &AdapterDesc| a.device_type == "Cpu" || a.backend == "Gl";
    for a in adapters {
        if skip(a) {
            continue;
        }
        let key = (a.name.as_str(), a.device_type.as_str());
        let r = backend_rank(&a.backend);
        best.entry(key)
            .and_modify(|cur| *cur = (*cur).min(r))
            .or_insert(r);
    }
    adapters
        .iter()
        .filter(|a| {
            !skip(a)
                && best
                    .get(&(a.name.as_str(), a.device_type.as_str()))
                    .is_some_and(|&r| backend_rank(&a.backend) == r)
        })
        .map(|a| a.index)
        .collect()
}

/// Every distinct physical adapter worth computing on, discrete cards
/// first — what a serving layer should open under an "all adapters"
/// policy.
pub fn usable_adapters() -> Vec<AdapterDesc> {
    let all = enumerate_adapters();
    let mut keep: Vec<AdapterDesc> = dedup_adapter_indices(&all)
        .into_iter()
        .map(|i| all[i].clone())
        .collect();
    keep.sort_by_key(|a| {
        (
            match a.device_type.as_str() {
                "DiscreteGpu" => 0usize,
                "IntegratedGpu" => 1,
                _ => 2,
            },
            a.index,
        )
    });
    keep
}

/// Resolve a human adapter selector to an index: a decimal string is
/// an explicit index, anything else is a case-insensitive substring
/// matched against name/vendor/backend (ties broken discrete-first).
/// Substrings ("nvidia", "radeon", "arc") are the robust choice —
/// numeric enumeration order can differ between processes.
pub fn resolve_adapter_selector(sel: &str) -> Option<usize> {
    if let Ok(i) = sel.trim().parse::<usize>() {
        return Some(i);
    }
    let needle = sel.trim().to_lowercase();
    let all = enumerate_adapters();
    all.iter()
        .filter(|a| {
            a.name.to_lowercase().contains(&needle)
                || a.vendor.to_lowercase().contains(&needle)
                || a.backend.to_lowercase().contains(&needle)
        })
        .min_by_key(|a| match a.device_type.as_str() {
            "DiscreteGpu" => 0usize,
            "IntegratedGpu" => 1,
            "Cpu" => 3,
            _ => 2,
        })
        .map(|a| a.index)
}

impl WgpuDevice {
    /// Open the adapter at `index` from [`enumerate_adapters`]'s order,
    /// or — when `None` — the most capable one (discrete first); see
    /// [`default_adapter_index`].
    pub fn new(index: Option<usize>) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let mut adapters = instance.enumerate_adapters(wgpu::Backends::all());
        if adapters.is_empty() {
            return Err(WgpuError::NoAdapter);
        }
        // Enumeration order is not stable across driver updates or even
        // process runs, so an unspecified index must NOT mean "first" —
        // prefer the most capable class: discrete > integrated/virtual >
        // software rasterizers.
        let rank = |a: &wgpu::Adapter| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => 2,
            wgpu::DeviceType::Cpu => 3,
        };
        let idx = index.unwrap_or_else(|| {
            adapters
                .iter()
                .enumerate()
                .min_by_key(|(i, a)| (rank(a), *i))
                .map(|(i, _)| i)
                .unwrap_or(0)
        });
        if idx >= adapters.len() {
            return Err(WgpuError::NoAdapter);
        }
        let adapter = adapters.swap_remove(idx);
        let raw = adapter.get_info();
        let info = AdapterDesc {
            index: idx,
            name: raw.name.clone(),
            vendor: vendor_name(raw.vendor),
            backend: format!("{:?}", raw.backend),
            device_type: format!("{:?}", raw.device_type),
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("callosum-wgpu"),
                required_features: wgpu::Features::empty(),
                // The adapter's own limits, not downlevel defaults —
                // large-vocab embedding tables and multi-GB weight
                // buffers need the real max_buffer_size (256 MiB under
                // the defaults, typically ≥ 2 GiB on discrete GPUs).
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| WgpuError::Device(e.to_string()))?;

        // Shader source: hand-written kernels + quant dot-units + the
        // per-format matmul/matvec entry points, which differ only in
        // the dot function they call and are therefore generated.
        let mut src = String::new();
        src.push_str(include_str!("shaders.wgsl"));
        src.push_str(include_str!("quant.wgsl"));
        for fmt in QuantDtype::ALL {
            src.push_str(&quant_entry_points(fmt.fn_suffix()));
        }
        src.push_str(MATVEC_F32);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("callosum-wgpu kernels"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        // One explicit layout for every kernel: auto layouts drop
        // bindings an entry point doesn't reference (silu/softmax skip
        // `b`), which would force per-kernel bind-group shapes.
        let storage_ro = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("callosum-wgpu bindings"),
            entries: &[
                storage_ro(0),
                storage_ro(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro(4),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let mut quant = std::collections::HashMap::new();
        let mut expert = std::collections::HashMap::new();
        for fmt in QuantDtype::ALL {
            quant.insert((fmt, false), mk(&format!("matmul_t_{}", fmt.fn_suffix())));
            quant.insert((fmt, true), mk(&format!("matvec_{}", fmt.fn_suffix())));
            expert.insert(fmt, mk(&format!("matmul_exp_{}", fmt.fn_suffix())));
        }
        let pipelines = Pipelines {
            matmul: mk("matmul"),
            matmul_t: mk("matmul_t"),
            matvec_f32: mk("matvec_f32"),
            quant,
            add: mk("add"),
            add_bias: mk("add_bias"),
            mul: mk("mul"),
            mul_bias: mk("mul_bias"),
            gelu: mk("gelu"),
            gelu_mul: mk("gelu_mul"),
            softcap: mk("softcap"),
            slice_cols: mk("slice_cols"),
            moe_topk: mk("moe_topk"),
            moe_combine: mk("moe_combine"),
            expert,
            expert_f32: mk("matmul_exp_f32"),
            silu: mk("silu"),
            rms_norm: mk("rms_norm"),
            softmax: mk("softmax"),
            embed_gather: mk("embed_gather"),
            rope_interleaved: mk("rope_interleaved"),
            rope_half: mk("rope_half"),
            attn_scores: mk("attn_scores"),
            attn_out: mk("attn_out"),
            copy_to: mk("copy_to"),
            silu_mul: mk("silu_mul"),
        };
        Ok(Self {
            inner: Arc::new(DeviceInner {
                device,
                queue,
                pipelines,
                layout: bgl,
                batch: std::sync::Mutex::new(None),
                out_pool: std::sync::Mutex::new(std::collections::HashMap::new()),
                uniform_pool: std::sync::Mutex::new((Vec::new(), 0)),
                bind_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
                info,
            }),
        })
    }

    pub fn info(&self) -> &AdapterDesc {
        &self.inner.info
    }

    /// Upload an f32 slice into a GPU storage buffer.
    pub fn upload(&self, data: &[f32]) -> GpuBuffer {
        let buf = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        GpuBuffer {
            buf,
            len: data.len(),
            id: next_buf_id(),
            pool: None,
        }
    }

    fn alloc_out(&self, len: usize) -> GpuBuffer {
        if let Some((buf, id)) = self
            .inner
            .out_pool
            .lock()
            .unwrap()
            .get_mut(&len)
            .and_then(Vec::pop)
        {
            return GpuBuffer {
                buf,
                len,
                id,
                pool: Some((std::sync::Arc::downgrade(&self.inner), len)),
            };
        }
        let buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GpuBuffer {
            buf,
            len,
            id: next_buf_id(),
            pool: Some((std::sync::Arc::downgrade(&self.inner), len)),
        }
    }

    /// Read a GPU buffer back to host memory. Flushes any open batch
    /// first so recorded work is visible.
    pub fn download(&self, src: &GpuBuffer) -> Result<Vec<f32>> {
        self.flush();
        let dev = &self.inner.device;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (src.len * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&src.buf, 0, &staging, 0, (src.len * 4) as u64);
        self.inner.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        dev.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| WgpuError::Readback(e.to_string()))?
            .map_err(|e| WgpuError::Readback(e.to_string()))?;
        let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
        staging.unmap();
        Ok(out)
    }

    /// Three-input dispatch; kernels that ignore the auxiliary `c`
    /// binding go through [`Self::dispatch`], which re-binds `a` there.
    #[allow(clippy::too_many_arguments)]
    fn dispatch4(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        out: &GpuBuffer,
        params: Params,
        groups: (u32, u32, u32),
    ) {
        let dev = &self.inner.device;
        let (pbuf, uslot) = {
            let mut pool = self.inner.uniform_pool.lock().unwrap();
            let (bufs, cursor) = &mut *pool;
            if *cursor >= bufs.len() {
                bufs.push(dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("params"),
                    size: std::mem::size_of::<Params>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let buf = bufs[*cursor].clone();
            let slot = *cursor as u64;
            *cursor += 1;
            (buf, slot)
        };
        self.inner
            .queue
            .write_buffer(&pbuf, 0, bytemuck::bytes_of(&params));
        let key = (a.id, b.id, c.id, out.id, uslot | (1u64 << 63));
        let bind = {
            let mut cache = self.inner.bind_cache.lock().unwrap();
            cache
                .entry(key)
                .or_insert_with(|| {
                    std::sync::Arc::new(dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.inner.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a.buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b.buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out.buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: pbuf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: c.buf.as_entire_binding(),
                            },
                        ],
                    }))
                })
                .clone()
        };
        let bind = &*bind;
        let mut batch = self.inner.batch.lock().unwrap();
        if let Some(b) = batch.as_mut() {
            let pass = b.pass();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        } else {
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, groups.2);
            }
            self.inner.queue.submit([enc.finish()]);
        }
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a: &GpuBuffer,
        b: &GpuBuffer,
        out: &GpuBuffer,
        params: Params,
        groups: (u32, u32, u32),
    ) {
        self.dispatch4(pipeline, a, b, a, out, params, groups);
    }

    /// Start recording ops into a single command buffer. Ends at
    /// [`Self::flush`]; readbacks flush implicitly. Any batch already
    /// open is flushed first.
    pub fn begin_batch(&self) {
        self.flush();
        self.inner.uniform_pool.lock().unwrap().1 = 0;
        let enc = self
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("callosum-wgpu batch"),
            });
        *self.inner.batch.lock().unwrap() = Some(Batch { enc, pass: None });
    }

    /// Submit any batched work. No-op when nothing is recording.
    pub fn flush(&self) {
        if let Some(mut b) = self.inner.batch.lock().unwrap().take() {
            b.end_pass();
            self.inner.queue.submit([b.enc.finish()]);
        }
    }

    /// C[m,n] = A[m,k] × B[k,n], all row-major f32.
    pub fn matmul(
        &self,
        a: &GpuBuffer,
        b: &GpuBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuBuffer> {
        if a.len != m * k || b.len != k * n {
            return Err(WgpuError::Shape(format!(
                "matmul {m}x{k} × {k}x{n} vs buffers {} / {}",
                a.len, b.len
            )));
        }
        let out = self.alloc_out(m * n);
        let params = Params {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ..Default::default()
        };
        let gx = (n as u32).div_ceil(16);
        let gy = (m as u32).div_ceil(16);
        self.dispatch(
            &self.inner.pipelines.matmul,
            a,
            b,
            &out,
            params,
            (gx, gy, 1),
        );
        Ok(out)
    }

    fn elementwise(
        &self,
        pipeline: &wgpu::ComputePipeline,
        a: &GpuBuffer,
        b: &GpuBuffer,
    ) -> Result<GpuBuffer> {
        let out = self.alloc_out(a.len);
        let params = Params {
            len: a.len as u32,
            ..Default::default()
        };
        let groups = (a.len as u32).div_ceil(256);
        self.dispatch(pipeline, a, b, &out, params, (groups, 1, 1));
        Ok(out)
    }

    pub fn add(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        if a.len != b.len {
            return Err(WgpuError::Shape("add: length mismatch".into()));
        }
        self.elementwise(&self.inner.pipelines.add, a, b)
    }

    /// out[r, c] = a[r, c] + bias[c] over rows of width `bias.len`.
    pub fn add_bias(&self, a: &GpuBuffer, bias: &GpuBuffer) -> Result<GpuBuffer> {
        if bias.len == 0 || !a.len.is_multiple_of(bias.len) {
            return Err(WgpuError::Shape(format!(
                "add_bias: {} rows of {} don't tile {}",
                a.len / bias.len.max(1),
                bias.len,
                a.len
            )));
        }
        let out = self.alloc_out(a.len);
        let params = Params {
            len: a.len as u32,
            n: bias.len as u32,
            ..Default::default()
        };
        let groups = (a.len as u32).div_ceil(256);
        self.dispatch(
            &self.inner.pipelines.add_bias,
            a,
            bias,
            &out,
            params,
            (groups, 1, 1),
        );
        Ok(out)
    }

    pub fn mul(&self, a: &GpuBuffer, b: &GpuBuffer) -> Result<GpuBuffer> {
        if a.len != b.len {
            return Err(WgpuError::Shape("mul: length mismatch".into()));
        }
        self.elementwise(&self.inner.pipelines.mul, a, b)
    }

    /// out[r, c] = a[r, c] * bias[c] over rows of width `bias.len`.
    pub fn mul_bias(&self, a: &GpuBuffer, bias: &GpuBuffer) -> Result<GpuBuffer> {
        if bias.len == 0 || !a.len.is_multiple_of(bias.len) {
            return Err(WgpuError::Shape("mul_bias: shape mismatch".into()));
        }
        let out = self.alloc_out(a.len);
        let params = Params {
            len: a.len as u32,
            n: bias.len as u32,
            ..Default::default()
        };
        let groups = (a.len as u32).div_ceil(256);
        self.dispatch(
            &self.inner.pipelines.mul_bias,
            a,
            bias,
            &out,
            params,
            (groups, 1, 1),
        );
        Ok(out)
    }

    /// Tanh-approximation GELU (matches callosum-core's `gelu` unary).
    pub fn gelu(&self, a: &GpuBuffer) -> Result<GpuBuffer> {
        self.elementwise(&self.inner.pipelines.gelu, a, a)
    }

    /// out = gelu(gate) * up — fused GeGLU (gemma FFN).
    pub fn gelu_mul(&self, gate: &GpuBuffer, up: &GpuBuffer) -> Result<GpuBuffer> {
        if gate.len != up.len {
            return Err(WgpuError::Shape("gelu_mul: length mismatch".into()));
        }
        self.elementwise(&self.inner.pipelines.gelu_mul, gate, up)
    }

    /// out = cap * tanh(a / cap) — gemma-2 logit soft-capping.
    pub fn softcap(&self, a: &GpuBuffer, cap: f32) -> Result<GpuBuffer> {
        let out = self.alloc_out(a.len);
        let params = Params {
            len: a.len as u32,
            cap,
            ..Default::default()
        };
        let groups = (a.len as u32).div_ceil(256);
        self.dispatch(
            &self.inner.pipelines.softcap,
            a,
            a,
            &out,
            params,
            (groups, 1, 1),
        );
        Ok(out)
    }

    /// out[r, 0..width] = a[r, off..off+width] over `rows` rows of
    /// stride `stride` — a strided column slice.
    pub fn slice_cols(
        &self,
        a: &GpuBuffer,
        rows: usize,
        stride: usize,
        off: usize,
        width: usize,
    ) -> Result<GpuBuffer> {
        if a.len != rows * stride || off + width > stride {
            return Err(WgpuError::Shape("slice_cols: shape mismatch".into()));
        }
        let out = self.alloc_out(rows * width);
        let params = Params {
            m: rows as u32,
            k: stride as u32,
            n: width as u32,
            pos0: off as u32,
            ..Default::default()
        };
        let groups = ((rows * width) as u32).div_ceil(256);
        self.dispatch(
            &self.inner.pipelines.slice_cols,
            a,
            a,
            &out,
            params,
            (groups, 1, 1),
        );
        Ok(out)
    }

    /// MoE routing: full-axis softmax over `logits` [m, n_experts],
    /// iterative top-`slots` selection, weights renormalised over the
    /// selected set. Returns [m, slots, 2] rows of (expert_id, weight).
    pub fn moe_topk(
        &self,
        logits: &GpuBuffer,
        m: usize,
        n_experts: usize,
        slots: usize,
    ) -> Result<GpuBuffer> {
        if logits.len != m * n_experts || slots == 0 || slots > n_experts {
            return Err(WgpuError::Shape("moe_topk: shape mismatch".into()));
        }
        let out = self.alloc_out(m * slots * 2);
        let params = Params {
            m: m as u32,
            k: n_experts as u32,
            n_heads: slots as u32,
            ..Default::default()
        };
        self.dispatch(
            &self.inner.pipelines.moe_topk,
            logits,
            logits,
            &out,
            params,
            (m as u32, 1, 1),
        );
        Ok(out)
    }

    /// Expert-indexed matmul over a fused quantized expert tensor
    /// (`w` holds n_experts × rows stacked row-major). For every
    /// (token, slot) pair in `routing` ([m, slots, 2]), computes
    /// x[token] · W[expert]ᵀ into out[(token*slots+slot), rows].
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_expert(
        &self,
        x: &GpuBuffer,
        w: &QuantBuffer,
        routing: &GpuBuffer,
        m: usize,
        slots: usize,
        rows_per_expert: usize,
        k: usize,
        x_per_slot: bool,
    ) -> Result<GpuBuffer> {
        let want_x = if x_per_slot { m * slots * k } else { m * k };
        if x.len != want_x || w.k != k || routing.len != m * slots * 2 {
            return Err(WgpuError::Shape("matmul_expert: shape mismatch".into()));
        }
        let out = self.alloc_out(m * slots * rows_per_expert);
        let params = Params {
            m: m as u32,
            n: rows_per_expert as u32,
            k: k as u32,
            len: w.row_words as u32,
            n_heads: slots as u32,
            flags: if x_per_slot { 1 } else { 0 },
            ..Default::default()
        };
        let pipeline = &self.inner.pipelines.expert[&w.dtype];
        self.dispatch4(
            pipeline,
            x,
            &w.buf,
            routing,
            &out,
            params,
            matvec_groups(m * slots * rows_per_expert),
        );
        Ok(out)
    }

    /// f32 fallback of [`Self::matmul_expert`] for expert tensors in
    /// formats without an in-shader dequant kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_expert_f32(
        &self,
        x: &GpuBuffer,
        w: &GpuBuffer,
        routing: &GpuBuffer,
        m: usize,
        slots: usize,
        rows_per_expert: usize,
        k: usize,
        x_per_slot: bool,
    ) -> Result<GpuBuffer> {
        let want_x = if x_per_slot { m * slots * k } else { m * k };
        if x.len != want_x || routing.len != m * slots * 2 {
            return Err(WgpuError::Shape("matmul_expert_f32: shape mismatch".into()));
        }
        let out = self.alloc_out(m * slots * rows_per_expert);
        let params = Params {
            m: m as u32,
            n: rows_per_expert as u32,
            k: k as u32,
            n_heads: slots as u32,
            flags: if x_per_slot { 1 } else { 0 },
            ..Default::default()
        };
        self.dispatch4(
            &self.inner.pipelines.expert_f32,
            x,
            w,
            routing,
            &out,
            params,
            matvec_groups(m * slots * rows_per_expert),
        );
        Ok(out)
    }

    /// Weighted sum of per-slot expert outputs back onto tokens:
    /// out[t, h] = Σ_s routing_weight(t, s) · y[(t*slots+s), h].
    pub fn moe_combine(
        &self,
        y: &GpuBuffer,
        routing: &GpuBuffer,
        m: usize,
        slots: usize,
        hidden: usize,
    ) -> Result<GpuBuffer> {
        if y.len != m * slots * hidden || routing.len != m * slots * 2 {
            return Err(WgpuError::Shape("moe_combine: shape mismatch".into()));
        }
        let out = self.alloc_out(m * hidden);
        let params = Params {
            m: m as u32,
            k: hidden as u32,
            n_heads: slots as u32,
            ..Default::default()
        };
        let groups = ((m * hidden) as u32).div_ceil(256);
        self.dispatch4(
            &self.inner.pipelines.moe_combine,
            y,
            y,
            routing,
            &out,
            params,
            (groups, 1, 1),
        );
        Ok(out)
    }

    pub fn silu(&self, a: &GpuBuffer) -> Result<GpuBuffer> {
        // b is unused by the kernel; bind a itself to satisfy the layout.
        self.elementwise(&self.inner.pipelines.silu, a, a)
    }

    /// RMSNorm over rows of length `k` with weight `w` (len k).
    pub fn rms_norm(
        &self,
        a: &GpuBuffer,
        w: &GpuBuffer,
        rows: usize,
        k: usize,
        eps: f32,
    ) -> Result<GpuBuffer> {
        if a.len != rows * k || w.len != k {
            return Err(WgpuError::Shape("rms_norm: shape mismatch".into()));
        }
        let out = self.alloc_out(a.len);
        let params = Params {
            m: rows as u32,
            k: k as u32,
            eps,
            ..Default::default()
        };
        self.dispatch(
            &self.inner.pipelines.rms_norm,
            a,
            w,
            &out,
            params,
            (rows as u32, 1, 1),
        );
        Ok(out)
    }

    /// Numerically-stable softmax over rows of length `k`.
    pub fn softmax(&self, a: &GpuBuffer, rows: usize, k: usize) -> Result<GpuBuffer> {
        if a.len != rows * k {
            return Err(WgpuError::Shape("softmax: shape mismatch".into()));
        }
        let out = self.alloc_out(a.len);
        let params = Params {
            m: rows as u32,
            k: k as u32,
            ..Default::default()
        };
        self.dispatch(
            &self.inner.pipelines.softmax,
            a,
            a,
            &out,
            params,
            (rows as u32, 1, 1),
        );
        Ok(out)
    }
}

impl WgpuDevice {
    /// C[m,n] = A[m,k] × B^T with B row-major [n,k] (weights as stored:
    /// [out_features, in_features]).
    pub fn matmul_t(
        &self,
        x: &GpuBuffer,
        w: &GpuBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuBuffer> {
        if x.len != m * k || w.len != n * k {
            return Err(WgpuError::Shape(format!(
                "matmul_t {m}x{k} × ({n}x{k})^T vs buffers {} / {}",
                x.len, w.len
            )));
        }
        if m == 1 {
            return self.matvec_f32(x, w, k, n);
        }
        let out = self.alloc_out(m * n);
        let params = Params {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ..Default::default()
        };
        let gx = (n as u32).div_ceil(16);
        let gy = (m as u32).div_ceil(16);
        self.dispatch(
            &self.inner.pipelines.matmul_t,
            x,
            w,
            &out,
            params,
            (gx, gy, 1),
        );
        Ok(out)
    }

    /// y[m,n] = x[m,k] × Wq^T over an in-shader-dequantized weight.
    /// Decode (m == 1) routes to the matvec reduction kernels — one
    /// 256-thread workgroup per output element — instead of the m×n
    /// grid, which at m=1 would leave the GPU almost entirely idle.
    pub fn matmul_t_quant(
        &self,
        x: &GpuBuffer,
        w: &QuantBuffer,
        m: usize,
        k: usize,
    ) -> Result<GpuBuffer> {
        if x.len != m * k || w.k != k {
            return Err(WgpuError::Shape(format!(
                "matmul_t_quant {m}x{k} vs weight [{}x{}]",
                w.n, w.k
            )));
        }
        let out = self.alloc_out(m * w.n);
        let params = Params {
            m: m as u32,
            n: w.n as u32,
            k: k as u32,
            len: w.row_words as u32,
            ..Default::default()
        };
        let matvec = m == 1;
        let pipeline = &self.inner.pipelines.quant[&(w.dtype, matvec)];
        let groups = if matvec {
            matvec_groups(w.n)
        } else {
            ((w.n as u32).div_ceil(16), (m as u32).div_ceil(16), 1)
        };
        self.dispatch(pipeline, x, &w.buf, &out, params, groups);
        Ok(out)
    }

    /// Backwards-compatible q8_0 entry (see [`Self::matmul_t_quant`]).
    pub fn matmul_t_q8_0(
        &self,
        x: &GpuBuffer,
        w: &QuantBuffer,
        m: usize,
        k: usize,
    ) -> Result<GpuBuffer> {
        self.matmul_t_quant(x, w, m, k)
    }

    /// f32 matvec: y[n] = W[n,k] · x[k] with a 256-thread reduction per
    /// output element. Used automatically by [`Self::matmul_t`] at m=1.
    fn matvec_f32(&self, x: &GpuBuffer, w: &GpuBuffer, k: usize, n: usize) -> Result<GpuBuffer> {
        let out = self.alloc_out(n);
        let params = Params {
            m: 1,
            n: n as u32,
            k: k as u32,
            ..Default::default()
        };
        self.dispatch(
            &self.inner.pipelines.matvec_f32,
            x,
            w,
            &out,
            params,
            matvec_groups(n),
        );
        Ok(out)
    }

    /// Gather embedding rows: ids (exact-integer f32s, len = seq) ×
    /// table [vocab, hidden] → [seq, hidden].
    pub fn embed_gather(
        &self,
        ids: &GpuBuffer,
        table: &GpuBuffer,
        seq: usize,
        hidden: usize,
    ) -> Result<GpuBuffer> {
        if ids.len != seq || table.len % hidden != 0 {
            return Err(WgpuError::Shape("embed_gather: shape mismatch".into()));
        }
        let out = self.alloc_out(seq * hidden);
        let params = Params {
            m: seq as u32,
            k: hidden as u32,
            ..Default::default()
        };
        let groups = ((seq * hidden) as u32).div_ceil(256);
        self.dispatch(
            &self.inner.pipelines.embed_gather,
            ids,
            table,
            &out,
            params,
            (groups, 1, 1),
        );
        Ok(out)
    }

    /// RoPE over [seq, heads, head_dim]; `interleaved` picks the
    /// Llama/Mistral pair convention, otherwise rotate-half (neox).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &GpuBuffer,
        seq: usize,
        heads: usize,
        head_dim: usize,
        pos0: usize,
        theta: f32,
        interleaved: bool,
    ) -> Result<GpuBuffer> {
        if x.len != seq * heads * head_dim {
            return Err(WgpuError::Shape("rope: shape mismatch".into()));
        }
        let out = self.alloc_out(x.len);
        let params = Params {
            m: seq as u32,
            n_heads: heads as u32,
            head_dim: head_dim as u32,
            pos0: pos0 as u32,
            theta,
            fscale: 1.0,
            ..Default::default()
        };
        let pairs = (seq * heads * head_dim / 2) as u32;
        let pipeline = if interleaved {
            &self.inner.pipelines.rope_interleaved
        } else {
            &self.inner.pipelines.rope_half
        };
        self.dispatch(pipeline, x, x, &out, params, (pairs.div_ceil(256), 1, 1));
        Ok(out)
    }

    /// RoPE with the full option set: linear position scale (`fscale`,
    /// 1.0 = off) and optional per-frequency divisors (`freqs`, length
    /// head_dim/2 — gemma's `rope_freqs.weight`).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_scaled(
        &self,
        x: &GpuBuffer,
        seq: usize,
        heads: usize,
        head_dim: usize,
        pos0: usize,
        theta: f32,
        interleaved: bool,
        fscale: f32,
        freqs: Option<&GpuBuffer>,
    ) -> Result<GpuBuffer> {
        if x.len != seq * heads * head_dim {
            return Err(WgpuError::Shape("rope: shape mismatch".into()));
        }
        if let Some(f) = freqs {
            if f.len < head_dim / 2 {
                return Err(WgpuError::Shape("rope: freq table too short".into()));
            }
        }
        let out = self.alloc_out(x.len);
        let params = Params {
            m: seq as u32,
            n_heads: heads as u32,
            head_dim: head_dim as u32,
            pos0: pos0 as u32,
            theta,
            fscale,
            flags: if freqs.is_some() { 1 } else { 0 },
            ..Default::default()
        };
        let pairs = (seq * heads * head_dim / 2) as u32;
        let pipeline = if interleaved {
            &self.inner.pipelines.rope_interleaved
        } else {
            &self.inner.pipelines.rope_half
        };
        self.dispatch(
            pipeline,
            x,
            freqs.unwrap_or(x),
            &out,
            params,
            (pairs.div_ceil(256), 1, 1),
        );
        Ok(out)
    }

    /// Causal GQA attention scores: Q [seq_q, heads, hd] × K-cache
    /// [kv_len, kv_heads, hd] → [heads, seq_q, kv_len], scaled and
    /// masked (queries sit at absolute positions pos0..pos0+seq_q).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_scores(
        &self,
        q: &GpuBuffer,
        k_cache: &GpuBuffer,
        seq_q: usize,
        kv_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos0: usize,
    ) -> Result<GpuBuffer> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        self.attn_scores_opt(
            q, k_cache, seq_q, kv_len, heads, kv_heads, head_dim, pos0, scale, 0,
        )
    }

    /// [`Self::attn_scores`] with an explicit scale (gemma 3n/4 use 1.0)
    /// and an optional sliding window (0 = full causal).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_scores_opt(
        &self,
        q: &GpuBuffer,
        k_cache: &GpuBuffer,
        seq_q: usize,
        kv_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos0: usize,
        scale: f32,
        window: usize,
    ) -> Result<GpuBuffer> {
        if q.len != seq_q * heads * head_dim || k_cache.len < kv_len * kv_heads * head_dim {
            return Err(WgpuError::Shape("attn_scores: shape mismatch".into()));
        }
        let out = self.alloc_out(heads * seq_q * kv_len);
        let params = Params {
            m: seq_q as u32,
            k: kv_len as u32,
            n_heads: heads as u32,
            n_kv_heads: kv_heads as u32,
            head_dim: head_dim as u32,
            pos0: pos0 as u32,
            scale,
            window: window as u32,
            ..Default::default()
        };
        let total = (heads * seq_q * kv_len) as u32;
        self.dispatch(
            &self.inner.pipelines.attn_scores,
            q,
            k_cache,
            &out,
            params,
            (total.div_ceil(256), 1, 1),
        );
        Ok(out)
    }

    /// probs [heads, seq_q, kv_len] × V-cache [kv_len, kv_heads, hd] →
    /// [seq_q, heads, hd].
    #[allow(clippy::too_many_arguments)]
    pub fn attn_out(
        &self,
        probs: &GpuBuffer,
        v_cache: &GpuBuffer,
        seq_q: usize,
        kv_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<GpuBuffer> {
        if probs.len != heads * seq_q * kv_len || v_cache.len < kv_len * kv_heads * head_dim {
            return Err(WgpuError::Shape("attn_out: shape mismatch".into()));
        }
        let out = self.alloc_out(seq_q * heads * head_dim);
        let params = Params {
            m: seq_q as u32,
            k: kv_len as u32,
            n_heads: heads as u32,
            n_kv_heads: kv_heads as u32,
            head_dim: head_dim as u32,
            ..Default::default()
        };
        let total = (seq_q * heads * head_dim) as u32;
        self.dispatch(
            &self.inner.pipelines.attn_out,
            probs,
            v_cache,
            &out,
            params,
            (total.div_ceil(256), 1, 1),
        );
        Ok(out)
    }

    /// Copy `rows` rows of `row_elems` f32 each from `src` into `dst`
    /// starting at row `dst_row` — KV-cache append.
    pub fn copy_rows(
        &self,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        dst_row: usize,
        rows: usize,
        row_elems: usize,
    ) -> Result<()> {
        if src.len < rows * row_elems || dst.len < (dst_row + rows) * row_elems {
            return Err(WgpuError::Shape("copy_rows: out of bounds".into()));
        }
        // A dispatch, not an encoder-level copy: stays inside the
        // batch's shared compute pass (a real copy would force the pass
        // closed and reopened around every KV append).
        let count = rows * row_elems;
        let params = Params {
            len: count as u32,
            pos0: (dst_row * row_elems) as u32,
            ..Default::default()
        };
        self.dispatch(
            &self.inner.pipelines.copy_to,
            src,
            src,
            dst,
            params,
            ((count as u32).div_ceil(256), 1, 1),
        );
        Ok(())
    }

    /// Fused SwiGLU elementwise: silu(gate) * up in one dispatch.
    pub fn silu_mul(&self, gate: &GpuBuffer, up: &GpuBuffer) -> Result<GpuBuffer> {
        if gate.len != up.len {
            return Err(WgpuError::Shape("silu_mul: length mismatch".into()));
        }
        self.elementwise(&self.inner.pipelines.silu_mul, gate, up)
    }

    /// Allocate an uninitialized f32 buffer (e.g. a KV cache).
    pub fn alloc(&self, len: usize) -> GpuBuffer {
        self.alloc_out(len)
    }

    /// Upload raw GGML rows for an [n, k] weight in any supported
    /// quant format (`k` must be a whole number of blocks). `raw` is
    /// the on-disk tensor payload; rows are re-packed with word-aligned
    /// starts so the shaders can index rows independently.
    pub fn upload_quantized(
        &self,
        raw: &[u8],
        n: usize,
        k: usize,
        dtype: QuantDtype,
    ) -> Result<QuantBuffer> {
        if k % dtype.block_elems() != 0 {
            return Err(WgpuError::Shape(format!(
                "{dtype:?}: k={k} not a multiple of {}",
                dtype.block_elems()
            )));
        }
        let row_bytes = k / dtype.block_elems() * dtype.block_bytes();
        if raw.len() != n * row_bytes {
            return Err(WgpuError::Shape(format!(
                "{dtype:?} payload {} bytes, expected {}",
                raw.len(),
                n * row_bytes
            )));
        }
        let row_words = row_bytes.div_ceil(4);
        let mut packed = vec![0u8; n * row_words * 4];
        for r in 0..n {
            packed[r * row_words * 4..r * row_words * 4 + row_bytes]
                .copy_from_slice(&raw[r * row_bytes..(r + 1) * row_bytes]);
        }
        let buf = self
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quantized weights"),
                contents: &packed,
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(QuantBuffer {
            buf: GpuBuffer {
                buf,
                len: n * row_words,
                id: next_buf_id(),
                pool: None,
            },
            n,
            k,
            dtype,
            row_words,
        })
    }

    /// Backwards-compatible q8_0 upload (see [`Self::upload_quantized`]).
    pub fn upload_q8_0(&self, raw: &[u8], n: usize, k: usize) -> Result<QuantBuffer> {
        self.upload_quantized(raw, n, k, QuantDtype::Q8_0)
    }
}

/// GGML quant formats with in-shader dequant kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantDtype {
    Q4_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl QuantDtype {
    pub const ALL: [QuantDtype; 5] = [
        QuantDtype::Q4_0,
        QuantDtype::Q8_0,
        QuantDtype::Q4K,
        QuantDtype::Q5K,
        QuantDtype::Q6K,
    ];

    pub fn block_elems(self) -> usize {
        match self {
            QuantDtype::Q4_0 | QuantDtype::Q8_0 => 32,
            QuantDtype::Q4K | QuantDtype::Q5K | QuantDtype::Q6K => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            QuantDtype::Q4_0 => 18,
            QuantDtype::Q8_0 => 34,
            QuantDtype::Q4K => 144,
            QuantDtype::Q5K => 176,
            QuantDtype::Q6K => 210,
        }
    }

    fn fn_suffix(self) -> &'static str {
        match self {
            QuantDtype::Q4_0 => "q4_0",
            QuantDtype::Q8_0 => "q8_0",
            QuantDtype::Q4K => "q4_k",
            QuantDtype::Q5K => "q5_k",
            QuantDtype::Q6K => "q6_k",
        }
    }
}

/// A quantized [n, k] weight resident on the GPU at its GGML on-disk
/// density (word-aligned row packing).
pub struct QuantBuffer {
    buf: GpuBuffer,
    pub n: usize,
    pub k: usize,
    pub dtype: QuantDtype,
    row_words: usize,
}

/// Generated per-format entry points: identical bodies, different dot
/// function. `params.len` carries the weight row stride in words.
fn quant_entry_points(f: &str) -> String {
    format!(
        r#"
@compute @workgroup_size(16, 16, 1)
fn matmul_t_{f}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) {{
        return;
    }}
    let units = params.k / 32u;
    let row_word = col * params.len;
    var acc: f32 = 0.0;
    for (var u: u32 = 0u; u < units; u = u + 1u) {{
        acc = acc + dot_{f}(row_word, u, row * params.k + u * 32u);
    }}
    out[row * params.n + col] = acc;
}}

@compute @workgroup_size(256, 1, 1)
fn matvec_{f}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let col = wid.y * 32768u + wid.x;
    if (col >= params.n) {{
        return;
    }}
    let units = params.k / 32u;
    let row_word = col * params.len;
    var acc: f32 = 0.0;
    var u = lid.x;
    loop {{
        if (u >= units) {{
            break;
        }}
        acc = acc + dot_{f}(row_word, u, u * 32u);
        u = u + 256u;
    }}
    scratch[lid.x] = acc;
    workgroupBarrier();
    var stride: u32 = 128u;
    while (stride > 0u) {{
        if (lid.x < stride) {{
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    if (lid.x == 0u && col < params.n) {{
        out[col] = scratch[0];
    }}
}}

// Expert-indexed matmul (MoE): output element oi covers
// [m tokens × n_heads slots × n rows]; the expert id for (token, slot)
// comes from the routing table in `c`. Weight rows for expert e start
// at row e*n of the fused [n_experts*n, k] quantized tensor.
@compute @workgroup_size(256, 1, 1)
fn matmul_exp_{f}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let oi = wid.y * 32768u + wid.x;
    let total = params.m * params.n_heads * params.n;
    if (oi >= total) {{
        return;
    }}
    let col = oi % params.n;
    let ts = oi / params.n;
    let t = ts / params.n_heads;
    // flags bit 0: input rows are per-(token, slot) — the down
    // projection consumes the per-slot SwiGLU outputs — instead of
    // per-token.
    let xrow = select(t, ts, (params.flags & 1u) != 0u);
    let eid = u32(c[ts * 2u]);
    let units = params.k / 32u;
    let row_word = (eid * params.n + col) * params.len;
    var acc: f32 = 0.0;
    var u = lid.x;
    loop {{
        if (u >= units) {{
            break;
        }}
        acc = acc + dot_{f}(row_word, u, xrow * params.k + u * 32u);
        u = u + 256u;
    }}
    scratch[lid.x] = acc;
    workgroupBarrier();
    var stride2: u32 = 128u;
    while (stride2 > 0u) {{
        if (lid.x < stride2) {{
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride2];
        }}
        workgroupBarrier();
        stride2 = stride2 / 2u;
    }}
    if (lid.x == 0u) {{
        out[ts * params.n + col] = scratch[0];
    }}
}}
"#
    )
}

/// Matvec kernels run one workgroup per output element, addressed as
/// `wid.y * 32768 + wid.x` — a single dispatch dimension caps at 65535
/// workgroups, which large-vocab lm_heads (e.g. qwen's 151936) exceed.
fn matvec_groups(n: usize) -> (u32, u32, u32) {
    const GX: u32 = 32768;
    ((n as u32).min(GX), (n as u32).div_ceil(GX), 1)
}

/// Monotonic buffer identity for the bind cache (wgpu itself exposes
/// no stable resource id).
fn next_buf_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// f32 matvec with the same reduction shape as the quant kernels.
const MATVEC_F32: &str = r#"
@compute @workgroup_size(256, 1, 1)
fn matvec_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wid.y * 32768u + wid.x;
    if (col >= params.n) {
        return;
    }
    let row_base = col * params.k;
    var acc: f32 = 0.0;
    var i = lid.x;
    loop {
        if (i >= params.k) {
            break;
        }
        acc = acc + b[row_base + i] * a[i];
        i = i + 256u;
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u && col < params.n) {
        out[col] = scratch[0];
    }
}

// f32 fallback of the expert-indexed matmul: `b` is the dense
// [n_experts * n, k] weight, `c` the routing table.
@compute @workgroup_size(256, 1, 1)
fn matmul_exp_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let oi = wid.y * 32768u + wid.x;
    let total = params.m * params.n_heads * params.n;
    if (oi >= total) {
        return;
    }
    let col = oi % params.n;
    let ts = oi / params.n;
    let t = ts / params.n_heads;
    let xrow = select(t, ts, (params.flags & 1u) != 0u);
    let eid = u32(c[ts * 2u]);
    let row_base = (eid * params.n + col) * params.k;
    var acc: f32 = 0.0;
    var i = lid.x;
    loop {
        if (i >= params.k) {
            break;
        }
        acc = acc + b[row_base + i] * a[xrow * params.k + i];
        i = i + 256u;
    }
    scratch[lid.x] = acc;
    workgroupBarrier();
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        out[ts * params.n + col] = scratch[0];
    }
}
"#;

/// An f32 storage buffer on the device. Pool-allocated buffers return
/// their storage on drop for reuse by later ops.
pub struct GpuBuffer {
    buf: wgpu::Buffer,
    pub len: usize,
    id: u64,
    pool: Option<(std::sync::Weak<DeviceInner>, usize)>,
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let Some((pool, len)) = self.pool.take() {
            if let Some(inner) = pool.upgrade() {
                inner
                    .out_pool
                    .lock()
                    .unwrap()
                    .entry(len)
                    .or_default()
                    .push((self.buf.clone(), self.id));
            }
        }
    }
}
