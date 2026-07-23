// callosum-wgpu compute kernels.
//
// Correctness-first WGSL: every kernel is bounds-checked and shaped by
// a uniform Params block rather than specialization constants, so one
// compiled pipeline serves every tensor shape. Perf work (tiling,
// vectorized loads, subgroup ops) comes after the op set is complete.

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
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// C[m,n] = A[m,k] × B[k,n]
@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) {
        return;
    }
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        acc = acc + a[row * params.k + i] * b[i * params.n + col];
    }
    out[row * params.n + col] = acc;
}

@compute @workgroup_size(256, 1, 1)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        out[i] = a[i] + b[i];
    }
}

// out[r, c] = a[r, c] + b[c] — row-broadcast bias (qwen2 QKV biases).
@compute @workgroup_size(256, 1, 1)
fn add_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        out[i] = a[i] + b[i % params.n];
    }
}

@compute @workgroup_size(256, 1, 1)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        out[i] = a[i] * b[i];
    }
}

// silu(x) = x * sigmoid(x)
@compute @workgroup_size(256, 1, 1)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        let x = a[i];
        out[i] = x / (1.0 + exp(-x));
    }
}

// out[params.pos0 + i] = a[i] — KV-cache append as a dispatch, so the
// shared compute pass never has to close for an encoder-level copy.
@compute @workgroup_size(256, 1, 1)
fn copy_to(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        out[params.pos0 + i] = a[i];
    }
}

// out = silu(a) * b — fused SwiGLU elementwise.
@compute @workgroup_size(256, 1, 1)
fn silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        let x = a[i];
        out[i] = (x / (1.0 + exp(-x))) * b[i];
    }
}

var<workgroup> scratch: array<f32, 256>;

// RMSNorm over the last dimension: rows of length k, weight in b.
// One workgroup per row; parallel reduction in shared memory.
@compute @workgroup_size(256, 1, 1)
fn rms_norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.m) {
        return;
    }
    let base = row * params.k;
    var sum: f32 = 0.0;
    for (var i: u32 = lid.x; i < params.k; i = i + 256u) {
        let x = a[base + i];
        sum = sum + x * x;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(scratch[0] / f32(params.k) + params.eps);
    for (var i: u32 = lid.x; i < params.k; i = i + 256u) {
        out[base + i] = a[base + i] * inv * b[i];
    }
}

// Numerically-stable softmax over the last dimension: rows of length k.
@compute @workgroup_size(256, 1, 1)
fn softmax(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.m) {
        return;
    }
    let base = row * params.k;

    // 1. row max
    var mx: f32 = -3.4e38;
    for (var i: u32 = lid.x; i < params.k; i = i + 256u) {
        mx = max(mx, a[base + i]);
    }
    scratch[lid.x] = mx;
    workgroupBarrier();
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_max = scratch[0];
    workgroupBarrier();

    // 2. exp + sum
    var sum: f32 = 0.0;
    for (var i: u32 = lid.x; i < params.k; i = i + 256u) {
        let e = exp(a[base + i] - row_max);
        out[base + i] = e;
        sum = sum + e;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();
    stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let denom = scratch[0];

    // 3. normalize
    for (var i: u32 = lid.x; i < params.k; i = i + 256u) {
        out[base + i] = out[base + i] / denom;
    }
}

// C[m,n] = A[m,k] × B^T where B is row-major [n,k] — the natural
// layout for weight matrices stored as [out_features, in_features],
// so no transpose pass is ever needed.
@compute @workgroup_size(16, 16, 1)
fn matmul_t(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) {
        return;
    }
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        acc = acc + a[row * params.k + i] * b[col * params.k + i];
    }
    out[row * params.n + col] = acc;
}

// out[s, :] = b[id(s), :] where a holds token ids as exact f32 values
// (vocab ids < 2^24 are exactly representable; a proper u32 binding
// replaces this once the layout grows a second bind group).
@compute @workgroup_size(256, 1, 1)
fn embed_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; // element index into [m rows × k cols]
    if (i >= params.m * params.k) {
        return;
    }
    let s = i / params.k;
    let c = i % params.k;
    let id = u32(a[s]);
    out[i] = b[id * params.k + c];
}

// Interleaved RoPE (Llama/Mistral/SmolLM2 convention): rotate pairs
// (2i, 2i+1) within each head. In/out over [seq, n_heads, head_dim]
// contiguous; absolute position = pos0 + s. `a` is the input, out the
// rotated copy. n_heads in params covers Q or KV depending on call.
@compute @workgroup_size(256, 1, 1)
fn rope_interleaved(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; // pair index across [seq × heads × head_dim/2]
    let half = params.head_dim / 2u;
    let total = params.m * params.n_heads * half;
    if (idx >= total) {
        return;
    }
    let d = idx % half;
    let rem = idx / half;
    let h = rem % params.n_heads;
    let s = rem / params.n_heads;
    let base = (s * params.n_heads + h) * params.head_dim;
    let pos = f32(params.pos0 + s);
    let inv_freq = pow(params.theta, -2.0 * f32(d) / f32(params.head_dim));
    let angle = pos * inv_freq;
    let c = cos(angle);
    let sn = sin(angle);
    let x0 = a[base + 2u * d];
    let x1 = a[base + 2u * d + 1u];
    out[base + 2u * d]      = x0 * c - x1 * sn;
    out[base + 2u * d + 1u] = x0 * sn + x1 * c;
}

// Rotate-half ("neox") RoPE (Qwen/Gemma/Phi convention): pairs
// (i, i + head_dim/2).
@compute @workgroup_size(256, 1, 1)
fn rope_half(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let half = params.head_dim / 2u;
    let total = params.m * params.n_heads * half;
    if (idx >= total) {
        return;
    }
    let d = idx % half;
    let rem = idx / half;
    let h = rem % params.n_heads;
    let s = rem / params.n_heads;
    let base = (s * params.n_heads + h) * params.head_dim;
    let pos = f32(params.pos0 + s);
    let inv_freq = pow(params.theta, -2.0 * f32(d) / f32(params.head_dim));
    let angle = pos * inv_freq;
    let c = cos(angle);
    let sn = sin(angle);
    let x0 = a[base + d];
    let x1 = a[base + d + half];
    out[base + d]        = x0 * c - x1 * sn;
    out[base + d + half] = x0 * sn + x1 * c;
}

// Causal attention scores with GQA:
//   scores[h, sq, sk] = scale · Σ_d Q[sq, h, d] · Kcache[sk, kv(h), d]
// masked to -inf where sk > pos0 + sq. Q is [m=seq_q, n_heads,
// head_dim]; `b` is the K cache [k = total_kv_len, n_kv_heads,
// head_dim]. Output rows (h·seq_q) feed straight into `softmax`.
@compute @workgroup_size(256, 1, 1)
fn attn_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.n_heads * params.m * params.k;
    if (idx >= total) {
        return;
    }
    let sk = idx % params.k;
    let rem = idx / params.k;
    let sq = rem % params.m;
    let h = rem / params.m;
    if (sk > params.pos0 + sq) {
        out[idx] = -3.0e38;
        return;
    }
    let kv_h = h / (params.n_heads / params.n_kv_heads);
    let qbase = (sq * params.n_heads + h) * params.head_dim;
    let kbase = (sk * params.n_kv_heads + kv_h) * params.head_dim;
    var acc: f32 = 0.0;
    for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
        acc = acc + a[qbase + d] * b[kbase + d];
    }
    out[idx] = acc * params.scale;
}

// Attention output:
//   out[sq, h, d] = Σ_sk probs[h, sq, sk] · Vcache[sk, kv(h), d]
// probs in `a` ([n_heads, seq_q, kv_len]), V cache in `b`.
@compute @workgroup_size(256, 1, 1)
fn attn_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.m * params.n_heads * params.head_dim;
    if (idx >= total) {
        return;
    }
    let d = idx % params.head_dim;
    let rem = idx / params.head_dim;
    let h = rem % params.n_heads;
    let sq = rem / params.n_heads;
    let kv_h = h / (params.n_heads / params.n_kv_heads);
    var acc: f32 = 0.0;
    for (var sk: u32 = 0u; sk < params.k; sk = sk + 1u) {
        let p = a[(h * params.m + sq) * params.k + sk];
        acc = acc + p * b[(sk * params.n_kv_heads + kv_h) * params.head_dim + d];
    }
    out[(sq * params.n_heads + h) * params.head_dim + d] = acc;
}


