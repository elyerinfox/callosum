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
    // Sliding-window size for attn_scores (0 = full causal).
    window: u32,
    // RoPE position scale (linear rope scaling); 1.0 = off.
    fscale: f32,
    // Soft-cap constant for `softcap` (tanh(x/cap)*cap).
    cap: f32,
    // Bit 0: rope divides inv_freq by the freq-factor table in `b`.
    flags: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
// Auxiliary read-only input for kernels needing a third tensor (MoE
// routing tables). Kernels that ignore it get a dummy binding; the
// shared explicit bind-group layout keeps the shape stable.
@group(0) @binding(4) var<storage, read> c: array<f32>;

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

// Tanh-approximation GELU — must match callosum-core's `gelu` unary
// (0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))), which the CUDA gemma path
// uses for the FFN and AltUp gates.
fn gelu_tanh(x: f32) -> f32 {
    let x3 = x * x * x;
    // Clamp: tanh saturates by |t| ~ 10, but some drivers (AMD Vulkan)
    // compute tanh via exp and return NaN once exp overflows f32
    // (|t| > ~44). Gemma activations reach 1e4+, so this bites hard.
    let t = clamp(0.7978845608028654 * (x + 0.044715 * x3), -20.0, 20.0);
    return 0.5 * x * (1.0 + tanh(t));
}

@compute @workgroup_size(256, 1, 1)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        out[i] = gelu_tanh(a[i]);
    }
}

// out = gelu(a) * b — fused GeGLU elementwise (gemma FFN).
@compute @workgroup_size(256, 1, 1)
fn gelu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        out[i] = gelu_tanh(a[i]) * b[i];
    }
}

// out[r, col] = a[r, col] * b[col] — row-broadcast multiply (AltUp
// layer_output_scale).
@compute @workgroup_size(256, 1, 1)
fn mul_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        out[i] = a[i] * b[i % params.n];
    }
}

// out = cap * tanh(a / cap) — gemma 2 logit soft-capping.
@compute @workgroup_size(256, 1, 1)
fn softcap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.len) {
        // Same overflow-safe clamp as gelu_tanh.
        out[i] = params.cap * tanh(clamp(a[i] / params.cap, -20.0, 20.0));
    }
}

// MoE router top-k: a = router logits [m tokens, k experts]; out =
// routing table [m, n_heads slots, 2] of (expert_id, weight). One
// workgroup per token; a single thread does the (tiny) full-row
// softmax, iterative top-k selection, and top-k renormalisation —
// mirroring run_moe in the CUDA backend exactly.
@compute @workgroup_size(1, 1, 1)
fn moe_topk(@builtin(workgroup_id) wid: vec3<u32>) {
    let t = wid.x;
    if (t >= params.m) {
        return;
    }
    let n_e = params.k;
    let slots = params.n_heads;
    let base = t * n_e;
    let sigmoid_gate = (params.flags & 1u) != 0u;
    let has_bias = (params.flags & 2u) != 0u;
    let renorm = (params.flags & 4u) != 0u;
    let n_group = max(params.window, 1u);
    let topk_group = max(params.head_dim, 1u);
    let route_scale = select(params.cap, 1.0, params.cap == 0.0);

    // Mixture scores: softmax (qwen/deepseek-v2) or sigmoid (v3/glm4moe).
    var mx: f32 = -3.4e38;
    for (var e: u32 = 0u; e < n_e; e = e + 1u) {
        mx = max(mx, a[base + e]);
    }
    var denom: f32 = 0.0;
    for (var e: u32 = 0u; e < n_e; e = e + 1u) {
        denom = denom + exp(a[base + e] - mx);
    }
    // score(e): mixture weight; sel(e): selection score (bias added).
    // Groups outside the topk_group best (by sum of each group's top-2
    // selection scores) are excluded from selection entirely.
    var group_mask: u32 = 0xffffffffu;
    if (n_group > 1u) {
        let gsize = n_e / n_group;
        group_mask = 0u;
        // Pick topk_group groups by their top-2 sum.
        for (var pick: u32 = 0u; pick < topk_group; pick = pick + 1u) {
            var best_g: u32 = 0u;
            var best_v: f32 = -3.4e38;
            for (var g: u32 = 0u; g < n_group; g = g + 1u) {
                if ((group_mask & (1u << g)) != 0u) {
                    continue;
                }
                var top1: f32 = -3.4e38;
                var top2: f32 = -3.4e38;
                for (var j: u32 = 0u; j < gsize; j = j + 1u) {
                    let e = g * gsize + j;
                    var sv = a[base + e];
                    if (sigmoid_gate) {
                        sv = 1.0 / (1.0 + exp(-sv));
                    } else {
                        sv = exp(sv - mx) / denom;
                    }
                    if (has_bias) {
                        sv = sv + b[e];
                    }
                    if (sv > top1) {
                        top2 = top1;
                        top1 = sv;
                    } else if (sv > top2) {
                        top2 = sv;
                    }
                }
                // Group score = sum of the group's top-2 selection
                // scores (single-expert groups contribute just top1).
                let gscore = top1 + select(top2, 0.0, top2 < -1.0e37);
                if (gscore > best_v) {
                    best_v = gscore;
                    best_g = g;
                }
            }
            group_mask = group_mask | (1u << best_g);
        }
    }

    var wsum: f32 = 0.0;
    for (var s: u32 = 0u; s < slots; s = s + 1u) {
        var best: u32 = 0xffffffffu;
        var best_v: f32 = -3.4e38;
        for (var e: u32 = 0u; e < n_e; e = e + 1u) {
            if (n_group > 1u) {
                let g = e / (n_e / n_group);
                if ((group_mask & (1u << g)) == 0u) {
                    continue;
                }
            }
            var taken = false;
            for (var s2: u32 = 0u; s2 < s; s2 = s2 + 1u) {
                if (u32(out[(t * slots + s2) * 2u]) == e) {
                    taken = true;
                }
            }
            if (taken) {
                continue;
            }
            var sv = a[base + e];
            if (sigmoid_gate) {
                sv = 1.0 / (1.0 + exp(-sv));
            } else {
                sv = exp(sv - mx) / denom;
            }
            if (has_bias) {
                sv = sv + b[e];
            }
            if (sv > best_v) {
                best_v = sv;
                best = e;
            }
        }
        // Mixture weight is the UNBIASED score of the selected expert.
        var p: f32;
        if (sigmoid_gate) {
            p = 1.0 / (1.0 + exp(-a[base + best]));
        } else {
            p = exp(a[base + best] - mx) / denom;
        }
        out[(t * slots + s) * 2u] = f32(best);
        out[(t * slots + s) * 2u + 1u] = p;
        wsum = wsum + p;
    }
    let d = select(1.0, wsum, renorm && wsum > 0.0);
    for (var s: u32 = 0u; s < slots; s = s + 1u) {
        out[(t * slots + s) * 2u + 1u] = out[(t * slots + s) * 2u + 1u] / d * route_scale;
    }
}

// MoE combine: a = expert outputs [m tokens * n_heads slots, k hidden];
// c = routing table [m, n_heads, 2]; out[t, h] = sum_s w(t,s)*a[t,s,h].
@compute @workgroup_size(256, 1, 1)
fn moe_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let hidden = params.k;
    let total = params.m * hidden;
    if (i >= total) {
        return;
    }
    let t = i / hidden;
    let h = i % hidden;
    let slots = params.n_heads;
    var acc: f32 = 0.0;
    for (var s: u32 = 0u; s < slots; s = s + 1u) {
        let w = c[(t * slots + s) * 2u + 1u];
        acc = acc + w * a[(t * slots + s) * hidden + h];
    }
    out[i] = acc;
}

// Column slice: out[r, 0..n] = a[r, pos0..pos0+n] over m rows of
// stride k — extracts one block's AltUp per-layer slice from the
// packed [seq, n_layers*hidden_per_layer] tensor.
@compute @workgroup_size(256, 1, 1)
fn slice_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.m * params.n;
    if (i >= total) {
        return;
    }
    let r = i / params.n;
    let j = i % params.n;
    out[i] = a[r * params.k + params.pos0 + j];
}

// out[r, pos0..pos0+n] = a[r / n_kv_heads, 0..n] over m output rows of
// stride k — writes a column span into a preallocated buffer, with the
// source row shared by groups of n_kv_heads consecutive output rows
// (n_kv_heads = 1 for a plain column write; = heads to broadcast MLA's
// single rope'd K head across all heads).
@compute @workgroup_size(256, 1, 1)
fn scatter_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.m * params.n;
    if (i >= total) {
        return;
    }
    let r = i / params.n;
    let j = i % params.n;
    let src = r / params.n_kv_heads;
    out[r * params.k + params.pos0 + j] = a[src * params.n + j];
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
    // Partial rotary (params.n = rot width, 0 = full): pairs past the
    // rotated span copy through unchanged; the inv-freq exponent runs
    // over the ROTATED width (GLM-4 semantics).
    let rot = select(params.head_dim, params.n, params.n != 0u);
    if (2u * d >= rot) {
        out[base + 2u * d]      = a[base + 2u * d];
        out[base + 2u * d + 1u] = a[base + 2u * d + 1u];
        return;
    }
    let pos = f32(params.pos0 + s) * params.fscale;
    var inv_freq = pow(params.theta, -2.0 * f32(d) / f32(rot));
    if ((params.flags & 1u) != 0u) {
        inv_freq = inv_freq / b[d];
    }
    let angle = pos * inv_freq;
    let cw = cos(angle);
    let sn = sin(angle);
    let x0 = a[base + 2u * d];
    let x1 = a[base + 2u * d + 1u];
    out[base + 2u * d]      = x0 * cw - x1 * sn;
    out[base + 2u * d + 1u] = x0 * sn + x1 * cw;
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
    // Partial rotary: rotate-half pairs are (i, i+rot/2) within the
    // rotated span; dims past it copy through.
    let rot = select(params.head_dim, params.n, params.n != 0u);
    let rhalf = rot / 2u;
    if (d >= rhalf) {
        // Threads past the rotated pairs each copy two pass-through
        // dims: [rot, head_dim) split evenly across (half - rhalf)
        // threads. (Both widths are even, so the split is exact.)
        let j = base + rot + 2u * (d - rhalf);
        out[j] = a[j];
        out[j + 1u] = a[j + 1u];
        return;
    }
    let pos = f32(params.pos0 + s) * params.fscale;
    var inv_freq = pow(params.theta, -2.0 * f32(d) / f32(rot));
    if ((params.flags & 1u) != 0u) {
        inv_freq = inv_freq / b[d];
    }
    let angle = pos * inv_freq;
    let cw = cos(angle);
    let sn = sin(angle);
    let x0 = a[base + d];
    let x1 = a[base + d + rhalf];
    out[base + d]         = x0 * cw - x1 * sn;
    out[base + d + rhalf] = x0 * sn + x1 * cw;
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
    if (params.window > 0u && sk + params.window <= params.pos0 + sq) {
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


