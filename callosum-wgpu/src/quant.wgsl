// Quantized matmul kernels: GGML block formats dequantized in-shader.
//
// Weight rows are word-aligned (`params.len` = words per row, set by
// the uploader) and bound as `b`, reinterpreted byte-wise. Formats
// mirror ggml exactly — q4_0 (18 B / 32), q8_0 (34 B / 32), and the
// 256-element K-quant super-blocks q4_K (144 B), q5_K (176 B),
// q6_K (210 B).
//
// Every format exposes a `dot unit` of 32 elements: `dot_<fmt>(row_word,
// unit, x_base)` returns the dot product of that unit's dequantized
// weights with 32 activations from `a`. On top of the units sit two
// entry points per format:
//   matmul_t_<fmt> — [m,k] × [n,k]^T grid, one thread per output
//   matvec_<fmt>   — m == 1 decode path: one 256-thread workgroup per
//                    output element, tree-reduced. This is what makes
//                    single-token decode use the whole GPU instead of
//                    n threads.

fn bget(row_word: u32, byte_off: u32) -> u32 {
    let w = bitcast<u32>(b[row_word + byte_off / 4u]);
    return (w >> ((byte_off % 4u) * 8u)) & 0xFFu;
}

// Word starting at an arbitrary byte offset, assembled from the two
// straddled words when misaligned. One load when aligned.
fn u32at(row_word: u32, byte_off: u32) -> u32 {
    let sh = (byte_off % 4u) * 8u;
    let w0 = bitcast<u32>(b[row_word + byte_off / 4u]);
    if (sh == 0u) {
        return w0;
    }
    let w1 = bitcast<u32>(b[row_word + byte_off / 4u + 1u]);
    return (w0 >> sh) | (w1 << (32u - sh));
}

fn f16at(row_word: u32, byte_off: u32) -> f32 {
    let w = bitcast<u32>(b[row_word + byte_off / 4u]);
    var bits: u32;
    if ((byte_off & 2u) == 0u) {
        bits = w & 0xFFFFu;
    } else {
        bits = (w >> 16u) & 0xFFFFu;
    }
    return unpack2x16float(bits).x;
}

fn i8at(row_word: u32, byte_off: u32) -> f32 {
    var v = i32(bget(row_word, byte_off));
    if (v >= 128) {
        v = v - 256;
    }
    return f32(v);
}

// ---- q4_0: 32 elems = f16 d + 16 nibble bytes (18 B) ----

fn dot_q4_0(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let base = unit * 18u;
    let d = f16at(row_word, base);
    var acc: f32 = 0.0;
    for (var l: u32 = 0u; l < 16u; l = l + 4u) {
        let w = u32at(row_word, base + 2u + l);
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let byte = (w >> (8u * j)) & 0xFFu;
            acc = acc + (f32(byte & 0xFu) - 8.0) * a[x_base + l + j];
            acc = acc + (f32(byte >> 4u) - 8.0) * a[x_base + 16u + l + j];
        }
    }
    return acc * d;
}

// ---- q8_0: 32 elems = f16 d + 32 int8 (34 B) ----

fn i8x4_dot(w: u32, x_base: u32, l: u32) -> f32 {
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        var v = i32((w >> (8u * j)) & 0xFFu);
        if (v >= 128) {
            v = v - 256;
        }
        acc = acc + f32(v) * a[x_base + l + j];
    }
    return acc;
}

fn dot_q8_0(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let base = unit * 34u;
    let d = f16at(row_word, base);
    var acc: f32 = 0.0;
    for (var l: u32 = 0u; l < 32u; l = l + 4u) {
        acc = acc + i8x4_dot(u32at(row_word, base + 2u + l), x_base, l);
    }
    return acc * d;
}

// ---- q4_1: 32 elems = f16 d + f16 m + 16 nibble bytes (20 B) ----

fn dot_q4_1(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let base = unit * 20u;
    let d = f16at(row_word, base);
    let m = f16at(row_word, base + 2u);
    var qdot: f32 = 0.0;
    var xsum: f32 = 0.0;
    for (var l: u32 = 0u; l < 16u; l = l + 4u) {
        let w = u32at(row_word, base + 4u + l);
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let byte = (w >> (8u * j)) & 0xFFu;
            let x0 = a[x_base + l + j];
            let x1 = a[x_base + 16u + l + j];
            qdot = qdot + f32(byte & 0xFu) * x0 + f32(byte >> 4u) * x1;
            xsum = xsum + x0 + x1;
        }
    }
    return d * qdot + m * xsum;
}

// ---- q5_0: 32 elems = f16 d + u32 qh + 16 nibble bytes (22 B) ----

fn dot_q5_0(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let base = unit * 22u;
    let d = f16at(row_word, base);
    let qh = u32at(row_word, base + 2u);
    var acc: f32 = 0.0;
    for (var l: u32 = 0u; l < 16u; l = l + 4u) {
        let w = u32at(row_word, base + 6u + l);
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let byte = (w >> (8u * j)) & 0xFFu;
            let q0 = (byte & 0xFu) | (((qh >> (l + j)) & 1u) << 4u);
            let q1 = (byte >> 4u) | (((qh >> (l + j + 16u)) & 1u) << 4u);
            acc = acc + (f32(q0) - 16.0) * a[x_base + l + j];
            acc = acc + (f32(q1) - 16.0) * a[x_base + 16u + l + j];
        }
    }
    return acc * d;
}

// ---- q5_1: 32 elems = f16 d + f16 m + u32 qh + 16 nibbles (24 B) ----

fn dot_q5_1(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let base = unit * 24u;
    let d = f16at(row_word, base);
    let m = f16at(row_word, base + 2u);
    let qh = u32at(row_word, base + 4u);
    var qdot: f32 = 0.0;
    var xsum: f32 = 0.0;
    for (var l: u32 = 0u; l < 16u; l = l + 4u) {
        let w = u32at(row_word, base + 8u + l);
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let byte = (w >> (8u * j)) & 0xFFu;
            let q0 = (byte & 0xFu) | (((qh >> (l + j)) & 1u) << 4u);
            let q1 = (byte >> 4u) | (((qh >> (l + j + 16u)) & 1u) << 4u);
            let x0 = a[x_base + l + j];
            let x1 = a[x_base + 16u + l + j];
            qdot = qdot + f32(q0) * x0 + f32(q1) * x1;
            xsum = xsum + x0 + x1;
        }
    }
    return d * qdot + m * xsum;
}

// ---- f16 / bf16: 32 elems stored dense (64 B) ----

fn dot_f16(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let base = unit * 64u;
    var acc: f32 = 0.0;
    for (var l: u32 = 0u; l < 32u; l = l + 2u) {
        let w = u32at(row_word, base + 2u * l);
        let v = unpack2x16float(w);
        acc = acc + v.x * a[x_base + l] + v.y * a[x_base + l + 1u];
    }
    return acc;
}

fn dot_bf16(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let base = unit * 64u;
    var acc: f32 = 0.0;
    for (var l: u32 = 0u; l < 32u; l = l + 2u) {
        let w = u32at(row_word, base + 2u * l);
        acc = acc + bitcast<f32>((w & 0xFFFFu) << 16u) * a[x_base + l];
        acc = acc + bitcast<f32>(w & 0xFFFF0000u) * a[x_base + l + 1u];
    }
    return acc;
}

// ---- q2_K: super-block of 256 = scales[16], qs[64], d, dmin (84 B).
// Per 16-element group: 4-bit scale (low nibble) on d, 4-bit min
// (high nibble) on dmin; 2-bit quants, shift = 2 * (group pair). ----

fn dot_q2_k(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let sb = unit / 8u;
    let u = unit % 8u;
    let base = sb * 84u;
    let d = f16at(row_word, base + 80u);
    let dmin = f16at(row_word, base + 82u);
    let half = u / 4u;
    let j2 = u % 4u;
    let shift = 2u * j2;
    let sc0 = bget(row_word, base + half * 8u + j2 * 2u);
    let sc1 = bget(row_word, base + half * 8u + j2 * 2u + 1u);
    let qb = base + 16u + 32u * half;
    var acc0: f32 = 0.0;
    var xs0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var xs1: f32 = 0.0;
    for (var l: u32 = 0u; l < 16u; l = l + 4u) {
        let w0 = u32at(row_word, qb + l);
        let w1 = u32at(row_word, qb + 16u + l);
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let x0 = a[x_base + l + j];
            let x1 = a[x_base + 16u + l + j];
            acc0 = acc0 + f32(((w0 >> (8u * j)) >> shift) & 3u) * x0;
            xs0 = xs0 + x0;
            acc1 = acc1 + f32(((w1 >> (8u * j)) >> shift) & 3u) * x1;
            xs1 = xs1 + x1;
        }
    }
    return d * (f32(sc0 & 0xFu) * acc0 + f32(sc1 & 0xFu) * acc1)
        - dmin * (f32(sc0 >> 4u) * xs0 + f32(sc1 >> 4u) * xs1);
}

// ---- q3_K: super-block of 256 = hmask[32], qs[64], scales[12], d
// (110 B). 3-bit quants: 2 low bits from qs, high bit from hmask
// (offset -4 when the mask bit is clear); 6-bit scales unpacked via
// ggml's kmask transform, value - 32. ----

fn q3k_scale(row_word: u32, base: u32, is: u32) -> f32 {
    let kmask1 = 0x03030303u;
    let kmask2 = 0x0f0f0f0fu;
    let a0 = u32at(row_word, base + 96u);
    let a1 = u32at(row_word, base + 100u);
    let t = u32at(row_word, base + 104u);
    var word: u32;
    switch is / 4u {
        case 0u: {
            word = (a0 & kmask2) | ((t & kmask1) << 4u);
        }
        case 1u: {
            word = (a1 & kmask2) | (((t >> 2u) & kmask1) << 4u);
        }
        case 2u: {
            word = ((a0 >> 4u) & kmask2) | (((t >> 4u) & kmask1) << 4u);
        }
        default: {
            word = ((a1 >> 4u) & kmask2) | (((t >> 6u) & kmask1) << 4u);
        }
    }
    return f32((word >> (8u * (is % 4u))) & 0xFFu) - 32.0;
}

fn dot_q3_k(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let sb = unit / 8u;
    let u = unit % 8u;
    let base = sb * 110u;
    let d = f16at(row_word, base + 108u);
    let half = u / 4u;
    let j2 = u % 4u;
    let shift = 2u * j2;
    let hbit = half * 4u + j2;
    let is = half * 8u + j2 * 2u;
    let s0 = q3k_scale(row_word, base, is);
    let s1 = q3k_scale(row_word, base, is + 1u);
    let qb = base + 32u + 32u * half;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    for (var l: u32 = 0u; l < 16u; l = l + 4u) {
        let w0 = u32at(row_word, qb + l);
        let w1 = u32at(row_word, qb + 16u + l);
        let h0 = u32at(row_word, base + l);
        let h1 = u32at(row_word, base + 16u + l);
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            var q0 = f32(((w0 >> (8u * j)) >> shift) & 3u);
            if ((((h0 >> (8u * j)) >> hbit) & 1u) == 0u) {
                q0 = q0 - 4.0;
            }
            var q1 = f32(((w1 >> (8u * j)) >> shift) & 3u);
            if ((((h1 >> (8u * j)) >> hbit) & 1u) == 0u) {
                q1 = q1 - 4.0;
            }
            acc0 = acc0 + q0 * a[x_base + l + j];
            acc1 = acc1 + q1 * a[x_base + 16u + l + j];
        }
    }
    return d * (s0 * acc0 + s1 * acc1);
}

// ---- K-quant shared: 6-bit (scale, min) pairs, ggml get_scale_min_k4 ----

fn scale_min_k4(row_word: u32, scales_off: u32, j: u32) -> vec2<f32> {
    var sc: u32;
    var mn: u32;
    if (j < 4u) {
        sc = bget(row_word, scales_off + j) & 63u;
        mn = bget(row_word, scales_off + j + 4u) & 63u;
    } else {
        let qj4 = bget(row_word, scales_off + j + 4u);
        sc = (qj4 & 0xFu) | ((bget(row_word, scales_off + j - 4u) >> 6u) << 4u);
        mn = (qj4 >> 4u) | ((bget(row_word, scales_off + j) >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(mn));
}

// ---- q4_K: super-block of 256 = d, dmin (f16), scales[12], qs[128] (144 B) ----

fn dot_q4_k(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let sb = unit / 8u;
    let j = unit % 8u; // 32-element sub-block, own (scale, min)
    let base = sb * 144u;
    let d = f16at(row_word, base);
    let dmin = f16at(row_word, base + 2u);
    let sm = scale_min_k4(row_word, base + 4u, j);
    // Nibble group: qs[32·(j/2)], low nibbles for even j, high for odd.
    let qs_word = row_word + (base + 16u + 32u * (j / 2u)) / 4u;
    let shift = select(0u, 4u, (j & 1u) == 1u);
    var qdot: f32 = 0.0;
    var xsum: f32 = 0.0;
    for (var l: u32 = 0u; l < 32u; l = l + 4u) {
        let w = bitcast<u32>(b[qs_word + l / 4u]);
        for (var jj: u32 = 0u; jj < 4u; jj = jj + 1u) {
            let q = ((w >> (8u * jj)) >> shift) & 0xFu;
            let x = a[x_base + l + jj];
            qdot = qdot + f32(q) * x;
            xsum = xsum + x;
        }
    }
    return d * sm.x * qdot - dmin * sm.y * xsum;
}

// Cooperative q4_K matvec warp body: 32 lanes share one row and read
// CONSECUTIVE qs words per super-block (full coalescing), vs the
// generic path's lane-strided unit walk. Lane l owns qs bytes
// [4l, 4l+4): nibble pair p = l/8, in-pair offset q0 = 4*(l%8).
fn q4k_row_coop(row_word: u32, sbs: u32, lane: u32, x_base: u32) -> f32 {
    var acc: f32 = 0.0;
    let p = lane / 8u;
    let q0 = 4u * (lane % 8u);
    for (var sb: u32 = 0u; sb < sbs; sb = sb + 1u) {
        let base = sb * 144u;
        let d = f16at(row_word, base);
        let dmin = f16at(row_word, base + 2u);
        let sm0 = scale_min_k4(row_word, base + 4u, 2u * p);
        let sm1 = scale_min_k4(row_word, base + 4u, 2u * p + 1u);
        // Coalesced: lanes 0..32 load the super-block's 32 qs words.
        let w = bitcast<u32>(b[row_word + (base + 16u) / 4u + lane]);
        var qdot0: f32 = 0.0;
        var xsum0: f32 = 0.0;
        var qdot1: f32 = 0.0;
        var xsum1: f32 = 0.0;
        let xb = x_base + sb * 256u + 64u * p + q0;
        for (var jj: u32 = 0u; jj < 4u; jj = jj + 1u) {
            let byte = (w >> (8u * jj)) & 0xFFu;
            let x0 = a[xb + jj];
            let x1 = a[xb + 32u + jj];
            qdot0 = qdot0 + f32(byte & 0xFu) * x0;
            xsum0 = xsum0 + x0;
            qdot1 = qdot1 + f32(byte >> 4u) * x1;
            xsum1 = xsum1 + x1;
        }
        acc = acc + d * (sm0.x * qdot0 + sm1.x * qdot1) - dmin * (sm0.y * xsum0 + sm1.y * xsum1);
    }
    return acc;
}

// Cooperative q6_K warp body: 32 lanes = 2 halves x 16 position-pairs.
// Each lane owns positions {2i, 2i+1} of its half and produces the 4
// quarter elements per position (ql low/high nibble + qh bit pairs),
// reading bytes that adjacent lanes access contiguously.
fn q6k_row_coop(row_word: u32, sbs: u32, lane: u32, x_base: u32) -> f32 {
    var acc: f32 = 0.0;
    let half = lane / 16u;
    let p0 = 2u * (lane % 16u);
    for (var sb: u32 = 0u; sb < sbs; sb = sb + 1u) {
        let base = sb * 210u;
        let d = f16at(row_word, base + 208u);
        let ql = base + 64u * half;
        let qh = base + 128u + 32u * half;
        let sc = base + 192u + 8u * half;
        let xb = x_base + sb * 256u + 128u * half;
        for (var pi: u32 = 0u; pi < 2u; pi = pi + 1u) {
            let l = p0 + pi;
            let lo0 = bget(row_word, ql + l);
            let lo32 = bget(row_word, ql + 32u + l);
            let hi = bget(row_word, qh + l);
            let r = l / 16u; // scale column within quarter pair
            let s0 = i8at(row_word, sc + r);
            let s1 = i8at(row_word, sc + 2u + r);
            let s2 = i8at(row_word, sc + 4u + r);
            let s3 = i8at(row_word, sc + 6u + r);
            let q1 = f32((lo0 & 0xFu) | (((hi >> 0u) & 3u) << 4u)) - 32.0;
            let q2 = f32((lo32 & 0xFu) | (((hi >> 2u) & 3u) << 4u)) - 32.0;
            let q3 = f32((lo0 >> 4u) | (((hi >> 4u) & 3u) << 4u)) - 32.0;
            let q4 = f32((lo32 >> 4u) | (((hi >> 6u) & 3u) << 4u)) - 32.0;
            acc = acc
                + d * s0 * q1 * a[xb + l]
                + d * s1 * q2 * a[xb + 32u + l]
                + d * s2 * q3 * a[xb + 64u + l]
                + d * s3 * q4 * a[xb + 96u + l];
        }
    }
    return acc;
}

// Cooperative q2_K warp body: lanes 0..16 take even super-block qs
// words, 16..32 the odd super-block (2 super-blocks per iteration);
// each qs byte yields one element per 2-bit shift (4 quarters).
fn q2k_row_coop(row_word: u32, sbs: u32, lane: u32, x_base: u32) -> f32 {
    var acc: f32 = 0.0;
    let sb_off = lane / 16u;
    let wq = lane % 16u; // qs word index within the super-block (of 16)
    let half = wq / 8u;
    let idx = (wq % 8u) * 4u; // first qs byte within the half (0..32)
    let sub16 = idx / 16u;
    for (var sb2: u32 = 0u; sb2 < sbs; sb2 = sb2 + 2u) {
        let sb = sb2 + sb_off;
        if (sb >= sbs) {
            continue;
        }
        let base = sb * 84u;
        let d = f16at(row_word, base + 80u);
        let dmin = f16at(row_word, base + 82u);
        let w = bitcast<u32>(b[row_word + (base + 16u) / 4u + wq]);
        let scb = base + half * 8u;
        let xq = x_base + sb * 256u + half * 128u + sub16 * 16u + (idx % 16u);
        for (var j2: u32 = 0u; j2 < 4u; j2 = j2 + 1u) {
            let sc = bget(row_word, scb + j2 * 2u + sub16);
            let dl = d * f32(sc & 0xFu);
            let ml = dmin * f32(sc >> 4u);
            let xb2 = xq + j2 * 32u;
            for (var jj: u32 = 0u; jj < 4u; jj = jj + 1u) {
                let q = ((w >> (8u * jj)) >> (2u * j2)) & 3u;
                let x = a[xb2 + jj];
                acc = acc + dl * f32(q) * x - ml * x;
            }
        }
    }
    return acc;
}

// Cooperative q3_K warp body: same shape as q2_K plus the hmask high
// bit (offset -4 when clear) and the 6-bit kmask scales.
fn q3k_row_coop(row_word: u32, sbs: u32, lane: u32, x_base: u32) -> f32 {
    var acc: f32 = 0.0;
    let sb_off = lane / 16u;
    let wq = lane % 16u;
    let half = wq / 8u;
    let idx = (wq % 8u) * 4u;
    let sub16 = idx / 16u;
    for (var sb2: u32 = 0u; sb2 < sbs; sb2 = sb2 + 2u) {
        let sb = sb2 + sb_off;
        if (sb >= sbs) {
            continue;
        }
        let base = sb * 110u;
        let d = f16at(row_word, base + 108u);
        let w = u32at(row_word, base + 32u + half * 32u + idx);
        let hm = u32at(row_word, base + sub16 * 16u + (idx % 16u));
        let xq = x_base + sb * 256u + half * 128u + sub16 * 16u + (idx % 16u);
        for (var j2: u32 = 0u; j2 < 4u; j2 = j2 + 1u) {
            let is = half * 8u + j2 * 2u + sub16;
            let s = q3k_scale(row_word, base, is);
            let dl = d * s;
            let hbit = half * 4u + j2;
            let xb2 = xq + j2 * 32u;
            for (var jj: u32 = 0u; jj < 4u; jj = jj + 1u) {
                var q = f32(((w >> (8u * jj)) >> (2u * j2)) & 3u);
                if ((((hm >> (8u * jj)) >> hbit) & 1u) == 0u) {
                    q = q - 4.0;
                }
                acc = acc + dl * q * a[xb2 + jj];
            }
        }
    }
    return acc;
}

// ---- q5_K: d, dmin, scales[12], qh[32], qs[128] (176 B) ----

fn dot_q5_k(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let sb = unit / 8u;
    let j = unit % 8u;
    let base = sb * 176u;
    let d = f16at(row_word, base);
    let dmin = f16at(row_word, base + 2u);
    let sm = scale_min_k4(row_word, base + 4u, j);
    let qh_word = row_word + (base + 16u) / 4u;
    let qs_word = row_word + (base + 48u + 32u * (j / 2u)) / 4u;
    let shift = select(0u, 4u, (j & 1u) == 1u);
    var qdot: f32 = 0.0;
    var xsum: f32 = 0.0;
    for (var l: u32 = 0u; l < 32u; l = l + 4u) {
        let w = bitcast<u32>(b[qs_word + l / 4u]);
        let wh = bitcast<u32>(b[qh_word + l / 4u]);
        for (var jj: u32 = 0u; jj < 4u; jj = jj + 1u) {
            var q = ((w >> (8u * jj)) >> shift) & 0xFu;
            q = q + ((((wh >> (8u * jj)) >> j) & 1u) << 4u);
            let x = a[x_base + l + jj];
            qdot = qdot + f32(q) * x;
            xsum = xsum + x;
        }
    }
    return d * sm.x * qdot - dmin * sm.y * xsum;
}

// ---- q6_K: ql[128], qh[64], int8 scales[16], f16 d (210 B) ----

fn dot_q6_k(row_word: u32, unit: u32, x_base: u32) -> f32 {
    let sb = unit / 8u;
    let j = unit % 8u;
    let base = sb * 210u;
    let d = f16at(row_word, base + 208u);
    let half_idx = j / 4u; // 0..1: which 128-element half
    let r = j % 4u;        // 0..3: which 32-row within the half
    let ql = base + 64u * half_idx;
    let qh = base + 128u + 32u * half_idx;
    let sc = base + 192u + 8u * half_idx + 2u * r;
    let ql_off = select(ql + 32u, ql, r == 0u || r == 2u);
    let low_shift = select(4u, 0u, r < 2u);
    let s0 = i8at(row_word, sc);
    let s1 = i8at(row_word, sc + 1u);
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    for (var l: u32 = 0u; l < 32u; l = l + 4u) {
        let wl = u32at(row_word, ql_off + l);
        let wh = u32at(row_word, qh + l);
        for (var jj: u32 = 0u; jj < 4u; jj = jj + 1u) {
            let low = ((wl >> (8u * jj)) >> low_shift) & 0xFu;
            let hbits = ((wh >> (8u * jj)) >> (2u * r)) & 3u;
            let q = f32(low | (hbits << 4u)) - 32.0;
            let xv = q * a[x_base + l + jj];
            if (l + jj < 16u) {
                acc0 = acc0 + xv;
            } else {
                acc1 = acc1 + xv;
            }
        }
    }
    return d * (s0 * acc0 + s1 * acc1);
}

// ---- entry points ----
// params.len = weight row stride in WORDS; params.k = row length in
// elements (multiple of 32; K-quants additionally a multiple of 256).
