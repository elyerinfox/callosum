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
