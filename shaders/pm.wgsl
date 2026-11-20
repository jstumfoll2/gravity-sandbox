// Particle-mesh gravity: O(N) deposit + FFT convolution with the softened 1/r² force kernel
// on a zero-padded (isolated, non-periodic) grid, then the step shader interpolates.
//
//   deposit:   cloud-in-cell mass onto M x M nodes, 64-bit fixed point via integer atomics
//              (order independent, so runs stay bit-reproducible)
//   fft_a:     rows y < M: FFT over x, written transposed into cb
//   fft_b:     rows of cb: FFT over y, multiply by the kernel spectrum, inverse FFT over y,
//              written transposed into ca (rows y < M only)
//   fft_c:     rows y < M of ca: inverse FFT over x, in place. ca[y*L + x] = (ax, ay).
//
// The kernel is complex: K = Kx + i Ky with K(d) = -d / (|d|² + ε²)^{3/2}. Since density is real,
// IFFT(ρ̂ K̂) = (ρ*Kx) + i (ρ*Ky): both acceleration components come out of one inverse FFT.
// Its spectrum is built once per grid / world size / softening (kern_init, fft_k1, fft_k2).
//
// Each FFT row lives entirely in workgroup memory: L = 2M complex values.

const M: u32 = __M__u;
const L: u32 = __L__u;
const LOG2L: u32 = __LOG2L__u;
const FP: f32 = 65536.0;            // deposit fixed point per unit weight
const TWO32: f32 = 4294967296.0;

@group(0) @binding(0) var<uniform> u: SimU;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> grid: array<atomic<u32>>;   // (lo, hi) per node
@group(0) @binding(3) var<storage, read_write> ca: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> cb: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> ck: array<vec2f>;
@group(0) @binding(6) var<storage, read> tw: array<vec2f>;               // exp(-2πi j/L), j < L/2

var<workgroup> s: array<vec2f, L>;

fn add_u64(node: u32, v: u32) {
    let old = atomicAdd(&grid[2u * node], v);
    if (old + v < old) { atomicAdd(&grid[2u * node + 1u], 1u); }
}

@compute @workgroup_size(256)
fn deposit(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = flat_group(wid, nwg) * WG + lid;
    if (i >= u.n) { return; }
    let p = src[i].xy;
    if (!is_alive(p)) { return; }
    let g = (p - vec2f(u.grid_origin)) * u.grid_inv - 0.5;
    let f0 = floor(g);
    let f = g - f0;
    let i0 = vec2i(f0);
    let w = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    let o = array<vec2i, 4>(vec2i(0, 0), vec2i(1, 0), vec2i(0, 1), vec2i(1, 1));
    for (var k = 0u; k < 4u; k++) {
        let c = i0 + o[k];
        if (c.x < 0 || c.y < 0 || c.x >= i32(M) || c.y >= i32(M)) { continue; }
        let q = u32(w[k] * FP + 0.5);
        if (q > 0u) { add_u64(u32(c.y) * M + u32(c.x), q); }
    }
}

fn bitrev(i: u32) -> u32 {
    return reverseBits(i) >> (32u - LOG2L);
}

fn cmul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// In-place radix-2 DIT over s[], which must hold its input in bit-reversed order.
fn fft_shared(lid: u32, inverse: bool) {
    for (var len = 2u; len <= L; len <<= 1u) {
        let half = len >> 1u;
        let stride = L / len;
        for (var b = lid; b < L / 2u; b += WG) {
            let grp = b / half;
            let k = b - grp * half;
            let i0 = grp * len + k;
            let i1 = i0 + half;
            var w = tw[k * stride];
            if (inverse) { w.y = -w.y; }
            let a = s[i0];
            let t = cmul(w, s[i1]);
            s[i0] = a + t;
            s[i1] = a - t;
        }
        workgroupBarrier();
    }
}

fn node_mass(node: u32) -> f32 {
    let lo = atomicLoad(&grid[2u * node]);
    let hi = atomicLoad(&grid[2u * node + 1u]);
    return (f32(hi) * TWO32 + f32(lo)) * (u.gm / FP);
}

@compute @workgroup_size(256)
fn fft_a(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;   // < M
    for (var x = lid; x < L; x += WG) {
        var v = vec2f(0.0);
        if (x < M) { v = vec2f(node_mass(y * M + x), 0.0); }
        s[bitrev(x)] = v;
    }
    workgroupBarrier();
    fft_shared(lid, false);
    for (var x = lid; x < L; x += WG) {
        cb[x * L + y] = s[x];
    }
}

@compute @workgroup_size(256)
fn fft_b(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let x = wid.x;   // < L (a kx row)
    for (var y = lid; y < L; y += WG) {
        s[bitrev(y)] = cb[x * L + y];
    }
    workgroupBarrier();
    fft_shared(lid, false);
    for (var y = lid; y < L; y += WG) {
        s[y] = cmul(s[y], ck[x * L + y]);
    }
    workgroupBarrier();
    for (var y = lid; y < L; y += WG) {
        let r = bitrev(y);
        if (y < r) {
            let t = s[y];
            s[y] = s[r];
            s[r] = t;
        }
    }
    workgroupBarrier();
    fft_shared(lid, true);
    for (var y = lid; y < M; y += WG) {
        ca[y * L + x] = s[y];
    }
}

@compute @workgroup_size(256)
fn fft_c(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;   // < M
    for (var x = lid; x < L; x += WG) {
        s[bitrev(x)] = ca[y * L + x];
    }
    workgroupBarrier();
    fft_shared(lid, true);
    let norm = 1.0 / (f32(L) * f32(L));
    for (var x = lid; x < M; x += WG) {
        ca[y * L + x] = s[x] * norm;
    }
}

// ---------------------------------------------------------------- kernel spectrum

@compute @workgroup_size(256)
fn kern_init(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let idx = flat_group(wid, nwg) * WG + lid;
    if (idx >= L * L) { return; }
    let x = idx % L;
    let y = idx / L;
    var dx = f32(x);
    if (x >= L / 2u) { dx -= f32(L); }
    var dy = f32(y);
    if (y >= L / 2u) { dy -= f32(L); }
    let d = vec2f(dx, dy) / u.grid_inv;
    let inv = inverseSqrt(dot(d, d) + u.soft2);
    ck[idx] = -d * (inv * inv * inv);
}

@compute @workgroup_size(256)
fn fft_k1(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;   // < L
    for (var x = lid; x < L; x += WG) {
        s[bitrev(x)] = ck[y * L + x];
    }
    workgroupBarrier();
    fft_shared(lid, false);
    for (var x = lid; x < L; x += WG) {
        cb[x * L + y] = s[x];
    }
}

@compute @workgroup_size(256)
fn fft_k2(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let x = wid.x;   // < L
    for (var y = lid; y < L; y += WG) {
        s[bitrev(y)] = cb[x * L + y];
    }
    workgroupBarrier();
    fft_shared(lid, false);
    for (var y = lid; y < L; y += WG) {
        ck[x * L + y] = s[y];
    }
}
