// One simulation step for every particle, reading the cell-sorted copy (src) and writing the
// live state (dst) in the same order.
//
// Gravity: interpolated from the PM grid (pm.wgsl), or for small N an exact all-pairs sum tiled
// through workgroup memory. Contacts: within the contact radius h, particles push apart and
// share momentum (viscosity), found through the 3x3 hashed cells around each particle.
// Black holes: direct pull; matter inside a horizon is swallowed. Each workgroup also reduces
// its particles' pull on every hole and what it swallowed, and adds that to the hole with
// 64-bit integer atomics, so the totals do not depend on thread order.

const VEL_FP: f32 = 1024.0;
const PULL_FP: f32 = 16777216.0;
const TWO32: f32 = 4294967296.0;

@group(0) @binding(0) var<uniform> u: SimU;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read> src_aux: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> dst_aux: array<u32>;
@group(0) @binding(5) var<storage, read> ends: array<u32>;        // bucket ends from the sort
@group(0) @binding(6) var<storage, read> accel: array<vec2f>;     // PM result, row stride 2M
// hole i: holes[2i] = (x, y, mass, radius), holes[2i+1] = (vx, vy, swallowed, flash)
@group(0) @binding(7) var<storage, read> holes: array<vec4f>;
// per hole, 12 words: swallowed count, sum vx, sum vy, pull x, pull y (each lo, hi), pad
@group(0) @binding(8) var<storage, read_write> absorb: array<atomic<u32>>;

var<workgroup> s_pos: array<vec2f, WG>;
var<workgroup> s_red: array<vec4f, WG>;
var<workgroup> s_cnt: array<f32, WG>;

fn bad(x: f32) -> bool {
    // NaN or infinity, tested on the bits so fast-math can't fold it away
    return (bitcast<u32>(x) & 0x7fffffffu) >= 0x7f800000u;
}

// Adds round(v) to a signed 64-bit (lo, hi) pair.
fn add_i64(at: u32, v: f32) {
    let r = round(v);
    let hi = floor(r / TWO32);
    let lo = u32(clamp(r - hi * TWO32, 0.0, 4294967295.0));
    let old = atomicAdd(&absorb[at], lo);
    var carry = 0u;
    if (old + lo < old) { carry = 1u; }
    let add_hi = bitcast<u32>(i32(hi)) + carry;
    if (add_hi != 0u) { atomicAdd(&absorb[at + 1u], add_hi); }
}

fn pm_accel(p: vec2f) -> vec2f {
    let m = i32(u.m_grid);
    let stride = 2u * u.m_grid;
    let g = (p - vec2f(u.grid_origin)) * u.grid_inv - 0.5;
    let f0 = floor(g);
    let f = g - f0;
    let i0 = vec2i(f0);
    var a = vec2f(0.0);
    let w = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    let o = array<vec2i, 4>(vec2i(0, 0), vec2i(1, 0), vec2i(0, 1), vec2i(1, 1));
    for (var k = 0u; k < 4u; k++) {
        let c = i0 + o[k];
        if (c.x < 0 || c.y < 0 || c.x >= m || c.y >= m) { continue; }
        a += w[k] * accel[u32(c.y) * stride + u32(c.x)];
    }
    return a;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
    let gi = flat_group(wid, nwg) * WG + lid;
    let valid = gi < u.n;
    let i = min(gi, u.n - 1u);
    let me = src[i];
    var p = me.xy;
    var v = me.zw;
    let alive = valid && is_alive(p);

    // ---- gravity
    var grav = vec2f(0.0);
    if (u.exact != 0u) {
        for (var tile = 0u; tile < u.n; tile += WG) {
            s_pos[lid] = src[tile + lid].xy;
            workgroupBarrier();
            for (var j = 0u; j < WG; j++) {
                let d = s_pos[j] - p;
                let inv = inverseSqrt(dot(d, d) + u.soft2);
                grav += d * (inv * inv * inv);
            }
            workgroupBarrier();
        }
        grav *= u.gm;
    } else if (alive) {
        grav = pm_accel(p);
    }

    // ---- contacts
    var acc = vec2f(0.0);
    var dens = 0.0;
    var heat = 0.0;
    if (alive) {
        let h2 = u.h * u.h;
        let inv_h = 1.0 / u.h;
        let c0 = cell_of(p);
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let c = c0 + vec2i(dx, dy);
                let b = bucket_of(c);
                var j = 0u;
                if (b > 0u) { j = ends[b - 1u]; }
                let end = ends[b];
                for (; j < end; j++) {
                    if (j == i) { continue; }
                    let q = src[j];
                    let d = q.xy - p;
                    let r2 = dot(d, d);
                    if (r2 >= h2 || r2 <= 1.0e-12) { continue; }
                    // other cells can share this bucket; count each neighbour once
                    if (any(cell_of(q.xy) != c)) { continue; }
                    let r = sqrt(r2);
                    let w = 1.0 - r * inv_h;
                    let dv = q.zw - v;
                    acc -= d * (u.k * w / r);
                    acc += dv * (u.visc * w);
                    dens += w * w;
                    heat += w * dot(dv, dv);
                }
            }
        }
    }
    acc += grav;

    // ---- black holes
    var swallowed_by = -1;
    if (alive) {
        for (var h = 0u; h < u.hole_count; h++) {
            let hp = holes[2u * h];
            if (hp.z <= 0.0) { continue; }
            let d = hp.xy - p;
            let r2 = dot(d, d);
            if (r2 < hp.w * hp.w) { swallowed_by = i32(h); break; }
            let inv = inverseSqrt(r2 + u.hole_soft2);
            acc += d * (u.g * hp.z * inv * inv * inv);
        }
    }

    // ---- pointer forces (pull / push)
    for (var t = 0u; t < u.touch_count; t++) {
        let tc = u.touch[t];
        let d = tc.xy - p;
        let r = length(d) + 1.0e-3;
        let f = tc.z * (1.0 - smoothstep(0.0, tc.w, r));
        acc += d / r * f;
    }

    let v0 = v;
    let p0 = p;
    v += acc * u.dt;
    v *= max(0.0, 1.0 - u.drag * u.dt);
    p += v * u.dt;
    let old = unpack2x16float(src_aux[i]);
    let temp = old.x * max(0.0, 1.0 - u.heat_decay * u.dt) + heat * u.heat_gain * u.dt;

    var out_alive = alive && swallowed_by < 0;
    if (dot(p, p) > u.bound * u.bound || bad(p.x) || bad(p.y)) { out_alive = false; }
    if (valid) {
        if (out_alive) {
            dst[i] = vec4f(p, v);
            dst_aux[i] = pack2x16float(vec2f(min(temp, 6.0e4), min(dens, 6.0e4)));
        } else {
            dst[i] = DEAD;
            dst_aux[i] = 0u;
        }
    }

    // ---- what this workgroup did to each hole (uniform loop: every thread takes part)
    for (var h = 0u; h < u.hole_count; h++) {
        let hp = holes[2u * h];
        var pull = vec2f(0.0);
        var eaten = 0.0;
        var ev = vec2f(0.0);
        if (alive && hp.z > 0.0) {
            if (swallowed_by == i32(h)) {
                eaten = 1.0;
                ev = clamp(v0, vec2f(-1.0e5), vec2f(1.0e5));
            } else {
                let d = p0 - hp.xy;
                let inv = inverseSqrt(dot(d, d) + u.hole_soft2);
                pull = d * (inv * inv * inv);
            }
        }
        s_red[lid] = vec4f(pull, ev);
        s_cnt[lid] = eaten;
        workgroupBarrier();
        for (var off = WG / 2u; off > 0u; off >>= 1u) {
            if (lid < off) {
                s_red[lid] += s_red[lid + off];
                s_cnt[lid] += s_cnt[lid + off];
            }
            workgroupBarrier();
        }
        if (lid == 0u) {
            let r = s_red[0];
            let base = 12u * h;
            if (s_cnt[0] > 0.0) {
                atomicAdd(&absorb[base], u32(s_cnt[0]));
                add_i64(base + 2u, r.z * VEL_FP);
                add_i64(base + 4u, r.w * VEL_FP);
            }
            add_i64(base + 6u, r.x * PULL_FP);
            add_i64(base + 8u, r.y * PULL_FP);
        }
        workgroupBarrier();
    }
}
