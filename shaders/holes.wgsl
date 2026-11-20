// Updates the black holes (at most 16) in one thread. The step shader has already summed, with
// 64-bit integer atomics, every particle's pull on each hole and what each hole swallowed:
// 1) add swallowed mass and momentum, 2) hole-hole gravity and mergers, 3) integrate.

const MAX_HOLES: u32 = 16u;
const VEL_FP: f32 = 1024.0;
const PULL_FP: f32 = 16777216.0;
const TWO32: f32 = 4294967296.0;
const DEAD_HOLE: vec4f = vec4f(1.0e6, 1.0e6, 0.0, 0.0);

struct HolesU {
    hole_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    dt: f32,
    g: f32,
    hole_soft2: f32,
    radius_per_mass: f32,
    min_radius: f32,
    bound: f32,
    mass: f32,           // one particle
    gm: f32,             // g * mass
}

@group(0) @binding(0) var<uniform> u: HolesU;
@group(0) @binding(1) var<storage, read_write> holes: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> absorb: array<u32>;

var<private> acc: array<vec2f, MAX_HOLES>;

fn i64_at(at: u32) -> f32 {
    return f32(bitcast<i32>(absorb[at + 1u])) * TWO32 + f32(absorb[at]);
}

@compute @workgroup_size(1)
fn main() {
    // swallowed matter, and the particles' pull
    for (var h = 0u; h < u.hole_count; h++) {
        let base = 12u * h;
        let n = f32(absorb[base]);
        let sv = vec2f(i64_at(base + 2u), i64_at(base + 4u)) / VEL_FP;
        acc[h] = vec2f(i64_at(base + 6u), i64_at(base + 8u)) * (u.gm / PULL_FP);
        for (var k = 0u; k < 12u; k++) { absorb[base + k] = 0u; }
        var hp = holes[2u * h];
        var hv = holes[2u * h + 1u];
        let dm = n * u.mass;
        if (hp.z > 0.0 && dm > 0.0) {
            let m_new = hp.z + dm;
            hv = vec4f((hv.xy * hp.z + sv * u.mass) / m_new, hv.z + dm, hv.w);   // z: running total swallowed (glow)
            hp.z = m_new;
            holes[2u * h] = hp;
            holes[2u * h + 1u] = hv;
        }
    }

    // hole-hole gravity and mergers
    for (var h = 0u; h < u.hole_count; h++) {
        var hp = holes[2u * h];
        if (hp.z <= 0.0) { continue; }
        var a = acc[h];
        for (var k = 0u; k < u.hole_count; k++) {
            if (k == h) { continue; }
            let kp = holes[2u * k];
            if (kp.z <= 0.0) { continue; }
            let d = kp.xy - hp.xy;
            let r2 = dot(d, d);
            let rr = max(hp.w, kp.w);
            if (r2 < rr * rr && hp.z >= kp.z) {
                var hv = holes[2u * h + 1u];
                let kv = holes[2u * k + 1u];
                let m_new = hp.z + kp.z;
                hv = vec4f((hv.xy * hp.z + kv.xy * kp.z) / m_new, hv.z + kv.z, 1.0);   // w: merge flash
                hp = vec4f((hp.xy * hp.z + kp.xy * kp.z) / m_new, m_new, hp.w);
                holes[2u * h + 1u] = hv;
                holes[2u * k] = DEAD_HOLE;
                holes[2u * k + 1u] = vec4f(0.0);
                continue;
            }
            let inv = inverseSqrt(r2 + u.hole_soft2);
            a += d * (u.g * kp.z * inv * inv * inv);
        }
        acc[h] = a;
        holes[2u * h] = hp;
    }

    for (var h = 0u; h < u.hole_count; h++) {
        var hp = holes[2u * h];
        if (hp.z <= 0.0) { continue; }
        var hv = holes[2u * h + 1u];
        let vel = hv.xy + acc[h] * u.dt;
        hv = vec4f(vel, hv.z, hv.w * 0.97);
        hp = vec4f(hp.xy + vel * u.dt, hp.z, max(u.min_radius, u.radius_per_mass * hp.z));
        if (dot(hp.xy, hp.xy) > u.bound * u.bound) {
            hp = DEAD_HOLE;
            hv = vec4f(0.0);
        }
        holes[2u * h] = hp;
        holes[2u * h + 1u] = hv;
    }
}
