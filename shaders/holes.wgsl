// Updates the black holes (at most 16) in a single workgroup (port of holes.comp):
// 1) every thread sums the pull of a slice of the particles on each hole,
// 2) a tree reduction in workgroup memory,
// 3) thread 0 adds swallowed mass and momentum, applies hole-hole gravity,
//    merges holes that touch, and integrates.

const T: u32 = 256u;
const MAX_HOLES: u32 = 16u;
const MASS_FP: f32 = 1.0e7;
const MOM_FP: f32 = 1.0e5;
const DEAD: vec4f = vec4f(1.0e6, 1.0e6, 0.0, 0.0);

struct Particle { pos: vec4f, vel: vec4f }

struct HolesU {
    count: u32,
    hole_count: u32,
    _pad0: u32,
    _pad1: u32,
    dt: f32,
    g: f32,
    hole_soft2: f32,
    radius_per_mass: f32,
    min_radius: f32,
    bound: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> parts: array<Particle>;
@group(0) @binding(2) var<storage, read_write> holes: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> absorb: array<atomic<i32>>;
@group(0) @binding(4) var<uniform> u: HolesU;

var<workgroup> s_acc: array<vec2f, T>;
var<workgroup> s_hole_acc: array<vec2f, MAX_HOLES>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid3: vec3u) {
    let lid = lid3.x;
    for (var h = 0u; h < u.hole_count; h++) {
        let hp = holes[2u * h];
        var a = vec2f(0.0);
        for (var i = lid; i < u.count; i += T) {
            let pp = parts[i].pos;
            let d = pp.xy - hp.xy;
            let inv = inverseSqrt(dot(d, d) + u.hole_soft2);
            a += d * (pp.z * inv * inv * inv);
        }
        s_acc[lid] = a;
        workgroupBarrier();
        for (var s = T / 2u; s > 0u; s >>= 1u) {
            if (lid < s) { s_acc[lid] += s_acc[lid + s]; }
            workgroupBarrier();
        }
        if (lid == 0u) { s_hole_acc[h] = s_acc[0] * u.g; }
        workgroupBarrier();
    }

    if (lid != 0u) { return; }

    // swallowed matter
    for (var h = 0u; h < u.hole_count; h++) {
        var hp = holes[2u * h];
        var hv = holes[2u * h + 1u];
        let dm = f32(atomicLoad(&absorb[3u * h])) / MASS_FP;
        let dp = vec2f(f32(atomicLoad(&absorb[3u * h + 1u])), f32(atomicLoad(&absorb[3u * h + 2u]))) / MOM_FP;
        atomicStore(&absorb[3u * h], 0);
        atomicStore(&absorb[3u * h + 1u], 0);
        atomicStore(&absorb[3u * h + 2u], 0);
        if (hp.z > 0.0 && dm > 0.0) {
            let m_new = hp.z + dm;
            hv = vec4f((hv.xy * hp.z + dp) / m_new, hv.z + dm, hv.w);   // z: running total swallowed (glow)
            hp.z = m_new;
            holes[2u * h] = hp;
            holes[2u * h + 1u] = hv;
        }
    }

    // hole-hole gravity and mergers
    for (var h = 0u; h < u.hole_count; h++) {
        var hp = holes[2u * h];
        if (hp.z <= 0.0) { continue; }
        var a = s_hole_acc[h];
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
                holes[2u * k] = DEAD;
                holes[2u * k + 1u] = vec4f(0.0);
                continue;
            }
            let inv = inverseSqrt(r2 + u.hole_soft2);
            a += d * (u.g * kp.z * inv * inv * inv);
        }
        s_hole_acc[h] = a;
        holes[2u * h] = hp;
    }

    for (var h = 0u; h < u.hole_count; h++) {
        var hp = holes[2u * h];
        if (hp.z <= 0.0) { continue; }
        var hv = holes[2u * h + 1u];
        let vel = hv.xy + s_hole_acc[h] * u.dt;
        hv = vec4f(vel, hv.z, hv.w * 0.97);
        hp = vec4f(hp.xy + vel * u.dt, hp.z, max(u.min_radius, u.radius_per_mass * hp.z));
        if (dot(hp.xy, hp.xy) > u.bound * u.bound) {
            hp = DEAD;
            hv = vec4f(0.0);
        }
        holes[2u * h] = hp;
        holes[2u * h + 1u] = hv;
    }
}
