// One simulation step for every particle (port of the Android step.comp).
// Gravity is an exact all-pairs sum, tiled through workgroup memory (the classic
// GPU Gems 3 ch.31 layout). Close neighbours (r < h) also get a soft pressure
// push and viscosity, which is what lets dust settle into solid clumps
// ("planets") instead of passing through itself.

const TILE: u32 = 256u;
const MAX_TOUCH: u32 = 10u;
const MASS_FP: f32 = 1.0e7;
const MOM_FP: f32 = 1.0e5;
const DEAD_POS: vec4f = vec4f(1.0e6, 1.0e6, 0.0, 0.0);

// pos = (x, y, mass, alive)   vel = (vx, vy, heat, density)
struct Particle { pos: vec4f, vel: vec4f }

struct StepU {
    count: u32,          // multiple of TILE
    hole_count: u32,
    touch_count: u32,
    _pad0: u32,
    dt: f32,
    g: f32,
    soft2: f32,          // gravity softening squared
    h: f32,              // pressure / viscosity radius
    k: f32,              // pressure stiffness
    visc: f32,
    drag: f32,           // global velocity damping
    hole_soft2: f32,
    bound: f32,          // particles beyond this radius are removed
    heat_gain: f32,
    heat_decay: f32,
    _pad1: f32,
    touch: array<vec4f, MAX_TOUCH>,   // (x, y, strength, radius); strength > 0 pulls
}

@group(0) @binding(0) var<storage, read> src: array<Particle>;
@group(0) @binding(1) var<storage, read_write> dst: array<Particle>;
// hole i: holes[2i] = (x, y, mass, radius), holes[2i+1] = (vx, vy, swallowed, flash)
@group(0) @binding(2) var<storage, read> holes: array<vec4f>;
// per hole: swallowed mass, x momentum, y momentum (fixed point, integer atomics)
@group(0) @binding(3) var<storage, read_write> absorb: array<atomic<i32>>;
@group(0) @binding(4) var<uniform> u: StepU;

var<workgroup> s_pos: array<vec4f, TILE>;
var<workgroup> s_vel: array<vec2f, TILE>;

fn bad(x: f32) -> bool {
    // NaN or infinity, tested on the bits so fast-math can't fold it away
    return (bitcast<u32>(x) & 0x7fffffffu) >= 0x7f800000u;
}

fn kill(i: u32) {
    dst[i] = Particle(DEAD_POS, vec4f(0.0));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let i = gid.x;
    let lid = lid3.x;
    let me = src[i];
    var p = me.pos.xy;
    var v = me.vel.xy;
    let m = me.pos.z;
    let alive = me.pos.w > 0.5;

    var grav = vec2f(0.0);
    var acc = vec2f(0.0);
    var dens = 0.0;
    var heat = 0.0;
    let h2 = u.h * u.h;
    let inv_h = 1.0 / u.h;

    for (var tile = 0u; tile < u.count; tile += TILE) {
        let q = src[tile + lid];
        s_pos[lid] = q.pos;
        s_vel[lid] = q.vel.xy;
        workgroupBarrier();
        for (var j = 0u; j < TILE; j++) {
            let qp = s_pos[j];
            let d = qp.xy - p;
            let r2 = dot(d, d);
            let inv = inverseSqrt(r2 + u.soft2);
            grav += d * (qp.z * inv * inv * inv);
            if (r2 < h2 && r2 > 1.0e-10) {
                let r = sqrt(r2);
                let w = 1.0 - r * inv_h;
                let dv = s_vel[j] - v;
                acc -= d * (u.k * w / r);
                acc += dv * (u.visc * w);
                dens += w * w;
                heat += w * dot(dv, dv);
            }
        }
        workgroupBarrier();
    }

    if (!alive) { kill(i); return; }

    acc += grav * u.g;

    // black holes
    var swallowed_by = -1;
    for (var h = 0u; h < u.hole_count; h++) {
        let hp = holes[2u * h];
        if (hp.z <= 0.0) { continue; }
        let d = hp.xy - p;
        let r2 = dot(d, d);
        if (r2 < hp.w * hp.w) { swallowed_by = i32(h); break; }
        let inv = inverseSqrt(r2 + u.hole_soft2);
        acc += d * (u.g * hp.z * inv * inv * inv);
    }
    if (swallowed_by >= 0) {
        let b = 3u * u32(swallowed_by);
        atomicAdd(&absorb[b + 0u], i32(m * MASS_FP + 0.5));
        atomicAdd(&absorb[b + 1u], i32(round(m * v.x * MOM_FP)));
        atomicAdd(&absorb[b + 2u], i32(round(m * v.y * MOM_FP)));
        kill(i);
        return;
    }

    // pointer forces (pull / push)
    for (var t = 0u; t < u.touch_count; t++) {
        let tc = u.touch[t];
        let d = tc.xy - p;
        let r = length(d) + 1.0e-3;
        let f = tc.z * (1.0 - smoothstep(0.0, tc.w, r));
        acc += d / r * f;
    }

    v += acc * u.dt;
    v *= max(0.0, 1.0 - u.drag * u.dt);
    p += v * u.dt;

    let temp = me.vel.z * max(0.0, 1.0 - u.heat_decay * u.dt) + heat * u.heat_gain * u.dt;

    if (dot(p, p) > u.bound * u.bound || bad(p.x) || bad(p.y)) { kill(i); return; }
    dst[i] = Particle(vec4f(p, m, 1.0), vec4f(v, min(temp, 1.0e4), dens));
}
