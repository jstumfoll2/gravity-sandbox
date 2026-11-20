// Drawing: particles and black holes as screen-space quads (WGSL has no point sprites),
// the trail fade, and the tone-mapped present.

struct RenderU {
    scale: vec2f,        // world -> clip
    center: vec2f,       // camera centre, world units
    viewport: vec2f,     // pixels
    point_px: f32,
    px_per_unit: f32,
    dens_glow: f32,
    heat_scale: f32,
    brightness: f32,
    keep: f32,           // trail fade: dst *= keep each frame
    exposure: f32,
    srgb_out: f32,       // 1 if the swapchain re-encodes to sRGB
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: RenderU;
@group(1) @binding(1) var trail_tex: texture_2d<f32>;
@group(1) @binding(2) var trail_samp: sampler;

fn corner(vi: u32) -> vec2f {
    var c = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
        vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
    );
    return c[vi];
}

fn offscreen() -> vec4f { return vec4f(4.0, 4.0, 4.0, 1.0); }

// ---------------------------------------------------------------- particles

struct PartOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec3f,
    @location(1) c: vec2f,
}

fn palette(t: f32) -> vec3f {
    // cold dust -> blue-white -> yellow -> orange-white star
    let c0 = vec3f(0.35, 0.45, 0.85);
    let c1 = vec3f(0.85, 0.90, 1.00);
    let c2 = vec3f(1.00, 0.80, 0.40);
    let c3 = vec3f(1.00, 0.45, 0.15);
    if (t < 0.33) { return mix(c0, c1, t / 0.33); }
    if (t < 0.66) { return mix(c1, c2, (t - 0.33) / 0.33); }
    return mix(c2, c3, clamp((t - 0.66) / 0.34, 0.0, 1.0));
}

@vertex
fn vs_particle(@builtin(vertex_index) vi: u32,
               @location(0) st: vec4f,      // x, y, vx, vy
               @location(1) aux: u32) -> PartOut {  // pack2x16float(heat, density)
    var o: PartOut;
    o.c = corner(vi);
    if (st.x > 1.0e5) {
        o.clip = offscreen();
        o.color = vec3f(0.0);
        return o;
    }
    let hd = unpack2x16float(aux);
    let dens = hd.y;
    let heat = hd.x * u.heat_scale;
    let t = clamp(0.15 * dens + heat, 0.0, 1.0);
    o.color = palette(t) * (1.0 + u.dens_glow * min(dens, 20.0));
    let size = max(1.0, u.point_px * (1.0 + 0.08 * min(dens, 20.0)));
    let cc = (st.xy - u.center) * u.scale;
    o.clip = vec4f(cc + o.c * size / u.viewport, 0.0, 1.0);
    return o;
}

@fragment
fn fs_particle(i: PartOut) -> @location(0) vec4f {
    let r2 = dot(i.c, i.c);
    if (r2 > 1.0) { discard; }
    let a = exp(-3.0 * r2);
    return vec4f(i.color * (a * u.brightness), 1.0);
}

// ---------------------------------------------------------------- black holes

struct HoleOut {
    @builtin(position) clip: vec4f,
    @location(0) c: vec2f,
    @location(1) flash: f32,
    @location(2) fed: f32,
}

@vertex
fn vs_hole(@builtin(vertex_index) vi: u32,
           @location(0) pos: vec4f,     // x, y, mass, radius
           @location(1) vel: vec4f) -> HoleOut {  // vx, vy, swallowed, flash
    var o: HoleOut;
    o.c = corner(vi);
    o.flash = vel.w;
    o.fed = vel.z;
    if (pos.z <= 0.0) {
        o.clip = offscreen();
        return o;
    }
    // sprite covers 3 event-horizon radii so the ring and glow fit
    let size = max(12.0, pos.w * u.px_per_unit * 6.0);
    let cc = (pos.xy - u.center) * u.scale;
    o.clip = vec4f(cc + o.c * size / u.viewport, 0.0, 1.0);
    return o;
}

@fragment
fn fs_hole(i: HoleOut) -> @location(0) vec4f {
    let r = length(i.c) * 3.0;          // in event-horizon radii
    if (r > 3.0) { discard; }
    let ring = vec3f(1.0, 0.62, 0.25) * (0.6 + min(i.fed * 2.0, 1.5)) + vec3f(0.6, 0.7, 1.0) * i.flash;
    let ring_a = exp(-((r - 1.15) * 5.0) * ((r - 1.15) * 5.0));
    let glow = exp(-(r - 1.0) * 2.2) * 0.25;
    var out: vec4f;
    if (r < 1.0) {
        out = vec4f(0.0, 0.0, 0.0, 1.0);
    } else {
        out = vec4f(encode(ring * (ring_a + glow)), clamp(ring_a + glow, 0.0, 1.0));
    }
    return out;
}

// ---------------------------------------------------------------- full-screen passes

struct FsOut {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FsOut {
    // one big triangle, no vertex buffer
    let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    var o: FsOut;
    o.clip = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2f(p.x, 1.0 - p.y);
    return o;
}

@fragment
fn fs_fade() -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 0.0, u.keep);
}

// The Android build wrote straight to a non-sRGB framebuffer. If the swapchain here
// re-encodes to sRGB, pre-decode so the final pixels come out the same.
fn encode(c: vec3f) -> vec3f {
    if (u.srgb_out > 0.5) { return pow(max(c, vec3f(0.0)), vec3f(2.2)); }
    return c;
}

@fragment
fn fs_present(i: FsOut) -> @location(0) vec4f {
    var c = textureSample(trail_tex, trail_samp, i.uv).rgb * u.exposure;
    c = vec3f(1.0) - exp(-c);           // soft tone map, so dense cores don't clip flat
    c = pow(c, vec3f(1.0 / 1.1));
    return vec4f(encode(c + vec3f(0.004, 0.005, 0.012)), 1.0);
}
