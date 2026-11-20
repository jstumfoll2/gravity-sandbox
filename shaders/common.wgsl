// Shared by every compute shader (prepended by sim.rs).
//
// Particle state is struct-of-arrays so it stays compact at tens of millions of particles:
//   state[i] = (x, y, vx, vy)                         16 bytes
//   aux[i]   = pack2x16float(heat, density)            4 bytes
// Every particle has the same mass (u.mass), so mass is not stored. A dead particle sits at
// DEAD_X, far outside any world radius.

const WG: u32 = 256u;
const MAX_TOUCH: u32 = 10u;
const DEAD_X: f32 = 1.0e6;
const DEAD: vec4f = vec4f(1.0e6, 1.0e6, 0.0, 0.0);

struct SimU {
    n: u32,              // particle capacity (multiple of WG)
    t: u32,              // hash table size (power of two); bucket t holds the dead
    m_grid: u32,         // PM grid is m_grid x m_grid nodes, FFTs are 2 m_grid long
    hole_count: u32,
    touch_count: u32,
    exact: u32,          // 1 = all-pairs gravity instead of the PM grid
    spawn_count: u32,
    _pad0: u32,
    dt: f32,
    g: f32,
    mass: f32,           // mass of one particle
    gm: f32,             // g * mass
    soft2: f32,
    h: f32,              // contact radius (already scaled for N)
    inv_cell: f32,       // 1 / hash cell size (cell size >= h)
    k: f32,
    visc: f32,
    drag: f32,
    hole_soft2: f32,
    bound: f32,
    heat_gain: f32,
    heat_decay: f32,
    grid_origin: f32,    // world coordinate of grid node 0's centre minus half a cell
    grid_inv: f32,       // 1 / grid cell size
    touch: array<vec4f, MAX_TOUCH>,   // (x, y, strength, radius); strength > 0 pulls
}

fn flat_group(wid: vec3u, nwg: vec3u) -> u32 {
    return wid.y * nwg.x + wid.x;
}

fn is_alive(p: vec2f) -> bool {
    return p.x < 1.0e5;
}

fn cell_of(p: vec2f) -> vec2i {
    return vec2i(floor(p * u.inv_cell));
}

fn bucket_of(c: vec2i) -> u32 {
    var h = (bitcast<u32>(c.x) * 0x8da6b343u) ^ (bitcast<u32>(c.y) * 0xd8163841u);
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    return h & (u.t - 1u);
}
