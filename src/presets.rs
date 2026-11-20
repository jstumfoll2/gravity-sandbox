//! Starting scenes. Total particle mass of a full preset is 1; G (a slider) is G × total mass.
//!
//! Particles are generated in fixed-size chunks, each from its own seeded RNG, so a scene of
//! tens of millions builds on every CPU core and streams to the GPU without one big host copy,
//! and the result does not depend on the number of threads.

use crate::rng::Rng;
use std::f32::consts::TAU;

/// Presets fill this share of the buffer, leaving room to spawn more before recycling.
pub const FILL: f32 = 0.7;
/// Particles per independently seeded chunk.
pub const CHUNK: usize = 1 << 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Disk,
    Cloud,
    Galaxies,
    Rings,
    Empty,
}

impl Preset {
    pub const ALL: [Preset; 5] = [
        Preset::Disk,
        Preset::Cloud,
        Preset::Galaxies,
        Preset::Rings,
        Preset::Empty,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Preset::Disk => "Disk + hole",
            Preset::Cloud => "Collapsing cloud",
            Preset::Galaxies => "Galaxy collision",
            Preset::Rings => "Rings",
            Preset::Empty => "Empty",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Disk => "DISK",
            Preset::Cloud => "CLOUD",
            Preset::Galaxies => "GALAXIES",
            Preset::Rings => "RINGS",
            Preset::Empty => "EMPTY",
        }
    }

    pub fn from_name(s: &str) -> Option<Preset> {
        Preset::ALL.into_iter().find(|p| p.name().eq_ignore_ascii_case(s))
    }
}

/// One particle as the shaders see it: (x, y, vx, vy). Mass is the same for all of them.
pub type State = [f32; 4];

pub const DEAD: State = [1e6, 1e6, 0.0, 0.0];

/// A black hole, two vec4s: (x, y, mass, radius), (vx, vy, swallowed, flash).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Hole {
    pub pos: [f32; 4],
    pub vel: [f32; 4],
}

pub const DEAD_HOLE: Hole = Hole {
    pos: [1e6, 1e6, 0.0, 0.0],
    vel: [0.0; 4],
};

/// Every particle has this mass, so a preset that fills 70% of the buffer weighs 1.
pub fn particle_mass(capacity: usize) -> f32 {
    1.0 / (capacity as f32 * FILL)
}

pub fn hole(x: f32, y: f32, vx: f32, vy: f32, mass: f32, radius_per_mass: f32, min_radius: f32) -> Hole {
    Hole {
        pos: [x, y, mass, min_radius.max(radius_per_mass * mass)],
        vel: [vx, vy, 0.0, 0.0],
    }
}

/// How many slots a preset fills.
pub fn used(preset: Preset, capacity: usize) -> usize {
    if preset == Preset::Empty {
        0
    } else {
        (capacity as f32 * FILL) as usize
    }
}

pub fn holes(preset: Preset, rpm: f32, min_r: f32) -> Vec<Hole> {
    match preset {
        Preset::Disk => vec![hole(0.0, 0.0, 0.0, 0.0, 1.0, rpm, min_r)],
        Preset::Galaxies => vec![
            hole(-60.0, -25.0, 9.0, 3.0, 0.5, rpm, min_r),
            hole(60.0, 25.0, -9.0, -3.0, 0.5, rpm, min_r),
        ],
        Preset::Rings => vec![hole(0.0, 0.0, 0.0, 0.0, 1.5, rpm, min_r)],
        Preset::Cloud | Preset::Empty => vec![],
    }
}

/// Fills `out` with particles `first..first + out.len()` of a scene. `first` must be a
/// multiple of [`CHUNK`].
pub fn fill(preset: Preset, capacity: usize, seed: u64, g: f32, first: usize, out: &mut [State]) {
    let n = used(preset, capacity);
    for (c, part) in out.chunks_mut(CHUNK).enumerate() {
        let start = first + c * CHUNK;
        let chunk_id = (start / CHUNK) as u64;
        let mut rnd = Rng::new(seed ^ chunk_id.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for (k, p) in part.iter_mut().enumerate() {
            let i = start + k;
            *p = if i < n {
                particle(preset, i, n, &mut rnd, g)
            } else {
                DEAD
            };
        }
    }
}

/// Fills `out` using every CPU core.
pub fn fill_parallel(preset: Preset, capacity: usize, seed: u64, g: f32, first: usize, out: &mut [State]) {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(32);
    let per = out.len().div_ceil(threads).div_ceil(CHUNK).max(1) * CHUNK;
    std::thread::scope(|s| {
        for (t, part) in out.chunks_mut(per).enumerate() {
            s.spawn(move || fill(preset, capacity, seed, g, first + t * per, part));
        }
    });
}

fn particle(preset: Preset, i: usize, n: usize, rnd: &mut Rng, g: f32) -> State {
    match preset {
        Preset::Disk => disk(rnd, [0.0, 0.0], [0.0, 0.0], 12.0, 85.0, 1.0, g, 1.0, 1.0),
        Preset::Cloud => {
            let r = 70.0 * rnd.f32().sqrt();
            let a = rnd.f32() * TAU;
            let (s, c) = a.sin_cos();
            // slow solid-body spin: ~30% of what would hold it up
            let menc = (r / 70.0) * (r / 70.0);
            let vc = (g * menc / (r + 1.0)).sqrt() * 0.3;
            [r * c, r * s, -s * vc, c * vc]
        }
        Preset::Galaxies => {
            if i < n / 2 {
                disk(rnd, [-60.0, -25.0], [9.0, 3.0], 5.0, 40.0, 0.5, g, 1.0, 0.5)
            } else {
                disk(rnd, [60.0, 25.0], [-9.0, -3.0], 5.0, 40.0, 0.5, g, -1.0, 0.5)
            }
        }
        Preset::Rings => {
            let m_bh = 1.5;
            let rings = [25.0f32, 45.0, 70.0];
            let r = rings[i % rings.len()] + rnd.gaussian() * 1.5;
            let a = rnd.f32() * TAU;
            let (s, c) = a.sin_cos();
            let vc = (g * (m_bh + 0.5) / r).sqrt();
            [r * c, r * s, -s * vc, c * vc]
        }
        Preset::Empty => DEAD,
    }
}

/// Keplerian disk around a central mass, including the disk's own enclosed mass.
#[allow(clippy::too_many_arguments)]
fn disk(
    rnd: &mut Rng,
    c: [f32; 2],
    cv: [f32; 2],
    r_in: f32,
    r_out: f32,
    central: f32,
    g: f32,
    dir: f32,
    disk_mass: f32,
) -> State {
    let u = rnd.f32();
    // surface density ~ 1/r: radius uniform in [r_in, r_out]
    let r = r_in + (r_out - r_in) * u;
    let a = rnd.f32() * TAU;
    let (s, co) = a.sin_cos();
    let menc = disk_mass * u;
    let vc = (g * (central + menc) / r).sqrt() * dir;
    // a little velocity dispersion so it is not perfectly cold
    let jx = rnd.gaussian() * 0.03 * vc;
    let jy = rnd.gaussian() * 0.03 * vc;
    [c[0] + r * co, c[1] + r * s, cv[0] - s * vc + jx, cv[1] + co * vc + jy]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(p: Preset, cap: usize, seed: u64) -> Vec<State> {
        let mut v = vec![DEAD; cap];
        fill(p, cap, seed, 40000.0, 0, &mut v);
        v
    }

    #[test]
    fn presets_fill_and_weigh_one() {
        for p in Preset::ALL {
            let s = build(p, 4096, 7);
            let live = s.iter().filter(|q| q[0] < 1e5).count();
            let mass = live as f32 * particle_mass(4096);
            if p == Preset::Empty {
                assert_eq!(live, 0);
            } else {
                assert!((mass - 1.0).abs() < 1e-3, "{p:?} mass {mass}");
            }
            assert!(s.iter().all(|q| q.iter().all(|x| x.is_finite())));
        }
    }

    #[test]
    fn deterministic_and_thread_independent() {
        let cap = CHUNK * 5 + 1024;
        let a = build(Preset::Galaxies, cap, 42);
        let mut b = vec![DEAD; cap];
        fill_parallel(Preset::Galaxies, cap, 42, 40000.0, 0, &mut b);
        assert!(a == b);
        // and in pieces, as the GPU upload does it
        let mut c = vec![DEAD; cap];
        let (x, y) = c.split_at_mut(CHUNK * 2);
        fill(Preset::Galaxies, cap, 42, 40000.0, 0, x);
        fill(Preset::Galaxies, cap, 42, 40000.0, CHUNK * 2, y);
        assert!(a == c);
    }
}
