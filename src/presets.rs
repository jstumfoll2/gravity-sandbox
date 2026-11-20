//! Starting scenes. Total particle mass of a full preset is 1; G (a slider) is G × total mass.

use crate::rng::Rng;
use std::f32::consts::TAU;

/// Presets fill this share of the buffer, leaving room to spawn more before recycling.
pub const FILL: f32 = 0.7;

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

/// Two vec4s, exactly as the shaders see them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub pos: [f32; 4], // x, y, mass, alive
    pub vel: [f32; 4], // vx, vy, heat, density
}

pub const DEAD: Particle = Particle {
    pos: [1e6, 1e6, 0.0, 0.0],
    vel: [0.0; 4],
};

/// Same layout as a particle: (x, y, mass, radius), (vx, vy, swallowed, flash).
pub type Hole = Particle;

pub struct Scene {
    pub particles: Vec<Particle>,
    pub used: usize,
    pub holes: Vec<Hole>,
}

/// Every particle has this mass, so a preset that fills 70% of the buffer weighs 1.
pub fn particle_mass(capacity: usize) -> f32 {
    1.0 / (capacity as f32 * FILL)
}

pub fn hole(x: f32, y: f32, vx: f32, vy: f32, mass: f32, radius_per_mass: f32, min_radius: f32) -> Hole {
    Particle {
        pos: [x, y, mass, min_radius.max(radius_per_mass * mass)],
        vel: [vx, vy, 0.0, 0.0],
    }
}

fn live(x: f32, y: f32, vx: f32, vy: f32, m: f32) -> Particle {
    Particle {
        pos: [x, y, m, 1.0],
        vel: [vx, vy, 0.0, 0.0],
    }
}

pub fn build(preset: Preset, capacity: usize, seed: u64, g: f32, rpm: f32, min_r: f32) -> Scene {
    let mut rnd = Rng::new(seed);
    let n = if preset == Preset::Empty {
        0
    } else {
        (capacity as f32 * FILL) as usize
    };
    let mut out = vec![DEAD; capacity];
    let m = particle_mass(capacity);
    let mut holes = Vec::new();
    match preset {
        Preset::Disk => {
            let m_bh = 1.0;
            holes.push(hole(0.0, 0.0, 0.0, 0.0, m_bh, rpm, min_r));
            disk(
                &mut out[0..n],
                &mut rnd,
                [0.0, 0.0],
                [0.0, 0.0],
                12.0,
                85.0,
                m_bh,
                m,
                g,
                1.0,
                1.0,
            );
        }
        Preset::Cloud => {
            for p in &mut out[0..n] {
                let r = 70.0 * rnd.f32().sqrt();
                let a = rnd.f32() * TAU;
                let (s, c) = a.sin_cos();
                // slow solid-body spin: ~30% of what would hold it up
                let menc = (r / 70.0) * (r / 70.0);
                let vc = (g * menc / (r + 1.0)).sqrt() * 0.3;
                *p = live(r * c, r * s, -s * vc, c * vc, m);
            }
        }
        Preset::Galaxies => {
            let half = n / 2;
            let m_bh = 0.5;
            holes.push(hole(-60.0, -25.0, 9.0, 3.0, m_bh, rpm, min_r));
            holes.push(hole(60.0, 25.0, -9.0, -3.0, m_bh, rpm, min_r));
            let (a, b) = out[0..n].split_at_mut(half);
            disk(a, &mut rnd, [-60.0, -25.0], [9.0, 3.0], 5.0, 40.0, m_bh, m, g, 1.0, 0.5);
            disk(
                b,
                &mut rnd,
                [60.0, 25.0],
                [-9.0, -3.0],
                5.0,
                40.0,
                m_bh,
                m,
                g,
                -1.0,
                0.5,
            );
        }
        Preset::Rings => {
            let m_bh = 1.5;
            holes.push(hole(0.0, 0.0, 0.0, 0.0, m_bh, rpm, min_r));
            let rings = [25.0f32, 45.0, 70.0];
            for (i, p) in out[0..n].iter_mut().enumerate() {
                let r = rings[i % rings.len()] + rnd.gaussian() * 1.5;
                let a = rnd.f32() * TAU;
                let (s, c) = a.sin_cos();
                let vc = (g * (m_bh + 0.5) / r).sqrt();
                *p = live(r * c, r * s, -s * vc, c * vc, m);
            }
        }
        Preset::Empty => {}
    }
    Scene {
        particles: out,
        used: n,
        holes,
    }
}

/// Keplerian disk around a central mass, including the disk's own enclosed mass.
#[allow(clippy::too_many_arguments)]
fn disk(
    out: &mut [Particle],
    rnd: &mut Rng,
    c: [f32; 2],
    cv: [f32; 2],
    r_in: f32,
    r_out: f32,
    central: f32,
    m: f32,
    g: f32,
    dir: f32,
    disk_mass: f32,
) {
    for p in out {
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
        *p = live(
            c[0] + r * co,
            c[1] + r * s,
            cv[0] - s * vc + jx,
            cv[1] + co * vc + jy,
            m,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_fill_and_weigh_one() {
        for p in Preset::ALL {
            let s = build(p, 4096, 7, 40000.0, 6.0, 1.5);
            assert_eq!(s.particles.len(), 4096);
            let mass: f32 = s.particles.iter().filter(|q| q.pos[3] > 0.5).map(|q| q.pos[2]).sum();
            if p == Preset::Empty {
                assert_eq!(mass, 0.0);
            } else {
                assert!((mass - 1.0).abs() < 1e-3, "{p:?} mass {mass}");
            }
            assert!(
                s.particles
                    .iter()
                    .all(|q| q.pos.iter().chain(&q.vel).all(|x| x.is_finite()))
            );
        }
    }

    #[test]
    fn deterministic() {
        let a = build(Preset::Galaxies, 1024, 42, 40000.0, 6.0, 1.5);
        let b = build(Preset::Galaxies, 1024, 42, 40000.0, 6.0, 1.5);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&a.particles),
            bytemuck::cast_slice::<_, u8>(&b.particles)
        );
    }
}
