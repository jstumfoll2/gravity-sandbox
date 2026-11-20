//! Every tunable constant, shown as a slider. Ported from the Android Params.kt.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum P {
    Count,
    G,
    Soft,
    Dt,
    Substeps,
    Drag,
    Bound,
    Grid,
    ExactBelow,
    H,
    K,
    Visc,
    HeatGain,
    HeatDecay,
    HoleMass,
    HoleRadius,
    HoleMinRadius,
    SpawnRate,
    SpawnSpread,
    Fling,
    Spin,
    Pull,
    PullRadius,
    PointPx,
    Brightness,
    Exposure,
    Trails,
    DensGlow,
    GpuTiming,
}

pub struct Param {
    pub key: &'static str,
    pub label: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub log: bool,
    pub integer: bool,
    pub step: f32,
    /// Only takes effect on the next reset.
    pub restart: bool,
    pub section: &'static str,
    pub help: &'static str,
    pub value: f32,
}

impl Param {
    pub fn clamp(&self, v: f32) -> f32 {
        let mut v = v;
        if self.step > 0.0 {
            v = (v / self.step).round() * self.step;
        }
        if self.integer {
            v = v.round();
        }
        v.clamp(self.min, self.max)
    }
}

pub struct Params {
    pub all: Vec<Param>,
}

#[allow(clippy::too_many_arguments)]
fn p(key: &'static str, label: &'static str, min: f32, max: f32, default: f32, section: &'static str) -> Param {
    Param {
        key,
        label,
        min,
        max,
        default,
        section,
        log: false,
        integer: false,
        step: 0.0,
        restart: false,
        help: "",
        value: default,
    }
}

trait With {
    fn log(self) -> Self;
    fn int(self) -> Self;
    fn step(self, s: f32) -> Self;
    fn restart(self) -> Self;
    fn help(self, h: &'static str) -> Self;
}

impl With for Param {
    fn log(mut self) -> Self {
        self.log = true;
        self
    }
    fn int(mut self) -> Self {
        self.integer = true;
        self
    }
    fn step(mut self, s: f32) -> Self {
        self.step = s;
        self
    }
    fn restart(mut self) -> Self {
        self.restart = true;
        self
    }
    fn help(mut self, h: &'static str) -> Self {
        self.help = h;
        self
    }
}

/// Upper limit of the Particles slider; sized for an 8 GB GPU (about 56 bytes per particle).
pub const MAX_COUNT: f32 = 100_000_000.0;

pub const SECTIONS: [&str; 5] = ["Simulation", "Matter", "Black holes", "Mouse", "Look"];

impl Params {
    pub fn new() -> Self {
        const SIM: &str = "Simulation";
        const MAT: &str = "Matter";
        const BH: &str = "Black holes";
        const IN: &str = "Mouse";
        const LOOK: &str = "Look";
        // Order must match enum P.
        let all = vec![
            p("count", "Particles", 1024.0, MAX_COUNT, 1_048_576.0, SIM)
                .log()
                .int()
                .step(1024.0)
                .restart()
                .help("Cost grows about linearly with N (grid gravity). Applies on reset; capped by GPU memory."),
            p("g", "Gravity G", 0.0, 150000.0, 40000.0, SIM).help("G × total mass of a preset"),
            p("soft", "Softening", 0.1, 6.0, 1.0, SIM)
                .log()
                .help("Gravity is smoothed inside this distance"),
            p("dt", "Time step", 0.001, 0.025, 1.0 / 120.0, SIM).log(),
            p("substeps", "Steps per frame", 1.0, 8.0, 2.0, SIM).int(),
            p("drag", "Drag", 0.0, 2.0, 0.0, SIM),
            p("bound", "World radius", 100.0, 2000.0, 600.0, SIM)
                .log()
                .help("Particles that fly further are removed. The gravity grid spans this."),
            p("grid", "Gravity grid", 256.0, 2048.0, 1024.0, SIM)
                .int()
                .restart()
                .help("Particle-mesh nodes per side (rounded to a power of two). Finer = sharper gravity, slower FFT."),
            p("exactBelow", "Exact gravity up to", 0.0, 131072.0, 32768.0, SIM)
                .log()
                .int()
                .step(1024.0)
                .restart()
                .help("At or below this many particles, gravity is the exact all-pairs sum instead of the grid"),
            p("h", "Contact radius", 0.2, 5.0, 1.5, MAT)
                .log()
                .help("At 16k particles. Shrinks as 1/sqrt(N) so each particle keeps about as many neighbours."),
            p("k", "Pressure", 0.0, 200.0, 30.0, MAT).help("0 = pure collisionless gravity"),
            p("visc", "Viscosity", 0.0, 12.0, 2.0, MAT).help("Sticky collisions; lets clumps settle"),
            p("heatGain", "Heating", 0.0, 0.2, 0.02, MAT).help("Collisions heat matter (redder glow)"),
            p("heatDecay", "Cooling", 0.0, 5.0, 0.6, MAT),
            p("holeMass", "New hole mass", 0.02, 5.0, 0.4, BH)
                .log()
                .help("In units of a preset's total mass"),
            p("holeRadius", "Horizon per mass", 0.5, 30.0, 6.0, BH)
                .log()
                .help("Event horizon radius = this × mass"),
            p("holeMinRadius", "Min horizon", 0.3, 10.0, 1.5, BH).log(),
            p("spawnRate", "Spawn per frame", 1.0, 400.0, 40.0, IN).log().int(),
            p("spawnSpread", "Spawn spread", 0.0, 15.0, 2.5, IN),
            p("fling", "Fling strength", 0.0, 3.0, 1.0, IN).help("Spawned velocity = mouse velocity × this"),
            p("spin", "Spawn orbit", 0.0, 1.5, 0.0, IN)
                .help("Adds circular-orbit speed around the centre to new particles"),
            p("pull", "Pull / push force", 0.0, 6000.0, 1200.0, IN),
            p("pullRadius", "Pull radius", 3.0, 120.0, 30.0, IN).log(),
            p("pointPx", "Particle size", 1.0, 14.0, 3.0, LOOK)
                .help("At 720 px window height and 16k particles; shrinks gently with more particles"),
            p("brightness", "Brightness", 0.02, 2.0, 0.35, LOOK).log(),
            p("exposure", "Exposure", 0.2, 6.0, 1.5, LOOK).log(),
            p("trails", "Trails", 0.0, 0.97, 0.55, LOOK),
            p("densGlow", "Clump glow", 0.0, 0.5, 0.08, LOOK),
            p("gpuTiming", "Measure GPU time", 0.0, 1.0, 0.0, LOOK)
                .int()
                .help("1 = wait for the GPU each frame to time the physics (costs a little fps)"),
        ];
        Params { all }
    }

    pub fn get(&self, p: P) -> f32 {
        self.all[p as usize].value
    }

    pub fn set(&mut self, p: P, v: f32) {
        let q = &mut self.all[p as usize];
        q.value = q.clamp(v);
    }

    pub fn by_key(&self, key: &str) -> Option<usize> {
        self.all.iter().position(|q| q.key == key)
    }

    pub fn reset_defaults(&mut self) {
        for q in &mut self.all {
            q.value = q.default;
        }
    }

    pub fn snapshot(&self) -> Vec<(String, f32)> {
        self.all.iter().map(|q| (q.key.to_string(), q.value)).collect()
    }

    pub fn apply(&mut self, values: &[(String, f32)]) {
        for (k, v) in values {
            if let Some(i) = self.by_key(k) {
                let q = &mut self.all[i];
                q.value = q.clamp(*v);
            }
        }
    }

    /// `key value` lines.
    pub fn to_text(&self) -> String {
        self.all.iter().map(|q| format!("{} {}\n", q.key, q.value)).collect()
    }

    pub fn load_text(&mut self, text: &str) {
        let vals: Vec<(String, f32)> = text
            .lines()
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                Some((it.next()?.to_string(), it.next()?.parse().ok()?))
            })
            .collect();
        self.apply(&vals);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_matches_enum() {
        let p = Params::new();
        assert_eq!(p.all[P::GpuTiming as usize].key, "gpuTiming");
        assert_eq!(p.all[P::PullRadius as usize].key, "pullRadius");
        assert_eq!(p.all[P::HoleMass as usize].key, "holeMass");
        assert_eq!(p.all[P::ExactBelow as usize].key, "exactBelow");
        assert_eq!(p.all[P::H as usize].key, "h");
    }

    #[test]
    fn text_roundtrip() {
        let mut a = Params::new();
        a.set(P::G, 1234.0);
        a.set(P::Count, 5000.0);
        a.set(P::Grid, 512.0);
        let mut b = Params::new();
        b.load_text(&a.to_text());
        assert_eq!(b.get(P::G), 1234.0);
        assert_eq!(b.get(P::Count), 5120.0);
        assert_eq!(b.get(P::Grid), 512.0);
    }
}
