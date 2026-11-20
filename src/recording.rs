//! Record / replay of input, and where the app keeps its files.
//!
//! A recording is the starting preset, seed and every slider value, then the input stream
//! in world units, each event stamped with the frame it applies to. The simulation is
//! fixed-step and spawns from a seeded RNG, so replay on the same GPU reproduces the run
//! (swallowed mass uses integer atomics, so thread order doesn't matter).

use crate::presets::Preset;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// What the left mouse button does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Spawn,
    Pull,
    Push,
    Hole,
}

impl Mode {
    pub const ALL: [Mode; 4] = [Mode::Spawn, Mode::Pull, Mode::Push, Mode::Hole];

    pub fn name(self) -> &'static str {
        match self {
            Mode::Spawn => "SPAWN",
            Mode::Pull => "PULL",
            Mode::Push => "PUSH",
            Mode::Hole => "HOLE",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Mode::Spawn => "Spawn",
            Mode::Pull => "Pull",
            Mode::Push => "Push",
            Mode::Hole => "Hole",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Mode::Spawn => "Drag to spray matter. Flick to throw it.",
            Mode::Pull => "Hold to pull matter toward the cursor",
            Mode::Push => "Hold to push matter away",
            Mode::Hole => "Click to drop a black hole, flick to throw one",
        }
    }

    fn from_name(s: &str) -> Option<Mode> {
        Mode::ALL.into_iter().find(|m| m.name() == s)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Down = 0,
    Move = 1,
    Up = 2,
    Param = 3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InputEvent {
    pub frame: u64,
    /// Event time in ms since the scene started (for pointer velocity).
    pub t_ms: i64,
    pub kind: Kind,
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub mode: Mode,
    /// Param key for `Kind::Param` (value in `x`).
    pub key: String,
}

#[derive(Clone, Debug)]
pub struct Recording {
    pub preset: Preset,
    pub seed: u64,
    pub params: Vec<(String, f32)>,
    pub events: Vec<InputEvent>,
    pub frames: u64,
}

impl Recording {
    pub fn to_text(&self) -> String {
        let mut s = String::from("gravity-sandbox-recording 1\n");
        s += &format!(
            "preset {}\nseed {}\nframes {}\n",
            self.preset.name(),
            self.seed,
            self.frames
        );
        for (k, v) in &self.params {
            s += &format!("param {k} {v}\n");
        }
        for e in &self.events {
            let key = if e.key.is_empty() { "-" } else { &e.key };
            s += &format!(
                "e {} {} {} {} {} {} {} {}\n",
                e.frame,
                e.t_ms,
                e.kind as i32,
                e.id,
                e.x,
                e.y,
                e.mode.name(),
                key
            );
        }
        s
    }

    pub fn parse(text: &str) -> Option<Recording> {
        let mut r = Recording {
            preset: Preset::Disk,
            seed: 1,
            params: vec![],
            events: vec![],
            frames: 0,
        };
        let mut lines = text.lines();
        if !lines.next()?.starts_with("gravity-sandbox-recording") {
            return None;
        }
        for line in lines {
            let t: Vec<&str> = line.split(' ').collect();
            match t[0] {
                "preset" => r.preset = Preset::from_name(t.get(1)?).unwrap_or(Preset::Disk),
                "seed" => r.seed = t.get(1)?.parse().ok()?,
                "frames" => r.frames = t.get(1)?.parse().ok()?,
                "param" => r.params.push((t.get(1)?.to_string(), t.get(2)?.parse().ok()?)),
                "e" if t.len() >= 9 => {
                    let kind = match t[3] {
                        "0" => Kind::Down,
                        "1" => Kind::Move,
                        "2" => Kind::Up,
                        "3" => Kind::Param,
                        _ => continue,
                    };
                    r.events.push(InputEvent {
                        frame: t[1].parse().ok()?,
                        t_ms: t[2].parse().ok()?,
                        kind,
                        id: t[4].parse().ok()?,
                        x: t[5].parse().ok()?,
                        y: t[6].parse().ok()?,
                        mode: Mode::from_name(t[7])?,
                        key: if t[8] == "-" { String::new() } else { t[8].to_string() },
                    });
                }
                _ => {}
            }
        }
        Some(r)
    }
}

/// Per-user data folder: settings and recordings.
pub fn data_dir() -> PathBuf {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        env("APPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Application Support"))
    } else {
        env("XDG_DATA_HOME").or_else(|| env("HOME").map(|h| h.join(".local/share")))
    };
    base.unwrap_or_else(|| PathBuf::from(".")).join("gravity-sandbox")
}

pub fn recordings_dir() -> PathBuf {
    data_dir().join("recordings")
}

pub fn write_file(path: &PathBuf, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::File::create(path)?.write_all(text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let r = Recording {
            preset: Preset::Galaxies,
            seed: 123456789,
            params: vec![("g".into(), 40000.0), ("dt".into(), 0.008333)],
            events: vec![
                InputEvent {
                    frame: 3,
                    t_ms: 50,
                    kind: Kind::Down,
                    id: 0,
                    x: 1.5,
                    y: -2.25,
                    mode: Mode::Hole,
                    key: String::new(),
                },
                InputEvent {
                    frame: 9,
                    t_ms: 0,
                    kind: Kind::Param,
                    id: 0,
                    x: 3.0,
                    y: 0.0,
                    mode: Mode::Spawn,
                    key: "k".into(),
                },
            ],
            frames: 600,
        };
        let back = Recording::parse(&r.to_text()).unwrap();
        assert_eq!(back.preset, r.preset);
        assert_eq!(back.seed, r.seed);
        assert_eq!(back.frames, 600);
        assert_eq!(back.params, r.params);
        assert_eq!(back.events, r.events);
    }
}
