// Gravity Sandbox: fling matter around and watch gravity make disks, planets and stars.
// Copyright (C) 2026 Jason Stumfoll. GPL-3.0-or-later; see LICENSE.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod gpu;
mod headless;
mod params;
mod physics;
mod presets;
mod recording;
mod rng;
mod sim;

use presets::Preset;

const USAGE: &str = "\
Gravity Sandbox

  gravity-sandbox                     open the window
  gravity-sandbox --replay FILE       open the window and replay a recording
  gravity-sandbox --preview [opts]    run headless and write a PNG
      --preset disk|cloud|galaxies|rings   (default galaxies)
      --n N          particles (default 4096)
      --steps S      physics steps (default 300)
      --size WxH     image size (default 960x600)
      --seed S       (default 1)
      --out FILE     (default preview.png)
  gravity-sandbox --bench [N ...]     time one physics step at several particle counts
      --preset P     scene (default galaxies)
      --warm S       let the scene evolve S steps first (clumps form; default 0)
  gravity-sandbox --gravcheck [N]     compare grid gravity with the exact sum (fails above 5%)
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| {
        a == "--preview" || a == "--bench" || a == "--gravcheck" || a == "--selftest" || a == "--help" || a == "-h"
    }) {
        attach_console();
    }
    if let Err(e) = real_main(&args) {
        eprintln!("gravity-sandbox: {e}");
        std::process::exit(1);
    }
}

fn real_main(args: &[String]) -> Result<(), String> {
    let get = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{USAGE}");
        return Ok(());
    }
    if args.iter().any(|a| a == "--selftest") {
        return headless::selftest();
    }
    if args.iter().any(|a| a == "--gravcheck") {
        let n = args.iter().find_map(|a| a.parse().ok()).unwrap_or(16384);
        let rms = headless::gravcheck(n, 1024)?;
        if rms > 0.05 {
            return Err(format!("grid gravity is off by {:.1}% RMS", rms * 100.0));
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--bench") {
        let warm: usize = get("--warm").map_or(Ok(0), |v| v.parse().map_err(|_| format!("bad --warm {v}")))?;
        let preset_name = get("--preset").unwrap_or_else(|| "galaxies".into());
        let preset = Preset::from_name(&preset_name).ok_or(format!("unknown preset {preset_name}"))?;
        let sizes: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(i, _)| *i == 0 || !matches!(args[i - 1].as_str(), "--warm" | "--preset"))
            .filter_map(|(_, a)| a.parse().ok())
            .collect();
        let sizes = if sizes.is_empty() {
            vec![16384, 65536, 262_144, 1_048_576, 4_194_304, 16_777_216]
        } else {
            sizes
        };
        return headless::bench(&sizes, preset, warm);
    }
    if args.iter().any(|a| a == "--preview") {
        let num = |name: &str, d: usize| -> Result<usize, String> {
            get(name).map_or(Ok(d), |v| v.parse().map_err(|_| format!("bad {name} {v}")))
        };
        let (width, height) = match get("--size") {
            Some(s) => {
                let (w, h) = s.split_once('x').ok_or("--size wants WxH")?;
                (
                    w.parse().map_err(|_| "bad width")?,
                    h.parse().map_err(|_| "bad height")?,
                )
            }
            None => (960, 600),
        };
        let preset_name = get("--preset").unwrap_or_else(|| "galaxies".into());
        let o = headless::Opts {
            preset: Preset::from_name(&preset_name).ok_or(format!("unknown preset {preset_name}"))?,
            n: num("--n", 4096)?,
            steps: num("--steps", 300)?,
            width,
            height,
            out: get("--out").unwrap_or_else(|| "preview.png".into()),
            seed: num("--seed", 1)? as u64,
        };
        return headless::preview(&o);
    }
    let replay = match get("--replay") {
        Some(f) => {
            let text = std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?;
            Some(recording::Recording::parse(&text).ok_or(format!("{f} is not a recording"))?)
        }
        None => None,
    };
    app::run(replay)
}

/// Release builds on Windows have no console; reattach to the terminal for CLI modes.
#[cfg(windows)]
fn attach_console() {
    unsafe extern "system" {
        fn AttachConsole(pid: u32) -> i32;
    }
    // SAFETY: plain Win32 call with ATTACH_PARENT_PROCESS.
    unsafe {
        AttachConsole(u32::MAX);
    }
}

#[cfg(not(windows))]
fn attach_console() {}
