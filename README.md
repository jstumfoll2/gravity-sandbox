# Gravity Sandbox

Fling millions of particles around with the mouse and watch gravity pull them into
disks, planets and stars. Drop black holes and watch them feed.

A hobby N-body simulation that runs on your GPU: Rust, [wgpu](https://wgpu.rs) compute shaders
(Vulkan, Metal or DX12), winit and egui. Windows, Linux and macOS. No network code, no ads,
no tracking.

## Download

Every push to `main` refreshes the **Latest PC build** release
([pc-latest](https://github.com/jstumfoll2/gravity-sandbox/releases/tag/pc-latest)):

| OS | File |
|---|---|
| Windows 10/11 x64 | [gravity-sandbox-windows-x64.zip](https://github.com/jstumfoll2/gravity-sandbox/releases/download/pc-latest/gravity-sandbox-windows-x64.zip) |
| Linux x64 | [gravity-sandbox-linux-x64.tar.gz](https://github.com/jstumfoll2/gravity-sandbox/releases/download/pc-latest/gravity-sandbox-linux-x64.tar.gz) |
| macOS (Apple silicon) | [gravity-sandbox-macos-arm64.tar.gz](https://github.com/jstumfoll2/gravity-sandbox/releases/download/pc-latest/gravity-sandbox-macos-arm64.tar.gz) |

Unpack and run `gravity-sandbox`. The builds are unsigned: on Windows click *More info → Run anyway*;
on macOS right-click → *Open* the first time. Each Actions run also keeps them as artifacts.

Or build it yourself: install Rust, then `cargo run --release`.

## Controls

| Input | Does |
|---|---|
| Left mouse | Whatever the toolbar mode is (keys **1–4**): **Spawn** drag to spray matter, flick to throw it · **Pull** / **Push** hold to pull matter in or blow it away · **Hole** click to drop a black hole, flick to throw it |
| Right mouse | Pull (with **Shift**: push), whatever the mode |
| Middle drag / wheel | Pan / zoom toward the cursor (**C** recentres) |
| **Space** | Pause |
| **R** | Restart the scene with a new seed |
| **T** | Tune panel: every physics and look constant as a slider (right-click a slider to reset it), plus a GPU benchmark |
| Scene | Disk + hole, collapsing cloud, galaxy collision, rings, empty |
| Rec / Replay | Record your input from a fresh scene, then replay it exactly |
| **F11** | Fullscreen |

Sliders are saved between runs. Settings and recordings live in
`%APPDATA%\gravity-sandbox` (Windows), `~/Library/Application Support/gravity-sandbox` (macOS)
or `~/.local/share/gravity-sandbox` (Linux).

## How the physics works

* **State:** struct-of-arrays, 20 bytes per particle: `(x, y, vx, vy)` plus `(heat, density)`
  packed as two f16. Every particle has the same mass, so it is not stored. Counting the cell-sorted
  copy, sort keys and the hash table, a particle costs about 56 bytes of GPU memory: 100M (the
  slider's limit, sized for an 8 GB card) is about 5.6 GB; 1B would need about 60 GB.
* **Sort:** every step the particles are counting-sorted by hashed grid cell (hash, multi-level
  prefix scan, scatter, a per-cell fix-up that makes the order deterministic, gather). After that a
  cell's particles are contiguous in memory, which is what makes the passes below O(N) and
  cache-friendly.
* **Gravity:** particle-mesh. Mass is deposited cloud-in-cell onto an M×M grid (default 1024²,
  spanning the world radius) with 64-bit integer atomics, convolved with the softened
  `a = G Σ m d / (r² + ε²)^{3/2}` kernel by FFT on a zero-padded 2M×2M grid (isolated, not periodic),
  and interpolated back. Both force components come out of one complex inverse FFT; each FFT row
  lives in workgroup memory, so a whole 2-D convolution is three dispatches. The cost is O(N) plus a
  fixed FFT. At or below *Exact gravity up to* (default 32k) particles it is the exact all-pairs
  sum instead (tiled through workgroup memory). `--gravcheck` measures the grid against the exact
  sum: about 0.2% RMS acceleration error on the 16k cloud.
* **Matter:** within the contact radius `h`, particles push apart (`k (1 − r/h)`) and share
  momentum through viscosity, found through the 3×3 hashed cells around each particle. That is what
  makes dust settle into solid clumps instead of passing through itself, i.e. planets. Viscous work
  heats matter; dense, hot clumps glow yellow-orange like stars. `h` is set for 16k particles and
  shrinks as `1/√N`, so each particle keeps about as many neighbours at any count.
* **Black holes:** up to 16. Each workgroup of the step reduces its particles' pull on every hole
  and what each hole swallowed, then adds that with 64-bit integer atomics (fixed point), so the
  result does not depend on thread order. A one-thread pass then integrates the holes.
  Holes that touch merge.
* **Integration:** semi-implicit (symplectic) Euler at a fixed step, several steps per frame.
* **Rendering:** additive sprites (instanced quads) into a 16-bit float target that fades each frame
  (trails), then tone-mapped to the screen. The particle buffers are read directly as vertex
  buffers, so nothing is copied back to the CPU. Points get smaller and dimmer as N grows, so the
  total light stays about the same.
* **Spawning:** new matter goes into the free slots after the live particles (the sort keeps the
  dead at the end); when the buffer is full it recycles live ones. Presets are generated in
  independently seeded chunks on every CPU core and streamed to the GPU.

## Record and replay

Recording restarts the current scene with a fresh seed and saves the seed, every slider value and
every mouse event (in world units, stamped with the frame it applied on) to `recordings/last.txt`
plus a timestamped copy. The simulation is fixed-step and spawns from a seeded RNG, so replay on the
same machine reproduces the run, which makes it a repeatable benchmark and a way to compare slider
settings on identical input. `gravity-sandbox --replay FILE` replays any saved recording.

## Command line

* `gravity-sandbox --preview --preset galaxies --n 4096 --steps 300 --out p.png` runs the real
  shaders headless and writes a PNG (CI does this on a software Vulkan driver).
* `gravity-sandbox --bench 65536 1048576 16777216` prints ms per physics step and particle-steps
  per second for your GPU (no arguments: 16k to 16M).
* `gravity-sandbox --gravcheck` compares grid gravity with the exact all-pairs sum.
* `gravity-sandbox --selftest` checks mouse-to-world mapping and black-hole placement end to end.

## Code

* `shaders/step.wgsl`, `shaders/holes.wgsl`: the physics. `shaders/render.wgsl`: drawing.
* `src/sim.rs`: GPU buffers, the per-frame step, input and record/replay. `src/app.rs`: window,
  mouse and the egui panels. `src/presets.rs`, `src/params.rs`: scenes and every tunable.
* `app/`: the original Android app (Kotlin, GLES 3.1). It stays until the PC version has
  everything it has; `tools/preview.py` previews its GLSL shaders.

## Android (v0.1)

The first version was an Android app. Its
[APK](https://github.com/jstumfoll2/gravity-sandbox/releases/download/latest/gravity-sandbox-latest.apk)
still builds in CI. `./gradlew assembleDebug` builds it locally (needs the Android SDK).

## Roadmap

1. Particle-mesh (FFT) or Barnes-Hut gravity for 100k+ particles.
2. "Pocket universe": early-universe noise growing into a cosmic web, with a dark matter slider.
3. Falling-sand style materials (rock, ice, gas) with different contact rules.
4. Screenshot and video export.

## License

GPL-3.0-or-later.
