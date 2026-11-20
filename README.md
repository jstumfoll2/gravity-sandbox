# Gravity Sandbox

Fling thousands of particles with your thumbs and watch gravity pull them into
disks, planets and stars. Drop black holes and watch them feed.

A hobby Android app: a real N-body simulation running on the phone's GPU in
OpenGL ES 3.1 compute shaders. No network permission, no ads, no tracking.

## Install

Every push builds a debug APK in GitHub Actions. Open the latest run under
**Actions**, download the `gravity-sandbox-apk` artifact, unzip, install.
All builds share one public debug key, so a new build installs over the old one.

Needs a GPU with OpenGL ES 3.1 (most phones from 2016 on).

## Controls

| Toolbar | Thumb does |
|---|---|
| Spawn | Drag to spray matter; flick to throw it |
| Pull / Push | Hold to pull matter in or blow it away |
| Hole | Tap to drop a black hole, flick to throw it |
| Scene | Disk + hole, collapsing cloud, galaxy collision, rings, empty |
| Tune | Every physics and look constant as a slider (long-press a name to reset it), plus a GPU benchmark |
| Rec / Replay | Record your inputs from a fresh scene, then replay them exactly |

Both thumbs work at once, each in the mode that was selected when it touched down.

## How the physics works

* **State:** each particle is two `vec4`s: `(x, y, mass, alive)` and `(vx, vy, heat, density)`,
  in two shader storage buffers that ping-pong every step (read A, write B, swap).
* **Gravity:** an exact all-pairs sum with Plummer softening,
  `a = G Σ m d / (r² + ε²)^{3/2}`, tiled through 256 entries of workgroup shared memory
  (the GPU Gems 3 layout). Cost is N² per step, so ~8k particles is the comfortable default and
  the Tune screen's benchmark reports ms/step and pair interactions per second for your GPU.
* **Matter:** within the contact radius `h`, particles push apart (`k (1 − r/h)`) and share
  momentum through viscosity. That is what makes dust settle into solid clumps instead of
  passing through itself, i.e. planets. Viscous work heats matter; dense, hot clumps glow
  yellow-orange like stars.
* **Black holes:** up to 16, updated in their own one-workgroup pass that sums the pull of all
  particles. Particles inside the horizon are removed and their mass and momentum are added to
  the hole with integer atomics (fixed point), so the result does not depend on thread order.
  Holes that touch merge.
* **Integration:** semi-implicit (symplectic) Euler at a fixed step, several steps per frame.
* **Rendering:** additive point sprites into a 16-bit float target that fades each frame
  (trails), then tone-mapped to the screen. The particle buffer is read directly as a vertex
  buffer, so nothing is copied back to the CPU.

## Record and replay

Recording restarts the current scene with a fresh seed and saves the seed, every slider value and
every touch (in world units, stamped with the frame it applied on). The simulation is fixed-step
and spawns from a seeded RNG, so replay on the same phone reproduces the run, which makes it a
repeatable benchmark and a way to compare slider settings on identical input.
Files live in the app's private storage under `recordings/`.

## Development

* `./gradlew assembleDebug` builds the APK (needs the Android SDK).
* `python3 tools/preview.py --preset galaxies` runs the real compute and render shaders
  headless on Mesa (desktop GL with `ARB_ES3_1_compatibility`) and writes a PNG. CI runs it
  and uploads `shader-preview`, along with a `glslangValidator` pass over every shader.

## Roadmap

1. Particle-mesh (FFT) or Barnes-Hut gravity for 100k+ particles.
2. Pinch zoom and pan that does not fight two-thumb play.
3. "Pocket universe": early-universe noise growing into a cosmic web, with a dark matter slider.
4. Falling-sand style materials (rock, ice, gas) with different contact rules.
5. Export recordings and screenshots via the share sheet.

## License

GPL-3.0-or-later.
