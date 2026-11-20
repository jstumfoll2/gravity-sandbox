//! Owns the GPU state and runs the simulation: the port of the Android SimRenderer.

use crate::params::{P, Params};
use crate::physics::{self, HolesU, MAX_HOLES, MAX_TOUCH, N_REF, Pipelines, SimU, World};
use crate::presets::{self, Hole, Preset, State};
use crate::recording::{self, InputEvent, Kind, Mode, Recording};
use crate::rng::Rng;
use std::collections::HashMap;
use std::time::Instant;

const HOLE_SOFT2: f32 = 0.25;
pub const TRAIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderU {
    scale: [f32; 2],
    center: [f32; 2],
    viewport: [f32; 2],
    point_px: f32,
    px_per_unit: f32,
    dens_glow: f32,
    heat_scale: f32,
    brightness: f32,
    keep: f32,
    exposure: f32,
    srgb_out: f32,
    _pad: [f32; 2],
}

struct Pointer {
    x: f32,
    y: f32,
    t: i64,
    mode: Mode,
    px: f32,
    py: f32,
    vx: f32,
    vy: f32,
}

pub struct Sim {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: Pipelines,
    fade_pipe: wgpu::RenderPipeline,
    part_pipe: wgpu::RenderPipeline,
    present_pipe: wgpu::RenderPipeline,
    hole_pipe: wgpu::RenderPipeline,
    bgl_tex: wgpu::BindGroupLayout,
    render_u: wgpu::Buffer,
    u_bg: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    trail_view: wgpu::TextureView,
    tex_bg: wgpu::BindGroup,
    clear_trail: bool,
    srgb_out: bool,
    world: Option<World>,
    spawn_batch: Vec<State>,

    pub width: u32,
    pub height: u32,
    half_w: f32,
    half_h: f32,
    /// Camera centre and zoom (view only; not part of the simulation).
    pub center: [f32; 2],
    pub zoom: f32,

    pub params: Params,
    pub mode: Mode,
    pub paused: bool,
    capacity: usize,
    holes_used: usize,
    hole_next: usize,
    pub preset: Preset,
    seed: u64,
    rnd: Rng,
    pub frame: u64,
    t0: Instant,

    pointers: HashMap<u32, Pointer>,
    pending: Vec<InputEvent>,
    recording: Option<Recording>,
    replay: Option<Recording>,
    replay_idx: usize,

    // timing
    last_stats: Option<Instant>,
    frames_since_stats: u32,
    gpu_ms_acc: f64,
    gpu_ms_n: u32,
    bench_left: u32,
    bench_ms: Vec<f64>,
    pub stats: String,
    pub message: Option<(String, Instant)>,
    pub gpu_info: String,
}

impl Sim {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        output_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        gpu_info: String,
    ) -> Sim {
        let pipes = Pipelines::new(&device);

        let render_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });
        let bgl_u = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bgl_tex = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trail texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout_u = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("u"),
            bind_group_layouts: &[Some(&bgl_u)],
            immediate_size: 0,
        });
        let layout_tex = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("u+tex"),
            bind_group_layouts: &[Some(&bgl_u), Some(&bgl_tex)],
            immediate_size: 0,
        });

        // particles: state (x, y, vx, vy) and packed (heat, density), struct-of-arrays
        let part_layout = [
            Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4],
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: 4,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![1 => Uint32],
            }),
        ];
        let hole_layout = [Some(wgpu::VertexBufferLayout {
            array_stride: 32,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4],
        })];
        let pipe = |label: &str,
                    layout: &wgpu::PipelineLayout,
                    vs: &str,
                    fs: &str,
                    buffers: &[Option<wgpu::VertexBufferLayout>],
                    format: wgpu::TextureFormat,
                    blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &render_mod,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &render_mod,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let comp = |src, dst| wgpu::BlendComponent {
            src_factor: src,
            dst_factor: dst,
            operation: wgpu::BlendOperation::Add,
        };
        use wgpu::BlendFactor as F;
        // trails: dst *= keep
        let fade_blend = wgpu::BlendState {
            color: comp(F::Zero, F::SrcAlpha),
            alpha: comp(F::Zero, F::SrcAlpha),
        };
        let additive = wgpu::BlendState {
            color: comp(F::One, F::One),
            alpha: comp(F::One, F::One),
        };
        let fade_pipe = pipe(
            "fade",
            &layout_u,
            "vs_fullscreen",
            "fs_fade",
            &[],
            TRAIL_FORMAT,
            Some(fade_blend),
        );
        let part_pipe = pipe(
            "particles",
            &layout_u,
            "vs_particle",
            "fs_particle",
            &part_layout,
            TRAIL_FORMAT,
            Some(additive),
        );
        let present_pipe = pipe(
            "present",
            &layout_tex,
            "vs_fullscreen",
            "fs_present",
            &[],
            output_format,
            None,
        );
        let hole_pipe = pipe(
            "holes",
            &layout_u,
            "vs_hole",
            "fs_hole",
            &hole_layout,
            output_format,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );

        let ubuf = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let render_u = ubuf("render u", std::mem::size_of::<RenderU>());
        let u_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render u"),
            layout: &bgl_u,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: render_u.as_entire_binding(),
            }],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (trail_view, tex_bg) = make_trail(&device, &bgl_tex, &sampler, width, height);

        let mut sim = Sim {
            device,
            queue,
            pipes,
            fade_pipe,
            part_pipe,
            present_pipe,
            hole_pipe,
            bgl_tex,
            render_u,
            u_bg,
            sampler,
            trail_view,
            tex_bg,
            clear_trail: true,
            srgb_out: output_format.is_srgb(),
            world: None,
            spawn_batch: Vec::new(),
            width: width.max(1),
            height: height.max(1),
            half_w: 100.0,
            half_h: 100.0,
            center: [0.0, 0.0],
            zoom: 1.0,
            params: Params::new(),
            mode: Mode::Spawn,
            paused: false,
            capacity: 0,
            holes_used: 0,
            hole_next: 0,
            preset: Preset::Disk,
            seed: 1,
            rnd: Rng::new(1),
            frame: 0,
            t0: Instant::now(),
            pointers: HashMap::new(),
            pending: Vec::new(),
            recording: None,
            replay: None,
            replay_idx: 0,
            last_stats: None,
            frames_since_stats: 0,
            gpu_ms_acc: 0.0,
            gpu_ms_n: 0,
            bench_left: 0,
            bench_ms: Vec::new(),
            stats: String::new(),
            message: None,
            gpu_info,
        };
        if let Ok(text) = std::fs::read_to_string(recording::data_dir().join("settings.txt")) {
            sim.params.load_text(&text);
        }
        sim.reset(Preset::Disk, 1);
        sim
    }

    pub fn save_settings(&self) {
        let _ = recording::write_file(&recording::data_dir().join("settings.txt"), &self.params.to_text());
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_recording(&self) -> bool {
        self.recording.is_some()
    }

    pub fn is_replaying(&self) -> bool {
        self.replay.is_some()
    }

    fn say(&mut self, s: impl Into<String>) {
        self.message = Some((s.into(), Instant::now()));
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        self.width = w.max(1);
        self.height = h.max(1);
        let (v, bg) = make_trail(&self.device, &self.bgl_tex, &self.sampler, self.width, self.height);
        self.trail_view = v;
        self.tex_bg = bg;
        self.clear_trail = true;
    }

    // ------------------------------------------------------------------ commands

    pub fn reset(&mut self, preset: Preset, seed: u64) {
        self.preset = preset;
        self.seed = seed;
        self.rnd = Rng::new(seed ^ 0x5DEE_CE66D);
        let want = (self.params.get(P::Count) as usize / 256).max(4) * 256;
        let m = (self.params.get(P::Grid) as usize)
            .next_power_of_two()
            .clamp(256, self.max_grid());
        let cap = self.max_particles();
        let mut n = want.min(cap);
        let exact_below = self.params.get(P::ExactBelow) as usize;
        self.world = None;
        loop {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let w = World::new(&self.device, &self.queue, &mut self.pipes, n, m, n <= exact_below);
            if pollster::block_on(scope.pop()).is_none() {
                self.world = Some(w);
                break;
            }
            drop(w);
            if n <= 4096 {
                panic!("out of GPU memory even at {n} particles");
            }
            n = (n / 2 / 256) * 256;
        }
        if n < want {
            self.say(format!(
                "{} particles would not fit this GPU; using {}",
                group(want),
                group(n)
            ));
        }
        self.capacity = n;
        let w = self.world.as_ref().unwrap();
        w.upload_scene(&self.queue, preset, seed, self.params.get(P::G));
        let scene_holes = presets::holes(
            preset,
            self.params.get(P::HoleRadius),
            self.params.get(P::HoleMinRadius),
        );
        self.spawn_batch.clear();
        self.holes_used = 0;
        self.hole_next = 0;
        for h in scene_holes {
            self.write_hole(h);
        }
        self.pointers.clear();
        self.pending.clear();
        self.frame = 0;
        self.t0 = Instant::now();
        self.clear_trail = true;
    }

    fn new_seed() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(7)
    }

    pub fn new_scene(&mut self, p: Preset) {
        self.stop_replay(false);
        if self.recording.is_some() {
            self.stop_recording();
        }
        self.reset(p, Self::new_seed());
    }

    pub fn restart(&mut self) {
        self.new_scene(self.preset);
    }

    pub fn screen_to_world(&self, x_px: f32, y_px: f32) -> [f32; 2] {
        [
            self.center[0] + (x_px / self.width as f32 * 2.0 - 1.0) * self.half_w,
            self.center[1] + (1.0 - y_px / self.height as f32 * 2.0) * self.half_h,
        ]
    }

    /// Zoom by `factor`, keeping the world point under the cursor fixed.
    pub fn zoom_at(&mut self, x_px: f32, y_px: f32, factor: f32) {
        let before = self.screen_to_world(x_px, y_px);
        self.zoom = (self.zoom * factor).clamp(0.05, 20.0);
        self.update_camera();
        let after = self.screen_to_world(x_px, y_px);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
    }

    /// Pan by a mouse drag in pixels.
    pub fn pan_px(&mut self, dx: f32, dy: f32) {
        self.center[0] -= dx / self.width as f32 * 2.0 * self.half_w;
        self.center[1] += dy / self.height as f32 * 2.0 * self.half_h;
    }

    pub fn reset_view(&mut self) {
        self.center = [0.0, 0.0];
        self.zoom = 1.0;
    }

    /// Raw pointer input in window pixels.
    pub fn pointer(&mut self, kind: Kind, id: u32, x_px: f32, y_px: f32, mode: Mode) {
        if self.replay.is_some() {
            if kind == Kind::Down {
                self.stop_replay(true);
            } else {
                return;
            }
        }
        let [x, y] = self.screen_to_world(x_px, y_px);
        if std::env::var_os("GS_DEBUG_INPUT").is_some() {
            eprintln!(
                "pointer {kind:?} id={id} px=({x_px},{y_px}) world=({x:.1},{y:.1}) {mode:?} {}x{}",
                self.width, self.height
            );
        }
        let t_ms = self.t0.elapsed().as_millis() as i64;
        self.pending.push(InputEvent {
            frame: 0,
            t_ms,
            kind,
            id,
            x,
            y,
            mode,
            key: String::new(),
        });
    }

    pub fn param_changed(&mut self, idx: usize) {
        let q = &self.params.all[idx];
        let (key, value) = (q.key.to_string(), q.value);
        if let Some(r) = &mut self.recording {
            r.events.push(InputEvent {
                frame: self.frame,
                t_ms: 0,
                kind: Kind::Param,
                id: 0,
                x: value,
                y: 0.0,
                mode: Mode::Spawn,
                key,
            });
        }
    }

    pub fn start_recording(&mut self) {
        self.stop_replay(false);
        let s = Self::new_seed();
        self.reset(self.preset, s);
        self.recording = Some(Recording {
            preset: self.preset,
            seed: s,
            params: self.params.snapshot(),
            events: vec![],
            frames: 0,
        });
        self.say(format!("Recording from a fresh {}", self.preset.label()));
    }

    pub fn stop_recording(&mut self) {
        let Some(mut r) = self.recording.take() else { return };
        r.frames = self.frame;
        let text = r.to_text();
        let dir = recording::recordings_dir();
        let name = format!("rec_{}.txt", Self::new_seed() / 1_000_000_000);
        let res = recording::write_file(&dir.join(name), &text)
            .and_then(|_| recording::write_file(&dir.join("last.txt"), &text));
        match res {
            Ok(()) => self.say(format!(
                "Saved {} events, {} frames to {}",
                r.events.len(),
                r.frames,
                dir.display()
            )),
            Err(e) => self.say(format!("Could not save the recording: {e}")),
        }
    }

    pub fn start_replay(&mut self) {
        if self.recording.is_some() {
            self.stop_recording();
        }
        let path = recording::recordings_dir().join("last.txt");
        let Some(r) = std::fs::read_to_string(&path).ok().and_then(|t| Recording::parse(&t)) else {
            self.say("Nothing recorded yet");
            return;
        };
        self.load_replay(r);
    }

    pub fn load_replay(&mut self, r: Recording) {
        self.params.apply(&r.params);
        self.reset(r.preset, r.seed);
        self.say(format!("Replaying {} frames (click to take over)", r.frames));
        self.replay = Some(r);
        self.replay_idx = 0;
    }

    fn stop_replay(&mut self, by_user: bool) {
        if self.replay.take().is_none() {
            return;
        }
        self.pointers.clear();
        if by_user {
            self.say("Replay stopped, you have control");
        }
    }

    pub fn benchmark(&mut self) {
        self.bench_left = 150;
        self.bench_ms.clear();
        self.say(format!("Benchmarking {} particles…", self.capacity));
    }

    // ------------------------------------------------------------------ per frame

    /// Input, spawning and physics for one frame. Call once per displayed frame, then [`Sim::render`].
    pub fn update(&mut self) {
        self.update_camera();
        if self.paused && self.replay.is_none() {
            // keep pointer state sane, but don't advance time
            self.pending.retain(|e| e.kind == Kind::Up);
            self.process_input();
            self.stats();
            return;
        }
        self.process_input();
        self.apply_spawners();

        let timing = self.params.get(P::GpuTiming) > 0.5 || self.bench_left > 0;
        if timing {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let t_start = Instant::now();
        let steps = (self.params.get(P::Substeps) as usize).clamp(1, 8);
        self.run_steps(steps);
        if timing {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            let ms = t_start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
            self.gpu_ms_acc += ms;
            self.gpu_ms_n += 1;
            if self.bench_left > 0 {
                if self.bench_left <= 120 {
                    self.bench_ms.push(ms); // skip warm-up frames
                }
                self.bench_left -= 1;
                if self.bench_left == 0 {
                    self.report_bench();
                }
            }
        }
        self.frame += 1;
        if let Some(r) = &self.replay
            && self.frame >= r.frames
            && self.replay_idx >= r.events.len()
        {
            self.stop_replay(false);
            self.say("Replay finished");
        }
        self.stats();
    }

    fn update_camera(&mut self) {
        let half = 100.0 / self.zoom;
        let (w, h) = (self.width as f32, self.height as f32);
        if w <= h {
            self.half_w = half;
            self.half_h = half * h / w;
        } else {
            self.half_h = half;
            self.half_w = half * w / h;
        }
    }

    fn process_input(&mut self) {
        let events: Vec<InputEvent> = if let Some(rp) = &self.replay {
            let mut list = Vec::new();
            while self.replay_idx < rp.events.len() && rp.events[self.replay_idx].frame <= self.frame {
                list.push(rp.events[self.replay_idx].clone());
                self.replay_idx += 1;
            }
            list
        } else {
            let frame = self.frame;
            let ev: Vec<InputEvent> = self.pending.drain(..).map(|e| InputEvent { frame, ..e }).collect();
            if let Some(r) = &mut self.recording {
                r.events.extend(ev.iter().cloned());
            }
            ev
        };
        for p in self.pointers.values_mut() {
            p.px = p.x;
            p.py = p.y;
        }
        for e in events {
            match e.kind {
                Kind::Down => {
                    self.pointers.insert(
                        e.id,
                        Pointer {
                            x: e.x,
                            y: e.y,
                            t: e.t_ms,
                            mode: e.mode,
                            px: e.x,
                            py: e.y,
                            vx: 0.0,
                            vy: 0.0,
                        },
                    );
                }
                Kind::Move => {
                    if let Some(p) = self.pointers.get_mut(&e.id) {
                        let dt = (e.t_ms - p.t) as f32 / 1000.0;
                        if dt > 0.001 {
                            let ivx = (e.x - p.x) / dt;
                            let ivy = (e.y - p.y) / dt;
                            p.vx += (ivx - p.vx) * 0.5;
                            p.vy += (ivy - p.vy) * 0.5;
                        }
                        p.x = e.x;
                        p.y = e.y;
                        p.t = e.t_ms;
                    }
                }
                Kind::Up => {
                    if let Some(mut p) = self.pointers.remove(&e.id) {
                        if e.t_ms - p.t > 80 {
                            // the pointer had stopped before release
                            p.vx = 0.0;
                            p.vy = 0.0;
                        }
                        if p.mode == Mode::Hole {
                            let f = self.params.get(P::Fling);
                            let h = presets::hole(
                                e.x,
                                e.y,
                                p.vx * f,
                                p.vy * f,
                                self.params.get(P::HoleMass),
                                self.params.get(P::HoleRadius),
                                self.params.get(P::HoleMinRadius),
                            );
                            self.write_hole(h);
                        }
                    }
                }
                Kind::Param => {
                    if let Some(i) = self.params.by_key(&e.key)
                        && !self.params.all[i].restart
                    {
                        let q = &mut self.params.all[i];
                        q.value = q.clamp(e.x);
                    }
                }
            }
        }
    }

    fn apply_spawners(&mut self) {
        let rate = (self.params.get(P::SpawnRate) as usize).clamp(1, 400);
        let spread = self.params.get(P::SpawnSpread);
        let fling = self.params.get(P::Fling);
        let spin = self.params.get(P::Spin);
        let g = self.params.get(P::G);
        // deterministic order for replays
        let mut ids: Vec<u32> = self
            .pointers
            .iter()
            .filter(|(_, p)| p.mode == Mode::Spawn)
            .map(|(k, _)| *k)
            .collect();
        ids.sort();
        for id in ids {
            let p = &self.pointers[&id];
            let (x0, y0, x1, y1, pvx, pvy) = (p.px, p.py, p.x, p.y, p.vx, p.vy);
            for _ in 0..rate {
                let t = self.rnd.f32();
                let x = x0 + (x1 - x0) * t + self.rnd.gaussian() * spread;
                let y = y0 + (y1 - y0) * t + self.rnd.gaussian() * spread;
                let mut vx = pvx * fling + self.rnd.gaussian() * 1.5;
                let mut vy = pvy * fling + self.rnd.gaussian() * 1.5;
                if spin > 0.0 {
                    let r = (x * x + y * y).sqrt() + 1.0;
                    let vc = (g / r).sqrt() * spin;
                    vx += -y / r * vc;
                    vy += x / r * vc;
                }
                self.spawn_batch.push([x, y, vx, vy]);
            }
        }
    }

    fn write_hole(&mut self, h: Hole) {
        let Some(w) = &self.world else { return };
        let slot = self.hole_next;
        self.hole_next = (self.hole_next + 1) % MAX_HOLES;
        self.holes_used = self.holes_used.max(slot + 1);
        w.write_hole(&self.queue, slot, h);
    }

    /// Runs `steps` physics steps in one submission.
    pub fn run_steps(&mut self, steps: usize) {
        let Some(w) = &mut self.world else { return };
        let pr = &self.params;
        let n = w.n as f32;
        let mass = presets::particle_mass(w.n);
        let g = pr.get(P::G);
        // contacts: keep about the same number of neighbours per particle at any N
        let h = pr.get(P::H) * (N_REF / n).sqrt().min(1.0);
        let half = pr.get(P::Bound) * 1.02;
        let mut su = SimU {
            n: w.n as u32,
            t: w.t as u32,
            m_grid: w.m as u32,
            hole_count: self.holes_used as u32,
            exact: w.exact as u32,
            dt: pr.get(P::Dt),
            g,
            mass,
            gm: g * mass,
            soft2: pr.get(P::Soft) * pr.get(P::Soft),
            h,
            inv_cell: 1.0 / h,
            k: pr.get(P::K),
            visc: pr.get(P::Visc),
            drag: pr.get(P::Drag),
            hole_soft2: HOLE_SOFT2,
            bound: pr.get(P::Bound),
            heat_gain: pr.get(P::HeatGain),
            heat_decay: pr.get(P::HeatDecay),
            grid_origin: -half,
            grid_inv: w.m as f32 / (2.0 * half),
            ..Default::default()
        };
        // pointers that pull or push
        let mut ids: Vec<&u32> = self.pointers.keys().collect();
        ids.sort();
        for id in ids {
            let p = &self.pointers[id];
            let sign = match p.mode {
                Mode::Pull => 1.0,
                Mode::Push => -1.0,
                _ => continue,
            };
            if su.touch_count as usize >= MAX_TOUCH {
                break;
            }
            su.touch[su.touch_count as usize] = [p.x, p.y, sign * pr.get(P::Pull), pr.get(P::PullRadius)];
            su.touch_count += 1;
        }
        let hu = HolesU {
            hole_count: self.holes_used as u32,
            dt: pr.get(P::Dt),
            g,
            hole_soft2: HOLE_SOFT2,
            radius_per_mass: pr.get(P::HoleRadius),
            min_radius: pr.get(P::HoleMinRadius),
            bound: pr.get(P::Bound),
            mass,
            gm: g * mass,
            ..Default::default()
        };
        let cb = w.encode(
            &self.device,
            &self.queue,
            &mut self.pipes,
            &su,
            &hu,
            &self.spawn_batch,
            steps,
        );
        self.spawn_batch.clear();
        self.queue.submit([cb]);
    }

    /// Draws into the trail target, then tone-maps it and the black holes onto `target`.
    pub fn render(&mut self, enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let Some(world) = &self.world else { return };
        let (w, h) = (self.width as f32, self.height as f32);
        // fewer, smaller and dimmer points as N grows, so the total light stays about the same
        let n_scale = (N_REF / self.capacity as f32).min(1.0);
        let size = (self.params.get(P::PointPx) * w.min(h) / 720.0).max(1.0);
        let size_eff = (size * n_scale.powf(0.25)).max(1.0);
        let pr = &self.params;
        let ru = RenderU {
            scale: [1.0 / self.half_w, 1.0 / self.half_h],
            center: self.center,
            viewport: [w, h],
            point_px: size_eff,
            px_per_unit: w / (2.0 * self.half_w),
            dens_glow: pr.get(P::DensGlow),
            heat_scale: 0.05,
            brightness: pr.get(P::Brightness) * n_scale * (size / size_eff).powi(2),
            keep: pr.get(P::Trails),
            exposure: pr.get(P::Exposure),
            srgb_out: if self.srgb_out { 1.0 } else { 0.0 },
            _pad: [0.0; 2],
        };
        self.queue.write_buffer(&self.render_u, 0, bytemuck::bytes_of(&ru));

        {
            let load = if self.clear_trail {
                wgpu::LoadOp::Clear(wgpu::Color::BLACK)
            } else {
                wgpu::LoadOp::Load
            };
            self.clear_trail = false;
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("trails"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.trail_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.u_bg, &[]);
            // fade the previous frame (trails)
            pass.set_pipeline(&self.fade_pipe);
            pass.draw(0..3, 0..1);
            // particles, additive
            pass.set_pipeline(&self.part_pipe);
            pass.set_vertex_buffer(0, world.state[0].slice(..));
            pass.set_vertex_buffer(1, world.aux[0].slice(..));
            pass.draw(0..6, 0..self.capacity as u32);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.u_bg, &[]);
            pass.set_bind_group(1, &self.tex_bg, &[]);
            pass.set_pipeline(&self.present_pipe);
            pass.draw(0..3, 0..1);
            // black holes on top, drawn crisp (no trails)
            if self.holes_used > 0 {
                pass.set_pipeline(&self.hole_pipe);
                pass.set_vertex_buffer(0, world.holes.slice(..));
                pass.draw(0..6, 0..self.holes_used as u32);
            }
        }
    }

    fn stats(&mut self) {
        self.frames_since_stats += 1;
        let now = Instant::now();
        let last = *self.last_stats.get_or_insert(now);
        let el = now.duration_since(last).as_secs_f64();
        if el < 0.5 {
            return;
        }
        let fps = self.frames_since_stats as f64 / el;
        let mut s = format!(
            "{fps:.0} fps · {} particles · {}",
            group(self.capacity),
            match &self.world {
                Some(w) if w.exact => "exact gravity".to_string(),
                Some(w) => format!("grid {}²", w.m),
                None => String::new(),
            }
        );
        if let Some(w) = &self.world {
            s += &format!(" · ~{:.2} GB", physics::bytes_needed(w.n, w.m) as f64 / 1e9);
        }
        if self.gpu_ms_n > 0 {
            s += &format!(" · physics {:.2} ms/step", self.gpu_ms_acc / self.gpu_ms_n as f64);
        }
        if self.holes_used > 0 {
            s += &format!(
                " · {} hole slot{}",
                self.holes_used,
                if self.holes_used > 1 { "s" } else { "" }
            );
        }
        if self.paused {
            s += " · paused";
        }
        if self.recording.is_some() {
            s += " · REC";
        }
        if let Some(r) = &self.replay {
            s += &format!(" · replay {}/{}", self.frame, r.frames);
        }
        self.stats = s;
        self.last_stats = Some(now);
        self.frames_since_stats = 0;
        self.gpu_ms_acc = 0.0;
        self.gpu_ms_n = 0;
    }

    fn report_bench(&mut self) {
        if let Some(text) = bench_summary(self.capacity, self.is_exact(), &mut self.bench_ms) {
            self.say(text);
        }
    }

    /// Reads the black holes back (slow; for tests).
    pub fn read_holes(&self) -> Vec<Hole> {
        let Some(w) = &self.world else { return vec![] };
        let mut v: Vec<Hole> = World::read(&self.device, &self.queue, &w.holes, MAX_HOLES);
        v.truncate(self.holes_used);
        v
    }

    /// Reads the live particle state back (slow; for checks).
    pub fn read_state(&self) -> Vec<State> {
        let Some(w) = &self.world else { return vec![] };
        World::read(&self.device, &self.queue, &w.state[0], w.n)
    }

    pub fn is_exact(&self) -> bool {
        self.world.as_ref().is_some_and(|w| w.exact)
    }

    /// Largest grid whose FFT rows (2M complex values, 16 bytes per node) fit in workgroup memory.
    fn max_grid(&self) -> usize {
        let bytes = self.device.limits().max_compute_workgroup_storage_size as usize;
        let m = (bytes / 16).max(256);
        // round down to a power of two
        (1usize << (usize::BITS - 1 - m.leading_zeros())).clamp(256, 2048)
    }

    /// Particles the device limits allow (memory itself is checked when allocating).
    fn max_particles(&self) -> usize {
        let l = self.device.limits();
        let per_binding = l.max_storage_buffer_binding_size.min(l.max_buffer_size) as usize;
        let by_state = per_binding / 16;
        let by_table = (per_binding / 4 - 1).next_power_of_two() / 2;
        (by_state.min(by_table).min(crate::params::MAX_COUNT as usize) / 256) * 256
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// "N=…: median … ms/step, p90 …, … M particle-steps/s"
pub fn bench_summary(capacity: usize, exact: bool, ms: &mut [f64]) -> Option<String> {
    if ms.is_empty() {
        return None;
    }
    ms.sort_by(|a, b| a.total_cmp(b));
    let med = ms[ms.len() / 2];
    let p90 = ms[(ms.len() * 9) / 10];
    let rate = capacity as f64 / (med / 1000.0) / 1e6;
    Some(format!(
        "N={}: median {med:.2} ms/step, p90 {p90:.2} ms, {rate:.1} M particle-steps/s ({})",
        group(capacity),
        if exact { "exact gravity" } else { "grid gravity" },
    ))
}

pub fn group(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn make_trail(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    w: u32,
    h: u32,
) -> (wgpu::TextureView, wgpu::BindGroup) {
    // 16-bit float target: thousands of faint additive sprites accumulate without banding
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trails"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TRAIL_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trails"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    (view, bg)
}

#[cfg(test)]
mod tests {
    #[test]
    fn grouping() {
        assert_eq!(super::group(16384), "16,384");
        assert_eq!(super::group(512), "512");
        assert_eq!(super::group(1048576), "1,048,576");
    }
}
