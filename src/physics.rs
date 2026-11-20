//! GPU physics for large particle counts: compact struct-of-arrays state, a per-step counting
//! sort by hashed grid cell, particle-mesh (FFT) gravity, and hashed-grid contacts.
//! Every pass is O(N) (plus a fixed-size FFT), so cost grows linearly with the particle count.
//!
//! Per step:
//!   sort.wgsl   hash -> scan -> scatter -> fix -> gather   (state[0] -> cell-sorted state[1])
//!   pm.wgsl     deposit -> fft_a -> fft_b -> fft_c         (state[1] -> acceleration grid)
//!   step.wgsl   gravity + contacts + holes + integrate      (state[1] -> state[0])
//!   holes.wgsl  black holes
//!
//! State[0] is always the live copy the renderer draws and the spawner writes into.

use crate::presets::{self, DEAD_HOLE, Hole, Preset, State};

pub const WG: u32 = 256;
pub const MAX_HOLES: usize = 16;
pub const MAX_TOUCH: usize = 10;
pub const MAX_SPAWN: usize = 16384;
const SCAN_BLOCK: usize = 1024;
/// Reference particle count: the contact radius and the look are tuned at this N.
pub const N_REF: f32 = 16384.0;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimU {
    pub n: u32,
    pub t: u32,
    pub m_grid: u32,
    pub hole_count: u32,
    pub touch_count: u32,
    pub exact: u32,
    pub spawn_count: u32,
    pub _pad0: u32,
    pub dt: f32,
    pub g: f32,
    pub mass: f32,
    pub gm: f32,
    pub soft2: f32,
    pub h: f32,
    pub inv_cell: f32,
    pub k: f32,
    pub visc: f32,
    pub drag: f32,
    pub hole_soft2: f32,
    pub bound: f32,
    pub heat_gain: f32,
    pub heat_decay: f32,
    pub grid_origin: f32,
    pub grid_inv: f32,
    pub touch: [[f32; 4]; MAX_TOUCH],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HolesU {
    pub hole_count: u32,
    pub _pad: [u32; 3],
    pub dt: f32,
    pub g: f32,
    pub hole_soft2: f32,
    pub radius_per_mass: f32,
    pub min_radius: f32,
    pub bound: f32,
    pub mass: f32,
    pub gm: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ScanU {
    n: u32,
    _pad: [u32; 3],
}

#[derive(Clone, Copy)]
enum B {
    Uni,
    Ro,
    Rw,
}

fn bgl(device: &wgpu::Device, label: &str, entries: &[(u32, B)]) -> wgpu::BindGroupLayout {
    let e: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .map(|&(binding, b)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match b {
                    B::Uni => wgpu::BufferBindingType::Uniform,
                    B::Ro => wgpu::BufferBindingType::Storage { read_only: true },
                    B::Rw => wgpu::BufferBindingType::Storage { read_only: false },
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &e,
    })
}

fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, bufs: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .map(|&(binding, b)| wgpu::BindGroupEntry {
            binding,
            resource: b.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

struct Module {
    bgl: wgpu::BindGroupLayout,
    pipes: Vec<wgpu::ComputePipeline>,
}

fn module(device: &wgpu::Device, label: &str, src: &str, layout: &[(u32, B)], entries: &[&str]) -> Module {
    let bgl = bgl(device, label, layout);
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipes = entries
        .iter()
        .map(|e| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&pl),
                module: &m,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect();
    Module { bgl, pipes }
}

const COMMON: &str = include_str!("../shaders/common.wgsl");

fn with_common(src: &str) -> String {
    format!("{COMMON}\n{src}")
}

/// Workgroup counts for `groups` 1-D groups, folded into 2-D past the per-dimension limit.
fn groups2d(groups: usize) -> (u32, u32) {
    let g = groups.max(1) as u32;
    let x = g.min(65535);
    (x, g.div_ceil(x))
}

fn dispatch(pass: &mut wgpu::ComputePass, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, threads: usize) {
    let (x, y) = groups2d(threads.div_ceil(WG as usize));
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

/// Rough GPU memory for `n` particles on an `m`-node gravity grid, in bytes.
pub fn bytes_needed(n: usize, m: usize) -> u64 {
    let t = table_size(n) as u64;
    let n = n as u64;
    let l = 2 * m as u64;
    // state + aux twice, keys + perm, hash table (+ scan levels), PM grid + 3 complex planes
    n * 20 * 2 + n * 8 + (t + 1) * 4 * 1001 / 1000 + (m * m) as u64 * 8 + l * l * 8 * 3
}

pub fn table_size(n: usize) -> usize {
    n.next_power_of_two().max(1024)
}

/// Pipelines that do not depend on the particle count.
pub struct Pipelines {
    sort: Module,
    scan: Module,
    step: Module,
    holes: Module,
    spawn: Module,
    pm: Option<(usize, Module)>,
}

impl Pipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        use B::*;
        let sort = module(
            device,
            "sort",
            &with_common(include_str!("../shaders/sort.wgsl")),
            &[(0, Uni), (1, Ro), (2, Ro), (3, Rw), (4, Rw), (5, Rw), (6, Rw), (7, Rw)],
            &["hash", "scatter", "fix", "gather"],
        );
        let scan = module(
            device,
            "scan",
            include_str!("../shaders/scan.wgsl"),
            &[(0, Rw), (1, Rw), (2, Uni)],
            &["scan_block", "scan_add"],
        );
        let step = module(
            device,
            "step",
            &with_common(include_str!("../shaders/step.wgsl")),
            &[
                (0, Uni),
                (1, Ro),
                (2, Ro),
                (3, Rw),
                (4, Rw),
                (5, Ro),
                (6, Ro),
                (7, Ro),
                (8, Rw),
            ],
            &["main"],
        );
        let holes = module(
            device,
            "holes",
            include_str!("../shaders/holes.wgsl"),
            &[(0, Uni), (1, Rw), (2, Rw)],
            &["main"],
        );
        let spawn = module(
            device,
            "spawn",
            &with_common(include_str!("../shaders/spawn.wgsl")),
            &[(0, Uni), (1, Rw), (2, Rw), (3, Rw), (4, Ro)],
            &["spawn", "bump"],
        );
        Pipelines {
            sort,
            scan,
            step,
            holes,
            spawn,
            pm: None,
        }
    }

    fn pm(&mut self, device: &wgpu::Device, m: usize) -> &Module {
        if self.pm.as_ref().is_none_or(|(mm, _)| *mm != m) {
            use B::*;
            let l = 2 * m;
            let src = include_str!("../shaders/pm.wgsl")
                .replace("__M__", &m.to_string())
                .replace("__LOG2L__", &l.trailing_zeros().to_string())
                .replace("__L__", &l.to_string());
            let md = module(
                device,
                "pm",
                &with_common(&src),
                &[(0, Uni), (1, Ro), (2, Rw), (3, Rw), (4, Rw), (5, Rw), (6, Ro)],
                &["deposit", "fft_a", "fft_b", "fft_c", "kern_init", "fft_k1", "fft_k2"],
            );
            self.pm = Some((m, md));
        }
        &self.pm.as_ref().unwrap().1
    }
}

/// Stages timed by the profiler, in encoding order.
pub const STAGES: [&str; 9] = [
    "hash", "scan", "scatter", "fix", "gather", "deposit", "fft", "step", "holes",
];

struct Profiler {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    period_ns: f32,
}

/// Buffers for one particle count.
pub struct World {
    pub n: usize,
    pub t: usize,
    pub m: usize,
    pub exact: bool,
    pub state: [wgpu::Buffer; 2],
    pub aux: [wgpu::Buffer; 2],
    pub holes: wgpu::Buffer,
    absorb: wgpu::Buffer,
    counts: wgpu::Buffer,
    grid: wgpu::Buffer,
    cb: wgpu::Buffer,
    spawn_in: wgpu::Buffer,
    sim_u: wgpu::Buffer,
    holes_u: wgpu::Buffer,
    scan: Vec<(usize, wgpu::BindGroup)>,
    sort_bg: wgpu::BindGroup,
    step_bg: wgpu::BindGroup,
    holes_bg: wgpu::BindGroup,
    spawn_bg: wgpu::BindGroup,
    pm_bg: wgpu::BindGroup,
    kernel_key: Option<[u32; 2]>,
    prof: Option<Profiler>,
    prof_pending: bool,
    _keep: Vec<wgpu::Buffer>,
}

impl World {
    /// Allocates everything for `n` particles (a multiple of 256) and an `m`-node grid.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipes: &mut Pipelines,
        n: usize,
        m: usize,
        exact: bool,
    ) -> World {
        let t = table_size(n);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buf = |label: &str, size: usize, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (size.max(16) as u64).next_multiple_of(16),
                usage: storage | extra,
                mapped_at_creation: false,
            })
        };
        let v = wgpu::BufferUsages::VERTEX;
        let none = wgpu::BufferUsages::empty();
        let state = [buf("state", n * 16, v), buf("sorted state", n * 16, none)];
        let aux = [buf("aux", n * 4, v), buf("sorted aux", n * 4, none)];
        let keys = buf("keys", n * 4, none);
        let perm = buf("perm", n * 4, none);
        // scan levels: counts, then block totals of each level
        let mut sizes = vec![t + 1];
        while *sizes.last().unwrap() > SCAN_BLOCK {
            let s = sizes.last().unwrap().div_ceil(SCAN_BLOCK);
            sizes.push(s);
        }
        let levels: Vec<wgpu::Buffer> = sizes.iter().map(|&s| buf("scan level", s * 4, none)).collect();
        let dummy = buf("scan top", 16, none);
        let counts = levels[0].clone();
        let holes = buf("holes", MAX_HOLES * 32, v);
        let absorb = buf("absorb", MAX_HOLES * 48, none);
        let l = 2 * m;
        let grid = buf("pm grid", m * m * 8, none);
        let ca = buf("pm a", l * l * 8, none);
        let cb = buf("pm b", l * l * 8, none);
        let ck = buf("pm kernel", l * l * 8, none);
        let tw_data: Vec<[f32; 2]> = (0..l / 2)
            .map(|j| {
                let a = -std::f64::consts::TAU * j as f64 / l as f64;
                [a.cos() as f32, a.sin() as f32]
            })
            .collect();
        let tw = buf("twiddles", l * 4, none);
        let spawn_in = buf("spawn", MAX_SPAWN * 16, none);
        let uni = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let sim_u = uni("sim u", std::mem::size_of::<SimU>());
        let holes_u = uni("holes u", std::mem::size_of::<HolesU>());

        let mut keep = vec![
            keys.clone(),
            perm.clone(),
            ca.clone(),
            ck.clone(),
            tw.clone(),
            dummy.clone(),
        ];
        let mut scan = Vec::new();
        for (k, &s) in sizes.iter().enumerate() {
            let su = uni("scan u", 16);
            let data = ScanU {
                n: s as u32,
                _pad: [0; 3],
            };
            let sums = levels.get(k + 1).unwrap_or(&dummy);
            let bg = bind(device, &pipes.scan.bgl, &[(0, &levels[k]), (1, sums), (2, &su)]);
            scan.push((s, bg));
            queue.write_buffer(&su, 0, bytemuck::bytes_of(&data));
            keep.push(su);
        }
        keep.extend(levels.iter().cloned());

        let sort_bg = bind(
            device,
            &pipes.sort.bgl,
            &[
                (0, &sim_u),
                (1, &state[0]),
                (2, &aux[0]),
                (3, &state[1]),
                (4, &aux[1]),
                (5, &keys),
                (6, &perm),
                (7, &counts),
            ],
        );
        let step_bg = bind(
            device,
            &pipes.step.bgl,
            &[
                (0, &sim_u),
                (1, &state[1]),
                (2, &aux[1]),
                (3, &state[0]),
                (4, &aux[0]),
                (5, &counts),
                (6, &ca),
                (7, &holes),
                (8, &absorb),
            ],
        );
        let holes_bg = bind(device, &pipes.holes.bgl, &[(0, &holes_u), (1, &holes), (2, &absorb)]);
        let spawn_bg = bind(
            device,
            &pipes.spawn.bgl,
            &[(0, &sim_u), (1, &state[0]), (2, &aux[0]), (3, &counts), (4, &spawn_in)],
        );
        let pm = pipes.pm(device, m);
        let pm_bg = bind(
            device,
            &pm.bgl,
            &[
                (0, &sim_u),
                (1, &state[1]),
                (2, &grid),
                (3, &ca),
                (4, &cb),
                (5, &ck),
                (6, &tw),
            ],
        );
        let w = World {
            n,
            t,
            m,
            exact,
            state,
            aux,
            holes,
            absorb,
            counts,
            grid,
            cb,
            spawn_in,
            sim_u,
            holes_u,
            scan,
            sort_bg,
            step_bg,
            holes_bg,
            spawn_bg,
            pm_bg,
            kernel_key: None,
            prof: None,
            prof_pending: false,
            _keep: keep,
        };
        queue.write_buffer(&tw, 0, bytemuck::cast_slice(&tw_data));
        w
    }

    /// Streams a preset into the live state (in chunks, generated on all cores) and sets the
    /// live/dead boundary the spawner uses.
    pub fn upload_scene(&self, queue: &wgpu::Queue, preset: Preset, seed: u64, g: f32) -> usize {
        let used = presets::used(preset, self.n);
        let chunk = (presets::CHUNK * 64).min(self.n.next_multiple_of(presets::CHUNK));
        let mut tmp = vec![presets::DEAD; chunk];
        let mut first = 0;
        while first < self.n {
            let len = chunk.min(self.n - first);
            presets::fill_parallel(preset, self.n, seed, g, first, &mut tmp[..len]);
            queue.write_buffer(&self.state[0], first as u64 * 16, bytemuck::cast_slice(&tmp[..len]));
            first += len;
        }
        queue.write_buffer(
            &self.counts,
            (self.t as u64 - 1) * 4,
            bytemuck::bytes_of(&(used as u32)),
        );
        queue.write_buffer(&self.holes, 0, bytemuck::cast_slice(&[DEAD_HOLE; MAX_HOLES]));
        queue.write_buffer(&self.absorb, 0, &[0u8; MAX_HOLES * 48]);
        used
    }

    pub fn write_hole(&self, queue: &wgpu::Queue, slot: usize, h: Hole) {
        queue.write_buffer(&self.holes, slot as u64 * 32, bytemuck::bytes_of(&h));
        queue.write_buffer(&self.absorb, slot as u64 * 48, &[0u8; 48]);
    }

    /// Encodes spawning (if any) and `steps` physics steps.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipes: &mut Pipelines,
        su: &SimU,
        hu: &HolesU,
        spawn: &[State],
        steps: usize,
    ) -> wgpu::CommandBuffer {
        let mut su = *su;
        let spawn = &spawn[..spawn.len().min(MAX_SPAWN)];
        su.spawn_count = spawn.len() as u32;
        queue.write_buffer(&self.sim_u, 0, bytemuck::bytes_of(&su));
        queue.write_buffer(&self.holes_u, 0, bytemuck::bytes_of(hu));
        if !spawn.is_empty() {
            queue.write_buffer(&self.spawn_in, 0, bytemuck::cast_slice(spawn));
        }
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("physics") });
        let m = self.m;
        let l = 2 * m;
        let kkey = [su.grid_inv.to_bits(), su.soft2.to_bits()];
        let need_kernel = !self.exact && self.kernel_key != Some(kkey);
        let pm = pipes.pm(device, m);
        let pm_pipes: Vec<wgpu::ComputePipeline> = pm.pipes.clone();
        if need_kernel {
            self.kernel_key = Some(kkey);
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                dispatch(&mut pass, &pm_pipes[4], &self.pm_bg, l * l);
                pass.set_pipeline(&pm_pipes[5]);
                pass.dispatch_workgroups(l as u32, 1, 1);
                pass.set_pipeline(&pm_pipes[6]);
                pass.dispatch_workgroups(l as u32, 1, 1);
            }
            // fft_a only writes columns < M of cb; the rest must stay zero
            enc.clear_buffer(&self.cb, 0, None);
        }
        if !spawn.is_empty() {
            let mut pass = enc.begin_compute_pass(&Default::default());
            dispatch(&mut pass, &pipes.spawn.pipes[0], &self.spawn_bg, spawn.len());
            pass.set_pipeline(&pipes.spawn.pipes[1]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let n = self.n;
        let sort = &pipes.sort.pipes;
        let profiling = self.prof.is_some();
        for step in 0..steps {
            enc.clear_buffer(&self.counts, 0, None);
            if !self.exact {
                enc.clear_buffer(&self.grid, 0, None);
            }
            // one compute pass per stage, so the first step can be timed stage by stage
            let mut stage = 0u32;
            let begin = |enc: &mut wgpu::CommandEncoder, stage: &mut u32| -> wgpu::ComputePass<'static> {
                let tw = match &self.prof {
                    Some(p) if step == 0 => Some(wgpu::ComputePassTimestampWrites {
                        query_set: &p.set,
                        beginning_of_pass_write_index: Some(2 * *stage),
                        end_of_pass_write_index: Some(2 * *stage + 1),
                    }),
                    _ => None,
                };
                *stage += 1;
                enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("stage"),
                    timestamp_writes: tw,
                })
                .forget_lifetime()
            };
            {
                let mut pass = begin(&mut enc, &mut stage);
                dispatch(&mut pass, &sort[0], &self.sort_bg, n);
            }
            {
                // prefix sum up the levels, then add block offsets back down
                let mut pass = begin(&mut enc, &mut stage);
                for (size, bg) in &self.scan {
                    let (x, y) = groups2d(size.div_ceil(SCAN_BLOCK));
                    pass.set_pipeline(&pipes.scan.pipes[0]);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(x, y, 1);
                }
                for (size, bg) in self.scan.iter().rev().skip(1) {
                    let (x, y) = groups2d(size.div_ceil(SCAN_BLOCK));
                    pass.set_pipeline(&pipes.scan.pipes[1]);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(x, y, 1);
                }
            }
            dispatch(&mut begin(&mut enc, &mut stage), &sort[1], &self.sort_bg, n);
            dispatch(&mut begin(&mut enc, &mut stage), &sort[2], &self.sort_bg, self.t);
            dispatch(&mut begin(&mut enc, &mut stage), &sort[3], &self.sort_bg, n);
            {
                let mut pass = begin(&mut enc, &mut stage);
                if !self.exact {
                    dispatch(&mut pass, &pm_pipes[0], &self.pm_bg, n);
                }
            }
            {
                let mut pass = begin(&mut enc, &mut stage);
                if !self.exact {
                    pass.set_bind_group(0, &self.pm_bg, &[]);
                    pass.set_pipeline(&pm_pipes[1]);
                    pass.dispatch_workgroups(m as u32, 1, 1);
                    pass.set_pipeline(&pm_pipes[2]);
                    pass.dispatch_workgroups(l as u32, 1, 1);
                    pass.set_pipeline(&pm_pipes[3]);
                    pass.dispatch_workgroups(m as u32, 1, 1);
                }
            }
            dispatch(&mut begin(&mut enc, &mut stage), &pipes.step.pipes[0], &self.step_bg, n);
            {
                let mut pass = begin(&mut enc, &mut stage);
                if su.hole_count > 0 {
                    pass.set_pipeline(&pipes.holes.pipes[0]);
                    pass.set_bind_group(0, &self.holes_bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            debug_assert_eq!(stage as usize, STAGES.len());
        }
        if profiling && steps > 0 {
            let p = self.prof.as_ref().unwrap();
            let q = 2 * STAGES.len() as u32;
            enc.resolve_query_set(&p.set, 0..q, &p.resolve, 0);
            enc.copy_buffer_to_buffer(&p.resolve, 0, &p.read, 0, q as u64 * 8);
            self.prof_pending = true;
        }
        enc.finish()
    }

    /// Turns per-stage GPU timing on (if the device has timestamp queries).
    pub fn enable_profiling(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return false;
        }
        if self.prof.is_none() {
            let q = 2 * STAGES.len() as u32;
            let set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("stage timing"),
                ty: wgpu::QueryType::Timestamp,
                count: q,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timing resolve"),
                size: q as u64 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timing read"),
                size: q as u64 * 8,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.prof = Some(Profiler {
                set,
                resolve,
                read,
                period_ns: queue.get_timestamp_period(),
            });
        }
        true
    }

    /// Milliseconds per stage of the last profiled step (blocks on the GPU).
    pub fn stage_ms(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Vec<(&'static str, f64)>> {
        if !self.prof_pending {
            return None;
        }
        self.prof_pending = false;
        let p = self.prof.as_ref()?;
        let ts: Vec<u64> = World::read(device, queue, &p.read, 2 * STAGES.len());
        if ts.len() < 2 * STAGES.len() {
            return None;
        }
        Some(
            STAGES
                .iter()
                .enumerate()
                .map(|(i, &name)| {
                    let d = ts[2 * i + 1].saturating_sub(ts[2 * i]);
                    (name, d as f64 * p.period_ns as f64 / 1e6)
                })
                .collect(),
        )
    }

    /// Particles in the fullest hash bucket, and how many particles sit in buckets of more than
    /// `crowd` (after the last step's sort). Slow; for diagnostics.
    pub fn crowding(&self, device: &wgpu::Device, queue: &wgpu::Queue, crowd: u32) -> (u32, usize) {
        let ends: Vec<u32> = World::read(device, queue, &self.counts, self.t);
        let mut prev = 0u32;
        let (mut max, mut crowded) = (0u32, 0usize);
        for &e in &ends {
            let c = e.saturating_sub(prev);
            prev = e;
            max = max.max(c);
            if c > crowd {
                crowded += c as usize;
            }
        }
        (max, crowded)
    }

    /// Reads a buffer back (slow; for tests and checks).
    pub fn read<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, b: &wgpu::Buffer, len: usize) -> Vec<T> {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(b, 0, &staging, 0, size);
        queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let out = match slice.get_mapped_range() {
            Ok(data) => bytemuck::cast_slice::<u8, T>(&data).to_vec(),
            Err(_) => vec![],
        };
        staging.unmap();
        out
    }
}
