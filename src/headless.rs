//! Runs the real shaders without a window: preview PNGs for CI, and a benchmark.

use crate::gpu;
use crate::params::P;
use crate::presets::Preset;
use crate::sim::{self, Sim};
use std::time::Instant;

pub struct Opts {
    pub preset: Preset,
    pub n: usize,
    pub steps: usize,
    pub width: u32,
    pub height: u32,
    pub out: String,
    pub seed: u64,
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn make_sim(n: usize, w: u32, h: u32) -> Result<Sim, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let (_adapter, device, queue, info) = pollster::block_on(gpu::open(&instance, None))?;
    eprintln!("GPU: {info}");
    let mut s = Sim::new(device, queue, FORMAT, w, h, info);
    s.params.reset_defaults();
    s.params.set(P::Count, n as f32);
    Ok(s)
}

pub fn preview(o: &Opts) -> Result<(), String> {
    let mut s = make_sim(o.n, o.width, o.height)?;
    s.reset(o.preset, o.seed);
    let target = s.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("preview"),
        size: wgpu::Extent3d {
            width: o.width,
            height: o.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let t = Instant::now();
    let frames = o.steps.div_ceil(2);
    for f in 0..frames {
        s.run_steps(2);
        // draw the last frames so trails build up like on screen
        if f + 40 >= frames {
            let mut enc = s.device().create_command_encoder(&Default::default());
            s.render(&mut enc, &view);
            s.queue().submit([enc.finish()]);
        }
    }
    let _ = s.device().poll(wgpu::PollType::wait_indefinitely());
    eprintln!(
        "{} steps of {} particles in {:.2} s",
        frames * 2,
        s.capacity(),
        t.elapsed().as_secs_f64()
    );

    // read back
    let bpr = (o.width * 4).div_ceil(256) * 256;
    let buf = s.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bpr * o.height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = s.device().create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: o.width,
            height: o.height,
            depth_or_array_layers: 1,
        },
    );
    s.queue().submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = s.device().poll(wgpu::PollType::wait_indefinitely());
    let mut pixels = Vec::with_capacity((o.width * o.height * 4) as usize);
    {
        let data = slice.get_mapped_range().map_err(|e| format!("readback: {e:?}"))?;
        for row in 0..o.height {
            let start = (row * bpr) as usize;
            pixels.extend_from_slice(&data[start..start + (o.width * 4) as usize]);
        }
    }
    buf.unmap();
    let lit = pixels
        .chunks(4)
        .filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 60)
        .count();
    eprintln!("{lit} bright pixels");

    let file = std::fs::File::create(&o.out).map_err(|e| format!("{}: {e}", o.out))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), o.width, o.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    w.write_image_data(&pixels).map_err(|e| e.to_string())?;
    w.finish().map_err(|e| e.to_string())?;
    eprintln!("wrote {}", o.out);
    if lit < 50 {
        return Err("the preview is nearly black; the simulation or drawing is broken".into());
    }
    Ok(())
}

/// Checks mouse -> world mapping and black-hole placement through the real input path.
pub fn selftest() -> Result<(), String> {
    let mut s = make_sim(1024, 1280, 800)?;
    s.reset(Preset::Empty, 1);
    s.mode = crate::recording::Mode::Hole;
    // 1280x800: half-height 100 world units, half-width 160
    for (px, py) in [(1000.0, 250.0), (500.0, 600.0)] {
        s.update();
        s.pointer(crate::recording::Kind::Down, 0, px, py, s.mode);
        s.pointer(crate::recording::Kind::Up, 0, px, py, s.mode);
    }
    for _ in 0..3 {
        s.update();
    }
    let holes = s.read_holes();
    eprintln!("holes: {holes:?}");
    let want = [
        [(1000.0 / 640.0 - 1.0) * 160.0, (1.0 - 250.0 / 400.0) * 100.0],
        [(500.0 / 640.0 - 1.0) * 160.0, (1.0 - 600.0 / 400.0) * 100.0],
    ];
    if holes.len() != 2 {
        return Err(format!("expected 2 holes, got {}", holes.len()));
    }
    for (h, w) in holes.iter().zip(want) {
        if (h.pos[0] - w[0]).abs() > 1.0 || (h.pos[1] - w[1]).abs() > 1.0 || h.pos[2] <= 0.0 {
            return Err(format!("hole at {:?}, expected near {w:?}", h.pos));
        }
    }
    eprintln!("selftest ok");
    Ok(())
}

/// Physics cost per step for a range of particle counts.
pub fn bench(sizes: &[usize]) -> Result<(), String> {
    let mut s = make_sim(1024, 64, 64)?;
    for &n in sizes {
        s.params.set(P::Count, n as f32);
        s.reset(Preset::Galaxies, 1);
        let mut ms = Vec::new();
        for i in 0..40 {
            let _ = s.device().poll(wgpu::PollType::wait_indefinitely());
            let t = Instant::now();
            s.run_steps(1);
            let _ = s.device().poll(wgpu::PollType::wait_indefinitely());
            if i >= 5 {
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
        println!("{}", sim::bench_summary(s.capacity(), &mut ms).unwrap_or_default());
    }
    Ok(())
}
