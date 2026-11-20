//! The window: winit event loop, mouse/keyboard input and the egui control panel.

use crate::gpu;
use crate::params::SECTIONS;
use crate::presets::Preset;
use crate::recording::{self, Kind, Mode, Recording};
use crate::sim::Sim;
use std::sync::Arc;
use std::time::Duration;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

const LEFT: u32 = 0;
const RIGHT: u32 = 1;

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    sim: Sim,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_rend: egui_wgpu::Renderer,
}

#[derive(Default)]
struct Ui {
    show_tune: bool,
    /// Where the panels were drawn last frame (points), so clicks on them don't reach the scene.
    panel_rects: Vec<egui::Rect>,
}

impl Ui {
    /// Uses positions rather than egui's hover state, which lags a frame behind fast mouse jumps.
    fn over_ui(&self, ctx: &egui::Context, x: f32, y: f32) -> bool {
        let ppp = ctx.pixels_per_point();
        let p = egui::pos2(x / ppp, y / ppp);
        self.panel_rects.iter().any(|r| r.contains(p))
            || ctx.layer_id_at(p).is_some_and(|l| l.order != egui::Order::Background)
    }
}

pub struct App {
    gfx: Option<Gfx>,
    replay_file: Option<Recording>,
    error: Option<String>,
    cursor: (f32, f32),
    left_down: bool,
    right_down: bool,
    middle_down: bool,
    shift: bool,
    ui: Ui,
}

pub fn run(replay_file: Option<Recording>) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    let mut app = App {
        gfx: None,
        replay_file,
        error: None,
        cursor: (0.0, 0.0),
        left_down: false,
        right_down: false,
        middle_down: false,
        shift: false,
        ui: Ui::default(),
    };
    event_loop.run_app(&mut app).map_err(|e| e.to_string())?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn pick_format(caps: &wgpu::SurfaceCapabilities) -> wgpu::TextureFormat {
    // Prefer a plain (non-sRGB) 8-bit format, like the Android framebuffer.
    caps.formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(caps.formats[0])
}

impl App {
    fn init(&mut self, el: &ActiveEventLoop) -> Result<(), String> {
        let attrs = Window::default_attributes()
            .with_title("Gravity Sandbox")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = Arc::new(el.create_window(attrs).map_err(|e| e.to_string())?);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(
            el.owned_display_handle(),
        )));
        let surface = instance.create_surface(window.clone()).map_err(|e| e.to_string())?;
        let (adapter, device, queue, info) = pollster::block_on(gpu::open(&instance, Some(&surface)))?;
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        if caps.formats.is_empty() {
            return Err("This GPU can't draw to the window".into());
        }
        let format = pick_format(&caps);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&device, &config);

        let egui_ctx = egui::Context::default();
        egui_ctx.set_visuals(egui::Visuals::dark());
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let egui_rend = egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

        let mut sim = Sim::new(device, queue, format, config.width, config.height, info);
        if let Some(r) = self.replay_file.take() {
            sim.load_replay(r);
        }
        self.gfx = Some(Gfx {
            window,
            surface,
            config,
            sim,
            egui_ctx,
            egui_state,
            egui_rend,
        });
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        if let Err(e) = self.init(el) {
            self.error = Some(e);
            el.exit();
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
    }

    fn exiting(&mut self, _el: &ActiveEventLoop) {
        if let Some(g) = &mut self.gfx {
            if g.sim.is_recording() {
                g.sim.stop_recording();
            }
            g.sim.save_settings();
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(g) = &mut self.gfx else { return };
        let _ = g.egui_state.on_window_event(&g.window, &event);
        let ui_wants_keys = g.egui_ctx.egui_wants_keyboard_input();
        let (cx, cy) = self.cursor;
        let ui_wants_pointer = self.ui.over_ui(&g.egui_ctx, cx, cy);
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                g.config.width = size.width.max(1);
                g.config.height = size.height.max(1);
                g.surface.configure(g.sim.device(), &g.config);
                g.sim.resize(g.config.width, g.config.height);
            }
            WindowEvent::ModifiersChanged(m) => self.shift = m.state().shift_key(),
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if self.middle_down {
                    g.sim.pan_px(x - cx, y - cy);
                }
                self.cursor = (x, y);
                let mode = g.sim.mode;
                if self.left_down {
                    g.sim.pointer(Kind::Move, LEFT, x, y, mode);
                }
                if self.right_down {
                    g.sim.pointer(Kind::Move, RIGHT, x, y, mode);
                }
            }
            WindowEvent::CursorLeft { .. } => {
                if self.left_down {
                    g.sim.pointer(Kind::Up, LEFT, cx, cy, g.sim.mode);
                    self.left_down = false;
                }
                if self.right_down {
                    g.sim.pointer(Kind::Up, RIGHT, cx, cy, g.sim.mode);
                    self.right_down = false;
                }
                self.middle_down = false;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        if pressed && !ui_wants_pointer {
                            self.left_down = true;
                            g.sim.pointer(Kind::Down, LEFT, cx, cy, g.sim.mode);
                        } else if !pressed && self.left_down {
                            self.left_down = false;
                            g.sim.pointer(Kind::Up, LEFT, cx, cy, g.sim.mode);
                        }
                    }
                    MouseButton::Right => {
                        // right button always pulls; shift + right pushes
                        if pressed && !ui_wants_pointer {
                            self.right_down = true;
                            let m = if self.shift { Mode::Push } else { Mode::Pull };
                            g.sim.pointer(Kind::Down, RIGHT, cx, cy, m);
                        } else if !pressed && self.right_down {
                            self.right_down = false;
                            g.sim.pointer(Kind::Up, RIGHT, cx, cy, Mode::Pull);
                        }
                    }
                    MouseButton::Middle => self.middle_down = pressed && !ui_wants_pointer,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !ui_wants_pointer {
                    let lines = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                    };
                    g.sim.zoom_at(cx, cy, 1.12f32.powf(lines));
                }
            }
            WindowEvent::KeyboardInput { event, .. } if !ui_wants_keys && event.state == ElementState::Pressed => {
                match event.logical_key.as_ref() {
                    Key::Character("1") => g.sim.mode = Mode::Spawn,
                    Key::Character("2") => g.sim.mode = Mode::Pull,
                    Key::Character("3") => g.sim.mode = Mode::Push,
                    Key::Character("4") => g.sim.mode = Mode::Hole,
                    Key::Character("r") | Key::Character("R") => g.sim.restart(),
                    Key::Character("t") | Key::Character("T") => self.ui.show_tune = !self.ui.show_tune,
                    Key::Character("c") | Key::Character("C") => g.sim.reset_view(),
                    Key::Named(NamedKey::Space) => g.sim.paused = !g.sim.paused,
                    Key::Named(NamedKey::F11) => {
                        let fs = if g.window.fullscreen().is_some() {
                            None
                        } else {
                            Some(Fullscreen::Borderless(None))
                        };
                        g.window.set_fullscreen(fs);
                    }
                    Key::Named(NamedKey::Escape) if g.window.fullscreen().is_some() => g.window.set_fullscreen(None),
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = frame(g, &mut self.ui) {
                    self.error = Some(e);
                    el.exit();
                }
            }
            _ => {}
        }
    }
}

fn frame(g: &mut Gfx, ui_state: &mut Ui) -> Result<(), String> {
    let surf = match g.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            g.surface.configure(g.sim.device(), &g.config);
            return Ok(());
        }
        wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
            std::thread::sleep(Duration::from_millis(16));
            return Ok(());
        }
        other => return Err(format!("Lost the window surface ({other:?})")),
    };
    g.sim.update();

    let raw = g.egui_state.take_egui_input(&g.window);
    let mut out = g.egui_ctx.run_ui(raw, |ui| controls(ui, &mut g.sim, ui_state));
    g.egui_state.handle_platform_output(&g.window, out.platform_output);
    let jobs = g.egui_ctx.tessellate(out.shapes, out.pixels_per_point);
    let sd = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [g.config.width, g.config.height],
        pixels_per_point: out.pixels_per_point,
    };
    for (id, deltas) in &out.textures_delta.set {
        for delta in deltas {
            g.egui_rend.update_texture(g.sim.device(), g.sim.queue(), *id, delta);
        }
    }

    let view = surf.texture.create_view(&Default::default());
    let mut enc = g.sim.device().create_command_encoder(&Default::default());
    g.sim.render(&mut enc, &view);
    let extra = g
        .egui_rend
        .update_buffers(g.sim.device(), g.sim.queue(), &mut enc, &jobs, &sd);
    {
        let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        g.egui_rend.render(&mut pass.forget_lifetime(), &jobs, &sd);
    }
    g.sim.queue().submit(extra.into_iter().chain([enc.finish()]));
    g.window.pre_present_notify();
    g.sim.queue().present(surf);
    for id in &out.textures_delta.free {
        g.egui_rend.free_texture(id);
    }
    out.textures_delta.clear();
    Ok(())
}

fn panel_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(egui::Color32::from_rgba_unmultiplied(12, 14, 24, 200))
        .inner_margin(egui::Margin::symmetric(8, 6))
}

fn controls(ui: &mut egui::Ui, sim: &mut Sim, st: &mut Ui) {
    st.panel_rects.clear();
    let top = egui::Panel::top("toolbar").frame(panel_frame()).show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            for m in Mode::ALL {
                let key = 1 + m as usize;
                if ui
                    .add(egui::Button::selectable(sim.mode == m, m.label()))
                    .on_hover_text(format!("{} (key {key})", m.hint()))
                    .clicked()
                {
                    sim.mode = m;
                }
            }
            ui.separator();
            egui::ComboBox::from_id_salt("scene")
                .selected_text(sim.preset.label())
                .show_ui(ui, |ui| {
                    for p in Preset::ALL {
                        if ui.add(egui::Button::selectable(sim.preset == p, p.label())).clicked() {
                            sim.new_scene(p);
                        }
                    }
                });
            if ui
                .button("Reset")
                .on_hover_text("Restart this scene with a new seed (R)")
                .clicked()
            {
                sim.restart();
            }
            let pause = if sim.paused { "Run" } else { "Pause" };
            if ui.button(pause).on_hover_text("Space").clicked() {
                sim.paused = !sim.paused;
            }
            if ui
                .add(egui::Button::selectable(st.show_tune, "Tune"))
                .on_hover_text("Every constant as a slider (T)")
                .clicked()
            {
                st.show_tune = !st.show_tune;
            }
            ui.separator();
            if sim.is_recording() {
                if ui
                    .button(egui::RichText::new("Stop recording").color(egui::Color32::from_rgb(255, 96, 96)))
                    .clicked()
                {
                    sim.stop_recording();
                }
            } else if ui
                .button("Rec")
                .on_hover_text("Record your input from a fresh scene")
                .clicked()
            {
                sim.start_recording();
            }
            let replay = egui::RichText::new("Replay");
            let replay = if sim.is_replaying() {
                replay.color(egui::Color32::from_rgb(120, 180, 255))
            } else {
                replay
            };
            if ui
                .button(replay)
                .on_hover_text("Replay the last recording exactly")
                .clicked()
            {
                sim.start_replay();
            }
        });
    });
    st.panel_rects.push(top.response.rect);

    let bottom = egui::Panel::bottom("status").frame(panel_frame()).show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(&sim.stats);
            if let Some((msg, t)) = &sim.message
                && t.elapsed() < Duration::from_secs(6)
            {
                {
                    ui.separator();
                    ui.label(egui::RichText::new(msg).color(egui::Color32::from_rgb(255, 210, 140)));
                }
            }
            ui.separator();
            ui.label(
                egui::RichText::new(format!(
                    "Left: {} · Right: pull (Shift: push) · Middle: pan · Wheel: zoom",
                    sim.mode.label().to_lowercase()
                ))
                .weak(),
            );
        });
    });
    st.panel_rects.push(bottom.response.rect);

    if st.show_tune {
        let right = egui::Panel::right("tune")
            .frame(panel_frame())
            .default_size(330.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for section in SECTIONS {
                        egui::CollapsingHeader::new(section).default_open(true).show(ui, |ui| {
                            for i in 0..sim.params.all.len() {
                                if sim.params.all[i].section != section {
                                    continue;
                                }
                                let q = &mut sim.params.all[i];
                                let mut v = q.value;
                                let mut slider = egui::Slider::new(&mut v, q.min..=q.max)
                                    .logarithmic(q.log)
                                    .text(q.label);
                                if q.integer {
                                    slider = slider.integer();
                                }
                                if q.step > 0.0 {
                                    slider = slider.step_by(q.step as f64);
                                }
                                let mut r = ui.add(slider);
                                let mut help = String::from(q.help);
                                if !help.is_empty() {
                                    help.push('\n');
                                }
                                help += &format!("Default {} · right-click to reset", q.default);
                                r = r.on_hover_text(help);
                                let mut changed = r.changed();
                                if r.secondary_clicked() {
                                    v = q.default;
                                    changed = true;
                                }
                                if changed {
                                    q.value = q.clamp(v);
                                    sim.param_changed(i);
                                }
                            }
                        });
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui
                            .button("Benchmark")
                            .on_hover_text("Times 120 physics frames at the current particle count")
                            .clicked()
                        {
                            sim.benchmark();
                        }
                        if ui.button("Defaults").clicked() {
                            sim.params.reset_defaults();
                            for i in 0..sim.params.all.len() {
                                sim.param_changed(i);
                            }
                        }
                        if ui
                            .button("Apply count")
                            .on_hover_text("Restart with the Particles slider's count")
                            .clicked()
                        {
                            sim.restart();
                        }
                    });
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(&sim.gpu_info).small().weak());
                    ui.label(
                        egui::RichText::new(format!("Settings and recordings: {}", recording::data_dir().display()))
                            .small()
                            .weak(),
                    );
                });
            });
        st.panel_rects.push(right.response.rect);
    }
}
