//! Adapter and device selection shared by the window and headless modes.

pub async fn open(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue, String), String> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| format!("No usable GPU found ({e}). Gravity Sandbox needs Vulkan, Metal, DX12 or OpenGL 4.3."))?;
    let info = adapter.get_info();
    let al = adapter.limits();
    // Ask for what the shaders need (256-thread workgroups, 4 storage buffers per stage),
    // and let large particle buffers through where the GPU allows it.
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
        max_buffer_size: al.max_buffer_size,
        ..wgpu::Limits::downlevel_defaults().using_resolution(al)
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("gravity-sandbox"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("Could not open the GPU: {e}"))?;
    let text = format!("{} · {:?} · {}", info.name, info.backend, info.driver_info);
    Ok((adapter, device, queue, text))
}
