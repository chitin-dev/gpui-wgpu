use anyhow::Context as _;

const SHADOW_SHADER: &str = include_str!("../src/platform/cross/shaders/shadows.wgsl");

#[test]
fn shadow_shader_should_validate() -> anyhow::Result<()> {
    let module = wgpu::naga::front::wgsl::parse_str(SHADOW_SHADER)?;
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)?;
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan GPU or software adapter"]
fn crisp_shadows_should_preserve_finite_alpha_and_rounded_coverage() -> anyhow::Result<()> {
    // Execute the production fragment function directly, retaining its clipping
    // and alpha conversion rather than testing a separate CPU approximation.
    let source = SHADOW_SHADER
        .replace(
            "@group(0) @binding(0) var<uniform> globals: Globals;",
            "var<private> globals: Globals;",
        )
        .replace(
            "@group(1) @binding(0) var<storage, read> b_shadows: array<Shadow>;",
            "var<private> b_shadows: array<Shadow, 1>;",
        )
        .replace("@fragment\nfn fs_shadow", "fn fs_shadow")
        .replace("-> @location(0) vec4<f32>", "-> vec4<f32>")
        + r#"
@group(0) @binding(0) var<storage, read_write> results: array<vec4<f32>>;
@compute @workgroup_size(1)
fn check_shadow(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    globals.premultiplied_alpha = 1u;
    var shadow: Shadow;
    shadow.bounds = Bounds(vec2<f32>(16.0), vec2<f32>(32.0));
    shadow.corner_radii = Corners(6.0, 6.0, 6.0, 6.0);
    shadow.content_mask = Bounds(vec2<f32>(0.0), vec2<f32>(64.0));
    shadow.color = Hsla(0.0, 0.0, 1.0, 1.0);
    var input: ShadowVarying;
    input.position = vec4<f32>(32.0, 32.0, 0.0, 1.0);
    input.clip_distances = vec4<f32>(1.0);
    if (index == 1u) { input.position = vec4<f32>(16.5, 16.5, 0.0, 1.0); }
    if (index == 2u) { input.position.x = 16.0; }
    if (index == 3u) { input.clip_distances.x = -1.0; }
    if (index == 4u) { shadow.color.a = 0.25; }
    if (index == 5u) { shadow.color.a = 0.0; }
    if (index == 6u) { shadow.blur_radius = 3.0; }
    if (index == 7u) {
        shadow.corner_radii = Corners(0.0, 0.0, 0.0, 0.0);
        input.position = vec4<f32>(16.5, 16.5, 0.0, 1.0);
    }
    if (index == 8u) {
        shadow.corner_radii.top_right = 0.0;
        input.position = vec4<f32>(47.5, 16.5, 0.0, 1.0);
    }
    b_shadows[0] = shadow;
    input.color = hsla_to_rgba(shadow.color);
    results[index] = fs_shadow(input);
}
"#;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
        .into_iter()
        .next()
        .context("no Vulkan adapter available for shadow regression test")?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shadow regression"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("shadow regression"),
        layout: None,
        module: &shader,
        entry_point: Some("check_shadow"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow results"),
        size: 9 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow readback"),
        size: output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(9, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            if let Err(error) = sender.send(result) {
                eprintln!("shadow readback receiver dropped: {error}");
            }
        });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv_timeout(std::time::Duration::from_secs(5))??;
    let bytes = readback.slice(..).get_mapped_range()?;
    let samples: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
    let expected = [
        Some(1.0),
        Some(0.0),
        Some(0.5),
        Some(0.0),
        Some(0.25),
        Some(0.0),
        None,
        Some(1.0),
        Some(1.0),
    ];
    for (index, (sample, expected)) in samples.iter().zip(expected).enumerate() {
        assert!(
            sample
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "invalid sample {index}: {sample:?}"
        );
        if let Some(expected) = expected {
            assert!(
                sample.iter().all(|value| (value - expected).abs() < 0.0001),
                "incorrect coverage/alpha for sample {index}: {sample:?}"
            );
        }
    }
    drop(bytes);
    readback.unmap();
    Ok(())
}
