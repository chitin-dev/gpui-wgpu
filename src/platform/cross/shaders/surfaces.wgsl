struct Globals {
    viewport_size: vec2<f32>,
    premultiplied_alpha: u32,
    pad: u32,
}

struct Bounds {
    origin: vec2<f32>,
    size: vec2<f32>,
}

struct SurfaceParams {
    bounds: Bounds,
    content_mask: Bounds,
    clip_bounds: Bounds,
    corner_radii: vec4<f32>,
}

struct SurfaceVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
    @location(1) clip_distances: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> params: SurfaceParams;
@group(1) @binding(1) var t_surface: texture_2d<f32>;
@group(1) @binding(2) var s_surface: sampler;

fn to_device_position(position: vec2<f32>) -> vec4<f32> {
    let device_position = position / globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(device_position, 0.0, 1.0);
}

@vertex
fn vs_surface(@builtin(vertex_index) vertex_id: u32) -> SurfaceVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let position = unit_vertex * params.bounds.size + params.bounds.origin;

    let clip_origin = params.content_mask.origin;
    let clip_size = params.content_mask.size;
    let tl = position - clip_origin;
    let br = clip_origin + clip_size - position;

    var out: SurfaceVarying;
    out.position = to_device_position(position);
    out.tex_coord = unit_vertex;
    out.clip_distances = vec4<f32>(tl.x, br.x, tl.y, br.y);
    return out;
}

// `t_surface` is sampled from an sRGB-format texture, so `textureSample` below
// auto-decodes sRGB -> linear. The swapchain is non-sRGB (see renderer.rs), so
// writing linear values keeps blending correct and lets the display handle the
// final gamma curve.
@fragment
fn fs_surface(input: SurfaceVarying) -> @location(0) vec4<f32> {
    let inside = !any(input.clip_distances < vec4<f32>(0.0));
    let color = textureSample(t_surface, s_surface, input.tex_coord);
    let half_size = params.clip_bounds.size * 0.5;
    let local = input.position.xy - params.clip_bounds.origin - half_size;
    let top_radius = select(params.corner_radii.x, params.corner_radii.y, local.x >= 0.0);
    let bottom_radius = select(params.corner_radii.w, params.corner_radii.z, local.x >= 0.0);
    let radius = select(top_radius, bottom_radius, local.y >= 0.0);
    let delta = abs(local) - half_size + vec2<f32>(radius);
    let distance = min(max(delta.x, delta.y), 0.0) + length(max(delta, vec2<f32>(0.0))) - radius;
    let coverage = clamp(0.5 - distance, 0.0, 1.0);
    let alpha = color.a * coverage;
    let multiplier = select(1.0, alpha, globals.premultiplied_alpha != 0u);
    let result = vec4<f32>(color.rgb * multiplier, alpha);
    return select(vec4<f32>(0.0), result, inside);
}
