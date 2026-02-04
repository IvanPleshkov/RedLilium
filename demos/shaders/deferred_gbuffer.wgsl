// Deferred Rendering - G-Buffer Pass
// Outputs material properties to multiple render targets for later lighting calculation.

#import redlilium::math::{PI}

// Camera uniforms
struct CameraUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

// Per-instance data
struct InstanceData {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) instance_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) base_color: vec4<f32>,
    @location(4) metallic: f32,
    @location(5) roughness: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let instance = instances[in.instance_id];
    let world_pos = instance.model * vec4<f32>(in.position, 1.0);
    let normal_matrix = mat3x3<f32>(
        instance.model[0].xyz,
        instance.model[1].xyz,
        instance.model[2].xyz
    );

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal = normalize(normal_matrix * in.normal);
    out.uv = in.uv;
    out.base_color = instance.base_color;
    out.metallic = instance.metallic_roughness.x;
    out.roughness = instance.metallic_roughness.y;
    return out;
}

// G-Buffer output structure (Multiple Render Targets)
struct GBufferOutput {
    // RT0: Albedo (RGB) - sRGB color space
    @location(0) albedo: vec4<f32>,
    // RT1: World Normal (RGB) + Metallic (A) - linear, high precision
    @location(1) normal_metallic: vec4<f32>,
    // RT2: World Position (RGB) + Roughness (A) - linear, high precision
    @location(2) position_roughness: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    let albedo = in.base_color.rgb;
    let metallic = in.metallic;
    let roughness = max(in.roughness, 0.04);
    let normal = normalize(in.world_normal);

    var out: GBufferOutput;

    // RT0: Albedo
    out.albedo = vec4<f32>(albedo, 1.0);

    // RT1: Normal (encoded to [0,1] range) + Metallic
    // Normal encoding: map from [-1,1] to [0,1] for storage
    let encoded_normal = normal * 0.5 + 0.5;
    out.normal_metallic = vec4<f32>(encoded_normal, metallic);

    // RT2: World Position + Roughness
    out.position_roughness = vec4<f32>(in.world_position, roughness);

    return out;
}
