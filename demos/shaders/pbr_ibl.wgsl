// PBR IBL Demo - Main PBR Shader
// Demonstrates PBR rendering with Image-Based Lighting using the RedLilium shader library.

// Import shader library modules with explicit items
#import redlilium::math::{PI, INV_PI}
#import redlilium::brdf::{calculate_f0, fresnel_schlick_roughness, pbr_direct_lighting}
#import redlilium::color::{tonemap_reinhard, gamma_correct}

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

// IBL textures
@group(1) @binding(0) var irradiance_map: texture_cube<f32>;
@group(1) @binding(1) var prefilter_map: texture_cube<f32>;
@group(1) @binding(2) var brdf_lut: texture_2d<f32>;
@group(1) @binding(3) var ibl_sampler: sampler;

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

// IBL constants
const MAX_REFLECTION_LOD: f32 = 4.0;

// Fragment output structure for MRT (Multiple Render Targets)
// Supports G-buffer output for deferred rendering visualization
struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) albedo: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let albedo = in.base_color.rgb;
    let metallic = in.metallic;
    let roughness = max(in.roughness, 0.04);

    let n = normalize(in.world_normal);
    let v = normalize(camera.camera_pos.xyz - in.world_position);
    let r = reflect(-v, n);

    let n_dot_v = max(dot(n, v), 0.0);

    // Calculate F0 using library function
    let f0 = calculate_f0(albedo, metallic);

    // === Direct lighting ===
    // Simple directional light (sun-like)
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 0.5));
    let light_color = vec3<f32>(1.0, 0.98, 0.95) * 3.0;

    // Use library function for direct lighting
    var lo = pbr_direct_lighting(n, v, light_dir, albedo, metallic, roughness, light_color);

    // Fill light (simple diffuse-only)
    let fill_light_dir = normalize(vec3<f32>(-0.5, -0.3, -1.0));
    let fill_light_color = vec3<f32>(0.3, 0.4, 0.5) * 0.5;
    let fill_n_dot_l = max(dot(n, fill_light_dir), 0.0);
    let kd_fill = (1.0 - metallic);
    lo = lo + kd_fill * albedo * INV_PI * fill_light_color * fill_n_dot_l;

    // === IBL ambient lighting ===
    // Use library functions for Fresnel
    let f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let ks_ibl = f_ibl;
    let kd_ibl = (vec3<f32>(1.0) - ks_ibl) * (1.0 - metallic);

    // Diffuse IBL from irradiance map
    let irradiance = textureSample(irradiance_map, ibl_sampler, n).rgb;
    let diffuse_ibl = irradiance * albedo;

    // Specular IBL from pre-filtered environment map + BRDF LUT
    let prefiltered_color = textureSampleLevel(prefilter_map, ibl_sampler, r, roughness * MAX_REFLECTION_LOD).rgb;
    let brdf = textureSample(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness)).rg;
    let specular_ibl = prefiltered_color * (f_ibl * brdf.x + brdf.y);

    let ambient = kd_ibl * diffuse_ibl + specular_ibl;

    // Combine
    var color = ambient + lo;

    // HDR tonemapping using library function (Reinhard)
    color = tonemap_reinhard(color);

    // Gamma correction using library function
    color = gamma_correct(color);

    var out: FragmentOutput;
    out.color = vec4<f32>(color, 1.0);
    out.albedo = vec4<f32>(albedo, 1.0);
    return out;
}
