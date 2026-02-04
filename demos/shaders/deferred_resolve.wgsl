// Deferred Rendering - Resolve/Lighting Pass
// Reads G-buffer and applies IBL lighting to produce final image.

#import redlilium::math::{PI, INV_PI}
#import redlilium::brdf::{calculate_f0, fresnel_schlick_roughness, pbr_direct_lighting}
#import redlilium::color::{tonemap_reinhard, gamma_correct}

// Camera uniforms for lighting calculations
struct ResolveUniforms {
    camera_pos: vec4<f32>,
    screen_size: vec4<f32>, // xy = screen dimensions, zw = 1/screen dimensions
};

@group(0) @binding(0) var<uniform> uniforms: ResolveUniforms;

// G-buffer textures
@group(1) @binding(0) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(1) var gbuffer_normal_metallic: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_position_roughness: texture_2d<f32>;
@group(1) @binding(3) var gbuffer_sampler: sampler;

// IBL textures
@group(2) @binding(0) var irradiance_map: texture_cube<f32>;
@group(2) @binding(1) var prefilter_map: texture_cube<f32>;
@group(2) @binding(2) var brdf_lut: texture_2d<f32>;
@group(2) @binding(3) var ibl_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle vertices generated in shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate fullscreen triangle
    // Vertex 0: (-1, -1), UV (0, 1)
    // Vertex 1: (3, -1),  UV (2, 1)
    // Vertex 2: (-1, 3),  UV (0, -1)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

// IBL constants
const MAX_REFLECTION_LOD: f32 = 4.0;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample G-buffer
    let albedo_sample = textureSample(gbuffer_albedo, gbuffer_sampler, in.uv);
    let normal_metallic_sample = textureSample(gbuffer_normal_metallic, gbuffer_sampler, in.uv);
    let position_roughness_sample = textureSample(gbuffer_position_roughness, gbuffer_sampler, in.uv);

    // Check if this pixel has geometry (albedo alpha > 0)
    if albedo_sample.a < 0.5 {
        // Background - return transparent/black
        discard;
    }

    // Decode G-buffer
    let albedo = albedo_sample.rgb;
    // Decode normal from [0,1] back to [-1,1]
    let normal = normalize(normal_metallic_sample.rgb * 2.0 - 1.0);
    let metallic = normal_metallic_sample.a;
    let world_position = position_roughness_sample.rgb;
    let roughness = max(position_roughness_sample.a, 0.04);

    // Calculate view direction
    let v = normalize(uniforms.camera_pos.xyz - world_position);
    let r = reflect(-v, normal);
    let n_dot_v = max(dot(normal, v), 0.0);

    // Calculate F0 using library function
    let f0 = calculate_f0(albedo, metallic);

    // === Direct lighting ===
    // Simple directional light (sun-like)
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 0.5));
    let light_color = vec3<f32>(1.0, 0.98, 0.95) * 3.0;

    // Use library function for direct lighting
    var lo = pbr_direct_lighting(normal, v, light_dir, albedo, metallic, roughness, light_color);

    // Fill light (simple diffuse-only)
    let fill_light_dir = normalize(vec3<f32>(-0.5, -0.3, -1.0));
    let fill_light_color = vec3<f32>(0.3, 0.4, 0.5) * 0.5;
    let fill_n_dot_l = max(dot(normal, fill_light_dir), 0.0);
    let kd_fill = (1.0 - metallic);
    lo = lo + kd_fill * albedo * INV_PI * fill_light_color * fill_n_dot_l;

    // === IBL ambient lighting ===
    // Use library functions for Fresnel
    let f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let ks_ibl = f_ibl;
    let kd_ibl = (vec3<f32>(1.0) - ks_ibl) * (1.0 - metallic);

    // Diffuse IBL from irradiance map
    let irradiance = textureSample(irradiance_map, ibl_sampler, normal).rgb;
    let diffuse_ibl = irradiance * albedo;

    // Specular IBL from pre-filtered environment map + BRDF LUT
    let prefiltered_color = textureSampleLevel(prefilter_map, ibl_sampler, r, roughness * MAX_REFLECTION_LOD).rgb;
    let brdf = textureSample(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness)).rg;
    let specular_ibl = prefiltered_color * (f_ibl * brdf.x + brdf.y);

    let ambient = kd_ibl * diffuse_ibl + specular_ibl;

    // Combine
    var color = ambient + lo;

#ifdef HDR_OUTPUT
    // HDR output: skip tone mapping and gamma correction
    // The display will handle the HDR-to-SDR conversion if needed
    // Clamp to reasonable HDR range (avoid extreme values)
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(10.0));
#else
    // SDR output: apply tonemapping and gamma correction
    // HDR tonemapping using library function (Reinhard)
    color = tonemap_reinhard(color);

    // Gamma correction using library function
    color = gamma_correct(color);
#endif

    return vec4<f32>(color, 1.0);
}
