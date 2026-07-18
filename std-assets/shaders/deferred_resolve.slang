// Standard deferred resolve/lighting pass (#144) — reads the G-buffer and
// applies direct + IBL lighting to produce the lit image. Drawn as a
// fullscreen triangle over the skybox background (background pixels are
// discarded via the G-buffer's albedo-alpha geometry marker).
//
// The key/fill light constants are a stopgap until lights come from ECS
// components (#146). Shadows arrive with #130.
//
// HDR_OUTPUT gates the tonemap: on = raw linear for a linear-HDR target
// (1.0 = SDR white), off = tonemap + gamma encode for an sRGB/LDR target.
// Selected by the pipeline from the camera target's format.
//#pragma variant_system HDR_OUTPUT

import math;
import brdf;
import color;

// =============================================================================
// Vertex Shader
// =============================================================================

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    VsOutput output;

    // Fullscreen triangle
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };

    float2 uvs[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };

    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

// =============================================================================
// Fragment Shader
// =============================================================================

[[vk::binding(0, 0)]]
cbuffer ResolveUniforms {
    float4 camera_pos;
    float4 light_dir;   // xyz = normalized direction toward the key light
};

// G-buffer textures
[[vk::binding(0, 1)]]
Texture2D gbuffer_albedo;
[[vk::binding(1, 1)]]
Texture2D gbuffer_normal_metallic;
[[vk::binding(2, 1)]]
Texture2D gbuffer_position_roughness;
[[vk::binding(3, 1)]]
SamplerState gbuffer_sampler;

// IBL textures
[[vk::binding(0, 2)]]
TextureCube irradiance_map;
[[vk::binding(1, 2)]]
TextureCube prefilter_map;
[[vk::binding(2, 2)]]
Texture2D brdf_lut;
[[vk::binding(3, 2)]]
SamplerState ibl_sampler;

// Highest prefiltered-environment mip the roughness axis maps onto.
static const float MAX_REFLECTION_LOD_VAL = 4.0;

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    // Sample G-buffer
    float4 albedo_sample = gbuffer_albedo.Sample(gbuffer_sampler, input.uv);
    float4 normal_metallic_sample = gbuffer_normal_metallic.Sample(gbuffer_sampler, input.uv);
    float4 position_roughness_sample = gbuffer_position_roughness.Sample(gbuffer_sampler, input.uv);

    // Background pixels carry no geometry (albedo alpha stays at the clear
    // value 0) — keep whatever the skybox pass wrote.
    if (albedo_sample.a < 0.5) {
        discard;
    }

    // Decode G-buffer
    float3 albedo = albedo_sample.rgb;
    float3 normal = normalize(normal_metallic_sample.rgb * 2.0 - 1.0);
    float metallic = normal_metallic_sample.a;
    float3 world_position = position_roughness_sample.rgb;
    float roughness = max(position_roughness_sample.a, 0.04);

    float3 v = normalize(camera_pos.xyz - world_position);
    float3 r = reflect(-v, normal);
    float n_dot_v = max(dot(normal, v), 0.0);
    float3 f0 = calculate_f0(albedo, metallic);

    // === Direct lighting ===
    // Key light (sun-like); toned down so the image-based lighting stays the
    // dominant look.
    float3 sun_dir = normalize(light_dir.xyz);
    float3 light_color = float3(1.0, 0.98, 0.95) * 1.6;
    float3 lo = pbr_direct_lighting(normal, v, sun_dir, albedo, metallic, roughness, light_color);

    // Fill light (simple diffuse-only)
    float3 fill_light_dir = normalize(float3(-0.5, -0.3, -1.0));
    float3 fill_light_color = float3(0.3, 0.4, 0.5) * 0.5;
    float fill_n_dot_l = max(dot(normal, fill_light_dir), 0.0);
    float kd_fill = (1.0 - metallic);
    lo = lo + kd_fill * albedo * INV_PI * fill_light_color * fill_n_dot_l;

    // === IBL ambient lighting ===
    float3 f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    float3 ks_ibl = f_ibl;
    float3 kd_ibl = (float3(1.0) - ks_ibl) * (1.0 - metallic);

    // Diffuse IBL from irradiance map
    float3 irradiance = irradiance_map.Sample(ibl_sampler, normal).rgb;
    float3 diffuse_ibl = irradiance * albedo;

    // Specular IBL from pre-filtered environment map + BRDF LUT
    float3 prefiltered_color = prefilter_map.SampleLevel(ibl_sampler, r, roughness * MAX_REFLECTION_LOD_VAL).rgb;
    float2 brdf_sample = brdf_lut.Sample(ibl_sampler, float2(n_dot_v, roughness)).rg;
    float3 specular_ibl = prefiltered_color * (f_ibl * brdf_sample.x + brdf_sample.y);

    float3 ambient = kd_ibl * diffuse_ibl + specular_ibl;

    float3 result_color = ambient + lo;

#ifdef HDR_OUTPUT
    // Linear-HDR target: pass radiance through (encoding here would be
    // decoded a second time by the display).
    result_color = clamp(result_color, float3(0.0), float3(10.0));
#else
    // SDR target: tonemap and gamma correct.
    result_color = tonemap_reinhard(result_color);
    result_color = gamma_correct(result_color);
#endif

    return float4(result_color, 1.0);
}
