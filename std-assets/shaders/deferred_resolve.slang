// Standard deferred resolve/lighting pass (#144) — reads the G-buffer and
// applies direct + IBL lighting to produce the lit image. Drawn as a
// fullscreen triangle over the skybox background (background pixels are
// discarded via the G-buffer's albedo-alpha geometry marker).
//
// Direct lights arrive per frame in the uniforms, gathered from the ECS
// light components (#146) — fixed-capacity arrays, counts in
// `light_counts` (x directional, y point, z spot). Without lights the image
// is IBL-only. Shadows arrive with #130.
//
// World position is reconstructed from the depth buffer (full 32-bit
// precision) instead of a G-buffer position target — see reconstruct_world
// in the motion module.
//
// Output is scene-referred linear radiance (1.0 = SDR white) into the
// `scene_color` intermediate (#142) — exposure, tonemap, and display
// encoding all live in display_output.slang.

import math;
import brdf;
import motion; // reconstruct_world (position from depth)

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

// Fixed uniform-array capacities; the CPU side truncates beyond these
// (MAX_*_LIGHTS in ecs/src/std/rendering/deferred.rs must match).
static const uint MAX_DIRECTIONAL_LIGHTS = 4;
static const uint MAX_POINT_LIGHTS = 16;
static const uint MAX_SPOT_LIGHTS = 8;

struct DirectionalLightData {
    float4 dir_to_light; // xyz = normalized direction toward the light
    float4 color;        // rgb = linear color premultiplied by intensity
};

struct PointLightData {
    float4 position_range; // xyz = world position, w = range (0 = unbounded)
    float4 color;          // rgb = linear color premultiplied by intensity
};

struct SpotLightData {
    float4 position_range; // xyz = world position, w = range (0 = unbounded)
    float4 direction_cos;  // xyz = direction the light travels, w = cos(outer)
    float4 color_cos;      // rgb = premultiplied color, a = cos(inner)
};

[[vk::binding(0, 0)]]
cbuffer ResolveUniforms {
    // Inverse of the view-projection the G-buffer was rasterized with (the
    // jittered one under TAA) — reconstructs world position from depth.
    column_major float4x4 inv_view_proj;
    // xyz = camera world position; w = max reflection LOD (highest prefilter
    // mip index of the bound environment, #145 — 0 for the fallback).
    float4 camera_pos;
    uint4 light_counts;  // x = directional, y = point, z = spot count
    DirectionalLightData dir_lights[MAX_DIRECTIONAL_LIGHTS];
    PointLightData point_lights[MAX_POINT_LIGHTS];
    SpotLightData spot_lights[MAX_SPOT_LIGHTS];
};

// G-buffer textures
[[vk::binding(0, 1)]]
Texture2D gbuffer_albedo;
[[vk::binding(1, 1)]]
Texture2D gbuffer_normal_metallic;
[[vk::binding(2, 1)]]
Texture2D gbuffer_roughness;
[[vk::binding(3, 1)]]
SamplerState gbuffer_sampler;
// The camera depth buffer (reversed-Z), an unfilterable-float binding read
// via Load — see load_texel_r. World position = reconstruct_world(depth).
[[vk::binding(4, 1)]]
Texture2D<float> depth_texture;

// IBL textures
[[vk::binding(0, 2)]]
TextureCube irradiance_map;
[[vk::binding(1, 2)]]
TextureCube prefilter_map;
[[vk::binding(2, 2)]]
Texture2D brdf_lut;
[[vk::binding(3, 2)]]
SamplerState ibl_sampler;

// Screen-space ambient occlusion (#150): the denoised SSAO factor, or a 1x1
// white texture (AO = 1) when the camera has no CameraAmbientOcclusion — so
// this pass is unconditional and the no-AO image is byte-identical.
[[vk::binding(0, 3)]]
Texture2D ssao_texture;
[[vk::binding(1, 3)]]
SamplerState ssao_sampler;


[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    // Sample G-buffer
    float4 albedo_sample = gbuffer_albedo.Sample(gbuffer_sampler, input.uv);
    float4 normal_metallic_sample = gbuffer_normal_metallic.Sample(gbuffer_sampler, input.uv);

    // RT0 alpha is the AO-as-geometry-marker (ADR-039): 0 = background (the
    // clear value — keep whatever the skybox pass wrote), otherwise
    // max(AO, 1/64) written by the G-buffer shaders.
    if (albedo_sample.a < 1.0 / 128.0) {
        discard;
    }
    // Material AO (baked, RT0.a) combined with screen-space AO (#150). The
    // SSAO texture is 1x1 white when the camera has no SSAO, so this is a
    // no-op there.
    float ao = albedo_sample.a * ssao_texture.Sample(ssao_sampler, input.uv).r;

    // Decode G-buffer; world position reconstructed from the depth buffer.
    float3 albedo = albedo_sample.rgb;
    float3 normal = normalize(normal_metallic_sample.rgb * 2.0 - 1.0);
    float metallic = normal_metallic_sample.a;
    float depth = load_texel_r(depth_texture, input.uv);
    float3 world_position = reconstruct_world(inv_view_proj, input.uv, depth);
    float roughness = max(gbuffer_roughness.Sample(gbuffer_sampler, input.uv).r, 0.04);

    float3 v = normalize(camera_pos.xyz - world_position);
    float3 r = reflect(-v, normal);
    float n_dot_v = max(dot(normal, v), 0.0);
    float3 f0 = calculate_f0(albedo, metallic);

    // === Direct lighting (ECS light components, #146) ===
    float3 lo = float3(0.0);
    for (uint i = 0; i < light_counts.x; i++) {
        lo += pbr_direct_lighting(
            normal, v, dir_lights[i].dir_to_light.xyz,
            albedo, metallic, roughness, dir_lights[i].color.rgb);
    }
    for (uint i = 0; i < light_counts.y; i++) {
        float3 to_light = point_lights[i].position_range.xyz - world_position;
        float dist_sq = max(dot(to_light, to_light), 1e-6);
        float dist = sqrt(dist_sq);
        float3 l = to_light / dist;
        // Inverse-square falloff (intensity = radiance at unit distance),
        // smoothly windowed to zero at `range` (UE-style (1-(d/r)^4)^2) so a
        // bounded light has no hard cutoff edge.
        float attenuation = 1.0 / dist_sq;
        float range = point_lights[i].position_range.w;
        if (range > 0.0) {
            float q = dist / range;
            q *= q;
            q *= q;
            float window = saturate(1.0 - q);
            attenuation *= window * window;
        }
        lo += pbr_direct_lighting(
            normal, v, l, albedo, metallic, roughness,
            point_lights[i].color.rgb * attenuation);
    }
    for (uint i = 0; i < light_counts.z; i++) {
        float3 to_light = spot_lights[i].position_range.xyz - world_position;
        float dist_sq = max(dot(to_light, to_light), 1e-6);
        float dist = sqrt(dist_sq);
        float3 l = to_light / dist;
        // Same inverse-square + range window as point lights.
        float attenuation = 1.0 / dist_sq;
        float range = spot_lights[i].position_range.w;
        if (range > 0.0) {
            float q = dist / range;
            q *= q;
            q *= q;
            float window = saturate(1.0 - q);
            attenuation *= window * window;
        }
        // Cone falloff (KHR_lights_punctual): linear ramp in the cosine of
        // the angle between the spot direction and the fragment, from zero at
        // the outer cone to full inside the inner cone.
        float cos_outer = spot_lights[i].direction_cos.w;
        float cos_inner = spot_lights[i].color_cos.a;
        float cd = dot(-l, spot_lights[i].direction_cos.xyz);
        attenuation *= saturate((cd - cos_outer) / max(cos_inner - cos_outer, 1e-4));
        lo += pbr_direct_lighting(
            normal, v, l, albedo, metallic, roughness,
            spot_lights[i].color_cos.rgb * attenuation);
    }

    // === IBL ambient lighting ===
    float3 f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    float3 ks_ibl = f_ibl;
    float3 kd_ibl = (float3(1.0) - ks_ibl) * (1.0 - metallic);

    // Diffuse IBL from irradiance map
    float3 irradiance = irradiance_map.Sample(ibl_sampler, normal).rgb;
    float3 diffuse_ibl = irradiance * albedo;

    // Specular IBL from pre-filtered environment map + BRDF LUT
    // Highest prefilter mip index rides the environment asset (#145):
    // camera_pos.w = mip_count - 1 of the bound prefilter chain.
    float3 prefiltered_color = prefilter_map.SampleLevel(ibl_sampler, r, roughness * camera_pos.w).rgb;
    float2 brdf_sample = brdf_lut.Sample(ibl_sampler, float2(n_dot_v, roughness)).rg;
    float3 specular_ibl = prefiltered_color * (f_ibl * brdf_sample.x + brdf_sample.y);

    // Ambient occlusion attenuates the image-based terms only — direct
    // lights are shadowed by real shadows (#130), not baked occlusion.
    float3 ambient = (kd_ibl * diffuse_ibl + specular_ibl) * ao;

    // Scene-referred linear out — display transform happens downstream.
    // Sanitize the write: a bad prefilter/irradiance texel or a large finite
    // radiance (bright point light, grazing specular) can overflow the f16
    // scene_color target to Inf and poison the bloom/TAA/composite chain (#152).
    // Clamp keeps scene_color finite at the source, same invariant as skybox.
    return float4(sanitize_finite3(ambient + lo), 1.0);
}
