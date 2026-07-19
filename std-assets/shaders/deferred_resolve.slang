// Standard deferred resolve/lighting pass (#144) — reads the G-buffer and
// applies direct + IBL lighting to produce the lit image. Drawn as a
// fullscreen triangle over the skybox background (background pixels are
// discarded via the G-buffer's albedo-alpha geometry marker).
//
// Direct lights arrive per frame in the uniforms, gathered from the ECS
// light components (#146) — fixed-capacity arrays, counts in
// `light_counts`. Without lights the image is IBL-only. Shadows arrive
// with #130; spot lights are not consumed yet.
//
// Output-transform axes, selected by the pipeline from the camera target's
// format (same scheme as the egui shader):
//   HDR_OUTPUT       - raw linear for a linear-HDR target (1.0 = SDR white)
//   SRGB_FRAMEBUFFER - tonemapped linear for an sRGB-typed target (the
//                      hardware applies the linear->sRGB encode on write;
//                      encoding manually would double-gamma the image)
//   (both off)       - tonemap + manual gamma encode for a non-sRGB SDR target
//#pragma variant_system HDR_OUTPUT
//#pragma variant_system SRGB_FRAMEBUFFER

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

// Fixed uniform-array capacities; the CPU side truncates beyond these
// (MAX_*_LIGHTS in ecs/src/std/rendering/deferred.rs must match).
static const uint MAX_DIRECTIONAL_LIGHTS = 4;
static const uint MAX_POINT_LIGHTS = 16;

struct DirectionalLightData {
    float4 dir_to_light; // xyz = normalized direction toward the light
    float4 color;        // rgb = linear color premultiplied by intensity
};

struct PointLightData {
    float4 position_range; // xyz = world position, w = range (0 = unbounded)
    float4 color;          // rgb = linear color premultiplied by intensity
};

[[vk::binding(0, 0)]]
cbuffer ResolveUniforms {
    float4 camera_pos;
    uint4 light_counts;  // x = directional count, y = point count
    DirectionalLightData dir_lights[MAX_DIRECTIONAL_LIGHTS];
    PointLightData point_lights[MAX_POINT_LIGHTS];
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

// Highest prefiltered-environment mip the roughness axis maps onto. Must be
// the baked pack's mip count minus one (`PREFILTER_MIPS` in xtask/src/ibl.rs
// bakes `roughness = mip / (mips - 1)`); a lower value makes rough surfaces
// sample too sharp a reflection. Rides the environment asset with #145.
static const float MAX_REFLECTION_LOD_VAL = 7.0;

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    // Sample G-buffer
    float4 albedo_sample = gbuffer_albedo.Sample(gbuffer_sampler, input.uv);
    float4 normal_metallic_sample = gbuffer_normal_metallic.Sample(gbuffer_sampler, input.uv);
    float4 position_roughness_sample = gbuffer_position_roughness.Sample(gbuffer_sampler, input.uv);

    // RT0 alpha is the AO-as-geometry-marker (ADR-039): 0 = background (the
    // clear value — keep whatever the skybox pass wrote), otherwise
    // max(AO, 1/64) written by the G-buffer shaders.
    if (albedo_sample.a < 1.0 / 128.0) {
        discard;
    }
    float ao = albedo_sample.a;

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

    // Ambient occlusion attenuates the image-based terms only — direct
    // lights are shadowed by real shadows (#130), not baked occlusion.
    float3 ambient = (kd_ibl * diffuse_ibl + specular_ibl) * ao;

    float3 result_color = ambient + lo;

#ifdef HDR_OUTPUT
    // Linear-HDR target: pass radiance through (encoding here would be
    // decoded a second time by the display).
    result_color = clamp(result_color, float3(0.0), float3(10.0));
#else
    // SDR target: tonemap (Khronos PBR Neutral — hue-preserving, so
    // saturated albedos don't wash toward white); gamma-encode manually only
    // when the target is not sRGB-typed (an sRGB texture encodes on write by
    // itself).
    result_color = tonemap_pbr_neutral(result_color);
#ifndef SRGB_FRAMEBUFFER
    // Exact sRGB OETF — the same transfer the hardware applies on sRGB
    // targets, so both SDR paths encode identically.
    result_color = linear_to_srgb(result_color);
#endif
#endif

    return float4(result_color, 1.0);
}
