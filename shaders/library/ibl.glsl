// RedLilium Shader Library - IBL Module
// Image-based lighting utilities.

#ifndef REDLILIUM_IBL_GLSL
#define REDLILIUM_IBL_GLSL

#include "redlilium/brdf.glsl"

// Maximum mip level for pre-filtered environment map.
const float MAX_REFLECTION_LOD = 4.0;

// Sample diffuse IBL from irradiance cubemap.
// Caller must pass separate textureCube and sampler, combined at usage with samplerCube().
vec3 sample_diffuse_ibl(
    textureCube irradiance_map,
    sampler irradiance_samp,
    vec3 n,
    vec3 albedo
) {
    vec3 irradiance = texture(samplerCube(irradiance_map, irradiance_samp), n).rgb;
    return irradiance * albedo;
}

// Sample specular IBL from pre-filtered environment map.
// Uses split-sum approximation with BRDF LUT.
vec3 sample_specular_ibl(
    textureCube prefilter_map,
    sampler prefilter_samp,
    texture2D brdf_lut,
    sampler brdf_samp,
    vec3 r,
    float n_dot_v,
    vec3 f0,
    float roughness
) {
    // Sample pre-filtered environment at roughness-based mip level
    vec3 prefiltered = textureLod(samplerCube(prefilter_map, prefilter_samp), r, roughness * MAX_REFLECTION_LOD).rgb;

    // Sample BRDF integration LUT
    vec2 brdf = texture(sampler2D(brdf_lut, brdf_samp), vec2(n_dot_v, roughness)).rg;

    // Combine using split-sum approximation
    return prefiltered * (f0 * brdf.x + brdf.y);
}

// Calculate full IBL ambient lighting.
vec3 ibl_ambient(
    textureCube irradiance_map,
    sampler irradiance_samp,
    textureCube prefilter_map,
    sampler prefilter_samp,
    texture2D brdf_lut,
    sampler brdf_samp,
    vec3 n,
    vec3 v,
    vec3 albedo,
    float metallic,
    float roughness
) {
    float n_dot_v = max(dot(n, v), 0.0);
    vec3 r = reflect(-v, n);

    vec3 f0 = calculate_f0(albedo, metallic);
    vec3 f = fresnel_schlick_roughness(n_dot_v, f0, roughness);

    // Energy conservation
    vec3 ks = f;
    vec3 kd = (vec3(1.0) - ks) * (1.0 - metallic);

    // Diffuse IBL
    vec3 diffuse = sample_diffuse_ibl(irradiance_map, irradiance_samp, n, albedo);

    // Specular IBL
    vec3 specular = sample_specular_ibl(
        prefilter_map,
        prefilter_samp,
        brdf_lut,
        brdf_samp,
        r,
        n_dot_v,
        f0,
        roughness
    );

    return kd * diffuse + specular;
}

#endif
