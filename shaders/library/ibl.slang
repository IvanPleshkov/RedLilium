// RedLilium Shader Library - IBL Module
// Image-based lighting utilities.

import brdf;

// Maximum mip level for pre-filtered environment map.
static const float MAX_REFLECTION_LOD = 4.0;

// Sample diffuse IBL from irradiance cubemap.
float3 sample_diffuse_ibl(
    TextureCube irradiance_map,
    SamplerState irradiance_samp,
    float3 n,
    float3 albedo
) {
    float3 irradiance = irradiance_map.Sample(irradiance_samp, n).rgb;
    return irradiance * albedo;
}

// Sample specular IBL from pre-filtered environment map.
// Uses split-sum approximation with BRDF LUT.
float3 sample_specular_ibl(
    TextureCube prefilter_map,
    SamplerState prefilter_samp,
    Texture2D brdf_lut,
    SamplerState brdf_samp,
    float3 r,
    float n_dot_v,
    float3 f0,
    float roughness
) {
    // Sample pre-filtered environment at roughness-based mip level
    float3 prefiltered = prefilter_map.SampleLevel(prefilter_samp, r, roughness * MAX_REFLECTION_LOD).rgb;

    // Sample BRDF integration LUT
    float2 brdf = brdf_lut.Sample(brdf_samp, float2(n_dot_v, roughness)).rg;

    // Combine using split-sum approximation
    return prefiltered * (f0 * brdf.x + brdf.y);
}

// Calculate full IBL ambient lighting.
float3 ibl_ambient(
    TextureCube irradiance_map,
    SamplerState irradiance_samp,
    TextureCube prefilter_map,
    SamplerState prefilter_samp,
    Texture2D brdf_lut,
    SamplerState brdf_samp,
    float3 n,
    float3 v,
    float3 albedo,
    float metallic,
    float roughness
) {
    float n_dot_v = max(dot(n, v), 0.0);
    float3 r = reflect(-v, n);

    float3 f0 = calculate_f0(albedo, metallic);
    float3 f = fresnel_schlick_roughness(n_dot_v, f0, roughness);

    // Energy conservation
    float3 ks = f;
    float3 kd = (float3(1.0) - ks) * (1.0 - metallic);

    // Diffuse IBL
    float3 diffuse = sample_diffuse_ibl(irradiance_map, irradiance_samp, n, albedo);

    // Specular IBL
    float3 specular = sample_specular_ibl(
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
