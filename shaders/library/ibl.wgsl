// RedLilium Shader Library - IBL Module
// Image-based lighting utilities.

#define_import_path redlilium::ibl

#import redlilium::brdf::{calculate_f0, fresnel_schlick_roughness}

/// Maximum mip level for pre-filtered environment map.
const MAX_REFLECTION_LOD: f32 = 4.0;

/// Sample diffuse IBL from irradiance cubemap.
fn sample_diffuse_ibl(
    irradiance_map: texture_cube<f32>,
    sampler_: sampler,
    n: vec3<f32>,
    albedo: vec3<f32>,
) -> vec3<f32> {
    let irradiance = textureSample(irradiance_map, sampler_, n).rgb;
    return irradiance * albedo;
}

/// Sample specular IBL from pre-filtered environment map.
///
/// Uses split-sum approximation with BRDF LUT.
fn sample_specular_ibl(
    prefilter_map: texture_cube<f32>,
    brdf_lut: texture_2d<f32>,
    sampler_: sampler,
    r: vec3<f32>,
    n_dot_v: f32,
    f0: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    // Sample pre-filtered environment at roughness-based mip level
    let prefiltered = textureSampleLevel(
        prefilter_map,
        sampler_,
        r,
        roughness * MAX_REFLECTION_LOD
    ).rgb;

    // Sample BRDF integration LUT
    let brdf = textureSample(brdf_lut, sampler_, vec2<f32>(n_dot_v, roughness)).rg;

    // Combine using split-sum approximation
    return prefiltered * (f0 * brdf.x + brdf.y);
}

/// Calculate full IBL ambient lighting.
fn ibl_ambient(
    irradiance_map: texture_cube<f32>,
    prefilter_map: texture_cube<f32>,
    brdf_lut: texture_2d<f32>,
    sampler_: sampler,
    n: vec3<f32>,
    v: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let n_dot_v = max(dot(n, v), 0.0);
    let r = reflect(-v, n);

    let f0 = calculate_f0(albedo, metallic);
    let f = fresnel_schlick_roughness(n_dot_v, f0, roughness);

    // Energy conservation
    let ks = f;
    let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);

    // Diffuse IBL
    let diffuse = sample_diffuse_ibl(irradiance_map, sampler_, n, albedo);

    // Specular IBL
    let specular = sample_specular_ibl(
        prefilter_map,
        brdf_lut,
        sampler_,
        r,
        n_dot_v,
        f0,
        roughness
    );

    return kd * diffuse + specular;
}
