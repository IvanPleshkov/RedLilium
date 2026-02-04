// RedLilium Shader Library - BRDF Module
// PBR BRDF functions (Cook-Torrance microfacet model).

#define_import_path redlilium::brdf

#import redlilium::math::{PI, INV_PI, EPSILON, saturate}

/// GGX/Trowbridge-Reitz normal distribution function.
///
/// Describes how microfacets are oriented - higher roughness means more
/// spread out distribution.
fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;

    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

/// Schlick-GGX geometry function for a single direction.
///
/// Describes self-shadowing of microfacets.
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

/// Smith geometry function combining view and light directions.
fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

/// Schlick approximation for Fresnel reflectance.
///
/// f0 is the reflectance at normal incidence (typically 0.04 for dielectrics,
/// or the albedo for metals).
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

/// Schlick Fresnel with roughness factor for IBL.
///
/// Reduces Fresnel effect at glancing angles for rough surfaces.
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

/// Calculate F0 (reflectance at normal incidence) for a material.
///
/// For dielectrics, f0 is typically 0.04.
/// For metals, f0 is the albedo color.
fn calculate_f0(albedo: vec3<f32>, metallic: f32) -> vec3<f32> {
    return mix(vec3<f32>(0.04), albedo, metallic);
}

/// Full Cook-Torrance specular BRDF.
///
/// Returns the specular reflection contribution.
fn cook_torrance_specular(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    h: vec3<f32>,
    roughness: f32,
    f0: vec3<f32>,
) -> vec3<f32> {
    let d = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, l, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);

    let numerator = d * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l + EPSILON;

    return numerator / denominator;
}

/// Lambertian diffuse BRDF.
fn lambertian_diffuse(albedo: vec3<f32>) -> vec3<f32> {
    return albedo * INV_PI;
}

/// Calculate diffuse and specular contributions with energy conservation.
///
/// Returns (kd * diffuse, ks * specular) where kd + ks <= 1.
fn pbr_direct_lighting(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    light_color: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let f0 = calculate_f0(albedo, metallic);

    // Specular BRDF
    let specular = cook_torrance_specular(n, v, l, h, roughness, f0);

    // Fresnel gives us ks (specular contribution)
    let ks = fresnel_schlick(max(dot(h, v), 0.0), f0);

    // kd is what's left, reduced by metallic (metals have no diffuse)
    let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);

    // Combine diffuse and specular
    let n_dot_l = max(dot(n, l), 0.0);
    return (kd * lambertian_diffuse(albedo) + specular) * light_color * n_dot_l;
}
