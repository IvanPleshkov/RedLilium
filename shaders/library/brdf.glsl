// RedLilium Shader Library - BRDF Module
// PBR BRDF functions (Cook-Torrance microfacet model).

#ifndef REDLILIUM_BRDF_GLSL
#define REDLILIUM_BRDF_GLSL

#include "redlilium/math.glsl"

// GGX/Trowbridge-Reitz normal distribution function.
// Describes how microfacets are oriented - higher roughness means more spread out distribution.
float distribution_ggx(vec3 n, vec3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = max(dot(n, h), 0.0);
    float n_dot_h2 = n_dot_h * n_dot_h;

    float denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry function for a single direction.
// Describes self-shadowing of microfacets.
float geometry_schlick_ggx(float n_dot_v, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

// Smith geometry function combining view and light directions.
float geometry_smith(vec3 n, vec3 v, vec3 l, float roughness) {
    float n_dot_v = max(dot(n, v), 0.0);
    float n_dot_l = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

// Schlick approximation for Fresnel reflectance.
// f0 is the reflectance at normal incidence (typically 0.04 for dielectrics,
// or the albedo for metals).
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(saturate_f(1.0 - cos_theta), 5.0);
}

// Schlick Fresnel with roughness factor for IBL.
// Reduces Fresnel effect at glancing angles for rough surfaces.
vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(saturate_f(1.0 - cos_theta), 5.0);
}

// Calculate F0 (reflectance at normal incidence) for a material.
// For dielectrics, f0 is typically 0.04.
// For metals, f0 is the albedo color.
vec3 calculate_f0(vec3 albedo, float metallic) {
    return mix(vec3(0.04), albedo, metallic);
}

// Full Cook-Torrance specular BRDF.
// Returns the specular reflection contribution.
vec3 cook_torrance_specular(
    vec3 n,
    vec3 v,
    vec3 l,
    vec3 h,
    float roughness,
    vec3 f0
) {
    float d = distribution_ggx(n, h, roughness);
    float g = geometry_smith(n, v, l, roughness);
    vec3 f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    float n_dot_v = max(dot(n, v), 0.0);
    float n_dot_l = max(dot(n, l), 0.0);

    vec3 numerator = d * g * f;
    float denominator = 4.0 * n_dot_v * n_dot_l + EPSILON;

    return numerator / denominator;
}

// Lambertian diffuse BRDF.
vec3 lambertian_diffuse(vec3 albedo) {
    return albedo * INV_PI;
}

// Calculate diffuse and specular contributions with energy conservation.
// Returns (kd * diffuse, ks * specular) where kd + ks <= 1.
vec3 pbr_direct_lighting(
    vec3 n,
    vec3 v,
    vec3 l,
    vec3 albedo,
    float metallic,
    float roughness,
    vec3 light_color
) {
    vec3 h = normalize(v + l);
    vec3 f0 = calculate_f0(albedo, metallic);

    // Specular BRDF
    vec3 specular = cook_torrance_specular(n, v, l, h, roughness, f0);

    // Fresnel gives us ks (specular contribution)
    vec3 ks = fresnel_schlick(max(dot(h, v), 0.0), f0);

    // kd is what's left, reduced by metallic (metals have no diffuse)
    vec3 kd = (vec3(1.0) - ks) * (1.0 - metallic);

    // Combine diffuse and specular
    float n_dot_l = max(dot(n, l), 0.0);
    return (kd * lambertian_diffuse(albedo) + specular) * light_color * n_dot_l;
}

#endif
