//! Built-in shader library modules.
//!
//! This module provides the standard RedLilium shader library with
//! common functions for math, PBR lighting, IBL, and color processing.
//!
//! # Available Modules
//!
//! | Import Path | Description |
//! |-------------|-------------|
//! | `redlilium::math` | Mathematical constants and utilities |
//! | `redlilium::color` | Color space conversions and tone mapping |
//! | `redlilium::brdf` | PBR BRDF functions (Cook-Torrance) |
//! | `redlilium::ibl` | Image-based lighting utilities |
//!
//! # Example
//!
//! ```wgsl
//! #import redlilium::math
//! #import redlilium::brdf
//!
//! @fragment
//! fn fs_main() -> @location(0) vec4<f32> {
//!     let f = fresnel_schlick(n_dot_v, f0);
//!     let d = distribution_ggx(n, h, roughness);
//!     // ...
//! }
//! ```

/// Collection of shader modules that can be imported.
pub struct ShaderLibrary {
    modules: Vec<(&'static str, &'static str)>,
}

impl ShaderLibrary {
    /// Create the standard RedLilium shader library.
    ///
    /// This includes all built-in modules:
    /// - `redlilium::math` - Mathematical utilities
    /// - `redlilium::color` - Color processing
    /// - `redlilium::brdf` - PBR BRDF functions
    /// - `redlilium::ibl` - Image-based lighting
    pub fn standard() -> Self {
        Self {
            modules: vec![
                ("redlilium::math", MATH_MODULE),
                ("redlilium::color", COLOR_MODULE),
                ("redlilium::brdf", BRDF_MODULE),
                ("redlilium::ibl", IBL_MODULE),
            ],
        }
    }

    /// Create an empty shader library.
    pub fn empty() -> Self {
        Self {
            modules: Vec::new(),
        }
    }

    /// Get an iterator over all modules (name, source).
    pub fn modules(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.modules.iter().copied()
    }

    /// Add a custom module to the library.
    pub fn with_module(mut self, name: &'static str, source: &'static str) -> Self {
        self.modules.push((name, source));
        self
    }
}

// =============================================================================
// Math Module
// =============================================================================

/// Mathematical constants and utility functions.
const MATH_MODULE: &str = r#"
#define_import_path redlilium::math

/// Pi constant.
const PI: f32 = 3.14159265359;

/// Tau (2 * Pi) constant.
const TAU: f32 = 6.28318530718;

/// Inverse of Pi.
const INV_PI: f32 = 0.31830988618;

/// Inverse of Tau.
const INV_TAU: f32 = 0.15915494309;

/// Small epsilon for avoiding division by zero.
const EPSILON: f32 = 0.0001;

/// Clamp a value to [0, 1].
fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

/// Clamp a vec2 to [0, 1].
fn saturate2(v: vec2<f32>) -> vec2<f32> {
    return clamp(v, vec2<f32>(0.0), vec2<f32>(1.0));
}

/// Clamp a vec3 to [0, 1].
fn saturate3(v: vec3<f32>) -> vec3<f32> {
    return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Clamp a vec4 to [0, 1].
fn saturate4(v: vec4<f32>) -> vec4<f32> {
    return clamp(v, vec4<f32>(0.0), vec4<f32>(1.0));
}

/// Linear interpolation.
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

/// Linear interpolation for vec3.
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

/// Smoothstep function.
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

/// Square of a value.
fn sq(x: f32) -> f32 {
    return x * x;
}

/// Square of a vec3.
fn sq3(v: vec3<f32>) -> vec3<f32> {
    return v * v;
}

/// Safe normalize that handles zero vectors.
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len > EPSILON {
        return v / len;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}
"#;

// =============================================================================
// Color Module
// =============================================================================

/// Color space conversions and tone mapping functions.
const COLOR_MODULE: &str = r#"
#define_import_path redlilium::color

#import redlilium::math::{saturate3, lerp3}

/// Convert linear RGB to sRGB.
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = pow(color, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

/// Convert sRGB to linear RGB.
fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    let low = color / 12.92;
    let high = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, color <= vec3<f32>(0.04045));
}

/// Simple gamma correction (gamma = 2.2).
fn gamma_correct(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / 2.2));
}

/// Remove gamma correction (gamma = 2.2).
fn gamma_uncorrect(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(2.2));
}

/// Reinhard tone mapping.
fn tonemap_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

/// Reinhard extended tone mapping with white point.
fn tonemap_reinhard_extended(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let white_sq = white_point * white_point;
    let numerator = color * (vec3<f32>(1.0) + color / white_sq);
    return numerator / (vec3<f32>(1.0) + color);
}

/// ACES filmic tone mapping (approximation).
fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate3((color * (a * color + b)) / (color * (c * color + d) + e));
}

/// Uncharted 2 tone mapping curve.
fn tonemap_uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

/// Uncharted 2 tone mapping.
fn tonemap_uncharted2(color: vec3<f32>) -> vec3<f32> {
    let exposure_bias = 2.0;
    let W = vec3<f32>(11.2);
    let curr = tonemap_uncharted2_partial(color * exposure_bias);
    let white_scale = vec3<f32>(1.0) / tonemap_uncharted2_partial(W);
    return curr * white_scale;
}

/// Luminance of a linear RGB color.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Desaturate a color by a factor (0 = original, 1 = grayscale).
fn desaturate(color: vec3<f32>, factor: f32) -> vec3<f32> {
    let gray = luminance(color);
    return lerp3(color, vec3<f32>(gray), factor);
}
"#;

// =============================================================================
// BRDF Module
// =============================================================================

/// PBR BRDF functions (Cook-Torrance microfacet model).
const BRDF_MODULE: &str = r#"
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
"#;

// =============================================================================
// IBL Module
// =============================================================================

/// Image-based lighting utilities.
const IBL_MODULE: &str = r#"
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
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_library_modules() {
        let library = ShaderLibrary::standard();
        let modules: Vec<_> = library.modules().collect();

        assert_eq!(modules.len(), 4);
        assert!(modules.iter().any(|(name, _)| *name == "redlilium::math"));
        assert!(modules.iter().any(|(name, _)| *name == "redlilium::color"));
        assert!(modules.iter().any(|(name, _)| *name == "redlilium::brdf"));
        assert!(modules.iter().any(|(name, _)| *name == "redlilium::ibl"));
    }

    #[test]
    fn test_empty_library() {
        let library = ShaderLibrary::empty();
        assert_eq!(library.modules().count(), 0);
    }

    #[test]
    fn test_custom_module() {
        let library = ShaderLibrary::empty().with_module(
            "custom::module",
            "#define_import_path custom::module\nfn foo() -> f32 { return 1.0; }",
        );
        assert_eq!(library.modules().count(), 1);
    }
}
