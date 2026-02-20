//! Built-in shader library modules.
//!
//! This module provides the standard RedLilium shader library with
//! common functions for math, PBR lighting, IBL, and color processing.
//!
//! # Shader Files
//!
//! The library modules are stored as `.glsl` files in `shaders/library/`:
//! - `math.glsl` - Mathematical constants and utilities
//! - `color.glsl` - Color space conversions and tone mapping
//! - `brdf.glsl` - PBR BRDF functions (Cook-Torrance)
//! - `ibl.glsl` - Image-based lighting utilities
//! - `egui.glsl` - Complete egui shader with types, utilities, and entry points
//!
//! # Available Modules
//!
//! | Include Path | Description |
//! |-------------|-------------|
//! | `redlilium/math.glsl` | Mathematical constants and utilities |
//! | `redlilium/color.glsl` | Color space conversions and tone mapping |
//! | `redlilium/brdf.glsl` | PBR BRDF functions (Cook-Torrance) |
//! | `redlilium/ibl.glsl` | Image-based lighting utilities |
//!
//! # Example
//!
//! ```glsl
//! #version 450
//! #include "redlilium/math.glsl"
//! #include "redlilium/brdf.glsl"
//!
//! layout(location = 0) out vec4 out_color;
//!
//! void main() {
//!     vec3 f = fresnel_schlick(n_dot_v, f0);
//!     float d = distribution_ggx(n, h, roughness);
//!     // ...
//! }
//! ```

// =============================================================================
// Shader Module Sources (loaded from files at compile time)
// =============================================================================

/// Mathematical constants and utility functions.
const MATH_MODULE: &str = include_str!("../../../shaders/library/math.glsl");

/// Color space conversions and tone mapping functions.
const COLOR_MODULE: &str = include_str!("../../../shaders/library/color.glsl");

/// PBR BRDF functions (Cook-Torrance microfacet model).
const BRDF_MODULE: &str = include_str!("../../../shaders/library/brdf.glsl");

/// Image-based lighting utilities.
const IBL_MODULE: &str = include_str!("../../../shaders/library/ibl.glsl");

/// Complete egui shader with types, utilities, and entry points.
/// This shader includes both vertex and fragment stages via `#ifdef VERTEX` / `#ifdef FRAGMENT`.
/// Use `EGUI_SHADER_SOURCE` to access the full shader for rendering.
const EGUI_MODULE: &str = include_str!("../../../shaders/library/egui.glsl");

/// Complete egui shader source with vertex and fragment entry points.
/// This is the same as `EGUI_MODULE` but exported for use by the egui renderer.
pub const EGUI_SHADER_SOURCE: &str = EGUI_MODULE;

// =============================================================================
// ShaderLibrary
// =============================================================================

/// Collection of shader modules that can be included via `#include`.
pub struct ShaderLibrary {
    modules: Vec<(&'static str, &'static str)>,
}

impl ShaderLibrary {
    /// Create the standard RedLilium shader library.
    ///
    /// This includes all built-in modules:
    /// - `redlilium/math.glsl` - Mathematical utilities
    /// - `redlilium/color.glsl` - Color processing
    /// - `redlilium/brdf.glsl` - PBR BRDF functions
    /// - `redlilium/ibl.glsl` - Image-based lighting
    pub fn standard() -> Self {
        Self {
            modules: vec![
                ("redlilium/math.glsl", MATH_MODULE),
                ("redlilium/color.glsl", COLOR_MODULE),
                ("redlilium/brdf.glsl", BRDF_MODULE),
                ("redlilium/ibl.glsl", IBL_MODULE),
            ],
        }
    }

    /// Create an empty shader library.
    pub fn empty() -> Self {
        Self {
            modules: Vec::new(),
        }
    }

    /// Get an iterator over all modules (include_path, source).
    pub fn modules(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.modules.iter().copied()
    }

    /// Add a custom module to the library.
    pub fn with_module(mut self, path: &'static str, source: &'static str) -> Self {
        self.modules.push((path, source));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_library_modules() {
        let library = ShaderLibrary::standard();
        let modules: Vec<_> = library.modules().collect();

        assert_eq!(modules.len(), 4);
        assert!(
            modules
                .iter()
                .any(|(name, _)| *name == "redlilium/math.glsl")
        );
        assert!(
            modules
                .iter()
                .any(|(name, _)| *name == "redlilium/color.glsl")
        );
        assert!(
            modules
                .iter()
                .any(|(name, _)| *name == "redlilium/brdf.glsl")
        );
        assert!(
            modules
                .iter()
                .any(|(name, _)| *name == "redlilium/ibl.glsl")
        );
    }

    #[test]
    fn test_empty_library() {
        let library = ShaderLibrary::empty();
        assert_eq!(library.modules().count(), 0);
    }

    #[test]
    fn test_custom_module() {
        let library =
            ShaderLibrary::empty().with_module("custom/module.glsl", "float foo() { return 1.0; }");
        assert_eq!(library.modules().count(), 1);
    }

    #[test]
    fn test_module_contents() {
        // Verify that included files contain expected GLSL content
        assert!(MATH_MODULE.contains("const float PI"));
        assert!(MATH_MODULE.contains("float saturate_f"));

        assert!(COLOR_MODULE.contains("vec3 tonemap_reinhard"));
        assert!(COLOR_MODULE.contains("vec3 gamma_correct"));

        assert!(BRDF_MODULE.contains("float distribution_ggx"));
        assert!(BRDF_MODULE.contains("vec3 fresnel_schlick"));

        assert!(IBL_MODULE.contains("vec3 ibl_ambient"));

        assert!(EGUI_MODULE.contains("void main()"));
        assert!(EGUI_MODULE.contains("#ifdef VERTEX"));
        assert!(EGUI_MODULE.contains("#ifdef FRAGMENT"));
    }
}
