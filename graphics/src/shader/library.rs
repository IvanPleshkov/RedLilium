//! Built-in shader library modules.
//!
//! This module provides the standard RedLilium shader library with
//! common functions for math, PBR lighting, IBL, and color processing.
//!
//! # Shader Files
//!
//! The library modules are stored as `.wgsl` files in `shaders/library/`:
//! - `math.wgsl` - Mathematical constants and utilities
//! - `color.wgsl` - Color space conversions and tone mapping
//! - `brdf.wgsl` - PBR BRDF functions (Cook-Torrance)
//! - `ibl.wgsl` - Image-based lighting utilities
//! - `egui.wgsl` - Complete egui shader with types, utilities, and entry points
//!
//! # Available Modules
//!
//! | Import Path | Description |
//! |-------------|-------------|
//! | `redlilium::math` | Mathematical constants and utilities |
//! | `redlilium::color` | Color space conversions and tone mapping |
//! | `redlilium::brdf` | PBR BRDF functions (Cook-Torrance) |
//! | `redlilium::ibl` | Image-based lighting utilities |
//! | `redlilium::egui` | Egui UI rendering utilities |
//!
//! # Example
//!
//! ```wgsl
//! #import redlilium::math::{PI, saturate}
//! #import redlilium::brdf::{fresnel_schlick, distribution_ggx}
//!
//! @fragment
//! fn fs_main() -> @location(0) vec4<f32> {
//!     let f = fresnel_schlick(n_dot_v, f0);
//!     let d = distribution_ggx(n, h, roughness);
//!     // ...
//! }
//! ```

// =============================================================================
// Shader Module Sources (loaded from files at compile time)
// =============================================================================

/// Mathematical constants and utility functions.
const MATH_MODULE: &str = include_str!("../../../shaders/library/math.wgsl");

/// Color space conversions and tone mapping functions.
const COLOR_MODULE: &str = include_str!("../../../shaders/library/color.wgsl");

/// PBR BRDF functions (Cook-Torrance microfacet model).
const BRDF_MODULE: &str = include_str!("../../../shaders/library/brdf.wgsl");

/// Image-based lighting utilities.
const IBL_MODULE: &str = include_str!("../../../shaders/library/ibl.wgsl");

/// Complete egui shader with types, utilities, and entry points.
/// This shader includes both the importable module (`redlilium::egui`) and the entry points.
/// Use `EGUI_SHADER_SOURCE` to access the full shader for rendering.
const EGUI_MODULE: &str = include_str!("../../../shaders/library/egui.wgsl");

/// Complete egui shader source with vertex and fragment entry points.
/// This is the same as `EGUI_MODULE` but exported for use by the egui renderer.
pub const EGUI_SHADER_SOURCE: &str = EGUI_MODULE;

// =============================================================================
// ShaderLibrary
// =============================================================================

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
    /// - `redlilium::egui` - Egui UI rendering utilities
    pub fn standard() -> Self {
        Self {
            modules: vec![
                ("redlilium::math", MATH_MODULE),
                ("redlilium::color", COLOR_MODULE),
                ("redlilium::brdf", BRDF_MODULE),
                ("redlilium::ibl", IBL_MODULE),
                ("redlilium::egui", EGUI_MODULE),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_library_modules() {
        let library = ShaderLibrary::standard();
        let modules: Vec<_> = library.modules().collect();

        assert_eq!(modules.len(), 5);
        assert!(modules.iter().any(|(name, _)| *name == "redlilium::math"));
        assert!(modules.iter().any(|(name, _)| *name == "redlilium::color"));
        assert!(modules.iter().any(|(name, _)| *name == "redlilium::brdf"));
        assert!(modules.iter().any(|(name, _)| *name == "redlilium::ibl"));
        assert!(modules.iter().any(|(name, _)| *name == "redlilium::egui"));
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

    #[test]
    fn test_module_contents() {
        // Verify that included files contain expected content
        assert!(MATH_MODULE.contains("#define_import_path redlilium::math"));
        assert!(MATH_MODULE.contains("const PI"));
        assert!(MATH_MODULE.contains("fn saturate"));

        assert!(COLOR_MODULE.contains("#define_import_path redlilium::color"));
        assert!(COLOR_MODULE.contains("fn tonemap_reinhard"));
        assert!(COLOR_MODULE.contains("fn gamma_correct"));

        assert!(BRDF_MODULE.contains("#define_import_path redlilium::brdf"));
        assert!(BRDF_MODULE.contains("fn distribution_ggx"));
        assert!(BRDF_MODULE.contains("fn fresnel_schlick"));

        assert!(IBL_MODULE.contains("#define_import_path redlilium::ibl"));
        assert!(IBL_MODULE.contains("fn ibl_ambient"));

        assert!(EGUI_MODULE.contains("#define_import_path redlilium::egui"));
        assert!(EGUI_MODULE.contains("struct EguiUniforms"));
        assert!(EGUI_MODULE.contains("fn srgb_to_linear"));
        // Verify egui shader contains entry points (now in same file)
        assert!(EGUI_SHADER_SOURCE.contains("fn vs_main"));
        assert!(EGUI_SHADER_SOURCE.contains("fn fs_main"));
    }
}
