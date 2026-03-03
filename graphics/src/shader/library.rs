//! Built-in shader library modules.
//!
//! This module provides the standard RedLilium shader library with
//! common functions for math, PBR lighting, IBL, and color processing.
//!
//! # Shader Files
//!
//! The library modules are stored as `.slang` files in `shaders/library/`:
//! - `math.slang` - Mathematical constants and utilities
//! - `color.slang` - Color space conversions and tone mapping
//! - `brdf.slang` - PBR BRDF functions (Cook-Torrance)
//! - `ibl.slang` - Image-based lighting utilities
//! - `egui.slang` - Complete egui shader with types, utilities, and entry points
//!
//! # Available Modules
//!
//! | Module Name | Description |
//! |-------------|-------------|
//! | `math` | Mathematical constants and utilities |
//! | `color` | Color space conversions and tone mapping |
//! | `brdf` | PBR BRDF functions (Cook-Torrance) |
//! | `ibl` | Image-based lighting utilities |
//!
//! Slang shaders use `import math;` to include library modules.

// =============================================================================
// Shader Module Sources (loaded from files at compile time)
// =============================================================================

/// Mathematical constants and utility functions (Slang).
const MATH_MODULE: &str = include_str!("../../../shaders/library/math.slang");

/// Color space conversions and tone mapping functions (Slang).
const COLOR_MODULE: &str = include_str!("../../../shaders/library/color.slang");

/// PBR BRDF functions (Cook-Torrance microfacet model) (Slang).
const BRDF_MODULE: &str = include_str!("../../../shaders/library/brdf.slang");

/// Image-based lighting utilities (Slang).
const IBL_MODULE: &str = include_str!("../../../shaders/library/ibl.slang");

/// Complete egui shader with vertex and fragment entry points (Slang).
/// Entry points: `vs_main` (vertex) and `fs_main` (fragment).
/// Use `EGUI_SHADER_SOURCE` to access the full shader for rendering.
const EGUI_MODULE: &str = include_str!("../../../shaders/library/egui.slang");

/// Complete egui shader source with vertex and fragment entry points.
/// This is the same as `EGUI_MODULE` but exported for use by the egui renderer.
pub const EGUI_SHADER_SOURCE: &str = EGUI_MODULE;

// =============================================================================
// ShaderLibrary
// =============================================================================

/// Collection of shader modules that can be included via `import` (Slang).
pub struct ShaderLibrary {
    modules: Vec<(&'static str, &'static str)>,
}

impl ShaderLibrary {
    /// Create the standard RedLilium shader library (Slang modules).
    ///
    /// This includes all built-in Slang modules:
    /// - `math` - Mathematical utilities
    /// - `color` - Color processing
    /// - `brdf` - PBR BRDF functions (includes math)
    /// - `ibl` - Image-based lighting (includes brdf)
    pub fn standard_slang() -> Self {
        Self {
            modules: vec![
                ("math", MATH_MODULE),
                ("color", COLOR_MODULE),
                ("brdf", BRDF_MODULE),
                ("ibl", IBL_MODULE),
            ],
        }
    }

    /// Create an empty shader library.
    pub fn empty() -> Self {
        Self {
            modules: Vec::new(),
        }
    }

    /// Get an iterator over all modules (module_name, source).
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
    fn test_standard_slang_library_modules() {
        let library = ShaderLibrary::standard_slang();
        let modules: Vec<_> = library.modules().collect();

        assert_eq!(modules.len(), 4);
        assert!(modules.iter().any(|(name, _)| *name == "math"));
        assert!(modules.iter().any(|(name, _)| *name == "color"));
        assert!(modules.iter().any(|(name, _)| *name == "brdf"));
        assert!(modules.iter().any(|(name, _)| *name == "ibl"));
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
    fn test_module_contents_slang() {
        // Verify that included Slang files contain expected content
        assert!(MATH_MODULE.contains("static const float PI"));
        assert!(MATH_MODULE.contains("float saturate_f"));

        assert!(COLOR_MODULE.contains("float3 tonemap_reinhard"));
        assert!(COLOR_MODULE.contains("float3 gamma_correct"));

        assert!(BRDF_MODULE.contains("float distribution_ggx"));
        assert!(BRDF_MODULE.contains("float3 fresnel_schlick"));

        assert!(IBL_MODULE.contains("float3 ibl_ambient"));

        assert!(EGUI_MODULE.contains("vs_main"));
        assert!(EGUI_MODULE.contains("fs_main"));
    }
}
