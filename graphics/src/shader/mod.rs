//! Shader composition and library system.
//!
//! This module provides a shader composition system built on [naga_oil](https://github.com/bevyengine/naga_oil),
//! allowing shaders to import reusable modules and reducing code duplication.
//!
//! # Overview
//!
//! The shader system consists of:
//! - [`ShaderComposer`] - Composes shaders with import resolution
//! - [`ShaderLibrary`] - Pre-built collection of reusable shader modules
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::shader::{ShaderComposer, ShaderLibrary};
//!
//! // Create composer with standard library
//! let mut composer = ShaderComposer::new();
//! composer.add_library(&ShaderLibrary::standard())?;
//!
//! // Compose a shader that imports library modules
//! let shader_source = r#"
//! #import redlilium::math
//! #import redlilium::brdf
//!
//! @fragment
//! fn fs_main() -> @location(0) vec4<f32> {
//!     let f = fresnel_schlick(0.5, vec3(0.04));
//!     return vec4(f, 1.0);
//! }
//! "#;
//!
//! let composed = composer.compose(shader_source, &[])?;
//! ```

pub mod library;

use std::collections::HashMap;

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, ComposerError, NagaModuleDescriptor, ShaderDefValue,
};

use crate::error::GraphicsError;
use redlilium_core::profiling::profile_scope;

pub use library::{EGUI_SHADER_SOURCE, ShaderLibrary};

/// Shader composer for resolving imports and composing final shaders.
///
/// The composer maintains a set of importable shader modules and can
/// compose final shaders by resolving `#import` directives.
///
/// # Import Syntax
///
/// Shaders can import modules using the `#import` directive:
///
/// ```wgsl
/// #import redlilium::math
/// #import redlilium::brdf
/// ```
///
/// # Shader Definitions
///
/// Shader definitions allow compile-time conditionals:
///
/// ```wgsl
/// #ifdef HAS_NORMAL_MAP
///     let normal = sample_normal_map(uv);
/// #else
///     let normal = in.world_normal;
/// #endif
/// ```
pub struct ShaderComposer {
    composer: Composer,
}

impl Default for ShaderComposer {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderComposer {
    /// Create a new empty shader composer.
    pub fn new() -> Self {
        Self {
            composer: Composer::default(),
        }
    }

    /// Create a shader composer with the standard library pre-loaded.
    pub fn with_standard_library() -> Result<Self, GraphicsError> {
        let mut composer = Self::new();
        composer.add_library(&ShaderLibrary::standard())?;
        Ok(composer)
    }

    /// Add a shader library to the composer.
    ///
    /// All modules in the library become available for import.
    pub fn add_library(&mut self, library: &ShaderLibrary) -> Result<(), GraphicsError> {
        for (name, source) in library.modules() {
            self.add_module(name, source)?;
        }
        Ok(())
    }

    /// Add a single composable module.
    ///
    /// The module must contain a `#define_import_path` directive.
    ///
    /// # Example
    ///
    /// ```ignore
    /// composer.add_module("my_utils", r#"
    /// #define_import_path my_project::utils
    ///
    /// fn my_helper() -> f32 {
    ///     return 42.0;
    /// }
    /// "#)?;
    /// ```
    pub fn add_module(&mut self, name: &str, source: &str) -> Result<(), GraphicsError> {
        self.composer
            .add_composable_module(ComposableModuleDescriptor {
                source,
                file_path: name,
                ..Default::default()
            })
            .map_err(|e| composer_error_to_graphics_error(e, name))?;

        Ok(())
    }

    /// Compose a shader, resolving all imports.
    ///
    /// Returns the composed WGSL source code with all imports inlined.
    ///
    /// # Arguments
    ///
    /// * `source` - The shader source with `#import` directives
    /// * `shader_defs` - Compile-time definitions for conditionals
    ///
    /// # Example
    ///
    /// ```ignore
    /// let composed = composer.compose(
    ///     my_shader_source,
    ///     &[("MAX_LIGHTS", ShaderDef::Int(8))],
    /// )?;
    /// ```
    pub fn compose(
        &mut self,
        source: &str,
        shader_defs: &[(&str, ShaderDef)],
    ) -> Result<String, GraphicsError> {
        profile_scope!("shader_compose");

        let defs: HashMap<String, ShaderDefValue> = shader_defs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone().into()))
            .collect();

        let naga_module = self
            .composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: "<composed>",
                shader_defs: defs,
                ..Default::default()
            })
            .map_err(|e| composer_error_to_graphics_error(e, "<composed>"))?;

        // Validate the module
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let module_info = validator.validate(&naga_module).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Validation error: {e}"))
        })?;

        // Write back to WGSL
        let wgsl = naga::back::wgsl::write_string(
            &naga_module,
            &module_info,
            naga::back::wgsl::WriterFlags::empty(),
        )
        .map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("WGSL generation error: {e}"))
        })?;

        Ok(wgsl)
    }

    /// Compose a shader and return the raw naga module.
    ///
    /// This is useful when you need direct access to the IR, for example
    /// to generate SPIR-V for Vulkan.
    pub fn compose_to_naga(
        &mut self,
        source: &str,
        shader_defs: &[(&str, ShaderDef)],
    ) -> Result<naga::Module, GraphicsError> {
        profile_scope!("shader_compose_to_naga");

        let defs: HashMap<String, ShaderDefValue> = shader_defs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone().into()))
            .collect();

        self.composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path: "<composed>",
                shader_defs: defs,
                ..Default::default()
            })
            .map_err(|e| composer_error_to_graphics_error(e, "<composed>"))
    }
}

/// Shader definition value for compile-time conditionals.
#[derive(Debug, Clone)]
pub enum ShaderDef {
    /// Boolean definition (`#ifdef`, `#ifndef`).
    Bool(bool),
    /// Integer definition (`#if VAR == 5`).
    Int(i32),
    /// Unsigned integer definition.
    UInt(u32),
}

impl From<ShaderDef> for ShaderDefValue {
    fn from(def: ShaderDef) -> Self {
        match def {
            ShaderDef::Bool(v) => ShaderDefValue::Bool(v),
            ShaderDef::Int(v) => ShaderDefValue::Int(v),
            ShaderDef::UInt(v) => ShaderDefValue::UInt(v),
        }
    }
}

impl From<bool> for ShaderDef {
    fn from(v: bool) -> Self {
        ShaderDef::Bool(v)
    }
}

impl From<i32> for ShaderDef {
    fn from(v: i32) -> Self {
        ShaderDef::Int(v)
    }
}

impl From<u32> for ShaderDef {
    fn from(v: u32) -> Self {
        ShaderDef::UInt(v)
    }
}

/// Convert naga_oil composer error to graphics error.
fn composer_error_to_graphics_error(error: ComposerError, context: &str) -> GraphicsError {
    GraphicsError::ShaderCompilationFailed(format!(
        "Shader composition failed for '{}': {}",
        context, error
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composer_creation() {
        let _composer = ShaderComposer::new();
        // Composer created successfully
    }

    #[test]
    fn test_add_module() {
        let mut composer = ShaderComposer::new();
        let result = composer.add_module(
            "test_module",
            r#"
            #define_import_path test::module

            fn test_fn() -> f32 {
                return 1.0;
            }
            "#,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_compose_with_import() {
        let mut composer = ShaderComposer::new();

        // Add a module - note: in naga_oil 0.17, imported items are accessed
        // via their module path or need explicit re-export
        composer
            .add_module(
                "math_module",
                r#"
                #define_import_path test::math

                const TEST_PI: f32 = 3.14159;

                fn test_saturate(x: f32) -> f32 {
                    return clamp(x, 0.0, 1.0);
                }
                "#,
            )
            .unwrap();

        // Compose a shader that imports it
        // naga_oil uses different syntax - items from imports are inlined
        let result = composer.compose(
            r#"
            #import test::math::{TEST_PI, test_saturate}

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                let x = test_saturate(1.5);
                return vec4<f32>(x, TEST_PI, 0.0, 1.0);
            }
            "#,
            &[],
        );

        assert!(result.is_ok(), "Composition failed: {:?}", result.err());
        let wgsl = result.unwrap();
        // The composed shader should contain the inlined function
        assert!(wgsl.contains("clamp"));
    }

    #[test]
    fn test_standard_library() {
        let composer = ShaderComposer::with_standard_library();
        assert!(
            composer.is_ok(),
            "Standard library failed: {:?}",
            composer.err()
        );
    }

    #[test]
    fn test_compose_with_standard_library() {
        let mut composer = ShaderComposer::with_standard_library()
            .expect("Failed to create composer with library");

        let result = composer.compose(
            r#"
            #import redlilium::math::{PI, saturate}

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                let x = saturate(1.5);
                return vec4<f32>(x, PI, 0.0, 1.0);
            }
            "#,
            &[],
        );

        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_shader_defs() {
        let mut composer = ShaderComposer::new();

        let result = composer.compose(
            r#"
            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                #ifdef USE_RED
                    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
                #else
                    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
                #endif
            }
            "#,
            &[("USE_RED", ShaderDef::Bool(true))],
        );

        assert!(result.is_ok());
    }
}
