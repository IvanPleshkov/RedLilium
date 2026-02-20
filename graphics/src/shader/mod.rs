//! Shader composition and library system.
//!
//! This module provides a GLSL shader composition system with `#include` resolution
//! and compile-time shader definitions via `#ifdef` / `#define`.
//!
//! # Overview
//!
//! The shader system consists of:
//! - [`ShaderComposer`] - Composes GLSL shaders with include resolution
//! - [`ShaderLibrary`] - Pre-built collection of reusable shader modules
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::shader::{ShaderComposer, ShaderLibrary};
//! use redlilium_graphics::ShaderStage;
//!
//! // Create composer with standard library
//! let mut composer = ShaderComposer::with_standard_library();
//!
//! // Compose a fragment shader that includes library modules
//! let shader_source = r#"
//! #version 450
//! #include "redlilium/math.glsl"
//! #include "redlilium/brdf.glsl"
//!
//! layout(location = 0) out vec4 out_color;
//!
//! void main() {
//!     vec3 f = fresnel_schlick(0.5, vec3(0.04));
//!     out_color = vec4(f, 1.0);
//! }
//! "#;
//!
//! let composed = composer.compose(shader_source, ShaderStage::Fragment, &[])?;
//! ```

pub mod library;

use std::collections::{HashMap, HashSet};

use crate::error::GraphicsError;
use crate::materials::ShaderStage;
use redlilium_core::profiling::profile_scope;

pub use library::{EGUI_SHADER_SOURCE, ShaderLibrary};

/// Shader composer for resolving includes and composing final GLSL shaders.
///
/// The composer maintains a set of includable shader modules and can
/// compose final shaders by resolving `#include` directives, then
/// parsing the resulting GLSL through naga to produce WGSL output
/// for the GPU backends.
///
/// # Include Syntax
///
/// Shaders can include library modules using the `#include` directive:
///
/// ```glsl
/// #include "redlilium/math.glsl"
/// #include "redlilium/brdf.glsl"
/// ```
///
/// # Shader Definitions
///
/// Shader definitions allow compile-time conditionals:
///
/// ```glsl
/// #ifdef HAS_NORMAL_MAP
///     vec3 normal = sample_normal_map(uv);
/// #else
///     vec3 normal = in_world_normal;
/// #endif
/// ```
///
/// # Multi-Stage Shaders
///
/// A single GLSL file can contain both vertex and fragment shaders
/// using `#ifdef VERTEX` / `#ifdef FRAGMENT` blocks. The composer
/// automatically defines the appropriate stage macro.
pub struct ShaderComposer {
    /// Registered include sources: path -> source text.
    includes: HashMap<String, String>,
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
            includes: HashMap::new(),
        }
    }

    /// Create a shader composer with the standard library pre-loaded.
    pub fn with_standard_library() -> Self {
        let mut composer = Self::new();
        composer.add_library(&ShaderLibrary::standard());
        composer
    }

    /// Add a shader library to the composer.
    ///
    /// All modules in the library become available for `#include`.
    pub fn add_library(&mut self, library: &ShaderLibrary) {
        for (path, source) in library.modules() {
            self.register_include(path, source);
        }
    }

    /// Register a single include source.
    ///
    /// The path is what appears in `#include "path"` directives.
    ///
    /// # Example
    ///
    /// ```ignore
    /// composer.register_include("my_project/utils.glsl", r#"
    /// float my_helper() {
    ///     return 42.0;
    /// }
    /// "#);
    /// ```
    pub fn register_include(&mut self, path: &str, source: &str) {
        self.includes.insert(path.to_string(), source.to_string());
    }

    /// Resolve `#include` directives in a GLSL source.
    ///
    /// Returns the GLSL with all includes expanded but no further
    /// processing (no naga parsing, no WGSL conversion).
    /// The caller is responsible for passing the result to a backend
    /// compiler (shaderc for Vulkan, naga for wgpu).
    pub fn resolve_glsl(&self, source: &str) -> Result<String, GraphicsError> {
        let mut included = HashSet::new();
        self.resolve_includes(source, &mut included)
    }

    /// Build the defines list for a given stage and user shader defs.
    ///
    /// Returns a `Vec<(name, value)>` suitable for passing to
    /// [`ShaderSource::glsl()`] or backend compilers directly.
    /// The stage define (VERTEX, FRAGMENT, COMPUTE) is included automatically.
    pub fn build_defines(
        stage: ShaderStage,
        shader_defs: &[(&str, ShaderDef)],
    ) -> Vec<(String, String)> {
        let mut defines = Vec::new();

        // Stage define
        match stage {
            ShaderStage::Vertex => defines.push(("VERTEX".into(), String::new())),
            ShaderStage::Fragment => defines.push(("FRAGMENT".into(), String::new())),
            ShaderStage::Compute => defines.push(("COMPUTE".into(), String::new())),
        }

        // User shader defs
        for (name, def) in shader_defs {
            match def {
                ShaderDef::Bool(true) => defines.push((name.to_string(), String::new())),
                ShaderDef::Bool(false) => { /* omit */ }
                ShaderDef::Int(v) => defines.push((name.to_string(), v.to_string())),
                ShaderDef::UInt(v) => defines.push((name.to_string(), v.to_string())),
            }
        }

        defines
    }

    /// Compose a GLSL shader, resolving all includes.
    ///
    /// Returns composed WGSL source code ready for GPU backends.
    /// The GLSL is parsed through naga and converted to WGSL.
    ///
    /// # Arguments
    ///
    /// * `source` - The GLSL shader source with `#include` directives
    /// * `stage` - The shader stage (Vertex, Fragment, or Compute)
    /// * `shader_defs` - Compile-time definitions for conditionals
    ///
    /// # Example
    ///
    /// ```ignore
    /// let composed = composer.compose(
    ///     my_shader_source,
    ///     ShaderStage::Fragment,
    ///     &[("MAX_LIGHTS", ShaderDef::Int(8))],
    /// )?;
    /// ```
    pub fn compose(
        &self,
        source: &str,
        stage: ShaderStage,
        shader_defs: &[(&str, ShaderDef)],
    ) -> Result<String, GraphicsError> {
        profile_scope!("shader_compose");

        let naga_module = self.compose_to_naga(source, stage, shader_defs)?;

        // Validate the module
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let module_info = validator.validate(&naga_module).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Validation error: {e}"))
        })?;

        // Write to WGSL
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

    /// Compose a GLSL shader and return the raw naga module.
    ///
    /// This is useful when you need direct access to the IR, for example
    /// to generate SPIR-V for Vulkan.
    pub fn compose_to_naga(
        &self,
        source: &str,
        stage: ShaderStage,
        shader_defs: &[(&str, ShaderDef)],
    ) -> Result<naga::Module, GraphicsError> {
        profile_scope!("shader_compose_to_naga");

        // Resolve all #include directives
        let mut included = HashSet::new();
        let resolved = self.resolve_includes(source, &mut included)?;

        // Build defines map for naga's preprocessor
        let mut defines = naga::FastHashMap::default();

        // Stage define
        match stage {
            ShaderStage::Vertex => {
                defines.insert("VERTEX".to_string(), String::new());
            }
            ShaderStage::Fragment => {
                defines.insert("FRAGMENT".to_string(), String::new());
            }
            ShaderStage::Compute => {
                defines.insert("COMPUTE".to_string(), String::new());
            }
        }

        // User shader defs
        for (name, def) in shader_defs {
            match def {
                ShaderDef::Bool(true) => {
                    defines.insert(name.to_string(), String::new());
                }
                ShaderDef::Bool(false) => {
                    // Omit — not defined
                }
                ShaderDef::Int(v) => {
                    defines.insert(name.to_string(), v.to_string());
                }
                ShaderDef::UInt(v) => {
                    defines.insert(name.to_string(), v.to_string());
                }
            }
        }

        // Parse GLSL
        let naga_stage = match stage {
            ShaderStage::Vertex => naga::ShaderStage::Vertex,
            ShaderStage::Fragment => naga::ShaderStage::Fragment,
            ShaderStage::Compute => naga::ShaderStage::Compute,
        };

        let options = naga::front::glsl::Options {
            stage: naga_stage,
            defines,
        };

        let mut frontend = naga::front::glsl::Frontend::default();
        let module = frontend.parse(&options, &resolved).map_err(|errors| {
            GraphicsError::ShaderCompilationFailed(format!("GLSL parse error:\n{errors}"))
        })?;

        Ok(module)
    }

    /// Resolve `#include "path"` directives recursively.
    fn resolve_includes(
        &self,
        source: &str,
        included: &mut HashSet<String>,
    ) -> Result<String, GraphicsError> {
        let mut result = String::with_capacity(source.len());

        for line in source.lines() {
            let trimmed = line.trim();
            if let Some(path) = parse_include_directive(trimmed) {
                // Skip if already included (prevent double-inclusion)
                if included.contains(path) {
                    continue;
                }
                included.insert(path.to_string());

                let include_source = self.includes.get(path).ok_or_else(|| {
                    GraphicsError::ShaderCompilationFailed(format!("Include not found: \"{path}\""))
                })?;

                // Recursively resolve includes in the included file
                let resolved = self.resolve_includes(include_source, included)?;
                result.push_str(&resolved);
                result.push('\n');
            } else {
                result.push_str(line);
                result.push('\n');
            }
        }

        Ok(result)
    }
}

/// Parse a `#include "path"` directive, returning the path if found.
fn parse_include_directive(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("#include")?;
    let rest = rest.trim();
    // Support both #include "path" and #include <path>
    if let Some(inner) = rest.strip_prefix('"') {
        inner.strip_suffix('"')
    } else if let Some(inner) = rest.strip_prefix('<') {
        inner.strip_suffix('>')
    } else {
        None
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composer_creation() {
        let _composer = ShaderComposer::new();
    }

    #[test]
    fn test_register_include() {
        let mut composer = ShaderComposer::new();
        composer.register_include("test/module.glsl", "float test_fn() { return 1.0; }");
        assert!(composer.includes.contains_key("test/module.glsl"));
    }

    #[test]
    fn test_include_resolution() {
        let mut composer = ShaderComposer::new();
        composer.register_include(
            "test/math.glsl",
            "float my_saturate(float x) { return clamp(x, 0.0, 1.0); }",
        );

        let source = r#"#version 450
#include "test/math.glsl"

layout(location = 0) out vec4 out_color;

void main() {
    float x = my_saturate(1.5);
    out_color = vec4(x, 0.0, 0.0, 1.0);
}
"#;

        let result = composer.compose(source, ShaderStage::Fragment, &[]);
        assert!(result.is_ok(), "Composition failed: {:?}", result.err());
    }

    #[test]
    fn test_compose_with_standard_library() {
        let composer = ShaderComposer::with_standard_library();

        let source = r#"#version 450
#include "redlilium/math.glsl"

layout(location = 0) out vec4 out_color;

void main() {
    float x = saturate_f(1.5);
    out_color = vec4(x, PI, 0.0, 1.0);
}
"#;

        let result = composer.compose(source, ShaderStage::Fragment, &[]);
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_shader_defs() {
        let composer = ShaderComposer::new();

        let source = r#"#version 450

layout(location = 0) out vec4 out_color;

void main() {
#ifdef USE_RED
    out_color = vec4(1.0, 0.0, 0.0, 1.0);
#else
    out_color = vec4(0.0, 1.0, 0.0, 1.0);
#endif
}
"#;

        let result = composer.compose(
            source,
            ShaderStage::Fragment,
            &[("USE_RED", ShaderDef::Bool(true))],
        );
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_stage_defines() {
        let composer = ShaderComposer::new();

        let source = r#"#version 450

#ifdef VERTEX
layout(location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
#endif

#ifdef FRAGMENT
layout(location = 0) out vec4 out_color;
void main() {
    out_color = vec4(1.0);
}
#endif
"#;

        let vs_result = composer.compose(source, ShaderStage::Vertex, &[]);
        assert!(vs_result.is_ok(), "VS failed: {:?}", vs_result.err());

        let fs_result = composer.compose(source, ShaderStage::Fragment, &[]);
        assert!(fs_result.is_ok(), "FS failed: {:?}", fs_result.err());
    }

    #[test]
    fn test_standard_library() {
        let composer = ShaderComposer::with_standard_library();
        assert!(!composer.includes.is_empty());
    }

    #[test]
    fn test_double_include_prevention() {
        let mut composer = ShaderComposer::new();
        composer.register_include("test/shared.glsl", "const float MY_CONST = 42.0;");

        let source = r#"#version 450
#include "test/shared.glsl"
#include "test/shared.glsl"

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(MY_CONST, 0.0, 0.0, 1.0);
}
"#;

        let result = composer.compose(source, ShaderStage::Fragment, &[]);
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }

    #[test]
    fn test_missing_include() {
        let composer = ShaderComposer::new();

        let source = r#"#version 450
#include "nonexistent/file.glsl"
void main() {}
"#;

        let result = composer.compose(source, ShaderStage::Fragment, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_include_directive() {
        assert_eq!(
            parse_include_directive(r#"#include "foo/bar.glsl""#),
            Some("foo/bar.glsl")
        );
        assert_eq!(
            parse_include_directive(r#"#include <foo/bar.glsl>"#),
            Some("foo/bar.glsl")
        );
        assert_eq!(parse_include_directive("#define FOO"), None);
        assert_eq!(parse_include_directive("// comment"), None);
    }
}
