//! Slang shader compiler wrapper.
//!
//! Provides a high-level interface for compiling Slang shaders to SPIR-V and WGSL,
//! with support for reflection-based binding layout generation and module serialization.

use std::ffi::CString;

use crate::error::GraphicsError;
use crate::materials::{
    BindingLayout, BindingLayoutEntry, BindingType, ShaderStage, ShaderStageFlags,
};

use shader_slang as slang;
use slang::Downcast;

/// Compiled Slang shader output for a single entry point.
pub struct CompiledShader {
    /// The compiled bytecode (SPIR-V words or WGSL text depending on target).
    pub bytecode: Vec<u8>,
    /// Binding layouts reflected from the shader.
    pub binding_layouts: Vec<BindingLayout>,
}

/// Input tuple for [`SlangCompiler::reflect_all_bindings`]:
/// `(source, entry_point, stage, defines)`.
pub type ShaderReflectInput<'a> = (&'a str, &'a str, ShaderStage, &'a [(&'a str, &'a str)]);

/// Slang shader compiler.
///
/// Wraps a Slang `GlobalSession` and provides methods for compiling shaders
/// to various targets (SPIR-V, WGSL) with optional reflection.
///
/// Create once and reuse — the global session caches internal state.
pub struct SlangCompiler {
    global_session: slang::GlobalSession,
}

impl SlangCompiler {
    /// Create a new Slang compiler instance.
    pub fn new() -> Result<Self, GraphicsError> {
        let global_session = slang::GlobalSession::new().ok_or_else(|| {
            GraphicsError::InitializationFailed("Failed to create Slang global session".into())
        })?;

        Ok(Self { global_session })
    }

    /// Compile a Slang source string to SPIR-V bytecode.
    ///
    /// Returns the compiled SPIR-V as a byte vector (aligned to u32 words).
    pub fn compile_to_spirv(
        &self,
        source: &str,
        entry_point_name: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<Vec<u8>, GraphicsError> {
        let blob = self.compile_entry_point(
            source,
            entry_point_name,
            slang::CompileTarget::Spirv,
            "spirv_1_5",
            search_paths,
            defines,
        )?;
        Ok(blob.as_slice().to_vec())
    }

    /// Compile a Slang source string to WGSL text.
    pub fn compile_to_wgsl(
        &self,
        source: &str,
        entry_point_name: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<String, GraphicsError> {
        let blob = self.compile_entry_point(
            source,
            entry_point_name,
            slang::CompileTarget::Wgsl,
            "sm_6_0",
            search_paths,
            defines,
        )?;
        let wgsl = blob.as_str().map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!(
                "Slang WGSL output is not valid UTF-8: {e}"
            ))
        })?;
        Ok(wgsl.to_string())
    }

    /// Compile a Slang source and reflect binding layouts.
    ///
    /// Returns both the compiled bytecode and the auto-generated binding layouts.
    pub fn compile_and_reflect(
        &self,
        source: &str,
        entry_point_name: &str,
        target: slang::CompileTarget,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        stage: ShaderStage,
    ) -> Result<CompiledShader, GraphicsError> {
        let profile = match target {
            slang::CompileTarget::Spirv => "spirv_1_5",
            _ => "sm_6_0",
        };

        let (linked, _session) = self.compile_linked(
            source,
            entry_point_name,
            target,
            profile,
            search_paths,
            defines,
        )?;

        let bytecode = linked
            .entry_point_code(0, 0)
            .map_err(|e| {
                GraphicsError::ShaderCompilationFailed(format!("Slang code generation failed: {e}"))
            })?
            .as_slice()
            .to_vec();

        let binding_layouts = self.reflect_bindings(&linked, stage)?;

        Ok(CompiledShader {
            bytecode,
            binding_layouts,
        })
    }

    /// Reflect binding layouts from a compiled Slang program.
    fn reflect_bindings(
        &self,
        linked: &slang::ComponentType,
        stage: ShaderStage,
    ) -> Result<Vec<BindingLayout>, GraphicsError> {
        let reflection = linked.layout(0).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Slang reflection failed: {e}"))
        })?;

        let visibility = match stage {
            ShaderStage::Vertex => ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => ShaderStageFlags::COMPUTE,
        };

        // Group parameters by binding space (group).
        let mut groups: std::collections::BTreeMap<u32, Vec<BindingLayoutEntry>> =
            std::collections::BTreeMap::new();

        for param in reflection.parameters() {
            let space = param.binding_space();
            let binding = param.binding_index();
            let type_layout = param.type_layout();
            let binding_type = self.slang_type_to_binding_type(type_layout);

            let label = param.name().map(|n| n.to_string());

            groups.entry(space).or_default().push(BindingLayoutEntry {
                binding,
                binding_type,
                visibility,
                label,
            });
        }

        let layouts: Vec<BindingLayout> = groups
            .into_values()
            .map(|entries| BindingLayout {
                entries,
                label: None,
            })
            .collect();

        Ok(layouts)
    }

    /// Reflect binding layouts from multiple shader stages, merging visibility.
    ///
    /// Each shader entry is `(source, entry_point, stage, defines)`. Shaders sharing
    /// the same binding slot (space, binding) have their visibility flags OR-ed together.
    ///
    /// Returns one `BindingLayout` per binding space, ordered by space index.
    pub fn reflect_all_bindings(
        &self,
        shaders: &[ShaderReflectInput<'_>],
    ) -> Result<Vec<BindingLayout>, GraphicsError> {
        use std::collections::BTreeMap;

        type BindingInfo = (BindingType, ShaderStageFlags, Option<String>);
        let mut merged: BTreeMap<u32, BTreeMap<u32, BindingInfo>> = BTreeMap::new();

        for &(source, entry_point, stage, defines) in shaders {
            let (linked, _session) = self.compile_linked(
                source,
                entry_point,
                slang::CompileTarget::Spirv,
                "spirv_1_5",
                &[],
                defines,
            )?;

            let layouts = self.reflect_bindings(&linked, stage)?;

            for (space_idx, layout) in layouts.into_iter().enumerate() {
                let space = space_idx as u32;
                let space_map = merged.entry(space).or_default();

                for entry in layout.entries {
                    match space_map.get(&entry.binding) {
                        Some(&(existing_type, _, _)) if existing_type != entry.binding_type => {
                            return Err(GraphicsError::ShaderCompilationFailed(format!(
                                "Conflicting binding types at space={space}, binding={}: {:?} vs {:?}",
                                entry.binding, existing_type, entry.binding_type
                            )));
                        }
                        Some(&(_, existing_vis, _)) => {
                            let label = space_map.get(&entry.binding).and_then(|e| e.2.clone());
                            space_map.insert(
                                entry.binding,
                                (entry.binding_type, existing_vis | entry.visibility, label),
                            );
                        }
                        None => {
                            space_map.insert(
                                entry.binding,
                                (entry.binding_type, entry.visibility, entry.label),
                            );
                        }
                    }
                }
            }
        }

        let layouts = merged
            .into_values()
            .map(|bindings| {
                let entries = bindings
                    .into_iter()
                    .map(
                        |(binding, (binding_type, visibility, label))| BindingLayoutEntry {
                            binding,
                            binding_type,
                            visibility,
                            label,
                        },
                    )
                    .collect();
                BindingLayout {
                    entries,
                    label: None,
                }
            })
            .collect();

        Ok(layouts)
    }

    /// Map a Slang type layout to our BindingType.
    fn slang_type_to_binding_type(
        &self,
        type_layout: &slang::reflection::TypeLayout,
    ) -> BindingType {
        use slang::TypeKind;

        let kind = type_layout.kind();
        match kind {
            TypeKind::ConstantBuffer | TypeKind::ParameterBlock => BindingType::UniformBuffer,
            TypeKind::Resource => {
                if let Some(shape) = type_layout.resource_shape() {
                    use slang::ResourceShape;
                    match shape {
                        ResourceShape::SlangTextureCube => BindingType::TextureCube,
                        _ => BindingType::Texture,
                    }
                } else {
                    BindingType::Texture
                }
            }
            TypeKind::SamplerState => BindingType::Sampler,
            TypeKind::TextureBuffer => BindingType::StorageBuffer,
            _ => {
                // Fallback: check the parameter category
                let category = type_layout.parameter_category();
                use slang::ParameterCategory;
                match category {
                    ParameterCategory::ConstantBuffer => BindingType::UniformBuffer,
                    ParameterCategory::ShaderResource => BindingType::Texture,
                    ParameterCategory::UnorderedAccess => BindingType::StorageBuffer,
                    ParameterCategory::SamplerState => BindingType::Sampler,
                    _ => BindingType::UniformBuffer,
                }
            }
        }
    }

    /// Write standard library modules to the temp shader directory.
    ///
    /// This makes them available for `import math;` etc. in Slang shaders.
    pub fn write_library_modules(
        &self,
        library: &crate::shader::ShaderLibrary,
    ) -> Result<(), GraphicsError> {
        let temp_dir = std::env::temp_dir().join("redlilium_shaders");
        std::fs::create_dir_all(&temp_dir).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Failed to create temp dir: {e}"))
        })?;

        for (name, source) in library.modules() {
            let filename = if name.ends_with(".slang") {
                name.to_string()
            } else {
                format!("{name}.slang")
            };
            let path = temp_dir.join(filename);
            std::fs::write(&path, source).map_err(|e| {
                GraphicsError::ShaderCompilationFailed(format!(
                    "Failed to write library module '{name}': {e}"
                ))
            })?;
        }

        Ok(())
    }

    /// Internal: compile and link a single entry point.
    fn compile_linked(
        &self,
        source: &str,
        entry_point_name: &str,
        target: slang::CompileTarget,
        profile: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<(slang::ComponentType, slang::Session), GraphicsError> {
        let profile_id = self.global_session.find_profile(profile);

        let mut compiler_options = slang::CompilerOptions::default()
            .optimization(slang::OptimizationLevel::High)
            .emit_spirv_directly(true);

        for &(key, value) in defines {
            compiler_options = compiler_options.macro_define(key, value);
        }

        let target_desc = slang::TargetDesc::default()
            .format(target)
            .profile(profile_id)
            .options(&compiler_options);

        let targets = [target_desc];

        // Write source to a temporary file — the Slang API loads modules by file name.
        // Use a unique directory per compilation to avoid race conditions in parallel tests.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        entry_point_name.hash(&mut hasher);
        std::thread::current().id().hash(&mut hasher);
        let hash = hasher.finish();

        let temp_dir = std::env::temp_dir().join(format!("redlilium_shaders_{hash:016x}"));
        std::fs::create_dir_all(&temp_dir).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Failed to create temp dir: {e}"))
        })?;

        let temp_file = temp_dir.join("_temp_shader.slang");
        std::fs::write(&temp_file, source).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Failed to write temp shader: {e}"))
        })?;

        // Build search paths: temp dir first, then the shared library dir, then user-provided paths
        let lib_dir = std::env::temp_dir().join("redlilium_shaders");
        let temp_dir_str = temp_dir.to_string_lossy().to_string();
        let lib_dir_str = lib_dir.to_string_lossy().to_string();
        let mut all_search_paths = vec![temp_dir_str.as_str(), lib_dir_str.as_str()];
        all_search_paths.extend_from_slice(search_paths);

        let c_search_paths: Vec<CString> = all_search_paths
            .iter()
            .map(|p| CString::new(*p).unwrap())
            .collect();
        let search_path_ptrs: Vec<*const i8> = c_search_paths.iter().map(|p| p.as_ptr()).collect();

        let session_desc = slang::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_path_ptrs)
            .options(&compiler_options);

        let session = self
            .global_session
            .create_session(&session_desc)
            .ok_or_else(|| {
                GraphicsError::ShaderCompilationFailed("Failed to create Slang session".into())
            })?;

        let module = session.load_module("_temp_shader").map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Slang module load failed: {e}"))
        })?;

        let entry_point = module
            .find_entry_point_by_name(entry_point_name)
            .ok_or_else(|| {
                GraphicsError::ShaderCompilationFailed(format!(
                    "Entry point '{entry_point_name}' not found in Slang module"
                ))
            })?;

        let program = session
            .create_composite_component_type(&[
                module.downcast().clone(),
                entry_point.downcast().clone(),
            ])
            .map_err(|e| {
                GraphicsError::ShaderCompilationFailed(format!("Slang composition failed: {e}"))
            })?;

        let linked = program.link().map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Slang linking failed: {e}"))
        })?;

        Ok((linked, session))
    }

    /// Internal: compile a single entry point and return the output blob.
    fn compile_entry_point(
        &self,
        source: &str,
        entry_point_name: &str,
        target: slang::CompileTarget,
        profile: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<slang::Blob, GraphicsError> {
        let (linked, _session) = self.compile_linked(
            source,
            entry_point_name,
            target,
            profile,
            search_paths,
            defines,
        )?;

        linked.entry_point_code(0, 0).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Slang code generation failed: {e}"))
        })
    }
}

impl Default for SlangCompiler {
    fn default() -> Self {
        Self::new().expect("Failed to create Slang compiler")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = SlangCompiler::new();
        assert!(
            compiler.is_ok(),
            "Failed to create SlangCompiler: {:?}",
            compiler.err()
        );
    }

    #[test]
    fn test_compile_simple_vertex_shader_to_spirv() {
        let compiler = SlangCompiler::new().unwrap();

        let source = r#"
struct VertexOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vs_main(float3 position : POSITION) {
    VertexOutput output;
    output.position = float4(position, 1.0);
    return output;
}
"#;

        let result = compiler.compile_to_spirv(source, "vs_main", &[], &[]);
        assert!(
            result.is_ok(),
            "SPIR-V compilation failed: {:?}",
            result.err()
        );
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn test_compile_simple_fragment_shader_to_wgsl() {
        let compiler = SlangCompiler::new().unwrap();

        let source = r#"
[shader("fragment")]
float4 fs_main() : SV_Target {
    return float4(1.0, 0.0, 0.0, 1.0);
}
"#;

        let result = compiler.compile_to_wgsl(source, "fs_main", &[], &[]);
        assert!(
            result.is_ok(),
            "WGSL compilation failed: {:?}",
            result.err()
        );
        let wgsl = result.unwrap();
        assert!(!wgsl.is_empty());
    }

    #[test]
    fn test_compile_with_defines() {
        let compiler = SlangCompiler::new().unwrap();

        let source = r#"
#ifndef MAX_LIGHTS
#define MAX_LIGHTS 4
#endif

[shader("fragment")]
float4 fs_main() : SV_Target {
    return float4(float(MAX_LIGHTS) / 16.0, 0.0, 0.0, 1.0);
}
"#;

        let result = compiler.compile_to_spirv(source, "fs_main", &[], &[("MAX_LIGHTS", "8")]);
        assert!(
            result.is_ok(),
            "Compilation with defines failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_compile_and_reflect() {
        let compiler = SlangCompiler::new().unwrap();

        let source = r#"
struct Uniforms {
    float4x4 view_projection;
    float4x4 model;
};

ConstantBuffer<Uniforms> uniforms : register(b0, space0);

struct VertexOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vs_main(float3 position : POSITION) {
    VertexOutput output;
    output.position = mul(uniforms.view_projection, mul(uniforms.model, float4(position, 1.0)));
    return output;
}
"#;

        let result = compiler.compile_and_reflect(
            source,
            "vs_main",
            slang::CompileTarget::Spirv,
            &[],
            &[],
            ShaderStage::Vertex,
        );
        assert!(
            result.is_ok(),
            "Compile and reflect failed: {:?}",
            result.err()
        );

        let compiled = result.unwrap();
        assert!(!compiled.bytecode.is_empty());
        assert!(!compiled.binding_layouts.is_empty());
    }

    #[test]
    fn test_reflect_all_bindings() {
        let compiler = SlangCompiler::new().unwrap();

        // Shader with cbuffer at space 0 (used by both VS and FS),
        // and Texture2D + SamplerState at space 1 (used by FS only).
        let source = r#"
[[vk::binding(0, 0)]]
cbuffer Uniforms {
    float4x4 mvp;
};

[[vk::binding(0, 1)]]
Texture2D my_texture;
[[vk::binding(1, 1)]]
SamplerState my_sampler;

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(float3 position : POSITION, float2 uv : TEXCOORD0) {
    VsOutput output;
    output.position = mul(mvp, float4(position, 1.0));
    output.uv = uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    return my_texture.Sample(my_sampler, input.uv);
}
"#;

        let shaders: Vec<ShaderReflectInput<'_>> = vec![
            (source, "vs_main", ShaderStage::Vertex, &[]),
            (source, "fs_main", ShaderStage::Fragment, &[]),
        ];

        let layouts = compiler
            .reflect_all_bindings(&shaders)
            .expect("reflect_all_bindings failed");

        // Should have 2 layouts (space 0 and space 1)
        assert_eq!(
            layouts.len(),
            2,
            "Expected 2 binding layouts, got {}",
            layouts.len()
        );

        // Space 0: cbuffer at binding 0, used by both VS and FS
        let space0 = &layouts[0];
        assert_eq!(space0.entries.len(), 1);
        assert_eq!(space0.entries[0].binding, 0);
        assert_eq!(space0.entries[0].binding_type, BindingType::UniformBuffer);
        assert!(
            space0.entries[0]
                .visibility
                .contains(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
            "Expected VERTEX | FRAGMENT visibility for shared cbuffer, got {:?}",
            space0.entries[0].visibility
        );

        // Space 1: Texture2D at binding 0 + SamplerState at binding 1
        // Note: Slang reflection reports all global parameters for the program,
        // so when both VS and FS share the same source, all bindings get merged
        // visibility from both stages. This is correct (overly permissive is fine).
        let space1 = &layouts[1];
        assert_eq!(space1.entries.len(), 2);
        assert_eq!(space1.entries[0].binding_type, BindingType::Texture);
        assert!(
            space1.entries[0]
                .visibility
                .contains(ShaderStageFlags::FRAGMENT)
        );
        assert_eq!(space1.entries[1].binding_type, BindingType::Sampler);
        assert!(
            space1.entries[1]
                .visibility
                .contains(ShaderStageFlags::FRAGMENT)
        );
    }
}
