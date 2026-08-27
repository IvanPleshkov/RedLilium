//! Slang shader compiler wrapper.
//!
//! Provides a high-level interface for compiling Slang shaders to SPIR-V and WGSL,
//! with support for reflection-based binding layout generation and module serialization.

use std::ffi::CString;

use crate::error::GraphicsError;
use crate::materials::{
    BindingLayout, BindingLayoutEntry, BindingType, ShaderStage, ShaderStageFlags, UpdateRate,
};

use shader_slang as slang;
use slang::Downcast;

/// Compiled Slang shader output for a single entry point.
pub struct CompiledShader {
    /// The compiled bytecode (SPIR-V words or WGSL text depending on target).
    pub bytecode: Vec<u8>,
    /// Binding layouts reflected from the shader.
    pub binding_layouts: Vec<BindingLayout>,
    /// Per-set update-frequency classes (parallel to `binding_layouts`; `None`
    /// for legacy sets not declared as rate-classified `ParameterBlock`s).
    pub update_rates: Vec<Option<UpdateRate>>,
}

/// Input tuple for [`SlangCompiler::reflect_all_bindings`]:
/// `(source, entry_point, stage, defines)`.
pub type ShaderReflectInput<'a> = (&'a str, &'a str, ShaderStage, &'a [(&'a str, &'a str)]);

/// Converts `(binding_space, layout, rate)` tuples (sorted ascending, unique
/// spaces) into dense parallel `Vec`s where the vector index equals the
/// binding space.
///
/// Bind-group layouts are consumed positionally downstream (vector index ==
/// `@group(N)` / `space N`). When a shader uses non-contiguous spaces (e.g. 0
/// and 2), the missing spaces are filled with empty layouts (rate `None`) so
/// each remaining layout stays at its correct group index instead of being
/// shifted down.
fn densify_binding_layouts(
    by_space: Vec<(u32, BindingLayout, Option<UpdateRate>)>,
) -> (Vec<BindingLayout>, Vec<Option<UpdateRate>>) {
    let mut dense: Vec<BindingLayout> = Vec::new();
    let mut rates: Vec<Option<UpdateRate>> = Vec::new();
    for (space, layout, rate) in by_space {
        let space = space as usize;
        while dense.len() < space {
            dense.push(BindingLayout {
                entries: Vec::new(),
                label: None,
            });
            rates.push(None);
        }
        dense.push(layout);
        rates.push(rate);
    }
    (dense, rates)
}

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

    /// The Slang build tag (compiler version string) of the underlying global
    /// session, e.g. `"v2024.1.30"`. Stamped into the offline-baked shader table
    /// (`xtask bake-shaders`, #33) so a compiler upgrade shows up as an explicit
    /// version message in `--check` rather than an opaque WGSL byte-diff.
    pub fn build_tag(&self) -> &str {
        self.global_session.build_tag_string()
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

        let (binding_layouts, update_rates) =
            densify_binding_layouts(self.reflect_bindings(&linked, stage)?);

        Ok(CompiledShader {
            bytecode,
            binding_layouts,
            update_rates,
        })
    }

    /// Reflect binding layouts from a compiled Slang program.
    ///
    /// Returns `(binding_space, layout, update_rate)` tuples keyed by the
    /// **real** Slang binding space, sorted ascending. Callers must map
    /// space → bind-group index themselves (see [`densify_binding_layouts`]);
    /// the space is *not* the position in the returned vector when spaces are
    /// non-contiguous.
    ///
    /// Two parameter shapes coexist:
    /// - `ParameterBlock<T>` globals — one whole space per block (the space is
    ///   the parameter's `SubElementRegisterSpace` offset). Uniform fields
    ///   become the implicit constant buffer at binding 0; opaque fields
    ///   (textures/samplers) get their `DescriptorTableSlot` offsets. The
    ///   block's `[UpdateRate("...")]` user attribute classifies the set
    ///   (`docs/MATERIAL_ASSETS.md` Decision 7).
    /// - legacy `[[vk::binding(b, s)]]` globals — space/binding as declared,
    ///   no rate class.
    fn reflect_bindings(
        &self,
        linked: &slang::ComponentType,
        stage: ShaderStage,
    ) -> Result<Vec<(u32, BindingLayout, Option<UpdateRate>)>, GraphicsError> {
        let reflection = linked.layout(0).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Slang reflection failed: {e}"))
        })?;

        let visibility = match stage {
            ShaderStage::Vertex => ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => ShaderStageFlags::COMPUTE,
            ShaderStage::Task => ShaderStageFlags::TASK,
            ShaderStage::Mesh => ShaderStageFlags::MESH,
        };

        // Group parameters by binding space (group).
        let mut groups: std::collections::BTreeMap<u32, Vec<BindingLayoutEntry>> =
            std::collections::BTreeMap::new();
        let mut rates: std::collections::BTreeMap<u32, UpdateRate> =
            std::collections::BTreeMap::new();

        for param in reflection.parameters() {
            if param.type_layout().kind() == slang::TypeKind::ParameterBlock {
                let space = param.offset(slang::ParameterCategory::SubElementRegisterSpace) as u32;
                if let Some(rate) = self.block_update_rate(param)? {
                    rates.insert(space, rate);
                }
                let entries = groups.entry(space).or_default();

                // Uniform fields live in the block's implicit constant buffer
                // at binding 0; each opaque field owns a descriptor slot.
                let element = param.type_layout().element_type_layout();
                let mut has_uniforms = false;
                let mut opaque: Vec<BindingLayoutEntry> = Vec::new();
                for field in element.fields() {
                    let mut is_uniform = false;
                    for i in 0..field.category_count() {
                        match field.category_by_index(i) {
                            slang::ParameterCategory::Uniform => is_uniform = true,
                            slang::ParameterCategory::DescriptorTableSlot => {
                                opaque.push(BindingLayoutEntry {
                                    binding: field
                                        .offset(slang::ParameterCategory::DescriptorTableSlot)
                                        as u32,
                                    binding_type: self
                                        .slang_type_to_binding_type(field.type_layout()),
                                    visibility,
                                    label: field.name().map(|n| n.to_string()),
                                });
                            }
                            _ => {}
                        }
                    }
                    has_uniforms |= is_uniform;
                }
                if has_uniforms {
                    entries.push(BindingLayoutEntry {
                        binding: 0,
                        binding_type: BindingType::UniformBuffer,
                        visibility,
                        label: param.name().map(|n| n.to_string()),
                    });
                }
                entries.extend(opaque);
                continue;
            }

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

        let layouts: Vec<(u32, BindingLayout, Option<UpdateRate>)> = groups
            .into_iter()
            .map(|(space, entries)| {
                (
                    space,
                    BindingLayout {
                        entries,
                        label: None,
                    },
                    rates.get(&space).copied(),
                )
            })
            .collect();

        Ok(layouts)
    }

    /// The `[UpdateRate("...")]` class of a `ParameterBlock` parameter, if
    /// declared. An unknown rate string is an error (a typo would otherwise
    /// silently degrade to legacy binding).
    fn block_update_rate(
        &self,
        param: &slang::reflection::VariableLayout,
    ) -> Result<Option<UpdateRate>, GraphicsError> {
        let Some(var) = param.variable() else {
            return Ok(None);
        };
        let Some(attr) = var.user_attributes().find(|a| a.name() == "UpdateRate") else {
            return Ok(None);
        };
        let value = attr.argument_value_string(0).ok_or_else(|| {
            GraphicsError::ShaderCompilationFailed(format!(
                "ParameterBlock '{}': [UpdateRate] needs a string argument",
                param.name().unwrap_or("?")
            ))
        })?;
        UpdateRate::parse(value).map(Some).ok_or_else(|| {
            GraphicsError::ShaderCompilationFailed(format!(
                "ParameterBlock '{}': unknown update rate '{value}' \
                 (expected external | dynamic | static)",
                param.name().unwrap_or("?")
            ))
        })
    }

    /// Reflect binding layouts from multiple shader stages, merging visibility
    /// and per-set update rates (a rate conflict across stages is an error).
    ///
    /// Each shader entry is `(source, entry_point, stage, defines)`. Shaders sharing
    /// the same binding slot (space, binding) have their visibility flags OR-ed together.
    ///
    /// Returns one `BindingLayout` per binding space, ordered by space index,
    /// with the parallel per-space [`UpdateRate`] classes.
    pub fn reflect_all_bindings(
        &self,
        shaders: &[ShaderReflectInput<'_>],
    ) -> Result<(Vec<BindingLayout>, Vec<Option<UpdateRate>>), GraphicsError> {
        use std::collections::BTreeMap;

        type BindingInfo = (BindingType, ShaderStageFlags, Option<String>);
        let mut merged: BTreeMap<u32, BTreeMap<u32, BindingInfo>> = BTreeMap::new();

        let mut space_rates: std::collections::BTreeMap<u32, UpdateRate> =
            std::collections::BTreeMap::new();
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

            for (space, layout, rate) in layouts.into_iter() {
                if let Some(rate) = rate {
                    match space_rates.get(&space) {
                        Some(&existing) if existing != rate => {
                            return Err(GraphicsError::ShaderCompilationFailed(format!(
                                "Conflicting [UpdateRate] at space {space}: \
                                 {existing:?} vs {rate:?} across stages"
                            )));
                        }
                        _ => {
                            space_rates.insert(space, rate);
                        }
                    }
                }
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

        let by_space: Vec<(u32, BindingLayout, Option<UpdateRate>)> = merged
            .into_iter()
            .map(|(space, bindings)| {
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
                (
                    space,
                    BindingLayout {
                        entries,
                        label: None,
                    },
                    space_rates.get(&space).copied(),
                )
            })
            .collect();

        Ok(densify_binding_layouts(by_space))
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
                        // A `Texture2DArray` is a sampled 2D texture with the
                        // array flag — the layered-texture material-property
                        // kind. Without this arm it fell through to a plain
                        // `Texture`, so the reflected layout mismatched the
                        // D2Array view the instance manager binds.
                        ResourceShape::SlangTexture2dArray => BindingType::Texture2DArray,
                        // Structured / byte-address buffers are SSBOs, not textures.
                        // Distinguish RW (RWStructuredBuffer) from read-only
                        // (StructuredBuffer): wgpu validates the access mode exactly.
                        ResourceShape::SlangStructuredBuffer
                        | ResourceShape::SlangByteAddressBuffer => {
                            if Self::is_read_write_access(type_layout) {
                                BindingType::StorageBuffer
                            } else {
                                BindingType::StorageBufferReadOnly
                            }
                        }
                        _ => BindingType::Texture,
                    }
                } else {
                    BindingType::Texture
                }
            }
            TypeKind::SamplerState => BindingType::Sampler,
            TypeKind::TextureBuffer => BindingType::StorageBufferReadOnly,
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

    /// Whether a resource type reflects with read-write (UAV) access.
    ///
    /// `RWStructuredBuffer` / `RWByteAddressBuffer` report `ReadWrite`;
    /// plain `StructuredBuffer` / `ByteAddressBuffer` report `Read`.
    fn is_read_write_access(type_layout: &slang::reflection::TypeLayout) -> bool {
        matches!(
            type_layout.resource_access(),
            Some(slang::ResourceAccess::ReadWrite)
        )
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

        // The module has been loaded and linked from disk; the per-compilation
        // temp directory is no longer needed. Remove it so repeated compiles
        // (hot-reload, many materials) don't accumulate temp dirs indefinitely.
        let _ = std::fs::remove_dir_all(&temp_dir);

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
    fn densify_fills_gaps_at_correct_space() {
        // Spaces 0 and 2 (1 missing): the space-2 layout must land at index 2,
        // not be shifted down to index 1.
        let l0 = BindingLayout {
            entries: vec![BindingLayoutEntry {
                binding: 0,
                binding_type: BindingType::UniformBuffer,
                visibility: ShaderStageFlags::VERTEX,
                label: Some("a".into()),
            }],
            label: None,
        };
        let l2 = BindingLayout {
            entries: vec![BindingLayoutEntry {
                binding: 0,
                binding_type: BindingType::UniformBuffer,
                visibility: ShaderStageFlags::FRAGMENT,
                label: Some("b".into()),
            }],
            label: None,
        };
        let (dense, rates) = densify_binding_layouts(vec![
            (0, l0, Some(UpdateRate::External)),
            (2, l2, Some(UpdateRate::Static)),
        ]);
        assert_eq!(dense.len(), 3);
        assert_eq!(dense[0].entries.len(), 1);
        assert_eq!(dense[0].entries[0].label.as_deref(), Some("a"));
        assert!(dense[1].entries.is_empty(), "gap space 1 must be empty");
        assert_eq!(dense[2].entries[0].label.as_deref(), Some("b"));
        assert_eq!(
            rates,
            vec![Some(UpdateRate::External), None, Some(UpdateRate::Static)],
            "rates ride along, gaps are None"
        );
    }

    #[test]
    fn densify_contiguous_is_identity_shaped() {
        let mk = |vis| BindingLayout {
            entries: vec![BindingLayoutEntry {
                binding: 0,
                binding_type: BindingType::UniformBuffer,
                visibility: vis,
                label: None,
            }],
            label: None,
        };
        let (dense, rates) = densify_binding_layouts(vec![
            (0, mk(ShaderStageFlags::VERTEX), None),
            (1, mk(ShaderStageFlags::FRAGMENT), None),
        ]);
        assert_eq!(dense.len(), 2);
        assert!(!dense[0].entries.is_empty());
        assert!(!dense[1].entries.is_empty());
        assert_eq!(rates, vec![None, None]);
    }

    /// Decision 7 probe: a slang user attribute (`[UpdateRate("...")]`) on
    /// `ParameterBlock` globals is readable through reflection — per parameter,
    /// via `VariableLayout::variable()` → `user_attributes()`. This pins the
    /// attribute-definition boilerplate and the reflection path the frequency-
    /// classified binding sets build on.
    #[test]
    fn user_attribute_update_rate_reflects() {
        let compiler = SlangCompiler::new().unwrap();

        let source = r#"
[__AttributeUsage(_AttributeTargets.Var)]
struct UpdateRateAttribute {
    string rate;
};

struct CameraParams {
    column_major float4x4 view_projection;
};
struct ModelParams {
    column_major float4x4 model;
};
struct MaterialParams {
    float4 base_color;
};

[UpdateRate("external")]
ParameterBlock<CameraParams> gCamera;

[UpdateRate("dynamic")]
ParameterBlock<ModelParams> gModel;

[UpdateRate("static")]
ParameterBlock<MaterialParams> gMaterial;

[shader("vertex")]
float4 vs_main(float3 position : POSITION) : SV_Position {
    return mul(gCamera.view_projection, mul(gModel.model, float4(position, 1.0)));
}

[shader("fragment")]
float4 fs_main(float4 pos : SV_Position) : SV_Target {
    return gMaterial.base_color;
}
"#;

        let (linked, _session) = compiler
            .compile_linked(
                source,
                "vs_main",
                slang::CompileTarget::Spirv,
                "spirv_1_5",
                &[],
                &[],
            )
            .expect("probe shader compiles");
        let reflection = linked.layout(0).expect("reflection");

        let mut rates: std::collections::BTreeMap<String, (u32, String)> = Default::default();
        for param in reflection.parameters() {
            let Some(var) = param.variable() else {
                continue;
            };
            let rate = var
                .user_attributes()
                .find(|a| a.name() == "UpdateRate")
                .and_then(|a| a.argument_value_string(0).map(str::to_owned));
            if let Some(rate) = rate {
                let space = param.offset(slang::ParameterCategory::SubElementRegisterSpace) as u32;
                rates.insert(param.name().unwrap_or("?").to_owned(), (space, rate));
            }
        }

        assert_eq!(
            rates.get("gCamera").map(|(_, r)| r.as_str()),
            Some("external"),
            "camera block carries its rate; got {rates:?}"
        );
        assert_eq!(
            rates.get("gModel").map(|(_, r)| r.as_str()),
            Some("dynamic")
        );
        assert_eq!(
            rates.get("gMaterial").map(|(_, r)| r.as_str()),
            Some("static")
        );
        // Each ParameterBlock occupies its own register space; the space
        // index is the parameter's offset in the SubElementRegisterSpace
        // category (NOT `binding_space()`, which stays 0 for blocks).
        assert_eq!(rates.get("gCamera").map(|(s, _)| *s), Some(0));
        assert_eq!(rates.get("gModel").map(|(s, _)| *s), Some(1));
        assert_eq!(rates.get("gMaterial").map(|(s, _)| *s), Some(2));
    }

    /// A `ParameterBlock` mixing uniform fields with textures/samplers
    /// reflects to our group-1 convention: the implicit constant buffer at
    /// binding 0, opaque fields at their descriptor slots — and the block's
    /// `[UpdateRate]` classifies the whole set (Decision 7).
    #[test]
    fn parameter_block_reflects_buffer_and_opaque_fields() {
        let compiler = SlangCompiler::new().unwrap();

        let source = r#"
[__AttributeUsage(_AttributeTargets.Var)]
struct UpdateRateAttribute {
    string rate;
};

struct MaterialParams {
    float4 base_color;
    Texture2D base_texture;
    SamplerState base_sampler;
};

[UpdateRate("static")]
ParameterBlock<MaterialParams> gMaterial;

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    return gMaterial.base_texture.Sample(gMaterial.base_sampler, uv) * gMaterial.base_color;
}
"#;

        let (layouts, rates) = compiler
            .reflect_all_bindings(&[(source, "fs_main", ShaderStage::Fragment, &[])])
            .expect("reflection");

        assert_eq!(layouts.len(), 1);
        assert_eq!(rates, vec![Some(UpdateRate::Static)]);
        let entries = &layouts[0].entries;
        assert_eq!(entries.len(), 3, "buffer + texture + sampler: {entries:?}");
        assert_eq!(entries[0].binding, 0);
        assert_eq!(entries[0].binding_type, BindingType::UniformBuffer);
        assert_eq!(entries[1].binding, 1);
        assert_eq!(entries[1].binding_type, BindingType::Texture);
        assert_eq!(entries[2].binding, 2);
        assert_eq!(entries[2].binding_type, BindingType::Sampler);
    }

    /// The `[UpdateRate]` attribute lives in the `engine` library module —
    /// shaders `import engine;` instead of redeclaring the boilerplate, and
    /// the attribute still reflects (the real std-shader path).
    #[test]
    fn update_rate_attribute_imports_from_engine_module() {
        let compiler = SlangCompiler::new().unwrap();
        compiler
            .write_library_modules(&crate::shader::ShaderLibrary::standard_slang())
            .unwrap();

        let source = r#"
import engine;

struct CameraParams {
    column_major float4x4 view_projection;
};

[UpdateRate("external")]
ParameterBlock<CameraParams> gCamera;

[shader("vertex")]
float4 vs_main(float3 position : POSITION) : SV_Position {
    return mul(gCamera.view_projection, float4(position, 1.0));
}
"#;

        let (layouts, rates) = compiler
            .reflect_all_bindings(&[(source, "vs_main", ShaderStage::Vertex, &[])])
            .expect("reflection through the engine module");
        assert_eq!(layouts.len(), 1);
        assert_eq!(rates, vec![Some(UpdateRate::External)]);
    }

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

        let (layouts, _rates) = compiler
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

    #[test]
    fn test_reflect_structured_buffer_is_storage() {
        // A StructuredBuffer must reflect as a storage buffer, not a texture
        // (regression: `Resource` kind previously mapped everything to Texture).
        let compiler = SlangCompiler::new().unwrap();

        let source = r#"
struct InstanceData { float4x4 model; };

[[vk::binding(0, 0)]]
cbuffer Uniforms { float4x4 vp; };

[[vk::binding(1, 0)]]
StructuredBuffer<InstanceData> instances;

[shader("vertex")]
float4 vs_main(float3 position : POSITION, uint id : SV_InstanceID) : SV_Position {
    return mul(mul(vp, instances[id].model), float4(position, 1.0));
}
"#;

        let shaders: Vec<ShaderReflectInput<'_>> =
            vec![(source, "vs_main", ShaderStage::Vertex, &[])];
        let (layouts, _rates) = compiler
            .reflect_all_bindings(&shaders)
            .expect("reflect_all_bindings failed");

        let space0 = &layouts[0];
        let instances = space0
            .entries
            .iter()
            .find(|e| e.binding == 1)
            .expect("binding 1 (instances) present");
        assert_eq!(
            instances.binding_type,
            BindingType::StorageBufferReadOnly,
            "StructuredBuffer must reflect as a read-only storage buffer, got {:?}",
            instances.binding_type
        );
    }

    #[test]
    fn test_reflect_rw_structured_buffer_is_read_write_storage() {
        // RWStructuredBuffer must keep read-write access: wgpu validates the
        // layout access mode exactly against the shader declaration.
        let compiler = SlangCompiler::new().unwrap();

        let source = r#"
struct Particle { float4 position; };

[[vk::binding(0, 0)]]
RWStructuredBuffer<Particle> particles;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    particles[id.x].position += float4(0.0, 1.0, 0.0, 0.0);
}
"#;

        let shaders: Vec<ShaderReflectInput<'_>> =
            vec![(source, "cs_main", ShaderStage::Compute, &[])];
        let (layouts, _rates) = compiler
            .reflect_all_bindings(&shaders)
            .expect("reflect_all_bindings failed");

        let particles = layouts[0]
            .entries
            .iter()
            .find(|e| e.binding == 0)
            .expect("binding 0 (particles) present");
        assert_eq!(
            particles.binding_type,
            BindingType::StorageBuffer,
            "RWStructuredBuffer must reflect as read-write StorageBuffer, got {:?}",
            particles.binding_type
        );
    }
}
