//! Backend-neutral inspection of the executable shader interface. Uses the same
//! Slang compilation / offline WGSL artifacts as pipeline creation.
use crate::{GraphicsError, ShaderSource, ShaderSourceLanguage, ShaderStage};

/// Scalar/vector type of a shader location (not a vertex-buffer storage format).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceType {
    Float32(u8),
    Sint32(u8),
    Uint32(u8),
    Other,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformBlock {
    pub group: u32,
    pub binding: u32,
    pub size: u32,
    /// Present only for a block consisting entirely of mat4-compatible f32 storage.
    /// Includes Slang std140 wrappers around four float4 columns. Matrix
    /// interpretation (column-major) remains the shader author's contract.
    pub mat4_offsets: Option<Vec<u32>>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderInterface {
    pub outputs: Vec<(u32, InterfaceType)>,
    pub writes_depth: bool,
    pub uniforms: Vec<UniformBlock>,
}

/// Resolve the actual portable program, including defines and offline artifacts.
pub fn to_wgsl(shader: &ShaderSource) -> Result<String, GraphicsError> {
    let source = std::str::from_utf8(&shader.source)
        .map_err(|e| GraphicsError::ShaderCompilationFailed(e.to_string()))?;
    if shader.language == ShaderSourceLanguage::Wgsl {
        return Ok(source.to_owned());
    }
    let defines: Vec<_> = shader
        .defines
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    #[cfg(feature = "slang-shaders")]
    {
        let compiler = super::SlangCompiler::new()?;
        compiler.write_library_modules(&super::ShaderLibrary::standard_slang())?;
        compiler.compile_to_wgsl(source, &shader.entry_point, &[], &defines)
    }
    #[cfg(not(feature = "slang-shaders"))]
    {
        super::baked::resolve(source, &shader.entry_point, &defines).map(|s| s.to_string())
            .ok_or_else(|| GraphicsError::ShaderCompilationFailed(format!(
                "no offline WGSL for '{}'; bake/register the shader set before loading materials", shader.entry_point)))
    }
}

impl ShaderInterface {
    pub fn inspect(shader: &ShaderSource) -> Result<Self, GraphicsError> {
        let code = to_wgsl(shader)?;
        Self::inspect_wgsl(&code, &shader.entry_point, shader.stage)
    }
    pub fn inspect_wgsl(
        code: &str,
        entry: &str,
        stage: ShaderStage,
    ) -> Result<Self, GraphicsError> {
        let fail = |message: String| GraphicsError::ShaderCompilationFailed(message);
        let module =
            naga::front::wgsl::parse_str(code).map_err(|e| fail(e.emit_to_string(code)))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| fail(e.to_string()))?;
        let stage = match stage {
            ShaderStage::Vertex => naga::ShaderStage::Vertex,
            ShaderStage::Fragment => naga::ShaderStage::Fragment,
            ShaderStage::Compute => naga::ShaderStage::Compute,
            _ => {
                return Err(fail(
                    "portable interface inspection requires vertex/fragment/compute".into(),
                ));
            }
        };
        let ep = module
            .entry_points
            .iter()
            .find(|e| e.name == entry && e.stage == stage)
            .ok_or_else(|| fail(format!("missing {stage:?} entry '{entry}'")))?;
        let mut result = Self {
            outputs: vec![],
            writes_depth: false,
            uniforms: vec![],
        };
        fn output(
            result: &mut ShaderInterface,
            module: &naga::Module,
            ty: naga::Handle<naga::Type>,
            binding: Option<&naga::Binding>,
        ) {
            match binding {
                Some(naga::Binding::Location { location, .. }) => {
                    let (scalar, count) = match module.types[ty].inner {
                        naga::TypeInner::Scalar(s) => (s, 1),
                        naga::TypeInner::Vector { size, scalar } => (scalar, size as u8),
                        _ => {
                            result.outputs.push((*location, InterfaceType::Other));
                            return;
                        }
                    };
                    let ty = match (scalar.kind, scalar.width) {
                        (naga::ScalarKind::Float, 4) => InterfaceType::Float32(count),
                        (naga::ScalarKind::Sint, 4) => InterfaceType::Sint32(count),
                        (naga::ScalarKind::Uint, 4) => InterfaceType::Uint32(count),
                        _ => InterfaceType::Other,
                    };
                    result.outputs.push((*location, ty));
                }
                Some(naga::Binding::BuiltIn(naga::BuiltIn::FragDepth)) => {
                    result.writes_depth = true
                }
                None => {
                    if let naga::TypeInner::Struct { ref members, .. } = module.types[ty].inner {
                        for member in members {
                            output(result, module, member.ty, member.binding.as_ref());
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(ret) = &ep.function.result {
            output(&mut result, &module, ret.ty, ret.binding.as_ref());
        }
        result.outputs.sort_by_key(|(location, _)| *location);
        fn matrix_storage(module: &naga::Module, ty: naga::Handle<naga::Type>) -> bool {
            match &module.types[ty].inner {
                naga::TypeInner::Matrix {
                    columns: naga::VectorSize::Quad,
                    rows: naga::VectorSize::Quad,
                    scalar:
                        naga::Scalar {
                            kind: naga::ScalarKind::Float,
                            width: 4,
                        },
                } => true,
                naga::TypeInner::Array {
                    base,
                    size: naga::ArraySize::Constant(n),
                    stride: 16,
                } if n.get() == 4 => {
                    matches!(
                        module.types[*base].inner,
                        naga::TypeInner::Vector {
                            size: naga::VectorSize::Quad,
                            scalar: naga::Scalar {
                                kind: naga::ScalarKind::Float,
                                width: 4
                            }
                        }
                    )
                }
                naga::TypeInner::Struct { members, span: 64 }
                    if members.len() == 1 && members[0].offset == 0 =>
                {
                    matrix_storage(module, members[0].ty)
                }
                _ => false,
            }
        }
        for (_, global) in module.global_variables.iter() {
            if global.space != naga::AddressSpace::Uniform {
                continue;
            }
            let Some(binding) = &global.binding else {
                continue;
            };
            if let naga::TypeInner::Struct { ref members, span } = module.types[global.ty].inner {
                let offsets = members
                    .iter()
                    .map(|m| matrix_storage(&module, m.ty).then_some(m.offset))
                    .collect();
                result.uniforms.push(UniformBlock {
                    group: binding.group,
                    binding: binding.binding,
                    size: span,
                    mat4_offsets: offsets,
                });
            }
        }
        Ok(result)
    }
}
