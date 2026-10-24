//! Versioned stock deferred surface contract. Registration declares the shader's
//! semantics; specialization validates its executable interface on every variant
//! and source version. See docs/DEFERRED_MATERIALS.md for packing and ownership.
use redlilium_graphics::shader::interface::{InterfaceType, ShaderInterface};
use redlilium_graphics::{
    BindingType, GraphicsError, Material, ShaderSource, TextureFormat, UpdateRate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GBufferContract {
    /// Four targets, reversed-Z depth, current-minus-previous unjittered NDC
    /// velocity. Negative RT1 alpha selects unlit RT0 RGB in the resolve.
    V1,
}
/// Attachment order of GBufferContract::V1. Alpha in RT0 is geometry/AO, not coverage.
pub const GBUFFER_FORMATS: [TextureFormat; 4] = [
    TextureFormat::Rgba8UnormSrgb,
    TextureFormat::Rgba16Float,
    TextureFormat::R8Unorm,
    TextureFormat::Rg16Float,
];
impl GBufferContract {
    pub(crate) fn validate_shaders(
        self,
        vertex: &ShaderSource,
        fragment: &ShaderSource,
    ) -> Result<(), GraphicsError> {
        let fail = |s: &str| GraphicsError::InvalidParameter(format!("G-buffer V1: {s}"));
        let fs = ShaderInterface::inspect(fragment)?;
        let expected = vec![
            (0, InterfaceType::Float32(4)),
            (1, InterfaceType::Float32(4)),
            (2, InterfaceType::Float32(1)),
            (3, InterfaceType::Float32(2)),
        ];
        if fs.outputs != expected || fs.writes_depth {
            return Err(fail(
                "fragment outputs must be exactly f32x4, f32x4, f32, f32x2 at locations 0..3, without custom depth",
            ));
        }
        let vs = ShaderInterface::inspect(vertex)?;
        for (group, matrices) in [(0, 3), (1, 2)] {
            let valid = vs.uniforms.iter().any(|b| {
                b.group == group
                    && b.binding == 0
                    && b.size == matrices * 64
                    && b.mat4_offsets.as_ref()
                        == Some(&(0..matrices).map(|i| i * 64).collect::<Vec<_>>())
            });
            if !valid {
                return Err(fail(
                    "vertex shader requires camera matrices at group 0/binding 0 (192 bytes) and model matrices at group 1/binding 0 (128 bytes)",
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn validate_material(self, material: &Material) -> Result<(), GraphicsError> {
        let rates = material.set_update_rates();
        let expected = [
            Some(UpdateRate::External),
            Some(UpdateRate::Dynamic),
            Some(UpdateRate::Static),
        ];
        let valid = (rates.len() == 2 || rates.len() == 3)
            && rates == &expected[..rates.len()]
            && material.binding_layouts().len() == rates.len()
            && material.binding_layouts()[..2].iter().all(|l| {
                l.entries.len() == 1
                    && l.entries[0].binding == 0
                    && l.entries[0].binding_type == BindingType::DynamicUniformBuffer
            });
        if !valid {
            return Err(GraphicsError::InvalidParameter("G-buffer V1 requires external camera set 0, dynamic model set 1, and optional static material set 2".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_graphics::ShaderStage;

    #[test]
    fn stock_surfaces_obey_v1() {
        for source in [
            include_str!("../../../../std-assets/shaders/deferred_gbuffer.slang"),
            include_str!("../../../../std-assets/shaders/deferred_gbuffer_textured.slang"),
        ] {
            GBufferContract::V1
                .validate_shaders(
                    &ShaderSource::slang(ShaderStage::Vertex, source, "vs_main", vec![]),
                    &ShaderSource::slang(ShaderStage::Fragment, source, "fs_main", vec![]),
                )
                .unwrap();
        }
    }
}

#[cfg(test)]
mod rejection_tests {
    use super::*;
    const VALID: &str = r#"
struct Camera { raster: mat4x4f, current: mat4x4f, previous: mat4x4f }
struct Model { current: mat4x4f, previous: mat4x4f }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> model: Model;
@vertex fn vs_main(@location(0) p: vec3f) -> @builtin(position) vec4f {
    return camera.raster * model.current * vec4f(p,1);
}
struct Output {
    @location(0) color: vec4f,
    @location(1) normal: vec4f,
    @location(2) roughness: f32,
    @location(3) velocity: vec2f,
}
@fragment fn fs_main() -> Output { return Output(vec4f(1),vec4f(1),1,vec2f(0)); }
"#;
    fn validate(code: &str) -> Result<(), GraphicsError> {
        GBufferContract::V1.validate_shaders(
            &ShaderSource::vertex(code, "vs_main"),
            &ShaderSource::fragment(code, "fs_main"),
        )
    }
    #[test]
    fn rejects_incompatible_outputs_and_camera_abi() {
        validate(VALID).unwrap();
        for bad in [
            VALID.replace("@location(3)", "@location(4)"),
            VALID
                .replace("velocity: vec2f", "velocity: vec4f")
                .replace("vec2f(0)", "vec4f(0)"),
            VALID.replace("@location(2) roughness", "@builtin(frag_depth) roughness"),
            VALID.replace(
                "current: mat4x4f, previous: mat4x4f }",
                "current: mat4x4f, previous: vec4f }",
            ),
            VALID.replace("@group(0)", "@group(4)"),
        ] {
            assert!(
                validate(&bad)
                    .unwrap_err()
                    .to_string()
                    .contains("G-buffer V1")
            );
        }
    }
}
