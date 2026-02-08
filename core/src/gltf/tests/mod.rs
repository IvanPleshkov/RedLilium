use std::sync::Arc;

use crate::gltf::GltfMaterial;
use crate::material::{CpuMaterial, CpuMaterialInstance, MaterialValue};
use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;

mod load_test;
mod roundtrip_test;

/// Default sampler callback for tests.
///
/// Simply wraps the parsed sampler in an `Arc`.
fn default_sampler_fn(sampler: &CpuSampler) -> Arc<CpuSampler> {
    Arc::new(sampler.clone())
}

/// Default PBR material callback for tests.
///
/// Creates a `CpuMaterialInstance` using `CpuMaterial::pbr_metallic_roughness`
/// and the native vertex layout from the glTF primitive.
fn default_pbr_material(mat: &GltfMaterial, layout: &VertexLayout) -> Arc<CpuMaterialInstance> {
    let declaration = CpuMaterial::pbr_metallic_roughness(
        Arc::new(layout.clone()),
        mat.alpha_mode,
        mat.double_sided,
        mat.base_color_texture.is_some(),
        mat.metallic_roughness_texture.is_some(),
        mat.normal_texture.is_some(),
        mat.occlusion_texture.is_some(),
        mat.emissive_texture.is_some(),
    );
    let decl = Arc::new(declaration);

    let mut values = vec![
        MaterialValue::Vec4(mat.base_color_factor),
        MaterialValue::Float(mat.metallic_factor),
        MaterialValue::Float(mat.roughness_factor),
        MaterialValue::Vec3(mat.emissive_factor),
        MaterialValue::Float(mat.normal_scale),
        MaterialValue::Float(mat.occlusion_strength),
    ];

    if let Some(t) = &mat.base_color_texture {
        values.push(MaterialValue::Texture(t.clone()));
    }
    if let Some(t) = &mat.metallic_roughness_texture {
        values.push(MaterialValue::Texture(t.clone()));
    }
    if let Some(t) = &mat.normal_texture {
        values.push(MaterialValue::Texture(t.clone()));
    }
    if let Some(t) = &mat.occlusion_texture {
        values.push(MaterialValue::Texture(t.clone()));
    }
    if let Some(t) = &mat.emissive_texture {
        values.push(MaterialValue::Texture(t.clone()));
    }

    Arc::new(CpuMaterialInstance {
        material: decl,
        name: mat.name.clone(),
        values,
    })
}
