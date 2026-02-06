use std::sync::Arc;

use crate::gltf::GltfMaterial;
use crate::material::{CpuMaterial, CpuMaterialInstance, MaterialValue};

mod load_test;
mod roundtrip_test;

/// Default PBR material callback for tests.
///
/// Creates a `CpuMaterialInstance` using `CpuMaterial::pbr_metallic_roughness`,
/// replicating the old built-in loader behavior.
fn default_pbr_material(mat: &GltfMaterial) -> Arc<CpuMaterialInstance> {
    let declaration = CpuMaterial::pbr_metallic_roughness(
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
