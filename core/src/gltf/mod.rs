//! glTF 2.0 loader and exporter.
//!
//! Loads `.gltf`/`.glb` files into CPU-side data structures including
//! meshes, materials, textures, animations, skins, and scene graphs.
//! Exports scenes and materials back to binary glTF (`.glb`) format.
//!
//! # Layout and Sampler Sharing
//!
//! The loader takes slices of `Arc<VertexLayout>` and `Arc<CpuSampler>` and
//! reuses matching instances via structural equality (ignoring labels/names).
//! New resources created during loading are returned in
//! [`GltfDocument::new_layouts`] and [`GltfDocument::new_samplers`].
//!
//! # Material Callback
//!
//! The loader delegates material creation to a user-provided callback. For each
//! glTF material, the loader extracts PBR properties into a [`GltfMaterial`]
//! and passes it to the callback, which returns an `Arc<CpuMaterialInstance>`.
//! This lets the caller map glTF material data to their own shader system.
//!
//! # Example
//!
//! ```ignore
//! use redlilium_core::gltf::{load_gltf, GltfMaterial};
//! use redlilium_core::material::*;
//! use std::sync::Arc;
//!
//! let data = std::fs::read("model.glb").unwrap();
//! let doc = load_gltf(&data, &[], &[], |mat: &GltfMaterial| {
//!     let layout = Arc::new(VertexLayout::new());
//!     let decl = CpuMaterial::pbr_metallic_roughness(
//!         layout, mat.alpha_mode, mat.double_sided,
//!         mat.base_color_texture.is_some(),
//!         mat.metallic_roughness_texture.is_some(),
//!         mat.normal_texture.is_some(),
//!         mat.occlusion_texture.is_some(),
//!         mat.emissive_texture.is_some(),
//!     );
//!     // ... build instance from decl and mat properties ...
//!     Arc::new(CpuMaterialInstance::new(Arc::new(decl)))
//! }).unwrap();
//!
//! println!("Scenes: {}", doc.scenes.len());
//! println!("Meshes: {}", doc.scenes[0].meshes.len());
//! ```

mod error;
mod exporter;
mod loader;
#[cfg(test)]
mod tests;
pub mod types;
mod vertex;

pub use error::GltfError;
pub use types::*;

use std::sync::Arc;

use crate::material::CpuMaterialInstance;
use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;
use crate::scene::Scene;

/// Load a glTF document from binary data.
///
/// Supports both binary glTF (`.glb`) and JSON glTF (`.gltf` with embedded
/// data URIs). External file references are not supported.
///
/// # Arguments
///
/// * `data` - Raw bytes of the `.glb` or `.gltf` file.
/// * `shared_layouts` - Existing vertex layouts to share. The loader will
///   reuse layouts that match structurally (same attributes, formats, offsets,
///   buffers — label is ignored).
/// * `shared_samplers` - Existing samplers to share. The loader will reuse
///   samplers that match structurally (same filter modes, address modes, LOD
///   clamps, compare function, anisotropy — name is ignored).
/// * `material_fn` - Callback invoked for each glTF material. Receives a
///   [`GltfMaterial`] with parsed PBR properties and resolved textures.
///   Returns an `Arc<CpuMaterialInstance>` to assign to meshes.
///
/// # Returns
///
/// A [`GltfDocument`] containing all loaded scenes with meshes, cameras,
/// skins, and animations. Material instances are embedded in each mesh via
/// `Arc<CpuMaterialInstance>`. New vertex layouts and samplers created during
/// loading are in [`GltfDocument::new_layouts`] and
/// [`GltfDocument::new_samplers`].
pub fn load_gltf(
    data: &[u8],
    shared_layouts: &[Arc<VertexLayout>],
    shared_samplers: &[Arc<CpuSampler>],
    mut material_fn: impl FnMut(&GltfMaterial) -> Arc<CpuMaterialInstance>,
) -> Result<GltfDocument, GltfError> {
    let gltf = gltf_dep::Gltf::from_slice(data)?;
    let blob = gltf.blob.clone();

    let buffers = loader::resolve_buffers(&gltf.document, blob)?;
    let mut ctx = loader::LoadContext::new(gltf.document, buffers, shared_layouts, shared_samplers);

    ctx.load_textures()?;
    ctx.load_samplers();
    ctx.load_materials(&mut material_fn);
    let cameras = ctx.load_cameras();
    let meshes = ctx.load_meshes()?;
    let skins = ctx.load_skins()?;
    let animations = ctx.load_animations()?;
    let scenes = ctx.load_scenes(meshes, cameras, skins, animations);
    let default_scene = ctx.default_scene();
    let (new_layouts, new_samplers) = ctx.into_new_resources();

    Ok(GltfDocument {
        scenes,
        default_scene,
        new_layouts,
        new_samplers,
    })
}

/// Export scenes to a binary glTF (`.glb`) file.
///
/// Material instances, textures, and samplers are collected from meshes via
/// their `Arc<CpuMaterialInstance>` references and deduplicated using Arc
/// pointer identity.
///
/// # Texture handling
///
/// - [`TextureSource::Cpu`] — encodes RGBA8 data as PNG and embeds it in the GLB.
/// - [`TextureSource::Named`] — saved as an external texture URI reference.
///
/// # Example
///
/// ```ignore
/// use redlilium_core::gltf::save_gltf;
///
/// let scenes: Vec<&Scene> = doc.scenes.iter().collect();
/// let glb = save_gltf(&scenes, doc.default_scene).unwrap();
/// std::fs::write("output.glb", &glb).unwrap();
/// ```
pub fn save_gltf(scenes: &[&Scene], default_scene: Option<usize>) -> Result<Vec<u8>, GltfError> {
    let mut ctx = exporter::ExportContext::new();

    ctx.collect_resources(scenes);
    ctx.build_images()?;
    ctx.build_samplers();
    ctx.build_materials();
    ctx.build_scenes(scenes)?;
    ctx.set_default_scene(default_scene);
    ctx.finalize_buffer();
    ctx.to_glb()
}
