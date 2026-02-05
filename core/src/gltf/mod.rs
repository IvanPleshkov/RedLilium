//! glTF 2.0 loader.
//!
//! Loads `.gltf`/`.glb` files into CPU-side data structures including
//! meshes, materials, textures, animations, skins, and scene graphs.
//!
//! # Layout and Sampler Sharing
//!
//! The loader takes slices of `Arc<VertexLayout>` and `Arc<CpuSampler>` and
//! reuses matching instances via structural equality (ignoring labels/names).
//! New layouts and samplers created during loading are returned in
//! [`GltfDocument::new_layouts`] and [`GltfDocument::new_samplers`].
//!
//! # Example
//!
//! ```ignore
//! use redlilium_core::gltf::load_gltf;
//! use redlilium_core::mesh::VertexLayout;
//!
//! let data = std::fs::read("model.glb").unwrap();
//! let shared_layouts = vec![VertexLayout::position_normal_uv()];
//! let doc = load_gltf(&data, &shared_layouts, &[]).unwrap();
//!
//! println!("Scenes: {}", doc.scenes.len());
//! println!("Meshes: {}", doc.scenes[0].meshes.len());
//! println!("New layouts: {}", doc.new_layouts.len());
//! println!("New samplers: {}", doc.new_samplers.len());
//! ```

mod error;
mod loader;
#[cfg(test)]
mod tests;
pub mod types;
mod vertex;

pub use error::GltfError;
pub use types::*;

use std::sync::Arc;

use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;

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
///
/// # Returns
///
/// A [`GltfDocument`] containing all loaded scenes, meshes, materials,
/// cameras, and skins. Textures and samplers are embedded in material
/// [`TextureRef`] entries via `Arc<CpuTexture>` and `Arc<CpuSampler>`.
/// Animations are embedded in each [`Scene`]. New vertex layouts and
/// samplers created during loading are in [`GltfDocument::new_layouts`]
/// and [`GltfDocument::new_samplers`].
pub fn load_gltf(
    data: &[u8],
    shared_layouts: &[Arc<VertexLayout>],
    shared_samplers: &[Arc<CpuSampler>],
) -> Result<GltfDocument, GltfError> {
    let gltf = gltf_dep::Gltf::from_slice(data)?;
    let blob = gltf.blob.clone();

    let buffers = loader::resolve_buffers(&gltf.document, blob)?;
    let mut ctx = loader::LoadContext::new(gltf.document, buffers, shared_layouts, shared_samplers);

    ctx.load_textures()?;
    ctx.load_samplers();
    let materials = ctx.load_materials();
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
        materials,
        new_layouts,
        new_samplers,
    })
}
