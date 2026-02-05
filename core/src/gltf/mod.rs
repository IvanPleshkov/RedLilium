//! glTF 2.0 loader.
//!
//! Loads `.gltf`/`.glb` files into CPU-side data structures including
//! meshes, materials, textures, animations, skins, and scene graphs.
//!
//! # Layout Sharing
//!
//! The loader takes a slice of `Arc<VertexLayout>` and reuses matching
//! layouts via structural equality (ignoring labels). New layouts created
//! during loading are returned in [`GltfDocument::new_layouts`].
//!
//! # Example
//!
//! ```ignore
//! use redlilium_core::gltf::load_gltf;
//! use redlilium_core::mesh::VertexLayout;
//!
//! let data = std::fs::read("model.glb").unwrap();
//! let shared = vec![VertexLayout::position_normal_uv()];
//! let doc = load_gltf(&data, &shared).unwrap();
//!
//! println!("Scenes: {}", doc.scenes.len());
//! println!("Meshes: {}", doc.meshes.len());
//! println!("New layouts: {}", doc.new_layouts.len());
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
///
/// # Returns
///
/// A [`GltfDocument`] containing all loaded scenes, meshes, materials,
/// textures, images, cameras, animations, and skins. New vertex layouts
/// created during loading are in [`GltfDocument::new_layouts`].
pub fn load_gltf(
    data: &[u8],
    shared_layouts: &[Arc<VertexLayout>],
) -> Result<GltfDocument, GltfError> {
    let gltf = gltf_dep::Gltf::from_slice(data)?;
    let blob = gltf.blob.clone();

    let buffers = loader::resolve_buffers(&gltf.document, blob)?;
    let mut ctx = loader::LoadContext::new(gltf.document, buffers, shared_layouts);

    let images = ctx.load_images()?;
    let samplers = ctx.load_samplers();
    let textures = ctx.load_textures();
    let materials = ctx.load_materials();
    let cameras = ctx.load_cameras();
    let meshes = ctx.load_meshes()?;
    let skins = ctx.load_skins()?;
    let animations = ctx.load_animations()?;
    let scenes = ctx.load_scenes();
    let default_scene = ctx.default_scene();
    let new_layouts = ctx.into_new_layouts();

    Ok(GltfDocument {
        scenes,
        default_scene,
        meshes,
        materials,
        textures,
        images,
        samplers,
        cameras,
        animations,
        skins,
        new_layouts,
    })
}
