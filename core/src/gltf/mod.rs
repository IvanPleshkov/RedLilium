//! glTF 2.0 loader and exporter.
//!
//! Loads `.gltf`/`.glb` files into CPU-side data structures including
//! meshes, materials, textures, animations, skins, and scene graphs.
//! Exports scenes and materials back to binary glTF (`.glb`) format.
//!
//! # Layout, Sampler, and Material Sharing
//!
//! The loader takes slices of `Arc<VertexLayout>`, `Arc<CpuSampler>`, and
//! `Arc<CpuMaterial>` and reuses matching instances via structural equality
//! (ignoring labels/names). New resources created during loading are returned
//! in [`GltfDocument::new_layouts`], [`GltfDocument::new_samplers`], and
//! [`GltfDocument::new_materials`].
//!
//! # Example
//!
//! ```ignore
//! use redlilium_core::gltf::load_gltf;
//! use redlilium_core::mesh::VertexLayout;
//!
//! let data = std::fs::read("model.glb").unwrap();
//! let shared_layouts = vec![VertexLayout::position_normal_uv()];
//! let doc = load_gltf(&data, &shared_layouts, &[], &[]).unwrap();
//!
//! println!("Scenes: {}", doc.scenes.len());
//! println!("Meshes: {}", doc.scenes[0].meshes.len());
//! println!("New layouts: {}", doc.new_layouts.len());
//! println!("New samplers: {}", doc.new_samplers.len());
//! println!("New materials: {}", doc.new_materials.len());
//! ```

mod error;
mod exporter;
mod loader;
#[cfg(test)]
mod tests;
pub mod types;
mod vertex;

pub use error::GltfError;
pub use exporter::save_gltf;
pub use types::*;

use std::sync::Arc;

use crate::material::CpuMaterial;
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
/// * `shared_materials` - Existing materials to share. The loader will reuse
///   materials that match structurally (same properties, alpha mode,
///   double-sided — name is ignored).
///
/// # Returns
///
/// A [`GltfDocument`] containing all loaded scenes with meshes, cameras,
/// skins, and animations. Materials are embedded in each mesh via
/// `Arc<CpuMaterial>`. New vertex layouts, samplers, and materials created
/// during loading are in [`GltfDocument::new_layouts`],
/// [`GltfDocument::new_samplers`], and [`GltfDocument::new_materials`].
pub fn load_gltf(
    data: &[u8],
    shared_layouts: &[Arc<VertexLayout>],
    shared_samplers: &[Arc<CpuSampler>],
    shared_materials: &[Arc<CpuMaterial>],
) -> Result<GltfDocument, GltfError> {
    let gltf = gltf_dep::Gltf::from_slice(data)?;
    let blob = gltf.blob.clone();

    let buffers = loader::resolve_buffers(&gltf.document, blob)?;
    let mut ctx = loader::LoadContext::new(
        gltf.document,
        buffers,
        shared_layouts,
        shared_samplers,
        shared_materials,
    );

    ctx.load_textures()?;
    ctx.load_samplers();
    ctx.load_materials();
    let cameras = ctx.load_cameras();
    let meshes = ctx.load_meshes()?;
    let skins = ctx.load_skins()?;
    let animations = ctx.load_animations()?;
    let scenes = ctx.load_scenes(meshes, cameras, skins, animations);
    let default_scene = ctx.default_scene();
    let (new_layouts, new_samplers, new_materials) = ctx.into_new_resources();

    Ok(GltfDocument {
        scenes,
        default_scene,
        new_layouts,
        new_samplers,
        new_materials,
    })
}
