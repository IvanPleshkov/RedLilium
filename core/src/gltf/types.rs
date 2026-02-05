//! Data types for glTF loading results.

use std::sync::Arc;

use crate::material::CpuMaterial;
use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;
use crate::scene::Scene;
use crate::texture::CpuTexture;

/// A loaded glTF document containing all scenes and resources.
///
/// Scenes hold their own meshes, cameras, skins, and animations (see [`Scene`]).
/// The document holds shared resources (materials, textures, samplers)
/// that are referenced by index from scene resources.
#[derive(Debug)]
pub struct GltfDocument {
    /// All scenes in the document.
    pub scenes: Vec<Scene>,
    /// Index of the default scene, if specified.
    pub default_scene: Option<usize>,
    /// All materials.
    pub materials: Vec<CpuMaterial>,
    /// All textures (CPU-side pixel data).
    pub textures: Vec<CpuTexture>,
    /// All samplers.
    pub samplers: Vec<CpuSampler>,
    /// New vertex layouts created during loading (not found in shared_layouts).
    pub new_layouts: Vec<Arc<VertexLayout>>,
}
