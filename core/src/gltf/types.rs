//! Data types for glTF loading results.

use std::sync::Arc;

use crate::material::CpuMaterial;
use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;
use crate::scene::Scene;

/// A loaded glTF document containing all scenes and resources.
///
/// Scenes hold their own meshes, cameras, skins, and animations (see [`Scene`]).
/// Materials hold their own textures and samplers via `Arc<CpuTexture>` and
/// `Arc<CpuSampler>` embedded in [`TextureRef`].
#[derive(Debug)]
pub struct GltfDocument {
    /// All scenes in the document.
    pub scenes: Vec<Scene>,
    /// Index of the default scene, if specified.
    pub default_scene: Option<usize>,
    /// All materials.
    pub materials: Vec<CpuMaterial>,
    /// New vertex layouts created during loading (not found in shared_layouts).
    pub new_layouts: Vec<Arc<VertexLayout>>,
    /// New samplers created during loading (not found in shared_samplers).
    pub new_samplers: Vec<Arc<CpuSampler>>,
}
