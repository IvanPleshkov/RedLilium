//! Data types for glTF loading results.

use std::sync::Arc;

use crate::material::CpuMaterialInstance;
use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;
use crate::scene::Scene;

/// A loaded glTF document containing all scenes and resources.
///
/// Scenes hold their own meshes, cameras, skins, and animations (see [`Scene`]).
/// Material instances are embedded in each [`CpuMesh`] via `Arc<CpuMaterialInstance>`,
/// shared across meshes that reference the same material. Textures and samplers
/// are embedded in material [`TextureRef`] entries via `Arc<CpuTexture>` and
/// `Arc<CpuSampler>`.
#[derive(Debug)]
pub struct GltfDocument {
    /// All scenes in the document.
    pub scenes: Vec<Scene>,
    /// Index of the default scene, if specified.
    pub default_scene: Option<usize>,
    /// New vertex layouts created during loading (not found in shared_layouts).
    pub new_layouts: Vec<Arc<VertexLayout>>,
    /// New samplers created during loading (not found in shared_samplers).
    pub new_samplers: Vec<Arc<CpuSampler>>,
    /// New material instances created during loading (not found in shared_instances).
    pub new_instances: Vec<Arc<CpuMaterialInstance>>,
}
