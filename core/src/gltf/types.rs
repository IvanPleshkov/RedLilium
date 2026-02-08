//! Data types for glTF loading results.

use std::sync::Arc;

use crate::material::{AlphaMode, TextureRef};
use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;
use crate::scene::Scene;

/// A loaded glTF document containing all scenes and resources.
///
/// Scenes hold their own meshes, cameras, skins, and animations (see [`Scene`]).
/// Material instances are stored in each [`Scene`]'s `materials` array.
/// Meshes reference materials by index via `CpuMesh::material()`. Textures
/// and samplers are embedded in material [`TextureRef`] entries via
/// `Arc<CpuTexture>` and `Arc<CpuSampler>`.
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
}

/// Parsed glTF PBR metallic-roughness material properties.
///
/// Passed to the user-provided material callback during loading so the caller
/// can map glTF material data to their own shader/material system.
///
/// Textures are already decoded and resolved as [`TextureRef`] values.
#[derive(Debug, Clone)]
pub struct GltfMaterial {
    /// Material name from the glTF file.
    pub name: Option<String>,
    /// Alpha rendering mode.
    pub alpha_mode: AlphaMode,
    /// Whether the material is double-sided.
    pub double_sided: bool,
    /// Base color factor (linear RGBA).
    pub base_color_factor: [f32; 4],
    /// Metallic factor.
    pub metallic_factor: f32,
    /// Roughness factor.
    pub roughness_factor: f32,
    /// Emissive factor (linear RGB).
    pub emissive_factor: [f32; 3],
    /// Normal map scale.
    pub normal_scale: f32,
    /// Occlusion strength.
    pub occlusion_strength: f32,
    /// Base color texture.
    pub base_color_texture: Option<TextureRef>,
    /// Metallic-roughness texture.
    pub metallic_roughness_texture: Option<TextureRef>,
    /// Normal map texture.
    pub normal_texture: Option<TextureRef>,
    /// Occlusion texture.
    pub occlusion_texture: Option<TextureRef>,
    /// Emissive texture.
    pub emissive_texture: Option<TextureRef>,
}
