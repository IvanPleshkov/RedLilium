//! Data types for glTF loading results.
//!
//! All types use plain arrays (`[f32; 3]`, `[f32; 4]`, etc.) instead of
//! math library types to keep the core crate dependency-free of `glam`.

use std::sync::Arc;

use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;
use crate::scene::Scene;
use crate::texture::CpuTexture;

/// A loaded glTF document containing all scenes and resources.
///
/// Scenes hold their own meshes, cameras, and skins (see [`Scene`]).
/// The document holds glTF-specific resources (materials, textures,
/// samplers, animations) that are shared across scenes.
#[derive(Debug)]
pub struct GltfDocument {
    /// All scenes in the document.
    pub scenes: Vec<Scene>,
    /// Index of the default scene, if specified.
    pub default_scene: Option<usize>,
    /// All materials.
    pub materials: Vec<GltfMaterial>,
    /// All textures (CPU-side pixel data).
    pub textures: Vec<CpuTexture>,
    /// All samplers.
    pub samplers: Vec<CpuSampler>,
    /// All animations.
    pub animations: Vec<GltfAnimation>,
    /// New vertex layouts created during loading (not found in shared_layouts).
    pub new_layouts: Vec<Arc<VertexLayout>>,
}

// -- Materials --

/// PBR metallic-roughness material.
#[derive(Debug, Clone)]
pub struct GltfMaterial {
    /// Material name.
    pub name: Option<String>,
    /// Base color factor [r, g, b, a].
    pub base_color_factor: [f32; 4],
    /// Base color texture.
    pub base_color_texture: Option<GltfTextureRef>,
    /// Metallic factor (0.0 = dielectric, 1.0 = metal).
    pub metallic_factor: f32,
    /// Roughness factor (0.0 = smooth, 1.0 = rough).
    pub roughness_factor: f32,
    /// Metallic-roughness texture (B=metallic, G=roughness).
    pub metallic_roughness_texture: Option<GltfTextureRef>,
    /// Normal map.
    pub normal_texture: Option<GltfNormalTextureRef>,
    /// Occlusion map.
    pub occlusion_texture: Option<GltfOcclusionTextureRef>,
    /// Emissive factor [r, g, b].
    pub emissive_factor: [f32; 3],
    /// Emissive texture.
    pub emissive_texture: Option<GltfTextureRef>,
    /// Alpha rendering mode.
    pub alpha_mode: GltfAlphaMode,
    /// Alpha cutoff threshold (for Mask mode).
    pub alpha_cutoff: f32,
    /// Whether the material is double-sided.
    pub double_sided: bool,
}

/// Reference to a texture with a specific tex coord set and optional sampler.
#[derive(Debug, Clone)]
pub struct GltfTextureRef {
    /// Index into `GltfDocument::textures`.
    pub texture: usize,
    /// Index into `GltfDocument::samplers`, if any.
    pub sampler: Option<usize>,
    /// Texture coordinate set index (0 or 1).
    pub tex_coord: u32,
}

/// Normal texture reference with scale.
#[derive(Debug, Clone)]
pub struct GltfNormalTextureRef {
    /// Index into `GltfDocument::textures`.
    pub texture: usize,
    /// Index into `GltfDocument::samplers`, if any.
    pub sampler: Option<usize>,
    /// Texture coordinate set index.
    pub tex_coord: u32,
    /// Normal map scale factor.
    pub scale: f32,
}

/// Occlusion texture reference with strength.
#[derive(Debug, Clone)]
pub struct GltfOcclusionTextureRef {
    /// Index into `GltfDocument::textures`.
    pub texture: usize,
    /// Index into `GltfDocument::samplers`, if any.
    pub sampler: Option<usize>,
    /// Texture coordinate set index.
    pub tex_coord: u32,
    /// Occlusion strength (0.0 = no occlusion, 1.0 = full occlusion).
    pub strength: f32,
}

/// Alpha rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GltfAlphaMode {
    /// Fully opaque (alpha ignored).
    #[default]
    Opaque,
    /// Alpha masking with cutoff threshold.
    Mask,
    /// Alpha blending.
    Blend,
}

// -- Animations --

/// An animation containing one or more channels.
#[derive(Debug, Clone)]
pub struct GltfAnimation {
    /// Animation name.
    pub name: Option<String>,
    /// Animation channels (one per target node + property).
    pub channels: Vec<GltfAnimationChannel>,
}

/// A single animation channel targeting a node property.
#[derive(Debug, Clone)]
pub struct GltfAnimationChannel {
    /// Target node index in the glTF document.
    pub target_node: usize,
    /// The property being animated.
    pub property: GltfAnimationProperty,
    /// Interpolation method.
    pub interpolation: GltfInterpolation,
    /// Keyframe timestamps in seconds.
    pub timestamps: Vec<f32>,
    /// Keyframe values (flat array, stride depends on property).
    pub values: Vec<f32>,
}

/// The property being animated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GltfAnimationProperty {
    /// Translation [x, y, z] — 3 floats per keyframe.
    Translation,
    /// Rotation quaternion [x, y, z, w] — 4 floats per keyframe.
    Rotation,
    /// Scale [x, y, z] — 3 floats per keyframe.
    Scale,
    /// Morph target weights — N floats per keyframe.
    MorphTargetWeights,
}

/// Animation interpolation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GltfInterpolation {
    /// Linear interpolation.
    #[default]
    Linear,
    /// Step (nearest) interpolation.
    Step,
    /// Cubic spline interpolation (with in/out tangents).
    CubicSpline,
}
