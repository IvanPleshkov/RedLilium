//! Data types for glTF loading results.
//!
//! All types use plain arrays (`[f32; 3]`, `[f32; 4]`, etc.) instead of
//! math library types to keep the core crate dependency-free of `glam`.

use std::sync::Arc;

use crate::mesh::{CpuMesh, VertexLayout};

/// A loaded glTF document containing all scenes and resources.
#[derive(Debug)]
pub struct GltfDocument {
    /// All scenes in the document.
    pub scenes: Vec<GltfScene>,
    /// Index of the default scene, if specified.
    pub default_scene: Option<usize>,
    /// All meshes as a flat list. Each `CpuMesh` carries its material
    /// index via [`CpuMesh::material()`]. Nodes reference meshes by
    /// indices into this array.
    pub meshes: Vec<CpuMesh>,
    /// All materials.
    pub materials: Vec<GltfMaterial>,
    /// All textures (reference an image + sampler).
    pub textures: Vec<GltfTexture>,
    /// All decoded images (RGBA8 pixel data).
    pub images: Vec<GltfImage>,
    /// All samplers.
    pub samplers: Vec<GltfSampler>,
    /// All cameras.
    pub cameras: Vec<GltfCamera>,
    /// All animations.
    pub animations: Vec<GltfAnimation>,
    /// All skins.
    pub skins: Vec<GltfSkin>,
    /// New vertex layouts created during loading (not found in shared_layouts).
    pub new_layouts: Vec<Arc<VertexLayout>>,
}

// -- Scene & Nodes --

/// A glTF scene containing a tree of nodes.
#[derive(Debug)]
pub struct GltfScene {
    /// Scene name.
    pub name: Option<String>,
    /// Root nodes of the scene.
    pub nodes: Vec<GltfNode>,
}

/// A node in the scene graph.
#[derive(Debug)]
pub struct GltfNode {
    /// Node name.
    pub name: Option<String>,
    /// Local transform (translation, rotation, scale).
    pub transform: GltfTransform,
    /// Indices into `GltfDocument::meshes` (one per primitive of the
    /// original glTF mesh). Empty if the node has no mesh.
    pub meshes: Vec<usize>,
    /// Index into `GltfDocument::cameras`.
    pub camera: Option<usize>,
    /// Index into `GltfDocument::skins`.
    pub skin: Option<usize>,
    /// Child nodes.
    pub children: Vec<GltfNode>,
}

/// Node transform decomposed into translation, rotation, and scale.
#[derive(Debug, Clone)]
pub struct GltfTransform {
    /// Translation [x, y, z].
    pub translation: [f32; 3],
    /// Rotation quaternion [x, y, z, w].
    pub rotation: [f32; 4],
    /// Scale [x, y, z].
    pub scale: [f32; 3],
}

impl Default for GltfTransform {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
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

/// Reference to a texture with a specific tex coord set.
#[derive(Debug, Clone)]
pub struct GltfTextureRef {
    /// Index into `GltfDocument::textures`.
    pub index: usize,
    /// Texture coordinate set index (0 or 1).
    pub tex_coord: u32,
}

/// Normal texture reference with scale.
#[derive(Debug, Clone)]
pub struct GltfNormalTextureRef {
    /// Index into `GltfDocument::textures`.
    pub index: usize,
    /// Texture coordinate set index.
    pub tex_coord: u32,
    /// Normal map scale factor.
    pub scale: f32,
}

/// Occlusion texture reference with strength.
#[derive(Debug, Clone)]
pub struct GltfOcclusionTextureRef {
    /// Index into `GltfDocument::textures`.
    pub index: usize,
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

// -- Textures & Images --

/// A texture combining an image and a sampler.
#[derive(Debug, Clone)]
pub struct GltfTexture {
    /// Texture name.
    pub name: Option<String>,
    /// Index into `GltfDocument::images`.
    pub image: usize,
    /// Index into `GltfDocument::samplers`, if any.
    pub sampler: Option<usize>,
}

/// A decoded image (always RGBA8).
#[derive(Debug, Clone)]
pub struct GltfImage {
    /// Image name.
    pub name: Option<String>,
    /// Raw RGBA8 pixel data.
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

/// Texture sampler parameters.
#[derive(Debug, Clone)]
pub struct GltfSampler {
    /// Magnification filter.
    pub mag_filter: Option<GltfFilter>,
    /// Minification filter.
    pub min_filter: Option<GltfFilter>,
    /// Wrapping mode for S (U) coordinate.
    pub wrap_s: GltfWrapping,
    /// Wrapping mode for T (V) coordinate.
    pub wrap_t: GltfWrapping,
}

/// Texture filter mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GltfFilter {
    /// Nearest-neighbor filtering.
    Nearest,
    /// Linear (bilinear) filtering.
    Linear,
}

/// Texture wrapping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GltfWrapping {
    /// Clamp to edge.
    ClampToEdge,
    /// Mirrored repeat.
    MirroredRepeat,
    /// Repeat (tile).
    #[default]
    Repeat,
}

// -- Cameras --

/// A camera definition.
#[derive(Debug, Clone)]
pub struct GltfCamera {
    /// Camera name.
    pub name: Option<String>,
    /// Projection type and parameters.
    pub projection: GltfCameraProjection,
}

/// Camera projection parameters.
#[derive(Debug, Clone)]
pub enum GltfCameraProjection {
    /// Perspective projection.
    Perspective {
        /// Vertical field of view in radians.
        yfov: f32,
        /// Aspect ratio (width/height), if specified.
        aspect: Option<f32>,
        /// Near clipping plane distance.
        znear: f32,
        /// Far clipping plane distance, if specified.
        zfar: Option<f32>,
    },
    /// Orthographic projection.
    Orthographic {
        /// Horizontal magnification.
        xmag: f32,
        /// Vertical magnification.
        ymag: f32,
        /// Near clipping plane distance.
        znear: f32,
        /// Far clipping plane distance.
        zfar: f32,
    },
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

// -- Skins --

/// A skin for skeletal animation.
#[derive(Debug, Clone)]
pub struct GltfSkin {
    /// Skin name.
    pub name: Option<String>,
    /// Joint node indices (in glTF document node order).
    pub joints: Vec<usize>,
    /// Inverse bind matrices (column-major 4x4, one per joint).
    pub inverse_bind_matrices: Vec<[f32; 16]>,
    /// Root skeleton node index, if specified.
    pub skeleton: Option<usize>,
}
