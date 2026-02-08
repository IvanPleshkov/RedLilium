//! Data types for glTF loading results.

use crate::material::{AlphaMode, TextureRef};
use crate::scene::Scene;

/// A loaded glTF document containing all scenes and resources.
///
/// Scenes hold their own meshes, cameras, skins, and animations (see [`Scene`]).
/// Material instances are stored in each [`Scene`]'s `materials` array.
/// Meshes reference materials by index via `CpuMesh::material()`. Textures
/// and samplers are embedded in material [`TextureRef`] entries via
/// `Arc<CpuTexture>` and `Arc<CpuSampler>`.
///
/// Vertex layouts are controlled entirely by the `material_fn` callback
/// passed to [`load_gltf`](super::load_gltf). Samplers are controlled by
/// the `sampler_fn` callback. Both callbacks receive raw glTF data and
/// return engine-side `Arc` resources.
#[derive(Debug)]
pub struct GltfDocument {
    /// All scenes in the document.
    pub scenes: Vec<Scene>,
    /// Index of the default scene, if specified.
    pub default_scene: Option<usize>,
}

/// Parsed glTF PBR metallic-roughness material properties.
///
/// Passed to the user-provided material callback along with the native
/// [`VertexLayout`](crate::mesh::VertexLayout) from the mesh primitive.
/// The callback can use the provided layout or choose a different one
/// (the loader will adapt the vertex data if needed).
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
