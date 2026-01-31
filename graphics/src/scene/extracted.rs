//! Extracted render data from ECS components.
//!
//! These types hold render-relevant data copied from ECS components during
//! the extract phase. They are designed to be compact and GPU-upload ready.

use glam::{Mat4, Vec3, Vec4};

/// Extracted transform data from GlobalTransform.
///
/// Contains the model matrix ready for GPU upload.
#[derive(Debug, Clone, Copy)]
pub struct ExtractedTransform {
    /// Model matrix for transforming vertices to world space.
    pub model_matrix: Mat4,
    /// World-space position for sorting and culling.
    pub world_position: Vec3,
}

impl ExtractedTransform {
    /// Creates an identity transform.
    pub const IDENTITY: Self = Self {
        model_matrix: Mat4::IDENTITY,
        world_position: Vec3::ZERO,
    };

    /// Creates an extracted transform from a model matrix.
    #[inline]
    pub fn from_matrix(model_matrix: Mat4) -> Self {
        Self {
            model_matrix,
            world_position: model_matrix.w_axis.truncate(),
        }
    }
}

impl Default for ExtractedTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Extracted mesh data from RenderMesh component.
#[derive(Debug, Clone, Copy)]
pub struct ExtractedMesh {
    /// Handle to the mesh resource (same as ECS MeshHandle).
    pub mesh_id: u64,
    /// Whether this mesh casts shadows.
    pub cast_shadows: bool,
    /// Whether this mesh receives shadows.
    pub receive_shadows: bool,
    /// Render layer mask.
    pub render_layers: u32,
}

impl ExtractedMesh {
    /// Creates a new extracted mesh.
    #[inline]
    pub fn new(mesh_id: u64) -> Self {
        Self {
            mesh_id,
            cast_shadows: true,
            receive_shadows: true,
            render_layers: 1, // DEFAULT layer
        }
    }
}

/// Extracted material data from Material component.
///
/// Contains PBR material parameters ready for GPU upload.
#[derive(Debug, Clone, Copy)]
pub struct ExtractedMaterial {
    /// Base color in linear RGBA.
    pub base_color: Vec4,
    /// Metallic factor (0.0 to 1.0).
    pub metallic: f32,
    /// Roughness factor (0.0 to 1.0).
    pub roughness: f32,
    /// Emissive color in linear RGB.
    pub emissive: Vec3,
    /// Alpha mode: 0 = Opaque, 1 = Mask, 2 = Blend.
    pub alpha_mode: u8,
    /// Alpha cutoff for mask mode (0-255).
    pub alpha_cutoff: u8,
    /// Whether double-sided rendering is enabled.
    pub double_sided: bool,
    /// Base color texture ID (0 = none).
    pub base_color_texture: u64,
    /// Normal map texture ID (0 = none).
    pub normal_texture: u64,
    /// Metallic-roughness texture ID (0 = none).
    pub metallic_roughness_texture: u64,
}

impl ExtractedMaterial {
    /// Creates a default white material.
    pub const DEFAULT: Self = Self {
        base_color: Vec4::ONE,
        metallic: 0.0,
        roughness: 0.5,
        emissive: Vec3::ZERO,
        alpha_mode: 0, // Opaque
        alpha_cutoff: 127,
        double_sided: false,
        base_color_texture: 0,
        normal_texture: 0,
        metallic_roughness_texture: 0,
    };

    /// Creates a simple colored material.
    #[inline]
    pub fn colored(color: Vec4) -> Self {
        Self {
            base_color: color,
            ..Self::DEFAULT
        }
    }
}

impl Default for ExtractedMaterial {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A complete render item combining transform, mesh, and material.
///
/// This represents a single drawable entity in the render world.
#[derive(Debug, Clone, Copy)]
pub struct RenderItem {
    /// Extracted transform data.
    pub transform: ExtractedTransform,
    /// Extracted mesh data.
    pub mesh: ExtractedMesh,
    /// Extracted material data.
    pub material: ExtractedMaterial,
    /// Entity ID for debugging and identification.
    pub entity_id: u64,
}

impl RenderItem {
    /// Creates a new render item.
    #[inline]
    pub fn new(
        entity_id: u64,
        transform: ExtractedTransform,
        mesh: ExtractedMesh,
        material: ExtractedMaterial,
    ) -> Self {
        Self {
            entity_id,
            transform,
            mesh,
            material,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_transform_from_matrix() {
        let matrix = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let extracted = ExtractedTransform::from_matrix(matrix);
        assert_eq!(extracted.world_position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn extracted_material_default() {
        let mat = ExtractedMaterial::DEFAULT;
        assert_eq!(mat.base_color, Vec4::ONE);
        assert_eq!(mat.alpha_mode, 0);
    }
}
