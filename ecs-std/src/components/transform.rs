use glam::{Mat4, Quat, Vec3};
use redlilium_core::scene::NodeTransform;

/// Local transform of an entity using glam types for runtime math.
///
/// Stores translation, rotation, and scale. Convertible to/from
/// core's [`NodeTransform`] which uses plain `[f32; N]` arrays.
///
/// Padding fields (`_pad*`) are required for `bytemuck::Pod` because
/// Quat has 16-byte SIMD alignment on x86_64.
#[derive(
    Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, redlilium_ecs::Component,
)]
#[repr(C)]
pub struct Transform {
    /// Translation in world units.
    pub translation: Vec3,
    _pad0: f32,
    /// Rotation as a unit quaternion.
    pub rotation: Quat,
    /// Non-uniform scale.
    pub scale: Vec3,
    _pad1: f32,
}

impl Transform {
    /// Identity transform: origin position, no rotation, unit scale.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        _pad0: 0.0,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        _pad1: 0.0,
    };

    /// Create from translation, rotation, and scale.
    pub fn new(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            translation,
            _pad0: 0.0,
            rotation,
            scale,
            _pad1: 0.0,
        }
    }

    /// Create from translation only (identity rotation and scale).
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    /// Create from rotation only (origin position and unit scale).
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            ..Self::IDENTITY
        }
    }

    /// Compute the local 4x4 transform matrix (T * R * S).
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<NodeTransform> for Transform {
    fn from(t: NodeTransform) -> Self {
        Self::new(
            Vec3::from(t.translation),
            Quat::from_array(t.rotation),
            Vec3::from(t.scale),
        )
    }
}

impl From<Transform> for NodeTransform {
    fn from(t: Transform) -> Self {
        Self {
            translation: t.translation.into(),
            rotation: t.rotation.to_array(),
            scale: t.scale.into(),
        }
    }
}

/// World-space transform as a 4x4 matrix.
///
/// Computed by the [`update_global_transforms`](crate::systems::update_global_transforms)
/// system. Without hierarchy, this equals the local [`Transform`]'s matrix.
/// With hierarchy (future), it will incorporate the parent chain.
#[derive(
    Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, redlilium_ecs::Component,
)]
#[repr(C)]
pub struct GlobalTransform(pub Mat4);

impl GlobalTransform {
    /// Identity global transform.
    pub const IDENTITY: Self = Self(Mat4::IDENTITY);

    /// Extract the world-space translation.
    pub fn translation(&self) -> Vec3 {
        self.0.w_axis.truncate()
    }

    /// Get the forward direction vector (-Z in right-handed coordinates).
    pub fn forward(&self) -> Vec3 {
        -self.0.z_axis.truncate().normalize()
    }

    /// Get the right direction vector (+X).
    pub fn right(&self) -> Vec3 {
        self.0.x_axis.truncate().normalize()
    }

    /// Get the up direction vector (+Y).
    pub fn up(&self) -> Vec3 {
        self.0.y_axis.truncate().normalize()
    }
}

impl Default for GlobalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn identity_transform() {
        let t = Transform::IDENTITY;
        assert_eq!(t.translation, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn identity_matrix() {
        let t = Transform::IDENTITY;
        assert_eq!(t.to_matrix(), Mat4::IDENTITY);
    }

    #[test]
    fn from_translation() {
        let t = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.translation, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn roundtrip_node_transform() {
        let original = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(FRAC_PI_2),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let node: NodeTransform = original.into();
        let restored: Transform = node.into();

        assert!((original.translation - restored.translation).length() < 1e-6);
        assert!((original.rotation - restored.rotation).length() < 1e-6);
        assert!((original.scale - restored.scale).length() < 1e-6);
    }

    #[test]
    fn to_matrix_translation() {
        let t = Transform::from_translation(Vec3::new(5.0, 10.0, 15.0));
        let mat = t.to_matrix();
        let gt = GlobalTransform(mat);
        assert!((gt.translation() - Vec3::new(5.0, 10.0, 15.0)).length() < 1e-6);
    }

    #[test]
    fn global_transform_directions() {
        let gt = GlobalTransform::IDENTITY;
        assert!((gt.forward() - Vec3::NEG_Z).length() < 1e-6);
        assert!((gt.right() - Vec3::X).length() < 1e-6);
        assert!((gt.up() - Vec3::Y).length() < 1e-6);
    }
}
