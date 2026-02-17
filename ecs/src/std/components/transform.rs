#[cfg(test)]
use redlilium_core::math::quat_from_rotation_y;
use redlilium_core::math::{
    Mat4, Quat, Vec3, mat4_from_scale_rotation_translation, quat_from_array, quat_to_array,
};
use redlilium_core::scene::NodeTransform;

/// Local transform of an entity.
///
/// Stores translation, rotation, and scale. Convertible to/from
/// core's [`NodeTransform`] which uses plain `[f32; N]` arrays.
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, crate::Component)]
#[repr(C)]
pub struct Transform {
    /// Translation in world units.
    pub translation: Vec3,
    /// Rotation as a quaternion.
    pub rotation: Quat,
    /// Non-uniform scale.
    pub scale: Vec3,
}

impl Transform {
    /// Identity transform: origin position, no rotation, unit scale.
    pub const IDENTITY: Self = Self {
        translation: Vec3::new(0.0, 0.0, 0.0),
        rotation: Quat::new(1.0, 0.0, 0.0, 0.0),
        scale: Vec3::new(1.0, 1.0, 1.0),
    };

    /// Create from translation, rotation, and scale.
    pub fn new(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            translation,
            rotation,
            scale,
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
        mat4_from_scale_rotation_translation(self.scale, self.rotation, self.translation)
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
            Vec3::new(t.translation[0], t.translation[1], t.translation[2]),
            quat_from_array(t.rotation),
            Vec3::new(t.scale[0], t.scale[1], t.scale[2]),
        )
    }
}

impl From<Transform> for NodeTransform {
    fn from(t: Transform) -> Self {
        Self {
            translation: [t.translation.x, t.translation.y, t.translation.z],
            rotation: quat_to_array(t.rotation),
            scale: [t.scale.x, t.scale.y, t.scale.z],
        }
    }
}

/// World-space transform as a 4x4 matrix.
///
/// Computed by the [`update_global_transforms`](crate::systems::update_global_transforms)
/// system. Without hierarchy, this equals the local [`Transform`]'s matrix.
/// With hierarchy (future), it will incorporate the parent chain.
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, crate::Component)]
#[repr(C)]
pub struct GlobalTransform(pub Mat4);

impl GlobalTransform {
    /// Identity global transform.
    #[rustfmt::skip]
    pub const IDENTITY: Self = Self(Mat4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ));

    /// Extract the world-space translation.
    pub fn translation(&self) -> Vec3 {
        Vec3::new(self.0[(0, 3)], self.0[(1, 3)], self.0[(2, 3)])
    }

    /// Get the forward direction vector (-Z in right-handed coordinates).
    pub fn forward(&self) -> Vec3 {
        let z = Vec3::new(self.0[(0, 2)], self.0[(1, 2)], self.0[(2, 2)]);
        (-z).normalize()
    }

    /// Get the right direction vector (+X).
    pub fn right(&self) -> Vec3 {
        Vec3::new(self.0[(0, 0)], self.0[(1, 0)], self.0[(2, 0)]).normalize()
    }

    /// Get the up direction vector (+Y).
    pub fn up(&self) -> Vec3 {
        Vec3::new(self.0[(0, 1)], self.0[(1, 1)], self.0[(2, 1)]).normalize()
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
        assert_eq!(t.translation, Vec3::zeros());
        assert_eq!(t.rotation, Quat::identity());
        assert_eq!(t.scale, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn identity_matrix() {
        let t = Transform::IDENTITY;
        assert!((t.to_matrix() - Mat4::identity()).norm() < 1e-6);
    }

    #[test]
    fn from_translation() {
        let t = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.translation, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(t.rotation, Quat::identity());
        assert_eq!(t.scale, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn roundtrip_node_transform() {
        let original = Transform::new(
            Vec3::new(1.0, 2.0, 3.0),
            quat_from_rotation_y(FRAC_PI_2),
            Vec3::new(2.0, 2.0, 2.0),
        );
        let node: NodeTransform = original.into();
        let restored: Transform = node.into();

        assert!((original.translation - restored.translation).norm() < 1e-6);
        assert!((original.rotation.coords - restored.rotation.coords).norm() < 1e-6);
        assert!((original.scale - restored.scale).norm() < 1e-6);
    }

    #[test]
    fn to_matrix_translation() {
        let t = Transform::from_translation(Vec3::new(5.0, 10.0, 15.0));
        let mat = t.to_matrix();
        let gt = GlobalTransform(mat);
        assert!((gt.translation() - Vec3::new(5.0, 10.0, 15.0)).norm() < 1e-6);
    }

    #[test]
    fn global_transform_directions() {
        let gt = GlobalTransform::IDENTITY;
        assert!((gt.forward() - Vec3::new(0.0, 0.0, -1.0)).norm() < 1e-6);
        assert!((gt.right() - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-6);
        assert!((gt.up() - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-6);
    }
}
