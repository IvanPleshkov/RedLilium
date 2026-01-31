//! Transform components for positioning entities in 3D space.
//!
//! This module provides [`Transform`] for local transforms relative to parent entities,
//! and [`GlobalTransform`] for world-space transforms computed by the hierarchy system.

use bevy_ecs::component::Component;
use glam::{Affine3A, Mat4, Quat, Vec3};

/// Local transform component describing position, rotation, and scale relative to a parent.
///
/// If the entity has no parent (no [`ChildOf`] component), this transform is relative to
/// world origin. For entities with parents, [`GlobalTransform`] holds the computed
/// world-space transform.
///
/// # Example
///
/// ```
/// use redlilium_ecs::components::Transform;
/// use glam::{Vec3, Quat};
///
/// let transform = Transform::from_xyz(1.0, 2.0, 3.0)
///     .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
///     .with_scale(Vec3::splat(2.0));
/// ```
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// Position relative to parent (or world origin if no parent).
    pub translation: Vec3,
    /// Rotation relative to parent.
    pub rotation: Quat,
    /// Scale relative to parent.
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// Identity transform with no translation, no rotation, and uniform scale of 1.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Creates a transform at the given position with default rotation and scale.
    #[inline]
    pub const fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_translation(Vec3::new(x, y, z))
    }

    /// Creates a transform with the given translation.
    #[inline]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Creates a transform with the given rotation.
    #[inline]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation,
            scale: Vec3::ONE,
        }
    }

    /// Creates a transform with the given scale.
    #[inline]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale,
        }
    }

    /// Creates a transform from a 4x4 matrix, extracting translation, rotation, and scale.
    #[inline]
    pub fn from_matrix(matrix: Mat4) -> Self {
        let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// Returns this transform with a different translation.
    #[inline]
    #[must_use]
    pub const fn with_translation(mut self, translation: Vec3) -> Self {
        self.translation = translation;
        self
    }

    /// Returns this transform with a different rotation.
    #[inline]
    #[must_use]
    pub const fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Returns this transform with a different scale.
    #[inline]
    #[must_use]
    pub const fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    /// Computes the transformation matrix for this transform.
    #[inline]
    pub fn compute_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Computes the affine transformation for this transform.
    #[inline]
    pub fn compute_affine(&self) -> Affine3A {
        Affine3A::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Returns the local forward direction (-Z axis).
    #[inline]
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// Returns the local back direction (+Z axis).
    #[inline]
    pub fn back(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }

    /// Returns the local right direction (+X axis).
    #[inline]
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Returns the local left direction (-X axis).
    #[inline]
    pub fn left(&self) -> Vec3 {
        self.rotation * Vec3::NEG_X
    }

    /// Returns the local up direction (+Y axis).
    #[inline]
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Returns the local down direction (-Y axis).
    #[inline]
    pub fn down(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Y
    }

    /// Rotates this transform so that its forward direction points at `target`.
    #[inline]
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let forward = (target - self.translation).normalize_or_zero();
        if forward.length_squared() < 1e-6 {
            return;
        }
        let right = up.cross(forward).normalize_or_zero();
        if right.length_squared() < 1e-6 {
            return;
        }
        let up = forward.cross(right);
        self.rotation = Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward));
    }

    /// Returns this transform with rotation set to look at `target`.
    #[inline]
    #[must_use]
    pub fn looking_at(mut self, target: Vec3, up: Vec3) -> Self {
        self.look_at(target, up);
        self
    }

    /// Multiplies this transform by another, combining them.
    /// The result represents applying `self` first, then `other`.
    #[inline]
    pub fn mul_transform(&self, other: &Transform) -> Transform {
        let translation = self.transform_point(other.translation);
        let rotation = self.rotation * other.rotation;
        let scale = self.scale * other.scale;
        Transform {
            translation,
            rotation,
            scale,
        }
    }

    /// Transforms a point from local space to the space of this transform.
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.rotation * (self.scale * point) + self.translation
    }

    /// Transforms a direction vector (ignores translation and scale).
    #[inline]
    pub fn transform_direction(&self, direction: Vec3) -> Vec3 {
        self.rotation * direction
    }
}

/// World-space transform computed from the entity hierarchy.
///
/// This component is automatically updated by the transform propagation system
/// based on the entity's [`Transform`] and its parent's [`GlobalTransform`].
///
/// For root entities (no parent), [`GlobalTransform`] equals the [`Transform`].
/// For child entities, it represents the combined transform of the entire hierarchy.
///
/// This component should not be modified directly; modify [`Transform`] instead.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct GlobalTransform(pub(crate) Affine3A);

impl Default for GlobalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl GlobalTransform {
    /// Identity global transform.
    pub const IDENTITY: Self = Self(Affine3A::IDENTITY);

    /// Creates a global transform from translation.
    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self(Affine3A::from_translation(translation))
    }

    /// Creates a global transform from rotation.
    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self(Affine3A::from_rotation_translation(rotation, Vec3::ZERO))
    }

    /// Creates a global transform from scale.
    #[inline]
    pub fn from_scale(scale: Vec3) -> Self {
        Self(Affine3A::from_scale(scale))
    }

    /// Creates a global transform from position coordinates.
    #[inline]
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_translation(Vec3::new(x, y, z))
    }

    /// Returns the underlying affine transformation.
    #[inline]
    pub fn affine(&self) -> Affine3A {
        self.0
    }

    /// Returns the transformation as a 4x4 matrix.
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from(self.0)
    }

    /// Extracts translation, rotation, and scale from this transform.
    /// Note: This operation can be lossy for sheared transforms.
    #[inline]
    pub fn to_scale_rotation_translation(&self) -> (Vec3, Quat, Vec3) {
        self.0.to_scale_rotation_translation()
    }

    /// Converts to a local [`Transform`].
    /// Note: This operation can be lossy for sheared transforms.
    #[inline]
    pub fn compute_transform(&self) -> Transform {
        let (scale, rotation, translation) = self.to_scale_rotation_translation();
        Transform {
            translation,
            rotation,
            scale,
        }
    }

    /// Returns the world-space translation.
    #[inline]
    pub fn translation(&self) -> Vec3 {
        self.0.translation.into()
    }

    /// Returns the world-space forward direction (-Z axis).
    #[inline]
    pub fn forward(&self) -> Vec3 {
        (self.0.matrix3 * Vec3::NEG_Z).normalize()
    }

    /// Returns the world-space back direction (+Z axis).
    #[inline]
    pub fn back(&self) -> Vec3 {
        (self.0.matrix3 * Vec3::Z).normalize()
    }

    /// Returns the world-space right direction (+X axis).
    #[inline]
    pub fn right(&self) -> Vec3 {
        (self.0.matrix3 * Vec3::X).normalize()
    }

    /// Returns the world-space left direction (-X axis).
    #[inline]
    pub fn left(&self) -> Vec3 {
        (self.0.matrix3 * Vec3::NEG_X).normalize()
    }

    /// Returns the world-space up direction (+Y axis).
    #[inline]
    pub fn up(&self) -> Vec3 {
        (self.0.matrix3 * Vec3::Y).normalize()
    }

    /// Returns the world-space down direction (-Y axis).
    #[inline]
    pub fn down(&self) -> Vec3 {
        (self.0.matrix3 * Vec3::NEG_Y).normalize()
    }

    /// Transforms a point from local space to world space.
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.0.transform_point3(point)
    }

    /// Multiplies this global transform by a local transform.
    #[inline]
    pub fn mul_transform(&self, transform: &Transform) -> GlobalTransform {
        GlobalTransform(self.0 * transform.compute_affine())
    }
}

impl From<Transform> for GlobalTransform {
    fn from(transform: Transform) -> Self {
        GlobalTransform(transform.compute_affine())
    }
}

impl From<Affine3A> for GlobalTransform {
    fn from(affine: Affine3A) -> Self {
        GlobalTransform(affine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn transform_identity() {
        let t = Transform::IDENTITY;
        assert_eq!(t.translation, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn transform_from_xyz() {
        let t = Transform::from_xyz(1.0, 2.0, 3.0);
        assert_eq!(t.translation, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn transform_directions() {
        let t = Transform::from_rotation(Quat::from_rotation_y(FRAC_PI_2));
        let forward = t.forward();
        // After 90 degree Y rotation (counter-clockwise looking down Y),
        // forward (-Z) should point towards -X
        assert!((forward - Vec3::NEG_X).length() < 1e-5);
    }

    #[test]
    fn transform_mul() {
        let parent = Transform::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let child = Transform::from_translation(Vec3::new(0.0, 5.0, 0.0));
        let combined = parent.mul_transform(&child);
        assert!((combined.translation - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn global_transform_from_local() {
        let local = Transform::from_xyz(1.0, 2.0, 3.0).with_scale(Vec3::splat(2.0));
        let global: GlobalTransform = local.into();
        let point = global.transform_point(Vec3::ONE);
        // With scale 2 and translation (1,2,3): (1,1,1) * 2 + (1,2,3) = (3,4,5)
        assert!((point - Vec3::new(3.0, 4.0, 5.0)).length() < 1e-5);
    }
}
