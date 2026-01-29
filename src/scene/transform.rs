//! Transform component

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};

/// Transform component for positioning objects in 3D space
#[derive(Component, Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    pub fn from_position_rotation(position: Vec3, rotation: Quat) -> Self {
        Self {
            position,
            rotation,
            ..Default::default()
        }
    }

    pub fn from_position_scale(position: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            scale,
            ..Default::default()
        }
    }

    /// Create transform from position, rotation (euler angles in radians), and scale
    pub fn from_components(position: Vec3, rotation_euler: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::from_euler(
                glam::EulerRot::XYZ,
                rotation_euler.x,
                rotation_euler.y,
                rotation_euler.z,
            ),
            scale,
        }
    }

    /// Get the model matrix for this transform
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    /// Get the normal matrix (inverse transpose of model matrix)
    pub fn normal_matrix(&self) -> Mat4 {
        self.matrix().inverse().transpose()
    }

    /// Get forward direction (local -Z in world space)
    pub fn forward(&self) -> Vec3 {
        self.rotation * -Vec3::Z
    }

    /// Get right direction (local +X in world space)
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Get up direction (local +Y in world space)
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Translate by an offset
    pub fn translate(&mut self, offset: Vec3) {
        self.position += offset;
    }

    /// Rotate by euler angles (radians)
    pub fn rotate_euler(&mut self, euler: Vec3) {
        let delta = Quat::from_euler(glam::EulerRot::XYZ, euler.x, euler.y, euler.z);
        self.rotation = delta * self.rotation;
    }

    /// Rotate around an axis
    pub fn rotate_axis(&mut self, axis: Vec3, angle: f32) {
        let delta = Quat::from_axis_angle(axis, angle);
        self.rotation = delta * self.rotation;
    }

    /// Look at a target position
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let forward = (target - self.position).normalize();
        let right = up.cross(forward).normalize();
        let up = forward.cross(right);

        self.rotation = Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward));
    }

    /// Build uniform data for shaders
    pub fn uniform_data(&self) -> TransformUniformData {
        let model = self.matrix();
        TransformUniformData {
            model,
            normal_matrix: model.inverse().transpose(),
        }
    }
}

/// Transform uniform data for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TransformUniformData {
    pub model: Mat4,
    pub normal_matrix: Mat4,
}
