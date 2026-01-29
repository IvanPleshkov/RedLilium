//! Camera system

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

/// Camera projection type
#[derive(Debug, Clone, Copy)]
pub enum Projection {
    Perspective {
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    },
}

impl Default for Projection {
    fn default() -> Self {
        Projection::Perspective {
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Projection {
    pub fn perspective(fov_y_degrees: f32, aspect: f32, near: f32, far: f32) -> Self {
        Projection::Perspective {
            fov_y: fov_y_degrees.to_radians(),
            aspect,
            near,
            far,
        }
    }

    pub fn orthographic(width: f32, height: f32, near: f32, far: f32) -> Self {
        let half_w = width / 2.0;
        let half_h = height / 2.0;
        Projection::Orthographic {
            left: -half_w,
            right: half_w,
            bottom: -half_h,
            top: half_h,
            near,
            far,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        match self {
            Projection::Perspective {
                fov_y,
                aspect,
                near,
                far,
            } => Mat4::perspective_rh(*fov_y, *aspect, *near, *far),
            Projection::Orthographic {
                left,
                right,
                bottom,
                top,
                near,
                far,
            } => Mat4::orthographic_rh(*left, *right, *bottom, *top, *near, *far),
        }
    }

    pub fn near(&self) -> f32 {
        match self {
            Projection::Perspective { near, .. } => *near,
            Projection::Orthographic { near, .. } => *near,
        }
    }

    pub fn far(&self) -> f32 {
        match self {
            Projection::Perspective { far, .. } => *far,
            Projection::Orthographic { far, .. } => *far,
        }
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        if let Projection::Perspective { aspect: a, .. } = self {
            *a = aspect;
        }
    }
}

/// Camera for viewing the scene
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub projection: Projection,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            projection: Projection::default(),
        }
    }
}

impl Camera {
    pub fn new(position: Vec3, target: Vec3) -> Self {
        Self {
            position,
            target,
            up: Vec3::Y,
            projection: Projection::default(),
        }
    }

    pub fn look_at(&mut self, target: Vec3) {
        self.target = target;
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    /// Get the view matrix
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Get the projection matrix
    pub fn projection_matrix(&self) -> Mat4 {
        self.projection.matrix()
    }

    /// Get combined view-projection matrix
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Get the forward direction
    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    /// Get the right direction
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize()
    }

    /// Build camera uniform data for shaders
    pub fn uniform_data(&self) -> CameraUniformData {
        let view = self.view_matrix();
        let proj = self.projection_matrix();
        let view_proj = proj * view;

        CameraUniformData {
            view,
            proj,
            view_proj,
            inv_view: view.inverse(),
            inv_proj: proj.inverse(),
            position: self.position.extend(1.0),
            near_far: Vec4::new(
                self.projection.near(),
                self.projection.far(),
                0.0,
                0.0,
            ),
        }
    }

    /// Update aspect ratio for perspective projection
    pub fn set_aspect(&mut self, width: f32, height: f32) {
        self.projection.set_aspect(width / height);
    }
}

/// Camera uniform data for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniformData {
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj: Mat4,
    pub inv_view: Mat4,
    pub inv_proj: Mat4,
    pub position: Vec4,
    pub near_far: Vec4,
}
