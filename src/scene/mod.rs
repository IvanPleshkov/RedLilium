//! Scene management with ECS

mod camera;
mod camera_controller;
mod light;
mod transform;

pub use camera::*;
pub use camera_controller::*;
pub use light::*;
pub use transform::*;

use bevy_ecs::prelude::*;
use glam::Vec3;

/// Component for rendering a mesh with a material
#[derive(Component, Debug, Clone)]
pub struct MeshRenderer {
    pub mesh_id: usize,
    pub material_id: usize,
}

impl MeshRenderer {
    pub fn new(mesh_id: usize, material_id: usize) -> Self {
        Self { mesh_id, material_id }
    }
}

/// Tag component to identify the main camera
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct MainCamera;

/// Resource for ambient light in the scene
#[derive(Resource, Debug, Clone, Copy)]
pub struct AmbientLight(pub Vec3);

impl Default for AmbientLight {
    fn default() -> Self {
        Self(Vec3::new(0.03, 0.03, 0.03))
    }
}
