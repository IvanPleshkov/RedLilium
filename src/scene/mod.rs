//! Scene management

mod camera;
mod camera_controller;
mod light;
mod transform;

pub use camera::*;
pub use camera_controller::*;
pub use light::*;
pub use transform::*;

use glam::Vec3;

/// A renderable object in the scene
#[derive(Debug, Clone)]
pub struct RenderObject {
    pub mesh_id: usize,
    pub material_id: usize,
    pub transform: Transform,
}

impl RenderObject {
    pub fn new(mesh_id: usize, material_id: usize) -> Self {
        Self {
            mesh_id,
            material_id,
            transform: Transform::default(),
        }
    }

    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_position(mut self, position: Vec3) -> Self {
        self.transform.position = position;
        self
    }

    pub fn with_scale(mut self, scale: Vec3) -> Self {
        self.transform.scale = scale;
        self
    }
}

/// The scene containing all renderable content
pub struct Scene {
    pub camera: Camera,
    pub lights: Vec<Light>,
    pub objects: Vec<RenderObject>,
    pub ambient_light: Vec3,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            camera: Camera::default(),
            lights: Vec::new(),
            objects: Vec::new(),
            ambient_light: Vec3::new(0.03, 0.03, 0.03),
        }
    }

    /// Add a point light to the scene
    pub fn add_point_light(&mut self, position: Vec3, color: Vec3, intensity: f32, radius: f32) {
        self.lights.push(Light::Point(PointLight {
            position,
            color,
            intensity,
            radius,
        }));
    }

    /// Add a spot light to the scene
    pub fn add_spot_light(
        &mut self,
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        radius: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) {
        self.lights.push(Light::Spot(SpotLight {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            radius,
            inner_angle,
            outer_angle,
        }));
    }

    /// Add a directional light to the scene
    pub fn add_directional_light(&mut self, direction: Vec3, color: Vec3, intensity: f32) {
        self.lights.push(Light::Directional(DirectionalLight {
            direction: direction.normalize(),
            color,
            intensity,
        }));
    }

    /// Add a render object to the scene
    pub fn add_object(&mut self, object: RenderObject) -> usize {
        let id = self.objects.len();
        self.objects.push(object);
        id
    }

    /// Get the number of point/spot lights (for Forward+)
    pub fn local_light_count(&self) -> usize {
        self.lights
            .iter()
            .filter(|l| matches!(l, Light::Point(_) | Light::Spot(_)))
            .count()
    }

    /// Get the directional light (if any)
    pub fn directional_light(&self) -> Option<&DirectionalLight> {
        self.lights.iter().find_map(|l| match l {
            Light::Directional(d) => Some(d),
            _ => None,
        })
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
