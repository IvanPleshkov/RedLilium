//! ECS scene setup for the PBR IBL demo.
//!
//! Creates a World with a camera entity and a grid of sphere entities,
//! each with PBR material properties.

use std::f32::consts::PI;

use ecs_std::{
    Camera, GlobalTransform, Transform, UpdateCameraMatrices, UpdateGlobalTransforms, Visibility,
};
use redlilium_core::math::{
    Mat4, Quat, Vec3, look_at_rh, mat4_from_translation, mat4_to_cols_array_2d, perspective_rh,
    to_scale_rotation_translation,
};
use redlilium_ecs::{EcsRunner, Entity, SystemsContainer, World};

use crate::uniforms::SphereInstance;
use crate::{GRID_SIZE, SPHERE_SPACING};

/// Per-sphere PBR material data, stored as an ECS component.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PbrSphere {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    _pad: [f32; 2],
}

/// Holds the ECS world and execution infrastructure for the PBR demo scene.
pub struct EcsScene {
    pub world: World,
    pub systems: SystemsContainer,
    pub runner: EcsRunner,
    pub camera_entity: Entity,
}

impl EcsScene {
    /// Creates a new ECS scene with a camera and a grid of PBR spheres.
    pub fn new(aspect_ratio: f32) -> Self {
        let mut world = World::new();

        // Register all standard components + our custom one
        ecs_std::register_std_components(&mut world);
        world.register_component::<PbrSphere>();

        // Spawn camera entity
        let camera_entity = world.spawn();
        let camera_transform = Transform::new(
            Vec3::new(
                8.0 * 0.4_f32.cos() * 0.5_f32.sin(),
                8.0 * 0.4_f32.sin(),
                8.0 * 0.4_f32.cos() * 0.5_f32.cos(),
            ),
            Quat::identity(),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world
            .insert(camera_entity, camera_transform)
            .expect("Transform not registered");
        world
            .insert(camera_entity, GlobalTransform(camera_transform.to_matrix()))
            .expect("GlobalTransform not registered");
        world
            .insert(
                camera_entity,
                Camera::perspective(PI / 4.0, aspect_ratio, 0.1, 100.0),
            )
            .expect("Camera not registered");

        // Spawn sphere grid
        let offset = (GRID_SIZE as f32 - 1.0) * SPHERE_SPACING / 2.0;
        for row in 0..GRID_SIZE {
            for col in 0..GRID_SIZE {
                let x = col as f32 * SPHERE_SPACING - offset;
                let z = row as f32 * SPHERE_SPACING - offset;

                let metallic = col as f32 / (GRID_SIZE - 1) as f32;
                let roughness = (row as f32 / (GRID_SIZE - 1) as f32).max(0.05);

                let entity = world.spawn();
                let transform = Transform::from_translation(Vec3::new(x, 0.0, z));
                world
                    .insert(entity, transform)
                    .expect("Transform not registered");
                world
                    .insert(entity, GlobalTransform(transform.to_matrix()))
                    .expect("GlobalTransform not registered");
                world
                    .insert(entity, Visibility::VISIBLE)
                    .expect("Visibility not registered");
                world
                    .insert(
                        entity,
                        PbrSphere {
                            base_color: [0.9, 0.1, 0.1, 1.0],
                            metallic,
                            roughness,
                            _pad: [0.0; 2],
                        },
                    )
                    .expect("PbrSphere not registered");
            }
        }

        // Setup systems
        let mut systems = SystemsContainer::new();
        systems.add(UpdateGlobalTransforms);
        systems.add(UpdateCameraMatrices);
        systems
            .add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>()
            .expect("No cycle possible with two systems");

        let runner = EcsRunner::single_thread();

        Self {
            world,
            systems,
            runner,
            camera_entity,
        }
    }

    /// Updates the camera entity's transform from orbit camera parameters.
    pub fn update_camera_transform(&mut self, position: Vec3, target: Vec3) {
        let view_matrix = look_at_rh(&position, &target, &Vec3::new(0.0, 1.0, 0.0));
        let camera_matrix = view_matrix.try_inverse().unwrap();
        let (_, rotation, translation) = to_scale_rotation_translation(&camera_matrix);

        if let Some(transform) = self.world.get_mut::<Transform>(self.camera_entity) {
            transform.translation = translation;
            transform.rotation = rotation;
        }
    }

    /// Updates the camera's projection matrix (e.g., on resize).
    pub fn update_camera_projection(&mut self, aspect_ratio: f32) {
        if let Some(camera) = self.world.get_mut::<Camera>(self.camera_entity) {
            camera.projection_matrix = perspective_rh(PI / 4.0, aspect_ratio, 0.1, 100.0);
        }
    }

    /// Runs the ECS systems (transform propagation + camera matrix update).
    pub fn run_systems(&mut self) {
        self.runner.run(&mut self.world, &self.systems);
    }

    /// Reads the camera's computed view and projection matrices.
    pub fn camera_matrices(&self) -> (Mat4, Mat4) {
        let camera = self.world.get::<Camera>(self.camera_entity).unwrap();
        (camera.view_matrix, camera.projection_matrix)
    }

    /// Reads the camera's world-space position from GlobalTransform.
    pub fn camera_position(&self) -> Vec3 {
        let gt = self
            .world
            .get::<GlobalTransform>(self.camera_entity)
            .unwrap();
        gt.translation()
    }

    /// Builds the sphere instance buffer data from ECS entities.
    pub fn build_sphere_instances(&self) -> Vec<SphereInstance> {
        let spheres = self
            .world
            .read::<PbrSphere>()
            .expect("PbrSphere not registered");
        let globals = self
            .world
            .read::<GlobalTransform>()
            .expect("GlobalTransform not registered");

        let mut instances = Vec::with_capacity(GRID_SIZE * GRID_SIZE);
        for (idx, sphere) in spheres.iter() {
            if let Some(global) = globals.get(idx) {
                instances.push(SphereInstance {
                    model: mat4_to_cols_array_2d(&global.0),
                    base_color: sphere.base_color,
                    metallic_roughness: [sphere.metallic, sphere.roughness, 0.0, 0.0],
                });
            }
        }
        instances
    }

    /// Updates sphere material properties (base_color, spacing) from UI state.
    pub fn update_spheres(&mut self, base_color: [f32; 4], spacing: f32) {
        let offset = (GRID_SIZE as f32 - 1.0) * spacing / 2.0;
        let mut spheres = self
            .world
            .write::<PbrSphere>()
            .expect("PbrSphere not registered");
        let mut transforms = self
            .world
            .write::<Transform>()
            .expect("Transform not registered");
        let mut globals = self
            .world
            .write::<GlobalTransform>()
            .expect("GlobalTransform not registered");

        for (i, (idx, sphere)) in spheres.iter_mut().enumerate() {
            sphere.base_color = base_color;

            let row = i / GRID_SIZE;
            let col = i % GRID_SIZE;
            let x = col as f32 * spacing - offset;
            let z = row as f32 * spacing - offset;

            if let Some(transform) = transforms.get_mut(idx) {
                transform.translation = Vec3::new(x, 0.0, z);
            }
            if let Some(global) = globals.get_mut(idx) {
                global.0 = mat4_from_translation(Vec3::new(x, 0.0, z));
            }
        }
    }
}
