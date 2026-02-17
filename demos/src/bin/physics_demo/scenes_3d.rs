//! 3D physics scene definitions using the reactive ECS approach.
//!
//! Each scene spawns entities with [`RigidBody3D`] + [`Collider3D`] + [`Transform`]
//! descriptor components. The sync systems automatically create rapier physics
//! objects from these descriptors. Joints use [`ImpulseJoint3D`] components
//! with entity references that are automatically remapped in prefabs.

use redlilium_core::math;
use redlilium_ecs::Transform;
use redlilium_ecs::World;
use redlilium_ecs::physics::components3d::{Collider3D, ImpulseJoint3D, RigidBody3D};
use redlilium_ecs::physics::physics3d::PhysicsWorld3D;
use redlilium_ecs::physics::rapier3d::prelude::*;

/// Trait for a 3D physics demo scene.
#[allow(dead_code)]
pub trait PhysicsScene3D: Send + Sync {
    /// Display name shown in the UI.
    fn name(&self) -> &str;

    /// Populate the world with physics bodies and ECS entities.
    fn setup(&self, world: &mut World);

    /// Optional per-frame callback (e.g. spawning, kinematic updates).
    fn update(&self, _world: &mut World) {}
}

/// Helper: spawn a physics entity with descriptor components.
fn spawn_entity(
    world: &mut World,
    body: RigidBody3D,
    collider: Collider3D,
    transform: Transform,
) -> redlilium_ecs::Entity {
    let entity = world.spawn();
    let _ = world.insert(entity, body);
    let _ = world.insert(entity, collider);
    let _ = world.insert(entity, transform);
    let _ = world.insert(entity, redlilium_ecs::GlobalTransform::IDENTITY);
    entity
}

// ---------------------------------------------------------------------------
// Balls — ground plane + many falling spheres
// ---------------------------------------------------------------------------

pub struct BallsScene;

impl PhysicsScene3D for BallsScene {
    fn name(&self) -> &str {
        "Balls"
    }

    fn setup(&self, world: &mut World) {
        // Ground
        spawn_entity(
            world,
            RigidBody3D::fixed(),
            Collider3D::cuboid(20.0, 0.1, 20.0).with_restitution(0.3),
            Transform::IDENTITY,
        );

        // Falling spheres
        let cols = 8;
        let rows = 8;
        for i in 0..cols {
            for j in 0..rows {
                let x = (i as f32 - cols as f32 / 2.0) * 1.2;
                let z = (j as f32 - rows as f32 / 2.0) * 1.2;
                let y = 5.0 + (i * rows + j) as f32 * 0.5;

                spawn_entity(
                    world,
                    RigidBody3D::dynamic(),
                    Collider3D::ball(0.5).with_restitution(0.7),
                    Transform::from_translation(math::Vec3::new(x, y, z)),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stacking — pyramid of boxes
// ---------------------------------------------------------------------------

pub struct StackingScene;

impl PhysicsScene3D for StackingScene {
    fn name(&self) -> &str {
        "Stacking"
    }

    fn setup(&self, world: &mut World) {
        // Ground
        spawn_entity(
            world,
            RigidBody3D::fixed(),
            Collider3D::cuboid(20.0, 0.1, 20.0),
            Transform::IDENTITY,
        );

        // Pyramid of boxes
        let layers = 10;
        let box_half = 0.5f32;
        let gap = 0.05f32;
        let step = box_half * 2.0 + gap;
        for layer in 0..layers {
            let count = layers - layer;
            let offset = count as f32 * step / 2.0 - step / 2.0;
            let y = box_half + layer as f32 * step + 0.1;
            for i in 0..count {
                let x = i as f32 * step - offset;
                spawn_entity(
                    world,
                    RigidBody3D::dynamic(),
                    Collider3D::cuboid(box_half, box_half, box_half),
                    Transform::from_translation(math::Vec3::new(x, y, 0.0)),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Joints — grid of balls connected with ball joints
// ---------------------------------------------------------------------------

pub struct JointsScene;

impl PhysicsScene3D for JointsScene {
    fn name(&self) -> &str {
        "Joints"
    }

    fn setup(&self, world: &mut World) {
        let size = 5;
        let spacing = 2.0f32;
        let half = spacing / 2.0;
        let mut entity_grid: Vec<Vec<redlilium_ecs::Entity>> = Vec::new();

        for i in 0..size {
            let mut row = Vec::new();
            for j in 0..size {
                let x = i as f32 * spacing - (size as f32 * spacing / 2.0);
                let z = j as f32 * spacing - (size as f32 * spacing / 2.0);
                let y = 10.0f32;

                let is_anchor = i == 0 || i == size - 1 || j == 0 || j == size - 1;
                let body = if is_anchor {
                    RigidBody3D::fixed()
                } else {
                    RigidBody3D::dynamic()
                };

                let entity = spawn_entity(
                    world,
                    body,
                    Collider3D::ball(0.3),
                    Transform::from_translation(math::Vec3::new(x, y, z)),
                );
                row.push(entity);
            }
            entity_grid.push(row);
        }

        // Connect neighbours with spherical joints via ImpulseJoint3D components
        for i in 0..size {
            for j in 0..size {
                if i + 1 < size {
                    let joint_entity = world.spawn();
                    let _ = world.insert(
                        joint_entity,
                        ImpulseJoint3D::spherical(
                            entity_grid[i][j],
                            entity_grid[i + 1][j],
                            math::Vec3::new(half, 0.0, 0.0),
                            math::Vec3::new(-half, 0.0, 0.0),
                        ),
                    );
                }
                if j + 1 < size {
                    let joint_entity = world.spawn();
                    let _ = world.insert(
                        joint_entity,
                        ImpulseJoint3D::spherical(
                            entity_grid[i][j],
                            entity_grid[i][j + 1],
                            math::Vec3::new(0.0, 0.0, half),
                            math::Vec3::new(0.0, 0.0, -half),
                        ),
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trimesh — heightfield terrain + falling objects
// ---------------------------------------------------------------------------

pub struct TrimeshScene;

impl PhysicsScene3D for TrimeshScene {
    fn name(&self) -> &str {
        "Trimesh"
    }

    fn setup(&self, world: &mut World) {
        // Spawn falling objects with descriptors
        for i in 0..30 {
            let x = (i % 6) as f32 * 2.0 - 5.0;
            let z = (i / 6) as f32 * 2.0 - 5.0;
            let y = 8.0 + i as f32 * 0.3;

            let collider = if i % 2 == 0 {
                Collider3D::ball(0.4).with_restitution(0.5)
            } else {
                Collider3D::cuboid(0.35, 0.35, 0.35).with_restitution(0.3)
            };

            spawn_entity(
                world,
                RigidBody3D::dynamic(),
                collider,
                Transform::from_translation(math::Vec3::new(x, y, z)),
            );
        }

        // Create resource early so sync system finds it and just adds descriptors
        let mut physics = PhysicsWorld3D::default();

        // Add trimesh ground directly via rapier (trimesh is a resource-heavy shape,
        // not representable as a simple Pod component)
        let grid = 20usize;
        let spacing = 1.0;
        let half_extent = grid as f64 * spacing / 2.0;
        let mut vertices = Vec::with_capacity((grid + 1) * (grid + 1));
        for i in 0..=grid {
            for j in 0..=grid {
                let x = i as f64 * spacing - half_extent;
                let z = j as f64 * spacing - half_extent;
                let y = (x * 0.3).sin() + (z * 0.3).cos();
                vertices.push(Vector::new(x, y, z));
            }
        }

        let mut indices = Vec::new();
        let stride = (grid + 1) as u32;
        for i in 0..grid as u32 {
            for j in 0..grid as u32 {
                let v0 = i * stride + j;
                let v1 = v0 + 1;
                let v2 = v0 + stride;
                let v3 = v2 + 1;
                indices.push([v0, v2, v1]);
                indices.push([v1, v2, v3]);
            }
        }

        let ground_handle = physics.add_body(RigidBodyBuilder::fixed().build());
        let trimesh = ColliderBuilder::trimesh(vertices, indices)
            .expect("valid trimesh")
            .restitution(0.3)
            .build();
        physics.add_collider(trimesh, ground_handle);

        world.insert_resource(physics);

        // Spawn ECS entity for the ground trimesh (handle only, no descriptors)
        let ground_entity = world.spawn();
        let _ = world.insert(
            ground_entity,
            redlilium_ecs::physics::physics3d::RigidBody3DHandle(ground_handle),
        );
        let _ = world.insert(ground_entity, Transform::IDENTITY);
        let _ = world.insert(ground_entity, redlilium_ecs::GlobalTransform::IDENTITY);
    }
}

// ---------------------------------------------------------------------------
// Character — kinematic character on a level
// ---------------------------------------------------------------------------

pub struct CharacterScene;

impl PhysicsScene3D for CharacterScene {
    fn name(&self) -> &str {
        "Character"
    }

    fn setup(&self, world: &mut World) {
        // Floor
        spawn_entity(
            world,
            RigidBody3D::fixed(),
            Collider3D::cuboid(20.0, 0.1, 20.0),
            Transform::IDENTITY,
        );

        // Ramp
        spawn_entity(
            world,
            RigidBody3D::fixed(),
            Collider3D::cuboid(4.0, 0.1, 3.0),
            Transform::new(
                math::Vec3::new(5.0, 1.0, 0.0),
                math::quat_from_rotation_z(0.3),
                math::Vec3::new(1.0, 1.0, 1.0),
            ),
        );

        // Platforms
        for i in 0..4 {
            spawn_entity(
                world,
                RigidBody3D::fixed(),
                Collider3D::cuboid(1.0, 0.1, 1.0),
                Transform::from_translation(math::Vec3::new(
                    -5.0 + i as f32 * 3.0,
                    2.0 + i as f32,
                    5.0,
                )),
            );
        }

        // Character capsule (kinematic)
        spawn_entity(
            world,
            RigidBody3D::kinematic_position(),
            Collider3D::capsule_y(0.5, 0.3),
            Transform::from_translation(math::Vec3::new(0.0, 2.0, 0.0)),
        );
    }
}

// ---------------------------------------------------------------------------
// Ragdoll — articulated body with joint constraints
// ---------------------------------------------------------------------------

pub struct RagdollScene;

impl PhysicsScene3D for RagdollScene {
    fn name(&self) -> &str {
        "Ragdoll"
    }

    fn setup(&self, world: &mut World) {
        // Ground
        spawn_entity(
            world,
            RigidBody3D::fixed(),
            Collider3D::cuboid(20.0, 0.1, 20.0),
            Transform::IDENTITY,
        );

        // Ragdoll parts
        let torso = spawn_entity(
            world,
            RigidBody3D::dynamic(),
            Collider3D::cuboid(0.3, 0.5, 0.2),
            Transform::from_translation(math::Vec3::new(0.0, 5.0, 0.0)),
        );

        let head = spawn_entity(
            world,
            RigidBody3D::dynamic(),
            Collider3D::ball(0.25),
            Transform::from_translation(math::Vec3::new(0.0, 6.0, 0.0)),
        );

        // Neck joint
        let neck_joint = world.spawn();
        let _ = world.insert(
            neck_joint,
            ImpulseJoint3D::spherical(
                torso,
                head,
                math::Vec3::new(0.0, 0.5, 0.0),
                math::Vec3::new(0.0, -0.25, 0.0),
            ),
        );

        for side in [-1.0f32, 1.0] {
            let upper_arm = spawn_entity(
                world,
                RigidBody3D::dynamic(),
                Collider3D::capsule_y(0.25, 0.08),
                Transform::from_translation(math::Vec3::new(side * 0.7, 5.3, 0.0)),
            );
            let forearm = spawn_entity(
                world,
                RigidBody3D::dynamic(),
                Collider3D::capsule_y(0.2, 0.07),
                Transform::from_translation(math::Vec3::new(side * 0.7, 4.7, 0.0)),
            );

            // Shoulder (spherical)
            let shoulder_joint = world.spawn();
            let _ = world.insert(
                shoulder_joint,
                ImpulseJoint3D::spherical(
                    torso,
                    upper_arm,
                    math::Vec3::new(side * 0.35, 0.4, 0.0),
                    math::Vec3::new(0.0, 0.25, 0.0),
                ),
            );

            // Elbow (revolute around Z)
            let elbow_joint = world.spawn();
            let _ = world.insert(
                elbow_joint,
                ImpulseJoint3D::revolute(
                    upper_arm,
                    forearm,
                    math::Vec3::new(0.0, 0.0, 1.0),
                    math::Vec3::new(0.0, -0.25, 0.0),
                    math::Vec3::new(0.0, 0.2, 0.0),
                ),
            );
        }

        for side in [-1.0f32, 1.0] {
            let thigh = spawn_entity(
                world,
                RigidBody3D::dynamic(),
                Collider3D::capsule_y(0.3, 0.1),
                Transform::from_translation(math::Vec3::new(side * 0.2, 4.2, 0.0)),
            );
            let shin = spawn_entity(
                world,
                RigidBody3D::dynamic(),
                Collider3D::capsule_y(0.25, 0.08),
                Transform::from_translation(math::Vec3::new(side * 0.2, 3.4, 0.0)),
            );

            // Hip (spherical)
            let hip_joint = world.spawn();
            let _ = world.insert(
                hip_joint,
                ImpulseJoint3D::spherical(
                    torso,
                    thigh,
                    math::Vec3::new(side * 0.2, -0.5, 0.0),
                    math::Vec3::new(0.0, 0.3, 0.0),
                ),
            );

            // Knee (revolute around X)
            let knee_joint = world.spawn();
            let _ = world.insert(
                knee_joint,
                ImpulseJoint3D::revolute(
                    thigh,
                    shin,
                    math::Vec3::new(1.0, 0.0, 0.0),
                    math::Vec3::new(0.0, -0.3, 0.0),
                    math::Vec3::new(0.0, 0.25, 0.0),
                ),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Vehicle — box chassis with wheel joints
// ---------------------------------------------------------------------------

pub struct VehicleScene;

impl PhysicsScene3D for VehicleScene {
    fn name(&self) -> &str {
        "Vehicle"
    }

    fn setup(&self, world: &mut World) {
        // Ground
        spawn_entity(
            world,
            RigidBody3D::fixed(),
            Collider3D::cuboid(40.0, 0.1, 40.0).with_friction(1.0),
            Transform::IDENTITY,
        );

        // Chassis
        let chassis_pos = math::Vec3::new(0.0, 2.0, 0.0);
        let chassis = spawn_entity(
            world,
            RigidBody3D::dynamic(),
            Collider3D::cuboid(1.0, 0.3, 0.5).with_density(2.0),
            Transform::from_translation(chassis_pos),
        );

        // Wheels
        let wheel_offsets = [
            math::Vec3::new(-0.8, -0.3, 0.6),
            math::Vec3::new(0.8, -0.3, 0.6),
            math::Vec3::new(-0.8, -0.3, -0.6),
            math::Vec3::new(0.8, -0.3, -0.6),
        ];

        for offset in &wheel_offsets {
            let wheel_pos = chassis_pos + *offset;
            let wheel = spawn_entity(
                world,
                RigidBody3D::dynamic(),
                Collider3D::ball(0.3).with_friction(1.5).with_density(0.5),
                Transform::from_translation(wheel_pos),
            );

            // Axle joint (revolute around Z)
            let axle_joint = world.spawn();
            let _ = world.insert(
                axle_joint,
                ImpulseJoint3D::revolute(
                    chassis,
                    wheel,
                    math::Vec3::new(0.0, 0.0, 1.0),
                    *offset,
                    math::Vec3::new(0.0, 0.0, 0.0),
                ),
            );
        }
    }
}

// ---------------------------------------------------------------------------

/// Returns all 3D demo scenes.
pub fn all_scenes_3d() -> Vec<Box<dyn PhysicsScene3D>> {
    vec![
        Box::new(BallsScene),
        Box::new(StackingScene),
        Box::new(JointsScene),
        Box::new(TrimeshScene),
        Box::new(CharacterScene),
        Box::new(RagdollScene),
        Box::new(VehicleScene),
    ]
}
