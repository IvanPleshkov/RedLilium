//! 3D physics scene definitions.
//!
//! Each scene populates an ECS world with a [`PhysicsWorld3D`] resource
//! and entities carrying rigid-body handle + transform components.

use ecs_std::Transform;
use ecs_std::physics::physics3d::{PhysicsWorld3D, RigidBody3DHandle};
use ecs_std::physics::rapier3d::prelude::*;
use glam::Vec3;
use redlilium_ecs::World;

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

/// Helper: spawn a physics entity (body + collider) and link it to an ECS entity.
fn spawn_physics_entity(
    world: &mut World,
    physics: &mut PhysicsWorld3D,
    body: RigidBody,
    collider: Collider,
) {
    let body_handle = physics.add_body(body);
    physics.add_collider(collider, body_handle);

    let pos = physics.bodies[body_handle].position();
    let t = pos.translation;
    let r = pos.rotation;
    let transform = Transform::new(
        Vec3::new(t.x as f32, t.y as f32, t.z as f32),
        glam::Quat::from_xyzw(r.x as f32, r.y as f32, r.z as f32, r.w as f32),
        Vec3::ONE,
    );

    let entity = world.spawn();
    let _ = world.insert(entity, RigidBody3DHandle(body_handle));
    let _ = world.insert(entity, transform);
    let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);
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
        let mut physics = PhysicsWorld3D::default();

        // Ground
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(20.0, 0.1, 20.0)
            .restitution(0.3)
            .build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Falling spheres
        let cols = 8;
        let rows = 8;
        for i in 0..cols {
            for j in 0..rows {
                let x = (i as f64 - cols as f64 / 2.0) * 1.2;
                let z = (j as f64 - rows as f64 / 2.0) * 1.2;
                let y = 5.0 + (i * rows + j) as f64 * 0.5;

                let body = RigidBodyBuilder::dynamic()
                    .translation(Vector::new(x, y, z))
                    .build();
                let collider = ColliderBuilder::ball(0.5).restitution(0.7).build();
                spawn_physics_entity(world, &mut physics, body, collider);
            }
        }

        world.insert_resource(physics);
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
        let mut physics = PhysicsWorld3D::default();

        // Ground
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(20.0, 0.1, 20.0).build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Pyramid of boxes
        let layers = 10;
        let box_half = 0.5;
        let gap = 0.05;
        let step = box_half * 2.0 + gap;
        for layer in 0..layers {
            let count = layers - layer;
            let offset = count as f64 * step / 2.0 - step / 2.0;
            let y = box_half + layer as f64 * step + 0.1;
            for i in 0..count {
                let x = i as f64 * step - offset;
                let body = RigidBodyBuilder::dynamic()
                    .translation(Vector::new(x, y, 0.0))
                    .build();
                let collider = ColliderBuilder::cuboid(box_half, box_half, box_half).build();
                spawn_physics_entity(world, &mut physics, body, collider);
            }
        }

        world.insert_resource(physics);
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
        let mut physics = PhysicsWorld3D::default();

        let size = 5;
        let spacing = 2.0;
        let mut handles = Vec::new();

        for i in 0..size {
            let mut row = Vec::new();
            for j in 0..size {
                let x = i as f64 * spacing - (size as f64 * spacing / 2.0);
                let z = j as f64 * spacing - (size as f64 * spacing / 2.0);
                let y = 10.0;

                let is_anchor = i == 0 || i == size - 1 || j == 0 || j == size - 1;
                let builder = if is_anchor {
                    RigidBodyBuilder::fixed()
                } else {
                    RigidBodyBuilder::dynamic()
                };

                let body = builder.translation(Vector::new(x, y, z)).build();
                let collider = ColliderBuilder::ball(0.3).build();
                let body_handle = physics.add_body(body);
                physics.add_collider(collider, body_handle);

                let pos = physics.bodies[body_handle].position();
                let t = pos.translation;
                let entity = world.spawn();
                let _ = world.insert(entity, RigidBody3DHandle(body_handle));
                let _ = world.insert(
                    entity,
                    Transform::from_translation(Vec3::new(t.x as f32, t.y as f32, t.z as f32)),
                );
                let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);

                row.push(body_handle);
            }
            handles.push(row);
        }

        // Connect neighbours with spherical joints
        let half = spacing / 2.0;
        for i in 0..size {
            for j in 0..size {
                if i + 1 < size {
                    let joint = SphericalJointBuilder::new()
                        .local_anchor1(Vector::new(half, 0.0, 0.0))
                        .local_anchor2(Vector::new(-half, 0.0, 0.0));
                    physics.add_impulse_joint(handles[i][j], handles[i + 1][j], joint);
                }
                if j + 1 < size {
                    let joint = SphericalJointBuilder::new()
                        .local_anchor1(Vector::new(0.0, 0.0, half))
                        .local_anchor2(Vector::new(0.0, 0.0, -half));
                    physics.add_impulse_joint(handles[i][j], handles[i][j + 1], joint);
                }
            }
        }

        world.insert_resource(physics);
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
        let mut physics = PhysicsWorld3D::default();

        // Build a wavy terrain from a triangle mesh
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

        let ground_body = RigidBodyBuilder::fixed().build();
        let ground_handle = physics.add_body(ground_body);
        let trimesh = ColliderBuilder::trimesh(vertices, indices)
            .expect("valid trimesh")
            .restitution(0.3)
            .build();
        physics.add_collider(trimesh, ground_handle);

        let entity = world.spawn();
        let _ = world.insert(entity, RigidBody3DHandle(ground_handle));
        let _ = world.insert(entity, Transform::IDENTITY);
        let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);

        // Falling objects
        for i in 0..30 {
            let x = (i % 6) as f64 * 2.0 - 5.0;
            let z = (i / 6) as f64 * 2.0 - 5.0;
            let y = 8.0 + i as f64 * 0.3;

            let body = RigidBodyBuilder::dynamic()
                .translation(Vector::new(x, y, z))
                .build();
            let collider = if i % 2 == 0 {
                ColliderBuilder::ball(0.4).restitution(0.5).build()
            } else {
                ColliderBuilder::cuboid(0.35, 0.35, 0.35)
                    .restitution(0.3)
                    .build()
            };
            spawn_physics_entity(world, &mut physics, body, collider);
        }

        world.insert_resource(physics);
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
        let mut physics = PhysicsWorld3D::default();

        // Floor
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(20.0, 0.1, 20.0).build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Ramp
        let ramp = RigidBodyBuilder::fixed()
            .translation(Vector::new(5.0, 1.0, 0.0))
            .rotation(Vector::new(0.0, 0.0, 0.3))
            .build();
        let ramp_col = ColliderBuilder::cuboid(4.0, 0.1, 3.0).build();
        spawn_physics_entity(world, &mut physics, ramp, ramp_col);

        // Platforms
        for i in 0..4 {
            let platform = RigidBodyBuilder::fixed()
                .translation(Vector::new(-5.0 + i as f64 * 3.0, 2.0 + i as f64, 5.0))
                .build();
            let col = ColliderBuilder::cuboid(1.0, 0.1, 1.0).build();
            spawn_physics_entity(world, &mut physics, platform, col);
        }

        // Character capsule (kinematic)
        let character = RigidBodyBuilder::kinematic_position_based()
            .translation(Vector::new(0.0, 2.0, 0.0))
            .build();
        let char_col = ColliderBuilder::capsule_y(0.5, 0.3).build();
        spawn_physics_entity(world, &mut physics, character, char_col);

        world.insert_resource(physics);
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
        let mut physics = PhysicsWorld3D::default();

        // Ground
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(20.0, 0.1, 20.0).build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Build a simple ragdoll
        let torso_body = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 5.0, 0.0))
            .build();
        let torso_handle = physics.add_body(torso_body);
        physics.add_collider(ColliderBuilder::cuboid(0.3, 0.5, 0.2).build(), torso_handle);

        let head_body = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 6.0, 0.0))
            .build();
        let head_handle = physics.add_body(head_body);
        physics.add_collider(ColliderBuilder::ball(0.25).build(), head_handle);

        // Head-torso joint
        let neck = SphericalJointBuilder::new()
            .local_anchor1(Vector::new(0.0, 0.5, 0.0))
            .local_anchor2(Vector::new(0.0, -0.25, 0.0));
        physics.add_impulse_joint(torso_handle, head_handle, neck);

        // Arms
        for side in [-1.0, 1.0] {
            let upper_arm = RigidBodyBuilder::dynamic()
                .translation(Vector::new(side * 0.7, 5.3, 0.0))
                .build();
            let ua_handle = physics.add_body(upper_arm);
            physics.add_collider(ColliderBuilder::capsule_y(0.25, 0.08).build(), ua_handle);

            let shoulder = SphericalJointBuilder::new()
                .local_anchor1(Vector::new(side * 0.35, 0.4, 0.0))
                .local_anchor2(Vector::new(0.0, 0.25, 0.0));
            physics.add_impulse_joint(torso_handle, ua_handle, shoulder);

            let forearm = RigidBodyBuilder::dynamic()
                .translation(Vector::new(side * 0.7, 4.7, 0.0))
                .build();
            let fa_handle = physics.add_body(forearm);
            physics.add_collider(ColliderBuilder::capsule_y(0.2, 0.07).build(), fa_handle);

            let elbow = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
                .local_anchor1(Vector::new(0.0, -0.25, 0.0))
                .local_anchor2(Vector::new(0.0, 0.2, 0.0));
            physics.add_impulse_joint(ua_handle, fa_handle, elbow);
        }

        // Legs
        for side in [-1.0, 1.0] {
            let thigh = RigidBodyBuilder::dynamic()
                .translation(Vector::new(side * 0.2, 4.2, 0.0))
                .build();
            let th_handle = physics.add_body(thigh);
            physics.add_collider(ColliderBuilder::capsule_y(0.3, 0.1).build(), th_handle);

            let hip = SphericalJointBuilder::new()
                .local_anchor1(Vector::new(side * 0.2, -0.5, 0.0))
                .local_anchor2(Vector::new(0.0, 0.3, 0.0));
            physics.add_impulse_joint(torso_handle, th_handle, hip);

            let shin = RigidBodyBuilder::dynamic()
                .translation(Vector::new(side * 0.2, 3.4, 0.0))
                .build();
            let sh_handle = physics.add_body(shin);
            physics.add_collider(ColliderBuilder::capsule_y(0.25, 0.08).build(), sh_handle);

            let knee = RevoluteJointBuilder::new(Vector::new(1.0, 0.0, 0.0))
                .local_anchor1(Vector::new(0.0, -0.3, 0.0))
                .local_anchor2(Vector::new(0.0, 0.25, 0.0));
            physics.add_impulse_joint(th_handle, sh_handle, knee);
        }

        // Spawn ECS entities for all bodies
        for (handle, body) in physics.bodies.iter() {
            let pos = body.position();
            let t = pos.translation;
            let r = pos.rotation;
            let entity = world.spawn();
            let _ = world.insert(entity, RigidBody3DHandle(handle));
            let _ = world.insert(
                entity,
                Transform::new(
                    Vec3::new(t.x as f32, t.y as f32, t.z as f32),
                    glam::Quat::from_xyzw(r.x as f32, r.y as f32, r.z as f32, r.w as f32),
                    Vec3::ONE,
                ),
            );
            let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);
        }

        world.insert_resource(physics);
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
        let mut physics = PhysicsWorld3D::default();

        // Ground
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(40.0, 0.1, 40.0)
            .friction(1.0)
            .build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Chassis
        let chassis = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 2.0, 0.0))
            .build();
        let chassis_handle = physics.add_body(chassis);
        physics.add_collider(
            ColliderBuilder::cuboid(1.0, 0.3, 0.5).density(2.0).build(),
            chassis_handle,
        );

        // Wheels
        let wheel_positions = [
            Vector::new(-0.8, -0.3, 0.6),
            Vector::new(0.8, -0.3, 0.6),
            Vector::new(-0.8, -0.3, -0.6),
            Vector::new(0.8, -0.3, -0.6),
        ];

        for wheel_offset in &wheel_positions {
            let wheel_pos = physics.bodies[chassis_handle].position().translation + *wheel_offset;
            let wheel = RigidBodyBuilder::dynamic().translation(wheel_pos).build();
            let wheel_handle = physics.add_body(wheel);
            physics.add_collider(
                ColliderBuilder::ball(0.3)
                    .friction(1.5)
                    .density(0.5)
                    .build(),
                wheel_handle,
            );

            // Hinge joint along Z axis (wheel spin axis)
            let axle = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
                .local_anchor1(*wheel_offset)
                .local_anchor2(Vector::ZERO);
            physics.add_impulse_joint(chassis_handle, wheel_handle, axle);
        }

        // Spawn ECS entities for all bodies
        for (handle, body) in physics.bodies.iter() {
            let pos = body.position();
            let t = pos.translation;
            let r = pos.rotation;
            let entity = world.spawn();
            let _ = world.insert(entity, RigidBody3DHandle(handle));
            let _ = world.insert(
                entity,
                Transform::new(
                    Vec3::new(t.x as f32, t.y as f32, t.z as f32),
                    glam::Quat::from_xyzw(r.x as f32, r.y as f32, r.z as f32, r.w as f32),
                    Vec3::ONE,
                ),
            );
            let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);
        }

        world.insert_resource(physics);
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
