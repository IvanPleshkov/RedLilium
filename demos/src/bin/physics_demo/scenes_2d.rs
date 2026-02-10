//! 2D physics scene definitions.
//!
//! Each scene populates an ECS world with a [`PhysicsWorld2D`] resource
//! and entities carrying rigid-body handle + transform components.

use ecs_std::Transform;
use ecs_std::physics::physics2d::{PhysicsWorld2D, RigidBody2DHandle};
use ecs_std::physics::rapier2d::prelude::*;
use glam::Vec3;
use redlilium_ecs::World;

/// Trait for a 2D physics demo scene.
#[allow(dead_code)]
pub trait PhysicsScene2D: Send + Sync {
    fn name(&self) -> &str;
    fn setup(&self, world: &mut World);
    fn update(&self, _world: &mut World) {}
}

/// Helper: spawn a 2D physics entity.
fn spawn_physics_entity(
    world: &mut World,
    physics: &mut PhysicsWorld2D,
    body: RigidBody,
    collider: Collider,
) {
    let body_handle = physics.add_body(body);
    physics.add_collider(collider, body_handle);

    let pos = physics.bodies[body_handle].position();
    let t = pos.translation;
    let angle = pos.rotation.angle() as f32;
    let transform = Transform::new(
        Vec3::new(t.x as f32, t.y as f32, 0.0),
        glam::Quat::from_rotation_z(angle),
        Vec3::ONE,
    );

    let entity = world.spawn();
    let _ = world.insert(entity, RigidBody2DHandle(body_handle));
    let _ = world.insert(entity, transform);
    let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);
}

// ---------------------------------------------------------------------------
// Balls — ground + falling circles
// ---------------------------------------------------------------------------

pub struct BallsScene2D;

impl PhysicsScene2D for BallsScene2D {
    fn name(&self) -> &str {
        "Balls"
    }

    fn setup(&self, world: &mut World) {
        let mut physics = PhysicsWorld2D::default();

        // Ground
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(20.0, 0.1).restitution(0.3).build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Falling circles
        for i in 0..40 {
            let x = (i % 8) as f64 * 1.2 - 4.0;
            let y = 5.0 + (i / 8) as f64 * 1.5;
            let body = RigidBodyBuilder::dynamic()
                .translation(Vector::new(x, y))
                .build();
            let collider = ColliderBuilder::ball(0.4).restitution(0.7).build();
            spawn_physics_entity(world, &mut physics, body, collider);
        }

        world.insert_resource(physics);
    }
}

// ---------------------------------------------------------------------------
// Stacking — pyramid of boxes
// ---------------------------------------------------------------------------

pub struct StackingScene2D;

impl PhysicsScene2D for StackingScene2D {
    fn name(&self) -> &str {
        "Stacking"
    }

    fn setup(&self, world: &mut World) {
        let mut physics = PhysicsWorld2D::default();

        // Ground
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(20.0, 0.1).build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Pyramid
        let layers = 10;
        let half = 0.5;
        let step = half * 2.0 + 0.05;
        for layer in 0..layers {
            let count = layers - layer;
            let offset = count as f64 * step / 2.0 - step / 2.0;
            let y = half + layer as f64 * step + 0.1;
            for i in 0..count {
                let x = i as f64 * step - offset;
                let body = RigidBodyBuilder::dynamic()
                    .translation(Vector::new(x, y))
                    .build();
                let collider = ColliderBuilder::cuboid(half, half).build();
                spawn_physics_entity(world, &mut physics, body, collider);
            }
        }

        world.insert_resource(physics);
    }
}

// ---------------------------------------------------------------------------
// Joints — chain of balls
// ---------------------------------------------------------------------------

pub struct JointsScene2D;

impl PhysicsScene2D for JointsScene2D {
    fn name(&self) -> &str {
        "Joints"
    }

    fn setup(&self, world: &mut World) {
        let mut physics = PhysicsWorld2D::default();

        let count = 12;
        let spacing = 1.0;
        let mut prev_handle = None;

        for i in 0..count {
            let x = i as f64 * spacing - (count as f64 * spacing / 2.0);
            let y = 10.0;
            let is_anchor = i == 0 || i == count - 1;

            let builder = if is_anchor {
                RigidBodyBuilder::fixed()
            } else {
                RigidBodyBuilder::dynamic()
            };
            let body = builder.translation(Vector::new(x, y)).build();
            let body_handle = physics.add_body(body);
            physics.add_collider(ColliderBuilder::ball(0.3).build(), body_handle);

            let pos = physics.bodies[body_handle].position();
            let t = pos.translation;
            let entity = world.spawn();
            let _ = world.insert(entity, RigidBody2DHandle(body_handle));
            let _ = world.insert(
                entity,
                Transform::from_translation(Vec3::new(t.x as f32, t.y as f32, 0.0)),
            );
            let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);

            if let Some(prev) = prev_handle {
                let joint = RevoluteJointBuilder::new()
                    .local_anchor1(Vector::new(spacing / 2.0, 0.0))
                    .local_anchor2(Vector::new(-spacing / 2.0, 0.0));
                physics.add_impulse_joint(prev, body_handle, joint);
            }
            prev_handle = Some(body_handle);
        }

        world.insert_resource(physics);
    }
}

// ---------------------------------------------------------------------------
// Trimesh — custom polygon ground + falling shapes
// ---------------------------------------------------------------------------

pub struct TrimeshScene2D;

impl PhysicsScene2D for TrimeshScene2D {
    fn name(&self) -> &str {
        "Trimesh"
    }

    fn setup(&self, world: &mut World) {
        let mut physics = PhysicsWorld2D::default();

        // Create polyline ground
        let vertices: Vec<_> = (0..40)
            .map(|i| {
                let x = i as f64 - 20.0;
                let y = (x * 0.3).sin() * 2.0;
                Vector::new(x, y)
            })
            .collect();
        let indices: Vec<_> = (0..vertices.len() as u32 - 1).map(|i| [i, i + 1]).collect();

        let ground = RigidBodyBuilder::fixed().build();
        let ground_handle = physics.add_body(ground);
        let polyline = ColliderBuilder::polyline(vertices, Some(indices)).build();
        physics.add_collider(polyline, ground_handle);

        let entity = world.spawn();
        let _ = world.insert(entity, RigidBody2DHandle(ground_handle));
        let _ = world.insert(entity, Transform::IDENTITY);
        let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);

        // Falling shapes
        for i in 0..25 {
            let x = (i % 5) as f64 * 2.0 - 4.0;
            let y = 8.0 + (i / 5) as f64 * 1.5;
            let body = RigidBodyBuilder::dynamic()
                .translation(Vector::new(x, y))
                .build();
            let collider = if i % 2 == 0 {
                ColliderBuilder::ball(0.4).restitution(0.5).build()
            } else {
                ColliderBuilder::cuboid(0.35, 0.35).restitution(0.3).build()
            };
            spawn_physics_entity(world, &mut physics, body, collider);
        }

        world.insert_resource(physics);
    }
}

// ---------------------------------------------------------------------------
// Character — platforms + kinematic capsule
// ---------------------------------------------------------------------------

pub struct CharacterScene2D;

impl PhysicsScene2D for CharacterScene2D {
    fn name(&self) -> &str {
        "Character"
    }

    fn setup(&self, world: &mut World) {
        let mut physics = PhysicsWorld2D::default();

        // Ground
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(20.0, 0.1).build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Platforms
        for i in 0..5 {
            let platform = RigidBodyBuilder::fixed()
                .translation(Vector::new(-4.0 + i as f64 * 2.5, 1.0 + i as f64 * 1.5))
                .build();
            let col = ColliderBuilder::cuboid(1.0, 0.1).build();
            spawn_physics_entity(world, &mut physics, platform, col);
        }

        // Character capsule
        let character = RigidBodyBuilder::kinematic_position_based()
            .translation(Vector::new(0.0, 3.0))
            .build();
        let char_col = ColliderBuilder::capsule_y(0.4, 0.25).build();
        spawn_physics_entity(world, &mut physics, character, char_col);

        world.insert_resource(physics);
    }
}

// ---------------------------------------------------------------------------
// Ragdoll — simple 2D ragdoll
// ---------------------------------------------------------------------------

pub struct RagdollScene2D;

impl PhysicsScene2D for RagdollScene2D {
    fn name(&self) -> &str {
        "Ragdoll"
    }

    fn setup(&self, world: &mut World) {
        let mut physics = PhysicsWorld2D::default();

        // Ground
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(20.0, 0.1).build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Torso
        let torso = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 6.0))
            .build();
        let torso_handle = physics.add_body(torso);
        physics.add_collider(ColliderBuilder::cuboid(0.3, 0.5).build(), torso_handle);

        // Head
        let head = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 7.0))
            .build();
        let head_handle = physics.add_body(head);
        physics.add_collider(ColliderBuilder::ball(0.25).build(), head_handle);

        let neck = RevoluteJointBuilder::new()
            .local_anchor1(Vector::new(0.0, 0.5))
            .local_anchor2(Vector::new(0.0, -0.25));
        physics.add_impulse_joint(torso_handle, head_handle, neck);

        // Arms
        for side in [-1.0, 1.0] {
            let arm = RigidBodyBuilder::dynamic()
                .translation(Vector::new(side * 0.7, 5.8))
                .build();
            let arm_handle = physics.add_body(arm);
            physics.add_collider(ColliderBuilder::capsule_y(0.3, 0.08).build(), arm_handle);

            let shoulder = RevoluteJointBuilder::new()
                .local_anchor1(Vector::new(side * 0.35, 0.4))
                .local_anchor2(Vector::new(0.0, 0.3));
            physics.add_impulse_joint(torso_handle, arm_handle, shoulder);
        }

        // Legs
        for side in [-1.0, 1.0] {
            let leg = RigidBodyBuilder::dynamic()
                .translation(Vector::new(side * 0.2, 5.0))
                .build();
            let leg_handle = physics.add_body(leg);
            physics.add_collider(ColliderBuilder::capsule_y(0.35, 0.1).build(), leg_handle);

            let hip = RevoluteJointBuilder::new()
                .local_anchor1(Vector::new(side * 0.2, -0.5))
                .local_anchor2(Vector::new(0.0, 0.35));
            physics.add_impulse_joint(torso_handle, leg_handle, hip);
        }

        // Spawn ECS entities for all bodies
        for (handle, body) in physics.bodies.iter() {
            let pos = body.position();
            let t = pos.translation;
            let angle = pos.rotation.angle() as f32;
            let entity = world.spawn();
            let _ = world.insert(entity, RigidBody2DHandle(handle));
            let _ = world.insert(
                entity,
                Transform::new(
                    Vec3::new(t.x as f32, t.y as f32, 0.0),
                    glam::Quat::from_rotation_z(angle),
                    Vec3::ONE,
                ),
            );
            let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);
        }

        world.insert_resource(physics);
    }
}

// ---------------------------------------------------------------------------
// Vehicle — box chassis + circle wheels
// ---------------------------------------------------------------------------

pub struct VehicleScene2D;

impl PhysicsScene2D for VehicleScene2D {
    fn name(&self) -> &str {
        "Vehicle"
    }

    fn setup(&self, world: &mut World) {
        let mut physics = PhysicsWorld2D::default();

        // Ground
        let ground = RigidBodyBuilder::fixed().build();
        let ground_col = ColliderBuilder::cuboid(40.0, 0.1).friction(1.0).build();
        spawn_physics_entity(world, &mut physics, ground, ground_col);

        // Chassis
        let chassis = RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 2.0))
            .build();
        let chassis_handle = physics.add_body(chassis);
        physics.add_collider(
            ColliderBuilder::cuboid(1.2, 0.3).density(2.0).build(),
            chassis_handle,
        );

        // Wheels
        let wheel_offsets = [Vector::new(-0.8, -0.4), Vector::new(0.8, -0.4)];
        for offset in &wheel_offsets {
            let wheel_pos = physics.bodies[chassis_handle].position().translation + *offset;
            let wheel = RigidBodyBuilder::dynamic().translation(wheel_pos).build();
            let wheel_handle = physics.add_body(wheel);
            physics.add_collider(
                ColliderBuilder::ball(0.3)
                    .friction(1.5)
                    .density(0.5)
                    .build(),
                wheel_handle,
            );

            let axle = RevoluteJointBuilder::new()
                .local_anchor1(*offset)
                .local_anchor2(Vector::ZERO);
            physics.add_impulse_joint(chassis_handle, wheel_handle, axle);
        }

        // Spawn ECS entities
        for (handle, body) in physics.bodies.iter() {
            let pos = body.position();
            let t = pos.translation;
            let angle = pos.rotation.angle() as f32;
            let entity = world.spawn();
            let _ = world.insert(entity, RigidBody2DHandle(handle));
            let _ = world.insert(
                entity,
                Transform::new(
                    Vec3::new(t.x as f32, t.y as f32, 0.0),
                    glam::Quat::from_rotation_z(angle),
                    Vec3::ONE,
                ),
            );
            let _ = world.insert(entity, ecs_std::GlobalTransform::IDENTITY);
        }

        world.insert_resource(physics);
    }
}

// ---------------------------------------------------------------------------

/// Returns all 2D demo scenes.
pub fn all_scenes_2d() -> Vec<Box<dyn PhysicsScene2D>> {
    vec![
        Box::new(BallsScene2D),
        Box::new(StackingScene2D),
        Box::new(JointsScene2D),
        Box::new(TrimeshScene2D),
        Box::new(CharacterScene2D),
        Box::new(RagdollScene2D),
        Box::new(VehicleScene2D),
    ]
}
