//! 2D physics scene definitions using the component approach.
//!
//! Each scene spawns entities with [`RigidBody2D`] + [`Collider2D`] + [`Transform`]
//! descriptor components, then calls [`build_physics_world_2d`] to materialize
//! them into rapier physics objects.

use redlilium_core::math;
use redlilium_ecs::Transform;
use redlilium_ecs::World;
use redlilium_ecs::physics::components2d::{Collider2D, RigidBody2D, build_physics_world_2d};
use redlilium_ecs::physics::physics2d::{PhysicsWorld2D, RigidBody2DHandle};
use redlilium_ecs::physics::rapier2d::prelude::*;

/// Trait for a 2D physics demo scene.
#[allow(dead_code)]
pub trait PhysicsScene2D: Send + Sync {
    fn name(&self) -> &str;
    fn setup(&self, world: &mut World);
    fn update(&self, _world: &mut World) {}
}

/// Helper: spawn a 2D physics entity with descriptor components.
fn spawn_entity(
    world: &mut World,
    body: RigidBody2D,
    collider: Collider2D,
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
// Balls — ground + falling circles
// ---------------------------------------------------------------------------

pub struct BallsScene2D;

impl PhysicsScene2D for BallsScene2D {
    fn name(&self) -> &str {
        "Balls"
    }

    fn setup(&self, world: &mut World) {
        // Ground
        spawn_entity(
            world,
            RigidBody2D::fixed(),
            Collider2D::cuboid(20.0, 0.1).with_restitution(0.3),
            Transform::IDENTITY,
        );

        // Falling circles
        for i in 0..40 {
            let x = (i % 8) as f32 * 1.2 - 4.0;
            let y = 5.0 + (i / 8) as f32 * 1.5;
            spawn_entity(
                world,
                RigidBody2D::dynamic(),
                Collider2D::ball(0.4).with_restitution(0.7),
                Transform::from_translation(math::Vec3::new(x, y, 0.0)),
            );
        }

        build_physics_world_2d(world);
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
        // Ground
        spawn_entity(
            world,
            RigidBody2D::fixed(),
            Collider2D::cuboid(20.0, 0.1),
            Transform::IDENTITY,
        );

        // Pyramid
        let layers = 10;
        let half = 0.5f32;
        let step = half * 2.0 + 0.05;
        for layer in 0..layers {
            let count = layers - layer;
            let offset = count as f32 * step / 2.0 - step / 2.0;
            let y = half + layer as f32 * step + 0.1;
            for i in 0..count {
                let x = i as f32 * step - offset;
                spawn_entity(
                    world,
                    RigidBody2D::dynamic(),
                    Collider2D::cuboid(half, half),
                    Transform::from_translation(math::Vec3::new(x, y, 0.0)),
                );
            }
        }

        build_physics_world_2d(world);
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
        let count = 12;
        let spacing = 1.0f32;
        let mut entities = Vec::new();

        for i in 0..count {
            let x = i as f32 * spacing - (count as f32 * spacing / 2.0);
            let y = 10.0f32;
            let is_anchor = i == 0 || i == count - 1;

            let body = if is_anchor {
                RigidBody2D::fixed()
            } else {
                RigidBody2D::dynamic()
            };

            let entity = spawn_entity(
                world,
                body,
                Collider2D::ball(0.3),
                Transform::from_translation(math::Vec3::new(x, y, 0.0)),
            );
            entities.push(entity);
        }

        // Build physics from descriptors
        build_physics_world_2d(world);

        // Connect with revolute joints
        let half_spacing = spacing as f64 / 2.0;
        for i in 0..count - 1 {
            let h1 = world.get::<RigidBody2DHandle>(entities[i]).unwrap().0;
            let h2 = world.get::<RigidBody2DHandle>(entities[i + 1]).unwrap().0;
            let joint = RevoluteJointBuilder::new()
                .local_anchor1(Vector::new(half_spacing, 0.0))
                .local_anchor2(Vector::new(-half_spacing, 0.0));
            let mut physics = world.resource_mut::<PhysicsWorld2D>();
            physics.add_impulse_joint(h1, h2, joint);
        }
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
        // Spawn falling shapes with descriptors
        for i in 0..25 {
            let x = (i % 5) as f32 * 2.0 - 4.0;
            let y = 8.0 + (i / 5) as f32 * 1.5;
            let collider = if i % 2 == 0 {
                Collider2D::ball(0.4).with_restitution(0.5)
            } else {
                Collider2D::cuboid(0.35, 0.35).with_restitution(0.3)
            };
            spawn_entity(
                world,
                RigidBody2D::dynamic(),
                collider,
                Transform::from_translation(math::Vec3::new(x, y, 0.0)),
            );
        }

        // Build physics from descriptors
        build_physics_world_2d(world);

        // Create polyline ground directly via rapier
        let vertices: Vec<_> = (0..40)
            .map(|i| {
                let x = i as f64 - 20.0;
                let y = (x * 0.3).sin() * 2.0;
                Vector::new(x, y)
            })
            .collect();
        let indices: Vec<_> = (0..vertices.len() as u32 - 1).map(|i| [i, i + 1]).collect();

        let ground_handle = {
            let mut physics = world.resource_mut::<PhysicsWorld2D>();
            let ground_handle = physics.add_body(RigidBodyBuilder::fixed().build());
            let polyline = ColliderBuilder::polyline(vertices, Some(indices)).build();
            physics.add_collider(polyline, ground_handle);
            ground_handle
        };

        let ground_entity = world.spawn();
        let _ = world.insert(ground_entity, RigidBody2DHandle(ground_handle));
        let _ = world.insert(ground_entity, Transform::IDENTITY);
        let _ = world.insert(ground_entity, redlilium_ecs::GlobalTransform::IDENTITY);
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
        // Ground
        spawn_entity(
            world,
            RigidBody2D::fixed(),
            Collider2D::cuboid(20.0, 0.1),
            Transform::IDENTITY,
        );

        // Platforms
        for i in 0..5 {
            spawn_entity(
                world,
                RigidBody2D::fixed(),
                Collider2D::cuboid(1.0, 0.1),
                Transform::from_translation(math::Vec3::new(
                    -4.0 + i as f32 * 2.5,
                    1.0 + i as f32 * 1.5,
                    0.0,
                )),
            );
        }

        // Character capsule
        spawn_entity(
            world,
            RigidBody2D::kinematic_position(),
            Collider2D::capsule_y(0.4, 0.25),
            Transform::from_translation(math::Vec3::new(0.0, 3.0, 0.0)),
        );

        build_physics_world_2d(world);
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
        // Ground
        spawn_entity(
            world,
            RigidBody2D::fixed(),
            Collider2D::cuboid(20.0, 0.1),
            Transform::IDENTITY,
        );

        // Ragdoll parts
        let torso = spawn_entity(
            world,
            RigidBody2D::dynamic(),
            Collider2D::cuboid(0.3, 0.5),
            Transform::from_translation(math::Vec3::new(0.0, 6.0, 0.0)),
        );

        let head = spawn_entity(
            world,
            RigidBody2D::dynamic(),
            Collider2D::ball(0.25),
            Transform::from_translation(math::Vec3::new(0.0, 7.0, 0.0)),
        );

        let mut arm_entities = Vec::new();
        for side in [-1.0f32, 1.0] {
            let arm = spawn_entity(
                world,
                RigidBody2D::dynamic(),
                Collider2D::capsule_y(0.3, 0.08),
                Transform::from_translation(math::Vec3::new(side * 0.7, 5.8, 0.0)),
            );
            arm_entities.push((side, arm));
        }

        let mut leg_entities = Vec::new();
        for side in [-1.0f32, 1.0] {
            let leg = spawn_entity(
                world,
                RigidBody2D::dynamic(),
                Collider2D::capsule_y(0.35, 0.1),
                Transform::from_translation(math::Vec3::new(side * 0.2, 5.0, 0.0)),
            );
            leg_entities.push((side, leg));
        }

        // Build physics
        build_physics_world_2d(world);

        // Create joints
        let torso_h = world.get::<RigidBody2DHandle>(torso).unwrap().0;
        let head_h = world.get::<RigidBody2DHandle>(head).unwrap().0;

        {
            let neck = RevoluteJointBuilder::new()
                .local_anchor1(Vector::new(0.0, 0.5))
                .local_anchor2(Vector::new(0.0, -0.25));
            let mut physics = world.resource_mut::<PhysicsWorld2D>();
            physics.add_impulse_joint(torso_h, head_h, neck);
        }

        for (side, arm) in &arm_entities {
            let side = *side as f64;
            let arm_h = world.get::<RigidBody2DHandle>(*arm).unwrap().0;
            let shoulder = RevoluteJointBuilder::new()
                .local_anchor1(Vector::new(side * 0.35, 0.4))
                .local_anchor2(Vector::new(0.0, 0.3));
            let mut physics = world.resource_mut::<PhysicsWorld2D>();
            physics.add_impulse_joint(torso_h, arm_h, shoulder);
        }

        for (side, leg) in &leg_entities {
            let side = *side as f64;
            let leg_h = world.get::<RigidBody2DHandle>(*leg).unwrap().0;
            let hip = RevoluteJointBuilder::new()
                .local_anchor1(Vector::new(side * 0.2, -0.5))
                .local_anchor2(Vector::new(0.0, 0.35));
            let mut physics = world.resource_mut::<PhysicsWorld2D>();
            physics.add_impulse_joint(torso_h, leg_h, hip);
        }
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
        // Ground
        spawn_entity(
            world,
            RigidBody2D::fixed(),
            Collider2D::cuboid(40.0, 0.1).with_friction(1.0),
            Transform::IDENTITY,
        );

        // Chassis
        let chassis_pos = math::Vec3::new(0.0, 2.0, 0.0);
        let chassis = spawn_entity(
            world,
            RigidBody2D::dynamic(),
            Collider2D::cuboid(1.2, 0.3).with_density(2.0),
            Transform::from_translation(chassis_pos),
        );

        // Wheels
        let wheel_offsets = [math::Vec2::new(-0.8, -0.4), math::Vec2::new(0.8, -0.4)];
        let mut wheel_entities = Vec::new();
        for offset in &wheel_offsets {
            let wheel_pos =
                math::Vec3::new(chassis_pos.x + offset.x, chassis_pos.y + offset.y, 0.0);
            let wheel = spawn_entity(
                world,
                RigidBody2D::dynamic(),
                Collider2D::ball(0.3).with_friction(1.5).with_density(0.5),
                Transform::from_translation(wheel_pos),
            );
            wheel_entities.push((*offset, wheel));
        }

        // Build physics
        build_physics_world_2d(world);

        // Create wheel joints
        let chassis_h = world.get::<RigidBody2DHandle>(chassis).unwrap().0;
        for (offset, wheel_entity) in &wheel_entities {
            let wheel_h = world.get::<RigidBody2DHandle>(*wheel_entity).unwrap().0;
            let axle = RevoluteJointBuilder::new()
                .local_anchor1(Vector::new(offset.x as f64, offset.y as f64))
                .local_anchor2(Vector::ZERO);
            let mut physics = world.resource_mut::<PhysicsWorld2D>();
            physics.add_impulse_joint(chassis_h, wheel_h, axle);
        }
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
