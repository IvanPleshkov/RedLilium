//! 3D physics scene definitions using the component approach.
//!
//! Each scene spawns entities with [`RigidBody3D`] + [`Collider3D`] + [`Transform`]
//! descriptor components, then calls [`build_physics_world_3d`] to materialize
//! them into rapier physics objects.

use redlilium_core::math;
use redlilium_ecs::Transform;
use redlilium_ecs::World;
use redlilium_ecs::physics::components3d::{Collider3D, RigidBody3D, build_physics_world_3d};
use redlilium_ecs::physics::physics3d::{PhysicsWorld3D, RigidBody3DHandle};
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

        build_physics_world_3d(world);
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

        build_physics_world_3d(world);
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

        // Build physics from descriptors
        build_physics_world_3d(world);

        // Connect neighbours with spherical joints
        let half = spacing as f64 / 2.0;
        for i in 0..size {
            for j in 0..size {
                if i + 1 < size {
                    let h1 = world.get::<RigidBody3DHandle>(entity_grid[i][j]).unwrap().0;
                    let h2 = world
                        .get::<RigidBody3DHandle>(entity_grid[i + 1][j])
                        .unwrap()
                        .0;
                    let joint = SphericalJointBuilder::new()
                        .local_anchor1(Vector::new(half, 0.0, 0.0))
                        .local_anchor2(Vector::new(-half, 0.0, 0.0));
                    let mut physics = world.resource_mut::<PhysicsWorld3D>();
                    physics.add_impulse_joint(h1, h2, joint);
                }
                if j + 1 < size {
                    let h1 = world.get::<RigidBody3DHandle>(entity_grid[i][j]).unwrap().0;
                    let h2 = world
                        .get::<RigidBody3DHandle>(entity_grid[i][j + 1])
                        .unwrap()
                        .0;
                    let joint = SphericalJointBuilder::new()
                        .local_anchor1(Vector::new(0.0, 0.0, half))
                        .local_anchor2(Vector::new(0.0, 0.0, -half));
                    let mut physics = world.resource_mut::<PhysicsWorld3D>();
                    physics.add_impulse_joint(h1, h2, joint);
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

        // Build physics from descriptor components
        build_physics_world_3d(world);

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

        let ground_handle = {
            let mut physics = world.resource_mut::<PhysicsWorld3D>();
            let ground_handle = physics.add_body(RigidBodyBuilder::fixed().build());
            let trimesh = ColliderBuilder::trimesh(vertices, indices)
                .expect("valid trimesh")
                .restitution(0.3)
                .build();
            physics.add_collider(trimesh, ground_handle);
            ground_handle
        };

        // Spawn ECS entity for the ground trimesh (handle only, no descriptors)
        let ground_entity = world.spawn();
        let _ = world.insert(ground_entity, RigidBody3DHandle(ground_handle));
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

        build_physics_world_3d(world);
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

        let mut arm_entities = Vec::new();
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
            arm_entities.push((side, upper_arm, forearm));
        }

        let mut leg_entities = Vec::new();
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
            leg_entities.push((side, thigh, shin));
        }

        // Build physics from descriptors
        build_physics_world_3d(world);

        // Create joints using handles
        let torso_h = world.get::<RigidBody3DHandle>(torso).unwrap().0;
        let head_h = world.get::<RigidBody3DHandle>(head).unwrap().0;

        {
            let neck = SphericalJointBuilder::new()
                .local_anchor1(Vector::new(0.0, 0.5, 0.0))
                .local_anchor2(Vector::new(0.0, -0.25, 0.0));
            let mut physics = world.resource_mut::<PhysicsWorld3D>();
            physics.add_impulse_joint(torso_h, head_h, neck);
        }

        for (side, upper_arm, forearm) in &arm_entities {
            let side = *side as f64;
            let ua_h = world.get::<RigidBody3DHandle>(*upper_arm).unwrap().0;
            let fa_h = world.get::<RigidBody3DHandle>(*forearm).unwrap().0;

            {
                let shoulder = SphericalJointBuilder::new()
                    .local_anchor1(Vector::new(side * 0.35, 0.4, 0.0))
                    .local_anchor2(Vector::new(0.0, 0.25, 0.0));
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
                physics.add_impulse_joint(torso_h, ua_h, shoulder);
            }
            {
                let elbow = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
                    .local_anchor1(Vector::new(0.0, -0.25, 0.0))
                    .local_anchor2(Vector::new(0.0, 0.2, 0.0));
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
                physics.add_impulse_joint(ua_h, fa_h, elbow);
            }
        }

        for (side, thigh, shin) in &leg_entities {
            let side = *side as f64;
            let th_h = world.get::<RigidBody3DHandle>(*thigh).unwrap().0;
            let sh_h = world.get::<RigidBody3DHandle>(*shin).unwrap().0;

            {
                let hip = SphericalJointBuilder::new()
                    .local_anchor1(Vector::new(side * 0.2, -0.5, 0.0))
                    .local_anchor2(Vector::new(0.0, 0.3, 0.0));
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
                physics.add_impulse_joint(torso_h, th_h, hip);
            }
            {
                let knee = RevoluteJointBuilder::new(Vector::new(1.0, 0.0, 0.0))
                    .local_anchor1(Vector::new(0.0, -0.3, 0.0))
                    .local_anchor2(Vector::new(0.0, 0.25, 0.0));
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
                physics.add_impulse_joint(th_h, sh_h, knee);
            }
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

        let mut wheel_entities = Vec::new();
        for offset in &wheel_offsets {
            let wheel_pos = chassis_pos + *offset;
            let wheel = spawn_entity(
                world,
                RigidBody3D::dynamic(),
                Collider3D::ball(0.3).with_friction(1.5).with_density(0.5),
                Transform::from_translation(wheel_pos),
            );
            wheel_entities.push((*offset, wheel));
        }

        // Build physics from descriptors
        build_physics_world_3d(world);

        // Create wheel joints
        let chassis_h = world.get::<RigidBody3DHandle>(chassis).unwrap().0;
        for (offset, wheel_entity) in &wheel_entities {
            let wheel_h = world.get::<RigidBody3DHandle>(*wheel_entity).unwrap().0;
            let axle = RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
                .local_anchor1(Vector::new(
                    offset.x as f64,
                    offset.y as f64,
                    offset.z as f64,
                ))
                .local_anchor2(Vector::ZERO);
            let mut physics = world.resource_mut::<PhysicsWorld3D>();
            physics.add_impulse_joint(chassis_h, wheel_h, axle);
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
