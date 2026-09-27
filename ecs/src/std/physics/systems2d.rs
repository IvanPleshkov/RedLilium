//! 2D physics ECS systems.
//!
//! Systems that step the 2D physics simulation, sync transforms,
//! and manage rigid body / joint creation and removal.

use super::rapier2d::prelude::*;
use super::world2d::{ImpulseJoint2DHandle, PhysicsWorld2D, RigidBody2DHandle};

// ---- StepPhysics2D system ----

/// ECS system that steps the 2D physics simulation and syncs body positions
/// back to ECS [`Transform`](crate::Transform) components.
///
/// For 2D, the X/Y rapier position maps to the Transform's X/Y translation,
/// and the rapier rotation angle maps to a Z-axis rotation quaternion.
pub struct StepPhysics2D;

impl crate::System for StepPhysics2D {
    type Result = ();
    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        let fixed_dt = {
            let world = ctx.raw_world();
            world
                .has_resource::<crate::Time>()
                .then(|| world.resource::<crate::Time>().fixed_delta())
        };
        ctx.lock::<(
            crate::ResMut<PhysicsWorld2D>,
            crate::Read<RigidBody2DHandle>,
            crate::Write<crate::Transform>,
        )>()
        .execute(|(mut physics, handles, mut transforms)| {
            redlilium_core::profile_scope!("ecs: step_physics_2d");

            if let Some(dt) = fixed_dt {
                physics.integration_parameters.dt = dt as Real;
            }
            physics.step();

            // Sync positions back to transforms
            for (idx, handle) in handles.iter() {
                if let Some(body) = physics.bodies.get(handle.0)
                    && (body.is_dynamic() || body.is_kinematic())
                    && let Some(mut transform) = transforms.get_mut(idx)
                {
                    let pos = body.position();
                    let t = pos.translation;
                    // Preserve Z: 2D physics only governs X/Y, but Z is commonly
                    // used for draw-order/layering. Overwriting it with 0 would
                    // silently flatten the scene's layering.
                    transform.translation.x = t.x as f32;
                    transform.translation.y = t.y as f32;
                    let angle = pos.rotation.angle() as f32;
                    transform.rotation = redlilium_core::math::quat_from_rotation_z(angle);
                }
            }
        });
        Ok(())
    }
}

// Remove through Rapier first, then reconcile joints against the actual set.
// Descriptor endpoints may already have changed, so they are not a reliable
// source for discovering which live joints the removed bodies owned.
fn remove_bodies(physics: &mut PhysicsWorld2D, stale: &[crate::Entity]) -> Vec<crate::Entity> {
    for entity in stale {
        if let Some(handle) = physics.entity_to_body.remove(entity) {
            physics.body_to_entity.remove(&handle);
            physics.remove_body(handle);
        }
    }
    let mut stale_joints = Vec::new();
    if !stale.is_empty() {
        physics.entity_to_joint.retain(|entity, handle| {
            let live = physics.impulse_joints.contains(*handle);
            if !live {
                stale_joints.push(*entity);
            }
            live
        });
    }
    stale_joints
}

fn remove_body_components(
    world: &mut crate::World,
    bodies: &[crate::Entity],
    joints: &[crate::Entity],
) {
    for &entity in joints {
        if world.is_alive(entity) {
            let _ = world.remove::<ImpulseJoint2DHandle>(entity);
        }
    }
    for &entity in bodies {
        if world.is_alive(entity) {
            let _ = world.remove::<RigidBody2DHandle>(entity);
        }
    }
}

// ---- SyncPhysicsBodies2D exclusive system ----

/// Exclusive system that creates/removes rapier bodies from ECS descriptor components.
///
/// Detects entities with [`RigidBody2D`](super::components2d::RigidBody2D) +
/// [`Collider2D`](super::components2d::Collider2D) +
/// [`Transform`](crate::Transform) and creates corresponding rapier objects.
/// Removing any required component, despawning or excluding an entity from game
/// queries removes its body, attached colliders and joints on the next sync.
pub struct SyncPhysicsBodies2D;

impl crate::ExclusiveSystem for SyncPhysicsBodies2D {
    type Result = ();

    fn run(&mut self, world: &mut crate::World) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_bodies_2d");

        // Ensure resource exists
        if !world.has_resource::<PhysicsWorld2D>() {
            world.insert_resource(PhysicsWorld2D::default());
        }

        // Phase 1: Find bodies whose entity is excluded, dead or missing prerequisites.
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld2D>();
            physics
                .entity_to_body
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_excluded_from_game(**e)
                        || world.get::<super::components2d::RigidBody2D>(**e).is_none()
                        || world.get::<super::components2d::Collider2D>(**e).is_none()
                        || world.get::<crate::Transform>(**e).is_none()
                })
                .copied()
                .collect()
        };

        if !stale.is_empty() {
            let stale_joints = remove_bodies(&mut world.resource_mut::<PhysicsWorld2D>(), &stale);
            remove_body_components(world, &stale, &stale_joints);
        }

        // Phase 2: Find new bodies (have descriptors, not in mapping, not disabled)
        let new_entities: Vec<(
            crate::Entity,
            super::components2d::RigidBody2D,
            super::components2d::Collider2D,
            crate::Transform,
        )> = {
            let physics = world.resource::<PhysicsWorld2D>();
            world
                .iter_entities()
                .filter(|e| {
                    !physics.entity_to_body.contains_key(e) && !world.is_excluded_from_game(*e)
                })
                .filter_map(|entity| {
                    let body = world
                        .get::<super::components2d::RigidBody2D>(entity)?
                        .clone();
                    let collider = world
                        .get::<super::components2d::Collider2D>(entity)?
                        .clone();
                    let transform = *world.get::<crate::Transform>(entity)?;
                    Some((entity, body, collider, transform))
                })
                .collect()
        };

        if !new_entities.is_empty() {
            let mut handles = Vec::with_capacity(new_entities.len());
            {
                let mut physics = world.resource_mut::<PhysicsWorld2D>();
                for (entity, body_desc, collider_desc, transform) in &new_entities {
                    let rapier_body = body_desc.to_rigid_body(transform);
                    let body_handle = physics.add_body(rapier_body);
                    let rapier_collider = collider_desc.to_collider();
                    physics.add_collider(rapier_collider, body_handle);
                    physics.entity_to_body.insert(*entity, body_handle);
                    physics.body_to_entity.insert(body_handle, *entity);
                    handles.push((*entity, body_handle));
                }
            }
            for (entity, handle) in handles {
                let _ = world.insert(entity, RigidBody2DHandle(handle));
            }
        }

        Ok(())
    }
}

// ---- SyncPhysicsJoints2D exclusive system ----

/// Exclusive system that creates/removes rapier joints from ECS descriptor components.
///
/// Detects entities with [`ImpulseJoint2D`](super::components2d::ImpulseJoint2D)
/// and creates corresponding rapier joints. Also detects removed/despawned joints.
///
/// Must run after [`SyncPhysicsBodies2D`] so that body handles are available.
pub struct SyncPhysicsJoints2D;

impl crate::ExclusiveSystem for SyncPhysicsJoints2D {
    type Result = ();

    fn run(&mut self, world: &mut crate::World) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_joints_2d");

        if !world.has_resource::<PhysicsWorld2D>() {
            return Ok(());
        }

        // Phase 1: Find stale joints (entity dead, disabled, or lost ImpulseJoint2D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld2D>();
            physics
                .entity_to_joint
                .iter()
                .filter(|(e, handle)| {
                    !physics.impulse_joints.contains(**handle)
                        || !world.is_alive(**e)
                        || world.is_excluded_from_game(**e)
                        || world
                            .get::<super::components2d::ImpulseJoint2D>(**e)
                            .is_none()
                })
                .map(|(entity, _)| *entity)
                .collect()
        };

        if !stale.is_empty() {
            {
                let mut physics = world.resource_mut::<PhysicsWorld2D>();
                for entity in &stale {
                    if let Some(jh) = physics.entity_to_joint.remove(entity) {
                        physics.remove_impulse_joint(jh, true);
                    }
                }
            }
            for entity in &stale {
                if world.is_alive(*entity) {
                    let _ = world.remove::<ImpulseJoint2DHandle>(*entity);
                }
            }
        }

        // Phase 2: Find new joints (not in mapping, not disabled)
        let new_joints: Vec<(crate::Entity, super::components2d::ImpulseJoint2D)> = {
            let physics = world.resource::<PhysicsWorld2D>();
            world
                .iter_entities()
                .filter(|e| {
                    !physics.entity_to_joint.contains_key(e) && !world.is_excluded_from_game(*e)
                })
                .filter_map(|entity| {
                    let joint = world
                        .get::<super::components2d::ImpulseJoint2D>(entity)?
                        .clone();
                    Some((entity, joint))
                })
                .collect()
        };

        if !new_joints.is_empty() {
            let mut handles = Vec::new();
            {
                let mut physics = world.resource_mut::<PhysicsWorld2D>();
                for (entity, joint_desc) in &new_joints {
                    let body1_handle = match physics.entity_to_body.get(&joint_desc.body1) {
                        Some(h) => *h,
                        None => continue,
                    };
                    let body2_handle = match physics.entity_to_body.get(&joint_desc.body2) {
                        Some(h) => *h,
                        None => continue,
                    };
                    let rapier_joint = joint_desc.to_rapier_joint();
                    let jh = physics.add_impulse_joint(body1_handle, body2_handle, rapier_joint);
                    physics.entity_to_joint.insert(*entity, jh);
                    handles.push((*entity, jh));
                }
            }
            for (entity, handle) in handles {
                let _ = world.insert(entity, ImpulseJoint2DHandle(handle));
            }
        }

        Ok(())
    }
}

// ---- Regular system variants ----

/// Regular system variant of [`SyncPhysicsBodies2D`].
///
/// Uses lock-execute + deferred commands instead of exclusive world access.
/// Allows parallel scheduling but joints may lag 1 frame behind body creation.
pub struct SyncPhysicsBodiesSystem2D;

impl crate::System for SyncPhysicsBodiesSystem2D {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_bodies_system_2d");

        let (new_entities, stale_entities, stale_joints) = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld2D>,
                crate::Read<super::components2d::RigidBody2D>,
                crate::Read<super::components2d::Collider2D>,
                crate::Read<crate::Transform>,
            )>()
            .execute(|(mut physics, bodies, colliders, transforms)| {
                // Remove stale: entity dead (full-identity check, so a recycled
                // slot does not keep the old body), excluded, or missing prerequisites.
                let stale: Vec<crate::Entity> = physics
                    .entity_to_body
                    .keys()
                    .filter(|e| {
                        !ctx.is_alive(**e)
                            || ctx.is_excluded_from_game(**e)
                            || bodies.get(e.index()).is_none()
                            || colliders.get(e.index()).is_none()
                            || transforms.get(e.index()).is_none()
                    })
                    .copied()
                    .collect();
                let stale_joints = remove_bodies(&mut physics, &stale);

                // Create new
                let mut new_pairs: Vec<(crate::Entity, RigidBodyHandle)> = Vec::new();
                for (idx, body_desc) in bodies.iter() {
                    if let Some(entity) = ctx.raw_world().entity_at_index(idx)
                        && !physics.entity_to_body.contains_key(&entity)
                        && let (Some(collider_desc), Some(transform)) =
                            (colliders.get(idx), transforms.get(idx))
                    {
                        let rapier_body = body_desc.to_rigid_body(transform);
                        let body_handle = physics.add_body(rapier_body);
                        let rapier_collider = collider_desc.to_collider();
                        physics.add_collider(rapier_collider, body_handle);
                        new_pairs.push((entity, body_handle));
                    }
                }

                (new_pairs, stale, stale_joints)
            });

        if !new_entities.is_empty() || !stale_entities.is_empty() || !stale_joints.is_empty() {
            ctx.commands(move |world| {
                remove_body_components(world, &stale_entities, &stale_joints);
                for (entity, handle) in new_entities {
                    // Earlier commands can despawn/recycle the slot or remove prerequisites.
                    // Never publish a pending body against a different entity generation.
                    let valid = world.is_alive(entity)
                        && !world.is_excluded_from_game(entity)
                        && world
                            .get::<super::components2d::RigidBody2D>(entity)
                            .is_some()
                        && world
                            .get::<super::components2d::Collider2D>(entity)
                            .is_some()
                        && world.get::<crate::Transform>(entity).is_some();
                    if valid {
                        let _ = world.insert(entity, RigidBody2DHandle(handle));
                        let mut physics = world.resource_mut::<PhysicsWorld2D>();
                        physics.entity_to_body.insert(entity, handle);
                        physics.body_to_entity.insert(handle, entity);
                    } else {
                        world.resource_mut::<PhysicsWorld2D>().remove_body(handle);
                    }
                }
            });
        }

        Ok(())
    }
}

/// Regular system variant of [`SyncPhysicsJoints2D`].
///
/// Uses lock-execute + deferred commands. Joint creation may lag 1 frame behind
/// body creation when both are spawned in the same frame.
pub struct SyncPhysicsJointsSystem2D;

impl crate::System for SyncPhysicsJointsSystem2D {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_joints_system_2d");

        let (new_entities, stale_entities) = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld2D>,
                crate::Read<super::components2d::ImpulseJoint2D>,
            )>()
            .execute(|(mut physics, joints)| {
                // Remove stale: entity dead (full-identity check), disabled, or
                // lost the ImpulseJoint2D component.
                let stale: Vec<crate::Entity> = physics
                    .entity_to_joint
                    .iter()
                    .filter(|(e, handle)| {
                        !physics.impulse_joints.contains(**handle)
                            || !ctx.is_alive(**e)
                            || ctx.is_excluded_from_game(**e)
                            || joints.get(e.index()).is_none()
                    })
                    .map(|(entity, _)| *entity)
                    .collect();
                for entity in &stale {
                    if let Some(jh) = physics.entity_to_joint.remove(entity) {
                        physics.remove_impulse_joint(jh, true);
                    }
                }

                // Create new
                let mut new_pairs: Vec<(crate::Entity, ImpulseJointHandle)> = Vec::new();
                for (idx, joint_desc) in joints.iter() {
                    if let Some(entity) = ctx.raw_world().entity_at_index(idx)
                        && !physics.entity_to_joint.contains_key(&entity)
                    {
                        let body1_handle = match physics.entity_to_body.get(&joint_desc.body1) {
                            Some(h) => *h,
                            None => continue,
                        };
                        let body2_handle = match physics.entity_to_body.get(&joint_desc.body2) {
                            Some(h) => *h,
                            None => continue,
                        };
                        let rapier_joint = joint_desc.to_rapier_joint();
                        let jh =
                            physics.add_impulse_joint(body1_handle, body2_handle, rapier_joint);
                        new_pairs.push((entity, jh));
                    }
                }

                (new_pairs, stale)
            });

        if !new_entities.is_empty() || !stale_entities.is_empty() {
            ctx.commands(move |world| {
                for entity in stale_entities {
                    if world.is_alive(entity) {
                        let _ = world.remove::<ImpulseJoint2DHandle>(entity);
                    }
                }
                for (entity, handle) in new_entities {
                    let valid = world.is_alive(entity)
                        && !world.is_excluded_from_game(entity)
                        && world
                            .get::<super::components2d::ImpulseJoint2D>(entity)
                            .is_some()
                        && world
                            .resource::<PhysicsWorld2D>()
                            .impulse_joints
                            .contains(handle);
                    if valid {
                        let _ = world.insert(entity, ImpulseJoint2DHandle(handle));
                        let mut physics = world.resource_mut::<PhysicsWorld2D>();
                        physics.entity_to_joint.insert(entity, handle);
                    } else {
                        world
                            .resource_mut::<PhysicsWorld2D>()
                            .remove_impulse_joint(handle, true);
                    }
                }
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_bodies_creates_and_removes_2d() {
        use crate::system::run_exclusive_system_once;
        use redlilium_core::math::Vec3;

        let mut world = crate::World::new();
        crate::register_std_components(&mut world);

        // Spawn a dynamic ball
        let e = world.spawn();
        let _ = world.insert(e, crate::physics::components2d::RigidBody2D::dynamic());
        let _ = world.insert(e, crate::physics::components2d::Collider2D::ball(0.5));
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        );

        // Run sync
        run_exclusive_system_once(&mut SyncPhysicsBodies2D, &mut world).unwrap();

        // Should have handle
        assert!(world.get::<RigidBody2DHandle>(e).is_some());
        {
            let physics = world.resource::<PhysicsWorld2D>();
            assert_eq!(physics.bodies.len(), 1);
            assert!(physics.entity_to_body.contains_key(&e));
        }

        // Now remove the descriptor
        let _ = world.remove::<crate::physics::components2d::RigidBody2D>(e);

        // Run sync again
        run_exclusive_system_once(&mut SyncPhysicsBodies2D, &mut world).unwrap();

        // Should be cleaned up
        assert!(world.get::<RigidBody2DHandle>(e).is_none());
        {
            let physics = world.resource::<PhysicsWorld2D>();
            assert_eq!(physics.bodies.len(), 0);
            assert!(!physics.entity_to_body.contains_key(&e));
        }
    }
}
