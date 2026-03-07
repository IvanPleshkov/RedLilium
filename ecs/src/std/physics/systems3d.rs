//! 3D physics ECS systems.
//!
//! Systems that step the 3D physics simulation, sync transforms,
//! and manage rigid body / joint creation and removal.

use super::rapier3d::prelude::*;
use super::world3d::{ImpulseJoint3DHandle, PhysicsWorld3D, RigidBody3DHandle};

// ---- StepPhysics3D system ----

/// ECS system that steps the 3D physics simulation and syncs body positions
/// back to ECS [`Transform`](crate::Transform) components.
///
/// Requires a [`PhysicsWorld3D`] resource and entities with
/// [`RigidBody3DHandle`] + [`Transform`](crate::Transform) components.
pub struct StepPhysics3D;

impl crate::System for StepPhysics3D {
    type Result = ();
    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::ResMut<PhysicsWorld3D>,
            crate::Read<RigidBody3DHandle>,
            crate::Write<crate::Transform>,
        )>()
        .execute(|(mut physics, handles, mut transforms)| {
            redlilium_core::profile_scope!("ecs: step_physics_3d");

            // Step simulation
            physics.step();

            // Sync positions back to transforms
            for (idx, handle) in handles.iter() {
                if let Some(body) = physics.bodies.get(handle.0)
                    && (body.is_dynamic() || body.is_kinematic())
                    && let Some(transform) = transforms.get_mut(idx)
                {
                    let pos = body.position();
                    let t = pos.translation;
                    transform.translation =
                        redlilium_core::math::Vec3::new(t.x as f32, t.y as f32, t.z as f32);
                    let r = pos.rotation;
                    transform.rotation = redlilium_core::math::quat_from_xyzw(
                        r.x as f32, r.y as f32, r.z as f32, r.w as f32,
                    );
                }
            }
        });
        Ok(())
    }
}

// ---- SyncPhysicsBodies3D exclusive system ----

/// Exclusive system that creates/removes rapier bodies from ECS descriptor components.
///
/// Detects entities with [`RigidBody3D`](super::components3d::RigidBody3D) +
/// [`Collider3D`](super::components3d::Collider3D) +
/// [`Transform`](crate::Transform) and creates corresponding rapier objects.
/// Also detects removed/despawned entities and cleans up.
///
/// # Example
///
/// ```ignore
/// let mut systems = SystemsContainer::new();
/// systems.add_exclusive(SyncPhysicsBodies3D);
/// systems.add(StepPhysics3D);
/// systems.add_edge::<SyncPhysicsBodies3D, StepPhysics3D>().unwrap();
/// ```
pub struct SyncPhysicsBodies3D;

impl crate::ExclusiveSystem for SyncPhysicsBodies3D {
    type Result = ();

    fn run(&mut self, world: &mut crate::World) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_bodies_3d");

        // Ensure resource exists
        if !world.has_resource::<PhysicsWorld3D>() {
            world.insert_resource(PhysicsWorld3D::default());
        }

        // Phase 1: Find stale bodies (entity dead, disabled, or lost RigidBody3D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld3D>();
            physics
                .entity_to_body
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_disabled(**e)
                        || world.get::<super::components3d::RigidBody3D>(**e).is_none()
                })
                .copied()
                .collect()
        };

        // Remove stale bodies from rapier and clean mappings
        if !stale.is_empty() {
            // Also find joints that reference stale bodies
            let stale_joints: Vec<crate::Entity> = {
                let physics = world.resource::<PhysicsWorld3D>();
                physics
                    .entity_to_joint
                    .keys()
                    .filter(|je| {
                        if let Some(joint_desc) =
                            world.get::<super::components3d::ImpulseJoint3D>(**je)
                        {
                            stale.contains(&joint_desc.body1) || stale.contains(&joint_desc.body2)
                        } else {
                            false
                        }
                    })
                    .copied()
                    .collect()
            };

            {
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
                for entity in &stale_joints {
                    if let Some(jh) = physics.entity_to_joint.remove(entity) {
                        physics.remove_impulse_joint(jh, true);
                    }
                }
                for entity in &stale {
                    if let Some(bh) = physics.entity_to_body.remove(entity) {
                        physics.body_to_entity.remove(&bh);
                        physics.remove_body(bh);
                    }
                }
            }

            for entity in &stale_joints {
                if world.is_alive(*entity) {
                    world.remove::<ImpulseJoint3DHandle>(*entity);
                }
            }
            for entity in &stale {
                if world.is_alive(*entity) {
                    world.remove::<RigidBody3DHandle>(*entity);
                }
            }
        }

        // Phase 2: Find new bodies (have descriptors, not in mapping, not disabled)
        let new_entities: Vec<(
            crate::Entity,
            super::components3d::RigidBody3D,
            super::components3d::Collider3D,
            crate::Transform,
        )> = {
            let physics = world.resource::<PhysicsWorld3D>();
            world
                .iter_entities()
                .filter(|e| !physics.entity_to_body.contains_key(e) && !world.is_disabled(*e))
                .filter_map(|entity| {
                    let body = world
                        .get::<super::components3d::RigidBody3D>(entity)?
                        .clone();
                    let collider = world
                        .get::<super::components3d::Collider3D>(entity)?
                        .clone();
                    let transform = *world.get::<crate::Transform>(entity)?;
                    Some((entity, body, collider, transform))
                })
                .collect()
        };

        if !new_entities.is_empty() {
            let mut handles = Vec::with_capacity(new_entities.len());
            {
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
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
                let _ = world.insert(entity, RigidBody3DHandle(handle));
            }
        }

        Ok(())
    }
}

// ---- SyncPhysicsJoints3D exclusive system ----

/// Exclusive system that creates/removes rapier joints from ECS descriptor components.
///
/// Detects entities with [`ImpulseJoint3D`](super::components3d::ImpulseJoint3D)
/// and creates corresponding rapier joints. Also detects removed/despawned joints.
///
/// Must run after [`SyncPhysicsBodies3D`] so that body handles are available.
pub struct SyncPhysicsJoints3D;

impl crate::ExclusiveSystem for SyncPhysicsJoints3D {
    type Result = ();

    fn run(&mut self, world: &mut crate::World) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_joints_3d");

        if !world.has_resource::<PhysicsWorld3D>() {
            return Ok(());
        }

        // Phase 1: Find stale joints (entity dead, disabled, or lost ImpulseJoint3D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld3D>();
            physics
                .entity_to_joint
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_disabled(**e)
                        || world
                            .get::<super::components3d::ImpulseJoint3D>(**e)
                            .is_none()
                })
                .copied()
                .collect()
        };

        if !stale.is_empty() {
            {
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
                for entity in &stale {
                    if let Some(jh) = physics.entity_to_joint.remove(entity) {
                        physics.remove_impulse_joint(jh, true);
                    }
                }
            }
            for entity in &stale {
                if world.is_alive(*entity) {
                    world.remove::<ImpulseJoint3DHandle>(*entity);
                }
            }
        }

        // Phase 2: Find new joints (not in mapping, not disabled)
        let new_joints: Vec<(crate::Entity, super::components3d::ImpulseJoint3D)> = {
            let physics = world.resource::<PhysicsWorld3D>();
            world
                .iter_entities()
                .filter(|e| !physics.entity_to_joint.contains_key(e) && !world.is_disabled(*e))
                .filter_map(|entity| {
                    let joint = world
                        .get::<super::components3d::ImpulseJoint3D>(entity)?
                        .clone();
                    Some((entity, joint))
                })
                .collect()
        };

        if !new_joints.is_empty() {
            let mut handles = Vec::new();
            {
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
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
                let _ = world.insert(entity, ImpulseJoint3DHandle(handle));
            }
        }

        Ok(())
    }
}

// ---- Regular system variants ----

/// Regular system variant of [`SyncPhysicsBodies3D`].
///
/// Uses lock-execute + deferred commands instead of exclusive world access.
/// Allows parallel scheduling but joints may lag 1 frame behind body creation.
pub struct SyncPhysicsBodiesSystem3D;

impl crate::System for SyncPhysicsBodiesSystem3D {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_bodies_system_3d");

        let (new_indices, stale_entities) = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld3D>,
                crate::Read<super::components3d::RigidBody3D>,
                crate::Read<super::components3d::Collider3D>,
                crate::Read<crate::Transform>,
            )>()
            .execute(|(mut physics, bodies, colliders, transforms)| {
                // Remove stale
                let stale: Vec<crate::Entity> = physics
                    .entity_to_body
                    .keys()
                    .filter(|e| bodies.get(e.index()).is_none())
                    .copied()
                    .collect();
                for entity in &stale {
                    if let Some(bh) = physics.entity_to_body.remove(entity) {
                        physics.body_to_entity.remove(&bh);
                        physics.remove_body(bh);
                    }
                }

                // Find tracked indices
                let tracked: std::collections::HashSet<u32> =
                    physics.entity_to_body.keys().map(|e| e.index()).collect();

                // Create new
                let mut new_pairs: Vec<(u32, RigidBodyHandle)> = Vec::new();
                for (idx, body_desc) in bodies.iter() {
                    if !tracked.contains(&idx)
                        && let (Some(collider_desc), Some(transform)) =
                            (colliders.get(idx), transforms.get(idx))
                    {
                        let rapier_body = body_desc.to_rigid_body(transform);
                        let body_handle = physics.add_body(rapier_body);
                        let rapier_collider = collider_desc.to_collider();
                        physics.add_collider(rapier_collider, body_handle);
                        new_pairs.push((idx, body_handle));
                    }
                }

                (new_pairs, stale)
            });

        if !new_indices.is_empty() || !stale_entities.is_empty() {
            ctx.commands(move |world| {
                // Remove handle components for stale entities
                for entity in stale_entities {
                    if world.is_alive(entity) {
                        world.remove::<RigidBody3DHandle>(entity);
                    }
                }
                // Insert handles and update mapping for new bodies
                for (idx, handle) in new_indices {
                    if let Some(entity) = world.entity_at_index(idx) {
                        let _ = world.insert(entity, RigidBody3DHandle(handle));
                        let mut physics = world.resource_mut::<PhysicsWorld3D>();
                        physics.entity_to_body.insert(entity, handle);
                        physics.body_to_entity.insert(handle, entity);
                    }
                }
            });
        }

        Ok(())
    }
}

/// Regular system variant of [`SyncPhysicsJoints3D`].
///
/// Uses lock-execute + deferred commands. Joint creation may lag 1 frame behind
/// body creation when both are spawned in the same frame.
pub struct SyncPhysicsJointsSystem3D;

impl crate::System for SyncPhysicsJointsSystem3D {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_joints_system_3d");

        let (new_indices, stale_entities) = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld3D>,
                crate::Read<super::components3d::ImpulseJoint3D>,
            )>()
            .execute(|(mut physics, joints)| {
                // Remove stale
                let stale: Vec<crate::Entity> = physics
                    .entity_to_joint
                    .keys()
                    .filter(|e| joints.get(e.index()).is_none())
                    .copied()
                    .collect();
                for entity in &stale {
                    if let Some(jh) = physics.entity_to_joint.remove(entity) {
                        physics.remove_impulse_joint(jh, true);
                    }
                }

                // Find tracked
                let tracked: std::collections::HashSet<u32> =
                    physics.entity_to_joint.keys().map(|e| e.index()).collect();

                // Create new
                let mut new_pairs: Vec<(u32, ImpulseJointHandle)> = Vec::new();
                for (idx, joint_desc) in joints.iter() {
                    if !tracked.contains(&idx) {
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
                        new_pairs.push((idx, jh));
                    }
                }

                (new_pairs, stale)
            });

        if !new_indices.is_empty() || !stale_entities.is_empty() {
            ctx.commands(move |world| {
                for entity in stale_entities {
                    if world.is_alive(entity) {
                        world.remove::<ImpulseJoint3DHandle>(entity);
                    }
                }
                for (idx, handle) in new_indices {
                    if let Some(entity) = world.entity_at_index(idx) {
                        let _ = world.insert(entity, ImpulseJoint3DHandle(handle));
                        let mut physics = world.resource_mut::<PhysicsWorld3D>();
                        physics.entity_to_joint.insert(entity, handle);
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
    fn sync_bodies_creates_and_removes() {
        use crate::system::run_exclusive_system_once;
        use redlilium_core::math::Vec3;

        let mut world = crate::World::new();
        crate::register_std_components(&mut world);

        // Spawn a dynamic ball
        let e = world.spawn();
        let _ = world.insert(e, super::super::components3d::RigidBody3D::dynamic());
        let _ = world.insert(e, super::super::components3d::Collider3D::ball(0.5));
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        );

        // Run sync
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();

        // Should have handle
        assert!(world.get::<RigidBody3DHandle>(e).is_some());
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 1);
            assert!(physics.entity_to_body.contains_key(&e));
        }

        // Now remove the descriptor
        world.remove::<super::super::components3d::RigidBody3D>(e);

        // Run sync again
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();

        // Should be cleaned up
        assert!(world.get::<RigidBody3DHandle>(e).is_none());
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 0);
            assert!(!physics.entity_to_body.contains_key(&e));
        }
    }

    #[test]
    fn sync_bodies_handles_disabled_entities() {
        use crate::system::run_exclusive_system_once;
        use redlilium_core::math::Vec3;

        let mut world = crate::World::new();
        crate::register_std_components(&mut world);

        // Spawn a dynamic ball
        let e = world.spawn();
        let _ = world.insert(e, super::super::components3d::RigidBody3D::dynamic());
        let _ = world.insert(e, super::super::components3d::Collider3D::ball(0.5));
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        );

        // Run sync — body should be created
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();
        assert!(world.get::<RigidBody3DHandle>(e).is_some());
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 1);
        }

        // Disable the entity
        world.set_entity_flags(e, crate::Entity::DISABLED);

        // Run sync — body should be removed from rapier
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 0);
            assert!(!physics.entity_to_body.contains_key(&e));
        }

        // Re-enable the entity
        world.clear_entity_flags(e, crate::Entity::DISABLED);

        // Run sync — body should be re-created
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();
        assert!(world.get::<RigidBody3DHandle>(e).is_some());
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 1);
            assert!(physics.entity_to_body.contains_key(&e));
        }
    }
}
