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
/// Publishes collision transitions and contact-force events after the step when each Events queue
/// is registered. Enabled tracking without that queue returns InvalidConfiguration
/// before simulation advances. See [`super::events2d::CollisionEvent2D`].
pub struct StepPhysics2D;

impl crate::System for StepPhysics2D {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a crate::SystemContext<'a>) -> Result<(), crate::SystemError> {
        use super::control2d::{KinematicTarget2D, KinematicVelocity2D, PhysicsPose2D};
        use super::events2d::{CollisionEvent2D, ContactForceEvent2D};
        let events_available = ctx
            .raw_world()
            .has_resource::<crate::Events<CollisionEvent2D>>();
        let force_events_available = ctx
            .raw_world()
            .has_resource::<crate::Events<ContactForceEvent2D>>();
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
            crate::Read<crate::Parent>,
            crate::Write<KinematicTarget2D>,
            crate::Write<KinematicVelocity2D>,
        )>()
        .execute(
            |(mut physics, handles, mut transforms, parents, mut targets, mut velocities)| {
                if !physics.mass_dirty.is_empty() {
                    return Err(crate::SystemError::InvalidConfiguration {
                        message: "physics mass_properties lost required collider geometry during deferred publication; correct descriptors and run body sync before stepping".into(),
                    });
                }
                if physics.collision_events.requires_queue() && !events_available {
                    return Err(crate::SystemError::InvalidConfiguration {
                        message:
                            "collision tracking requires world.add_event::<CollisionEvent2D>()"
                                .into(),
                    });
                }
                if physics.collision_events.forces.requires_queue() && !force_events_available {
                    return Err(crate::SystemError::InvalidConfiguration {
                        message: "contact force tracking requires world.add_event::<ContactForceEvent2D>()".into(),
                    });
                }
                // Validate the whole batch before changing motion or advancing time.
                for (idx, handle) in handles.iter() {
                    let Some(entity) = ctx.raw_world().entity_at_index(idx) else {
                        continue;
                    };
                    let Some(t) = transforms.get(idx) else {
                        continue;
                    };
                    super::validation::transform(entity, t, parents.get(idx).is_some())?;
                    if let Some(body) = physics.bodies.get(handle.0) {
                        if (body.body_type() == RigidBodyType::KinematicPositionBased)
                            && let Some(target) = targets.get(idx)
                        {
                            PhysicsPose2D::from(*target).validate(entity)?;
                        }
                        if (body.body_type() == RigidBodyType::KinematicVelocityBased)
                            && let Some(v) = velocities.get(idx)
                            && (!v.linear.iter().all(|x| x.is_finite()) || !(v.angular.is_finite()))
                        {
                            return Err(super::validation::invalid(
                                entity,
                                "kinematic velocity must be finite",
                            ));
                        }
                    }
                }
                for (idx, handle) in handles.iter() {
                    let Some(t) = transforms.get(idx) else {
                        continue;
                    };
                    let Some(body) = physics.bodies.get_mut(handle.0) else {
                        continue;
                    };
                    if body.is_fixed() {
                        let pose = PhysicsPose2D::from_transform(t).to_rapier();
                        if *body.position() != pose {
                            body.set_position(pose, true);
                        }
                    } else if body.body_type() == RigidBodyType::KinematicPositionBased {
                        let pose = targets
                            .get(idx)
                            .map(|t| PhysicsPose2D::from(*t).to_rapier())
                            .unwrap_or(*body.position());
                        body.set_next_kinematic_position(pose);
                    } else if body.body_type() == RigidBodyType::KinematicVelocityBased {
                        let v = velocities.get(idx).copied().unwrap_or_default();
                        body.set_linvel(Vector::new(v.linear.x as Real, v.linear.y as Real), true);
                        body.set_angvel(v.angular as Real, true);
                    }
                }
                let teleports = std::mem::take(&mut physics.teleports);
                for (entity, (handle, pose, velocity)) in teleports {
                    if !ctx.is_alive(entity)
                        || ctx.is_excluded_from_game(entity)
                        || physics.entity_to_body.get(&entity) != Some(&handle)
                    {
                        continue;
                    }
                    let Some(body) = physics.bodies.get_mut(handle) else {
                        continue;
                    };
                    body.set_position(pose.to_rapier(), true);
                    body.set_next_kinematic_position(pose.to_rapier());
                    if velocity == super::TeleportVelocity::Reset {
                        body.set_linvel(Vector::ZERO, true);
                        body.set_angvel(0.0, true);
                        if let Some(mut input) = velocities.get_mut(entity.index()) {
                            *input = Default::default();
                        }
                    }
                    if let Some(mut target) = targets.get_mut(entity.index()) {
                        *target = pose.into();
                    }
                    if let Some(mut transform) = transforms.get_mut(entity.index()) {
                        transform.translation.x = pose.translation.x;
                        transform.translation.y = pose.translation.y;
                        transform.rotation =
                            redlilium_core::math::quat_from_rotation_z(pose.rotation);
                    }
                }
                if let Some(dt) = fixed_dt {
                    physics.integration_parameters.dt = dt as Real;
                }
                physics.step();
                for (idx, handle) in handles.iter() {
                    if let Some(body) = physics.bodies.get(handle.0)
                        && (body.is_dynamic() || body.is_kinematic())
                        && let Some(mut transform) = transforms.get_mut(idx)
                    {
                        let pose = PhysicsPose2D::from_rapier(body.position());
                        transform.translation.x = pose.translation.x;
                        transform.translation.y = pose.translation.y;
                        transform.rotation =
                            redlilium_core::math::quat_from_rotation_z(pose.rotation);
                    }
                }
                Ok(())
            },
        )?;
        if events_available {
            ctx.lock::<(
                crate::ResMut<PhysicsWorld2D>,
                crate::ResMut<crate::Events<CollisionEvent2D>>,
            )>()
            .execute(|(mut physics, mut events)| {
                for event in physics.collision_events.pending.drain(..) {
                    events.send(event);
                }
            });
        }
        if force_events_available {
            ctx.lock::<(
                crate::ResMut<PhysicsWorld2D>,
                crate::ResMut<crate::Events<ContactForceEvent2D>>,
            )>()
            .execute(|(mut physics, mut events)| {
                for event in physics.collision_events.forces.pending.drain(..) {
                    events.send(event);
                }
            });
        }
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
    if !stale.is_empty() {
        physics
            .applied_joints
            .retain(|handle, _| physics.impulse_joints.contains(*handle));
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

use super::components2d::{Collider2D, ColliderBody2D, RigidBody2D};
use super::world2d::Collider2DHandle;
super::mass_support::mass_sync!(PhysicsWorld2D, RigidBody2D, Collider2D, ColliderBody2D);
super::sync_support::collider_sync!(
    PhysicsWorld2D,
    RigidBody2D,
    Collider2D,
    ColliderBody2D,
    RigidBody2DHandle,
    Collider2DHandle,
    SyncPhysicsBodies2D,
    SyncPhysicsBodiesSystem2D
);

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

        for entity in world
            .iter_entities()
            .filter(|e| !world.is_excluded_from_game(*e))
        {
            if let Some(joint) = world.get::<super::components2d::ImpulseJoint2D>(entity) {
                joint.validate(entity)?;
            }
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
                .filter(|e| !world.is_excluded_from_game(*e))
                .filter_map(|entity| {
                    let joint = world.get::<super::components2d::ImpulseJoint2D>(entity)?;
                    if physics
                        .entity_to_joint
                        .get(&entity)
                        .and_then(|h| physics.applied_joints.get(h))
                        == Some(joint)
                    {
                        return None;
                    }
                    Some((entity, joint.clone()))
                })
                .collect()
        };

        if !new_joints.is_empty() {
            let mut handles = Vec::new();
            {
                let mut physics = world.resource_mut::<PhysicsWorld2D>();
                for (entity, joint_desc) in &new_joints {
                    if let Some(handle) = physics.entity_to_joint.get(entity).copied() {
                        if physics.applied_joints.get(&handle) == Some(joint_desc) {
                            continue;
                        }
                        physics.entity_to_joint.remove(entity);
                        physics.remove_impulse_joint(handle, true);
                        handles.push((*entity, None));
                    }
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
                    physics.applied_joints.insert(jh, joint_desc.clone());
                    handles.push((*entity, Some(jh)));
                }
            }
            for (entity, handle) in handles {
                if let Some(handle) = handle {
                    let _ = world.insert(entity, ImpulseJoint2DHandle(handle));
                } else {
                    let _ = world.remove::<ImpulseJoint2DHandle>(entity);
                }
            }
        }

        Ok(())
    }
}

// ---- Regular system variants ----

/// Regular system variant of [`SyncPhysicsJoints2D`].
///
/// Uses lock-execute + deferred commands. Run after body sync; native body
/// mappings are available even before ECS handles are published.
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
                for (idx, joint) in joints.iter() {
                    if let Some(entity) = ctx.raw_world().entity_at_index(idx) {
                        joint.validate(entity)?;
                    }
                }
                // Remove stale: entity dead (full-identity check), disabled, or
                // lost the ImpulseJoint2D component.
                let mut stale: Vec<crate::Entity> = physics
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
                    if let Some(entity) = ctx.raw_world().entity_at_index(idx) {
                        if let Some(handle) = physics.entity_to_joint.get(&entity).copied() {
                            if physics.applied_joints.get(&handle) == Some(joint_desc) {
                                continue;
                            }
                            physics.entity_to_joint.remove(&entity);
                            physics.remove_impulse_joint(handle, true);
                            stale.push(entity);
                        }
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
                        physics.applied_joints.insert(jh, joint_desc.clone());
                        new_pairs.push((entity, jh));
                    }
                }

                Ok::<_, crate::SystemError>((new_pairs, stale))
            })?;

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
