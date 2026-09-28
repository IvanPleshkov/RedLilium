//! 3D physics ECS systems.
//!
//! Systems that step the 3D physics simulation, sync transforms,
//! and manage rigid body / joint creation and removal.

use super::rapier3d::prelude::*;
use super::world3d::{
    ImpulseJoint3DHandle, PhysicsInterpolation, PhysicsWorld3D, RigidBody3DHandle,
};
// The engine's `Quat` is f32 while this rapier build is f64, so name the f32
// `UnitQuaternion` explicitly — the rapier prelude glob above would otherwise
// supply the f64 one.
use redlilium_core::math::nalgebra::UnitQuaternion;

// ---- StepPhysics3D system ----

/// ECS system that steps the 3D physics simulation and syncs body positions
/// back to ECS [`Transform`](crate::Transform) components.
///
/// Requires a [`PhysicsWorld3D`] resource and bodies created by body sync.
/// Reads native ownership mappings, so regular sync may run immediately before
/// this system without a deferred-command flush. Transforms receive stepped poses.
/// Publishes collision transitions and contact-force events after the step when each Events queue
/// is registered. Enabled tracking without that queue returns InvalidConfiguration
/// before simulation advances. See [`super::events3d::CollisionEvent3D`].
pub struct StepPhysics3D;

impl crate::System for StepPhysics3D {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a crate::SystemContext<'a>) -> Result<(), crate::SystemError> {
        use super::control3d::{KinematicTarget3D, KinematicVelocity3D, PhysicsPose3D};
        use super::events3d::{CollisionEvent3D, ContactForceEvent3D};
        let events_available = ctx
            .raw_world()
            .has_resource::<crate::Events<CollisionEvent3D>>();
        let force_events_available = ctx
            .raw_world()
            .has_resource::<crate::Events<ContactForceEvent3D>>();
        let fixed_dt = {
            let world = ctx.raw_world();
            world
                .has_resource::<crate::Time>()
                .then(|| world.resource::<crate::Time>().fixed_delta())
        };
        ctx.lock::<(
            crate::ResMut<PhysicsWorld3D>,
            crate::Write<crate::Transform>,
            crate::Read<crate::Parent>,
            crate::Write<KinematicTarget3D>,
            crate::Write<KinematicVelocity3D>,
        )>()
        .execute(
            |(mut physics, mut transforms, parents, mut targets, mut velocities)| {
                let physics = &mut *physics;
                if !physics.mass_dirty.is_empty() {
                    return Err(crate::SystemError::InvalidConfiguration {
                        message: "physics mass_properties lost required collider geometry during deferred publication; correct descriptors and run body sync before stepping".into(),
                    });
                }
                if physics.collision_events.requires_queue() && !events_available {
                    return Err(crate::SystemError::InvalidConfiguration {
                        message:
                            "collision tracking requires world.add_event::<CollisionEvent3D>()"
                                .into(),
                    });
                }
                if physics.collision_events.forces.requires_queue() && !force_events_available {
                    return Err(crate::SystemError::InvalidConfiguration {
                        message: "contact force tracking requires world.add_event::<ContactForceEvent3D>()".into(),
                    });
                }
                // Validate the whole batch before changing motion or advancing time.
                for (&entity, &handle) in &physics.entity_to_body {
                    if !ctx.is_alive(entity) || ctx.is_excluded_from_game(entity) {
                        continue;
                    }
                    let idx = entity.index();
                    let Some(t) = transforms.get(idx) else {
                        continue;
                    };
                    super::validation::transform(entity, t, parents.get(idx).is_some())?;
                    if let Some(body) = physics.bodies.get(handle) {
                        if (body.body_type() == RigidBodyType::KinematicPositionBased)
                            && let Some(target) = targets.get(idx)
                        {
                            PhysicsPose3D::from(*target).validate(entity)?;
                        }
                        if (body.body_type() == RigidBodyType::KinematicVelocityBased)
                            && let Some(v) = velocities.get(idx)
                            && (!v.linear.iter().all(|x| x.is_finite())
                                || !(v.angular.iter().all(|x| x.is_finite())))
                        {
                            return Err(super::validation::invalid(
                                entity,
                                "kinematic velocity must be finite",
                            ));
                        }
                    }
                }
                for (&entity, &handle) in &physics.entity_to_body {
                    if !ctx.is_alive(entity) || ctx.is_excluded_from_game(entity) {
                        continue;
                    }
                    let idx = entity.index();
                    let Some(t) = transforms.get(idx) else {
                        continue;
                    };
                    let Some(body) = physics.bodies.get_mut(handle) else {
                        continue;
                    };
                    if body.is_fixed() {
                        let pose = PhysicsPose3D::from_transform(t).to_rapier();
                        if *body.position() != pose {
                            body.set_position(pose, true);
                        }
                    } else if body.body_type() == RigidBodyType::KinematicPositionBased {
                        let pose = targets
                            .get(idx)
                            .map(|t| PhysicsPose3D::from(*t).to_rapier())
                            .unwrap_or(*body.position());
                        body.set_next_kinematic_position(pose);
                    } else if body.body_type() == RigidBodyType::KinematicVelocityBased {
                        let v = velocities.get(idx).copied().unwrap_or_default();
                        body.set_linvel(
                            Vector::new(v.linear.x as Real, v.linear.y as Real, v.linear.z as Real),
                            true,
                        );
                        body.set_angvel(
                            Vector::new(
                                v.angular.x as Real,
                                v.angular.y as Real,
                                v.angular.z as Real,
                            ),
                            true,
                        );
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
                        body.set_angvel(Vector::ZERO, true);
                        if let Some(mut input) = velocities.get_mut(entity.index()) {
                            *input = Default::default();
                        }
                    }
                    if let Some(mut target) = targets.get_mut(entity.index()) {
                        *target = pose.into();
                    }
                    if let Some(mut transform) = transforms.get_mut(entity.index()) {
                        transform.translation = pose.translation;
                        transform.rotation = pose.rotation;
                    }
                    physics.pose_resets.insert(handle);
                }
                if let Some(dt) = fixed_dt {
                    physics.integration_parameters.dt = dt as Real;
                }
                physics.step();
                for (&entity, &handle) in &physics.entity_to_body {
                    if !ctx.is_alive(entity) || ctx.is_excluded_from_game(entity) {
                        continue;
                    }
                    let idx = entity.index();
                    if let Some(body) = physics.bodies.get(handle)
                        && (body.is_dynamic() || body.is_kinematic())
                        && let Some(mut transform) = transforms.get_mut(idx)
                    {
                        let pose = PhysicsPose3D::from_rapier(body.position());
                        transform.translation = pose.translation;
                        transform.rotation = pose.rotation;
                    }
                }
                Ok(())
            },
        )?;
        if events_available {
            ctx.lock::<(
                crate::ResMut<PhysicsWorld3D>,
                crate::ResMut<crate::Events<CollisionEvent3D>>,
            )>()
            .execute(|(mut physics, mut events)| {
                for event in physics.collision_events.pending.drain(..) {
                    events.send(event);
                }
            });
        }
        if force_events_available {
            ctx.lock::<(
                crate::ResMut<PhysicsWorld3D>,
                crate::ResMut<crate::Events<ContactForceEvent3D>>,
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

// ---- Fixed-step pose history + render interpolation ----

/// Records each body's authoritative fixed-step pose into its
/// [`PhysicsInterpolation`] history. Runs in `FixedUpdate` **after**
/// [`StepPhysics3D`]. Poses are read directly from Rapier, so presentation
/// changes cannot enter the history. It executes once per fixed step, including the
/// extra iterations of a catch-up frame, leaving the two most recent steps in
/// `prev`/`cur`.
///
/// Bodies without the component are seeded with `prev == cur`, so a freshly
/// spawned body renders at its first recorded pose. Native ownership mappings
/// include bodies created by regular sync before ECS handles are published.
pub struct RecordPhysicsPose;

impl crate::System for RecordPhysicsPose {
    type Result = ();
    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        let to_seed = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld3D>,
                crate::WriteAll<PhysicsInterpolation>,
            )>()
            .execute(|(mut physics, mut interps)| {
                let physics = &mut *physics;
                redlilium_core::profile_scope!("ecs: record_physics_pose_3d");
                let mut seed = Vec::new();
                for (&entity, &handle) in &physics.entity_to_body {
                    if !ctx.is_alive(entity) || ctx.is_excluded_from_game(entity) {
                        continue;
                    }
                    let idx = entity.index();
                    let Some(body) = physics.bodies.get(handle) else {
                        continue;
                    };
                    if body.is_fixed() {
                        continue;
                    }
                    let pose = super::control3d::PhysicsPose3D::from_rapier(body.position());
                    let reset = physics.pose_resets.remove(&handle);
                    if let Some(mut interp) = interps.get_mut(idx) {
                        interp.prev_translation = if reset {
                            pose.translation
                        } else {
                            interp.cur_translation
                        };
                        interp.prev_rotation = if reset {
                            pose.rotation
                        } else {
                            interp.cur_rotation
                        };
                        interp.cur_translation = pose.translation;
                        interp.cur_rotation = pose.rotation;
                    } else {
                        seed.push((entity, handle, pose.translation, pose.rotation));
                    }
                }
                seed
            });

        if !to_seed.is_empty() {
            ctx.commands(move |world| {
                for (entity, handle, translation, rotation) in to_seed {
                    if world.is_alive(entity)
                        && !world.is_excluded_from_game(entity)
                        && world.resource::<PhysicsWorld3D>().body_for_entity(entity)
                            == Some(handle)
                    {
                        let _ = world.insert(
                            entity,
                            PhysicsInterpolation {
                                prev_translation: translation,
                                prev_rotation: rotation,
                                cur_translation: translation,
                                cur_rotation: rotation,
                            },
                        );
                    }
                }
            });
        }
        Ok(())
    }
}

/// Blends each body's two most recent fixed-step poses into `Transform` for
/// rendering, by the frame's [`Time::fixed_alpha`](crate::Time::fixed_alpha).
///
/// Runs in `PostUpdate` **before** transform propagation, so the interpolated
/// pose is what `GlobalTransform` and rendering see. Only dynamic and kinematic
/// bodies are interpolated; their Transform is output only. Fixed bodies read
/// Transform as input before each step and are never interpolated.
pub struct InterpolatePhysics;

impl crate::System for InterpolatePhysics {
    type Result = ();
    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        // Worlds ticked without `run_frame` carry no `Time`; there is no
        // accumulator to blend against, so show the latest step.
        let alpha = {
            let world = ctx.raw_world();
            let banked = if world.has_resource::<crate::Time>() {
                world.resource::<crate::Time>().fixed_alpha() as f32
            } else {
                1.0
            };
            banked.clamp(0.0, 1.0)
        };

        ctx.lock::<(
            crate::Read<PhysicsInterpolation>,
            crate::Read<RigidBody3DHandle>,
            crate::Res<PhysicsWorld3D>,
            crate::WriteAll<crate::Transform>,
        )>()
        .execute(|(interps, handles, physics, mut transforms)| {
            redlilium_core::profile_scope!("ecs: interpolate_physics_3d");
            for (idx, interp) in interps.iter() {
                if !handles
                    .get(idx)
                    .and_then(|h| physics.bodies.get(h.0))
                    .is_some_and(|body| !body.is_fixed())
                {
                    continue;
                }
                let Some(mut transform) = transforms.get_mut(idx) else {
                    continue;
                };
                transform.translation =
                    interp.prev_translation.lerp(&interp.cur_translation, alpha);
                // Normalize before slerping: the recorded quaternions come from
                // rapier and drift is cheap to absorb here. Antipodal pairs
                // cannot arise between consecutive steps, but fall back to the
                // latest pose rather than panicking if they somehow do.
                let prev = UnitQuaternion::new_normalize(interp.prev_rotation);
                let cur = UnitQuaternion::new_normalize(interp.cur_rotation);
                let blended = prev.try_slerp(&cur, alpha, 1e-6).unwrap_or(cur);
                transform.rotation = *blended.quaternion();
            }
        });
        Ok(())
    }
}

// Remove through Rapier first, then reconcile joints against the actual set.
// Descriptor endpoints may already have changed, so they are not a reliable
// source for discovering which live joints the removed bodies owned.
fn remove_bodies(physics: &mut PhysicsWorld3D, stale: &[crate::Entity]) -> Vec<crate::Entity> {
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
            let _ = world.remove::<ImpulseJoint3DHandle>(entity);
        }
    }
    for &entity in bodies {
        if world.is_alive(entity) {
            let _ = world.remove::<RigidBody3DHandle>(entity);
            let _ = world.remove::<PhysicsInterpolation>(entity);
        }
    }
}

use super::components3d::{Collider3D, ColliderBody3D, RigidBody3D};
use super::world3d::Collider3DHandle;
super::mass_support::mass_sync!(PhysicsWorld3D, RigidBody3D, Collider3D, ColliderBody3D);
super::sync_support::collider_sync!(
    PhysicsWorld3D,
    RigidBody3D,
    Collider3D,
    ColliderBody3D,
    RigidBody3DHandle,
    Collider3DHandle,
    SyncPhysicsBodies3D,
    SyncPhysicsBodiesSystem3D
);

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

        for entity in world
            .iter_entities()
            .filter(|e| !world.is_excluded_from_game(*e))
        {
            if let Some(joint) = world.get::<super::components3d::ImpulseJoint3D>(entity) {
                joint.validate(entity)?;
            }
        }

        // Phase 1: Find stale joints (entity dead, excluded from game, or lost ImpulseJoint3D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld3D>();
            physics
                .entity_to_joint
                .iter()
                .filter(|(e, handle)| {
                    !physics.impulse_joints.contains(**handle)
                        || !world.is_alive(**e)
                        || world.is_excluded_from_game(**e)
                        || world
                            .get::<super::components3d::ImpulseJoint3D>(**e)
                            .is_none()
                })
                .map(|(entity, _)| *entity)
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
                    let _ = world.remove::<ImpulseJoint3DHandle>(*entity);
                }
            }
        }

        // Phase 2: Find new joints (new or changed descriptors, not excluded from game)
        let new_joints: Vec<(crate::Entity, super::components3d::ImpulseJoint3D)> = {
            let physics = world.resource::<PhysicsWorld3D>();
            world
                .iter_entities()
                .filter(|e| !world.is_excluded_from_game(*e))
                .filter_map(|entity| {
                    let joint = world.get::<super::components3d::ImpulseJoint3D>(entity)?;
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
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
                for (entity, joint_desc) in &new_joints {
                    if let Some(handle) = physics.entity_to_joint.get(entity).copied() {
                        if physics.update_joint_parameters(handle, joint_desc) {
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
                    let _ = world.insert(entity, ImpulseJoint3DHandle(handle));
                } else {
                    let _ = world.remove::<ImpulseJoint3DHandle>(entity);
                }
            }
        }

        Ok(())
    }
}

// ---- Regular system variants ----

/// Regular system variant of [`SyncPhysicsJoints3D`].
///
/// Uses lock-execute + deferred commands. Run after body sync; native body
/// mappings are available even before ECS handles are published.
pub struct SyncPhysicsJointsSystem3D;

impl crate::System for SyncPhysicsJointsSystem3D {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_joints_system_3d");

        let (new_entities, stale_entities) = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld3D>,
                crate::Read<super::components3d::ImpulseJoint3D>,
            )>()
            .execute(|(mut physics, joints)| {
                for (idx, joint) in joints.iter() {
                    if let Some(entity) = ctx.raw_world().entity_at_index(idx) {
                        joint.validate(entity)?;
                    }
                }
                // Remove stale: entity dead (full-identity check), disabled, or
                // lost the ImpulseJoint3D component.
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
                            if physics.update_joint_parameters(handle, joint_desc) {
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
                        physics.entity_to_joint.insert(entity, jh);
                        new_pairs.push((entity, jh));
                    }
                }

                Ok::<_, crate::SystemError>((new_pairs, stale))
            })?;

        if !new_entities.is_empty() || !stale_entities.is_empty() {
            ctx.commands(move |world| {
                for entity in stale_entities {
                    if world.is_alive(entity) {
                        let _ = world.remove::<ImpulseJoint3DHandle>(entity);
                    }
                }
                for (entity, handle) in new_entities {
                    if world.resource::<PhysicsWorld3D>().joint_for_entity(entity) != Some(handle) {
                        continue;
                    }
                    let valid = world.is_alive(entity)
                        && !world.is_excluded_from_game(entity)
                        && world
                            .get::<super::components3d::ImpulseJoint3D>(entity)
                            .is_some()
                        && world
                            .resource::<PhysicsWorld3D>()
                            .impulse_joints
                            .contains(handle);
                    if valid {
                        let _ = world.insert(entity, ImpulseJoint3DHandle(handle));
                    } else {
                        let mut physics = world.resource_mut::<PhysicsWorld3D>();
                        physics.entity_to_joint.remove(&entity);
                        physics.remove_impulse_joint(handle, true);
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
        let _ = world.remove::<super::super::components3d::RigidBody3D>(e);

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

    /// The pose history seeds itself on a body's first fixed step (`prev ==
    /// cur`, so nothing lerps in from a bogus origin) and thereafter shifts by
    /// exactly one step per run.
    #[test]
    fn record_physics_pose_seeds_then_shifts() {
        use crate::compute::{ComputePool, IoRuntime};
        use crate::system::{run_exclusive_system_once, run_system_once};
        use redlilium_core::math::Vec3;

        let mut world = crate::World::new();
        crate::register_std_components(&mut world);
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        let e = world.spawn();
        let _ = world.insert(e, super::super::components3d::RigidBody3D::dynamic());
        let _ = world.insert(e, super::super::components3d::Collider3D::ball(0.5));
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
        );
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();

        // First record seeds the history at the body's current pose.
        run_system_once(&RecordPhysicsPose, &mut world, &compute, &io).unwrap();
        {
            let interp = world.get::<PhysicsInterpolation>(e).expect("seeded");
            assert_eq!(
                interp.prev_translation, interp.cur_translation,
                "a fresh body must not interpolate from anywhere"
            );
            assert_eq!(interp.cur_translation.y, 1.0);
        }

        // A later step shifts cur -> prev and records the new pose.
        {
            let mut physics = world.resource_mut::<PhysicsWorld3D>();
            let handle = physics.entity_to_body[&e];
            physics.bodies[handle].set_translation(Vector::new(0.0, 2.0, 0.0), true);
        }
        run_system_once(&RecordPhysicsPose, &mut world, &compute, &io).unwrap();
        let interp = world.get::<PhysicsInterpolation>(e).expect("recorded");
        assert_eq!(interp.prev_translation.y, 1.0, "previous step retained");
        assert_eq!(interp.cur_translation.y, 2.0, "latest step recorded");
    }

    /// A render frame landing between two fixed steps shows the blend, not the
    /// latest step — this is what removes the staircase when the render rate
    /// and the fixed physics rate disagree.
    #[test]
    fn interpolate_physics_blends_fixed_step_poses() {
        use crate::{EcsRunner, PostUpdate, Schedules, Transform};
        use redlilium_core::math::{Quat, Vec3};

        let mut world = crate::World::new();
        crate::register_std_components(&mut world);

        let e = world.spawn();
        world.insert(e, Transform::default()).unwrap();
        world
            .insert(e, super::super::components3d::RigidBody3D::dynamic())
            .unwrap();
        world
            .insert(e, super::super::components3d::Collider3D::ball(0.5))
            .unwrap();
        crate::system::run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();
        world
            .insert(
                e,
                PhysicsInterpolation {
                    prev_translation: Vec3::new(0.0, 0.0, 0.0),
                    prev_rotation: Quat::identity(),
                    cur_translation: Vec3::new(4.0, 0.0, 0.0),
                    cur_rotation: Quat::identity(),
                },
            )
            .unwrap();

        let mut schedules = Schedules::new();
        schedules.get_mut::<PostUpdate>().add(InterpolatePhysics);
        // 1/50 s step, 1/100 s frame: no step retires, half a step is banked.
        schedules.set_fixed_timestep(1.0 / 50.0);
        schedules.run_frame(&mut world, &EcsRunner::single_thread(), 1.0 / 100.0);

        assert!(
            (world.resource::<crate::Time>().fixed_alpha() - 0.5).abs() < 1e-9,
            "half a fixed step banked"
        );
        let transform = world.get::<Transform>(e).expect("transform");
        assert!(
            (transform.translation.x - 2.0).abs() < 1e-5,
            "rendered pose must be the midpoint, got {}",
            transform.translation.x
        );
    }
}
