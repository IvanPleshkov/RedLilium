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
/// Requires a [`PhysicsWorld3D`] resource and entities with
/// [`RigidBody3DHandle`] + [`Transform`](crate::Transform) components.
pub struct StepPhysics3D;

impl crate::System for StepPhysics3D {
    type Result = ();
    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        // Integrator dt: physics is scheduled in `FixedUpdate`, so advance
        // Rapier by the fixed timestep (`Time::fixed_delta`) instead of its
        // hardcoded default `dt` — otherwise changing the fixed rate would
        // silently desync simulated time from real time. Worlds ticked without
        // `run_frame` (e.g. the physics demo) carry no `Time`; leave their
        // integrator dt untouched there.
        let fixed_dt = {
            let world = ctx.raw_world();
            world
                .has_resource::<crate::Time>()
                .then(|| world.resource::<crate::Time>().fixed_delta())
        };
        ctx.lock::<(
            crate::ResMut<PhysicsWorld3D>,
            crate::Read<RigidBody3DHandle>,
            crate::Write<crate::Transform>,
        )>()
        .execute(|(mut physics, handles, mut transforms)| {
            redlilium_core::profile_scope!("ecs: step_physics_3d");

            // Step simulation at the fixed timestep (see above).
            if let Some(dt) = fixed_dt {
                physics.integration_parameters.dt = dt;
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

// ---- Fixed-step pose history + render interpolation ----

/// Records each body's authoritative fixed-step pose into its
/// [`PhysicsInterpolation`] history. Runs in `FixedUpdate` **after**
/// [`StepPhysics3D`] (which has just written the post-step pose to
/// `Transform`), so it executes exactly once per fixed step — including the
/// extra iterations of a catch-up frame, leaving the two most recent steps in
/// `prev`/`cur`.
///
/// Bodies without the component are seeded with `prev == cur`, so a freshly
/// spawned body renders at its spawn pose instead of lerping in from wherever
/// the history would otherwise have started.
pub struct RecordPhysicsPose;

impl crate::System for RecordPhysicsPose {
    type Result = ();
    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        let to_seed = ctx
            .lock::<(
                crate::Read<RigidBody3DHandle>,
                crate::Read<crate::Transform>,
                crate::WriteAll<PhysicsInterpolation>,
            )>()
            .execute(|(handles, transforms, mut interps)| {
                redlilium_core::profile_scope!("ecs: record_physics_pose_3d");
                let mut seed = Vec::new();
                for (idx, _handle) in handles.iter() {
                    let Some(transform) = transforms.get(idx) else {
                        continue;
                    };
                    if let Some(mut interp) = interps.get_mut(idx) {
                        interp.prev_translation = interp.cur_translation;
                        interp.prev_rotation = interp.cur_rotation;
                        interp.cur_translation = transform.translation;
                        interp.cur_rotation = transform.rotation;
                    } else {
                        seed.push((idx, transform.translation, transform.rotation));
                    }
                }
                seed
            });

        if !to_seed.is_empty() {
            ctx.commands(move |world| {
                for (idx, translation, rotation) in to_seed {
                    if let Some(entity) = world.entity_at_index(idx) {
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
/// pose is what `GlobalTransform` — and therefore the rasterizer and the
/// motion-vector history — sees. Overwriting `Transform` is safe: Rapier owns
/// the authoritative poses and never reads `Transform` back for an existing
/// body, and [`RecordPhysicsPose`] snapshots the pose inside `FixedUpdate`
/// before this system ever runs.
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
            crate::WriteAll<crate::Transform>,
        )>()
        .execute(|(interps, mut transforms)| {
            redlilium_core::profile_scope!("ecs: interpolate_physics_3d");
            for (idx, interp) in interps.iter() {
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

        // Phase 1: Find stale bodies (entity dead, excluded from game, or lost RigidBody3D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld3D>();
            physics
                .entity_to_body
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_excluded_from_game(**e)
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
                    let _ = world.remove::<ImpulseJoint3DHandle>(*entity);
                }
            }
            for entity in &stale {
                if world.is_alive(*entity) {
                    let _ = world.remove::<RigidBody3DHandle>(*entity);
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
                .filter(|e| {
                    !physics.entity_to_body.contains_key(e) && !world.is_excluded_from_game(*e)
                })
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

        // Phase 1: Find stale joints (entity dead, excluded from game, or lost ImpulseJoint3D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld3D>();
            physics
                .entity_to_joint
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_excluded_from_game(**e)
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
                    let _ = world.remove::<ImpulseJoint3DHandle>(*entity);
                }
            }
        }

        // Phase 2: Find new joints (not in mapping, not excluded from game)
        let new_joints: Vec<(crate::Entity, super::components3d::ImpulseJoint3D)> = {
            let physics = world.resource::<PhysicsWorld3D>();
            world
                .iter_entities()
                .filter(|e| {
                    !physics.entity_to_joint.contains_key(e) && !world.is_excluded_from_game(*e)
                })
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
                // Remove stale: entity dead (full-identity check, so a recycled
                // slot does not keep the old body), excluded from game, or lost RigidBody3D.
                let stale: Vec<crate::Entity> = physics
                    .entity_to_body
                    .keys()
                    .filter(|e| {
                        !ctx.is_alive(**e)
                            || ctx.is_excluded_from_game(**e)
                            || bodies.get(e.index()).is_none()
                    })
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
                        let _ = world.remove::<RigidBody3DHandle>(entity);
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
                // Remove stale: entity dead (full-identity check), disabled, or
                // lost the ImpulseJoint3D component.
                let stale: Vec<crate::Entity> = physics
                    .entity_to_joint
                    .keys()
                    .filter(|e| {
                        !ctx.is_alive(**e)
                            || ctx.is_excluded_from_game(**e)
                            || joints.get(e.index()).is_none()
                    })
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
                        let _ = world.remove::<ImpulseJoint3DHandle>(entity);
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
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 2.0, 0.0)),
        );
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
