//! Shared implementation; dimensions expose distinct, statically typed checkpoints.
use crate::{Entity, World};
use std::collections::HashMap;

/// Legacy frame queues remain the default. Replay systems read the physics
/// world's last_step_events in TickOnly mode, without frame-event cursors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PhysicsEventDelivery {
    /// Publish to the legacy frame queues and retain the latest tick batch.
    #[default]
    FrameQueues,
    /// Retain only the latest tick batch; consume it after every physics step.
    TickOnly,
}

/// A rejected checkpoint operation leaves physics and ECS bindings unchanged.
/// This does not roll back earlier game mutations outside that operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicsCheckpointError {
    MissingRegistration(&'static str),
    Busy(&'static str),
    WrongWorld,
    WrongPhysicsResource,
    DescriptorsChanged(&'static str),
    EntityChanged(Entity),
    Unsynchronized(&'static str),
}
impl std::fmt::Display for PhysicsCheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "physics checkpoint: {self:?}")
    }
}
impl std::error::Error for PhysicsCheckpointError {}

pub(super) fn values<T: Clone + 'static>(
    world: &World,
) -> Result<Vec<(Entity, T)>, PhysicsCheckpointError> {
    let storage = world
        .try_read::<T>()
        .ok_or(PhysicsCheckpointError::MissingRegistration(
            std::any::type_name::<T>(),
        ))?;
    Ok(storage
        .iter_unfiltered()
        .filter_map(|(index, value)| world.entity_at_index(index).map(|e| (e, value.clone())))
        .collect())
}
pub(super) fn descriptors<T: Clone + 'static>(
    world: &World,
) -> Result<HashMap<Entity, T>, PhysicsCheckpointError> {
    Ok(values::<T>(world)?.into_iter().collect())
}

macro_rules! physics_checkpoint {
    ($physics:ident, $checkpoint:ident, $bindings:ident,
     $components:path, $controls:path, $events:path,
     $body:ident, $collider:ident, $owner:ident, $joint:ident,
     $bh:ident, $ch:ident, $jh:ident, $interp:ident, $target:ident, $velocity:ident,
     $collision:ident, $force:ident) => {
        mod checkpoint_api {
            use super::*;
            use crate::physics::PhysicsCheckpointError as Error;
            use crate::physics::checkpoint_support::{descriptors, values};
            use crate::{Entity, Events, Parent, Transform, World};
            use components::{$body, $collider, $joint, $owner};
            use controls::{$target, $velocity};
            use events::{$collision, $force};
            use std::sync::Arc;
            use $components as components;
            use $controls as controls;
            use $events as events;

            #[derive(Clone)]
            struct EcsState {
                bodies: HashMap<Entity, $body>,
                colliders: HashMap<Entity, $collider>,
                owners: HashMap<Entity, $owner>,
                joints: HashMap<Entity, $joint>,
                flags: HashMap<Entity, u32>,
                parents: HashMap<Entity, Option<Parent>>,
                transforms: Vec<(Entity, Transform)>,
                body_handles: Vec<(Entity, $bh)>,
                collider_handles: Vec<(Entity, $ch)>,
                joint_handles: Vec<(Entity, $jh)>,
                interpolation: Vec<(Entity, $interp)>,
                targets: Vec<(Entity, $target)>,
                velocities: Vec<(Entity, $velocity)>,
            }
            impl EcsState {
                fn capture(world: &World) -> Result<Self, Error> {
                    let bodies = descriptors::<$body>(world)?;
                    let colliders = descriptors::<$collider>(world)?;
                    let owners = descriptors::<$owner>(world)?;
                    let joints = descriptors::<$joint>(world)?;
                    let targets = values::<$target>(world)?;
                    let velocities = values::<$velocity>(world)?;
                    let entities: std::collections::HashSet<_> = bodies
                        .keys()
                        .chain(colliders.keys())
                        .chain(owners.keys())
                        .chain(joints.keys())
                        .copied()
                        .chain(targets.iter().map(|(e, _)| *e))
                        .chain(velocities.iter().map(|(e, _)| *e))
                        .collect();
                    let flags = entities
                        .iter()
                        .map(|e| (*e, world.get_entity_flags(*e)))
                        .collect();
                    let parents = entities
                        .iter()
                        .map(|e| (*e, world.get::<Parent>(*e).map(|v| *v)))
                        .collect();
                    let transforms = values::<Transform>(world)?
                        .into_iter()
                        .filter(|(e, _)| bodies.contains_key(e))
                        .collect();
                    Ok(Self {
                        bodies,
                        colliders,
                        owners,
                        joints,
                        flags,
                        parents,
                        transforms,
                        targets,
                        velocities,
                        body_handles: values::<$bh>(world)?,
                        collider_handles: values::<$ch>(world)?,
                        joint_handles: values::<$jh>(world)?,
                        interpolation: values::<$interp>(world)?,
                    })
                }
                fn validate(&self, world: &World, saved: &$physics) -> Result<(), Error> {
                    for (&entity, &flags) in &self.flags {
                        if !world.is_alive(entity)
                            || world.get_entity_flags(entity) != flags
                            || world.get::<Parent>(entity).map(|v| *v) != self.parents[&entity]
                        {
                            return Err(Error::EntityChanged(entity));
                        }
                    }
                    if descriptors::<$body>(world)? != self.bodies {
                        return Err(Error::DescriptorsChanged(stringify!($body)));
                    }
                    if descriptors::<$collider>(world)? != self.colliders {
                        return Err(Error::DescriptorsChanged(stringify!($collider)));
                    }
                    if descriptors::<$owner>(world)? != self.owners {
                        return Err(Error::DescriptorsChanged(stringify!($owner)));
                    }
                    if descriptors::<$joint>(world)? != self.joints {
                        return Err(Error::DescriptorsChanged(stringify!($joint)));
                    }
                    let transforms: HashMap<_, _> =
                        self.transforms.iter().map(|(e, t)| (*e, t)).collect();
                    for &entity in self.bodies.keys() {
                        let expected = transforms.get(&entity).copied();
                        let current = world.get::<Transform>(entity);
                        if current.is_some() != expected.is_some() {
                            return Err(Error::EntityChanged(entity));
                        }
                        if saved
                            .body_for_entity(entity)
                            .and_then(|h| saved.bodies.get(h))
                            .is_some_and(|b| b.is_fixed())
                            && current.as_deref() != expected
                        {
                            return Err(Error::DescriptorsChanged("fixed body Transform"));
                        }
                    }
                    for entity in self
                        .body_handles
                        .iter()
                        .map(|(e, _)| *e)
                        .chain(self.collider_handles.iter().map(|(e, _)| *e))
                        .chain(self.joint_handles.iter().map(|(e, _)| *e))
                        .chain(self.interpolation.iter().map(|(e, _)| *e))
                    {
                        if !world.is_alive(entity) {
                            return Err(Error::EntityChanged(entity));
                        }
                    }
                    // Preflight every destination before commit. These reads also
                    // ensure no runtime component registration disappeared.
                    values::<$bh>(world)?;
                    values::<$ch>(world)?;
                    values::<$jh>(world)?;
                    values::<$interp>(world)?;
                    values::<$target>(world)?;
                    values::<$velocity>(world)?;
                    Ok(())
                }
                fn apply(self, world: &mut World) {
                    // Only engine-owned derived state and explicit motion inputs.
                    // Restore without invoking user lifecycle hooks/observers.
                    world.restore_runtime_components(self.body_handles);
                    world.restore_runtime_components(self.collider_handles);
                    world.restore_runtime_components(self.joint_handles);
                    world.restore_runtime_components(self.interpolation);
                    world.restore_runtime_components(self.targets);
                    world.restore_runtime_components(self.velocities);
                    for (entity, transform) in self.transforms {
                        *world
                            .get_mut::<Transform>(entity)
                            .expect("prevalidated transform") = transform;
                    }
                }
            }

            /// Opaque reusable in-memory checkpoint, bound to one physics resource
            /// and World. Contains no references to mutable live solver state.
            pub struct $checkpoint {
                state: $physics,
                ecs: EcsState,
                world: std::sync::Weak<()>,
            }
            impl Clone for $checkpoint {
                fn clone(&self) -> Self {
                    Self {
                        state: self.state.clone_checkpoint_state(),
                        ecs: self.ecs.clone(),
                        world: self.world.clone(),
                    }
                }
            }

            /// Exclusive, restricted ECS boundary for physical checkpoint operations.
            /// Obtain with `with` between completed ticks, after commands/observers
            /// have flushed. The callback only receives physics and these bindings,
            /// never arbitrary World mutation. No active tasks may mutate this state.
            pub struct $bindings<'a> {
                world: &'a mut World,
                physics: *const $physics,
                collisions: Option<&'a mut Events<$collision>>,
                forces: Option<&'a mut Events<$force>>,
            }
            impl $bindings<'_> {
                /// Acquire physics and its optional frame event queues exclusively.
                /// All acquisitions precede the callback; a live resource borrow
                /// returns `Busy` instead of waiting. Standard component registration
                /// and a completed synchronization/command boundary are required.
                pub fn with<R>(
                    world: &mut World,
                    operation: impl FnOnce(&mut $physics, &mut $bindings<'_>) -> Result<R, Error>,
                ) -> Result<R, Error> {
                    if !world.has_resource::<$physics>() {
                        return Err(Error::MissingRegistration(stringify!($physics)));
                    }
                    let physics = world.resource_shared::<$physics>();
                    let collision = world
                        .has_resource::<Events<$collision>>()
                        .then(|| world.resource_shared::<Events<$collision>>());
                    let force = world
                        .has_resource::<Events<$force>>()
                        .then(|| world.resource_shared::<Events<$force>>());
                    let mut guard = physics
                        .try_write()
                        .ok_or(Error::Busy(stringify!($physics)))?;
                    let physics = guard
                        .as_any_mut()
                        .downcast_mut::<$physics>()
                        .expect("typed resource");
                    let mut collision_guard = collision
                        .as_ref()
                        .map(|h| h.try_write().ok_or(Error::Busy(stringify!($collision))))
                        .transpose()?;
                    let mut force_guard = force
                        .as_ref()
                        .map(|h| h.try_write().ok_or(Error::Busy(stringify!($force))))
                        .transpose()?;
                    let collisions = collision_guard
                        .as_mut()
                        .map(|g| g.as_any_mut().downcast_mut::<Events<$collision>>().unwrap());
                    let forces = force_guard
                        .as_mut()
                        .map(|g| g.as_any_mut().downcast_mut::<Events<$force>>().unwrap());
                    let mut bindings = $bindings {
                        world,
                        physics: physics as *const _,
                        collisions,
                        forces,
                    };
                    operation(physics, &mut bindings)
                }
            }
            impl $physics {
                fn clone_checkpoint_state(&self) -> Self {
                    Self {
                        checkpoint_identity: self.checkpoint_identity.clone(),
                        event_delivery: self.event_delivery,
                        last_step_events: self.last_step_events.clone(),
                        collision_events: self.collision_events.clone(),
                        teleports: self.teleports.clone(),
                        pose_resets: self.pose_resets.clone(),
                        mass_dirty: self.mass_dirty.clone(),
                        applied_bodies: self.applied_bodies.clone(),
                        applied_colliders: self.applied_colliders.clone(),
                        entity_to_collider: self.entity_to_collider.clone(),
                        collider_to_entity: self.collider_to_entity.clone(),
                        applied_joints: self.applied_joints.clone(),
                        gravity: self.gravity,
                        integration_parameters: self.integration_parameters,
                        // Rapier pipeline is workspace, not persistent solver state.
                        // A fresh pipeline invalidates its joint-selection memo.
                        pipeline: PhysicsPipeline::new(),
                        island_manager: self.island_manager.clone(),
                        broad_phase: self.broad_phase.clone(),
                        narrow_phase: self.narrow_phase.clone(),
                        bodies: self.bodies.clone(),
                        colliders: self.colliders.clone(),
                        impulse_joints: self.impulse_joints.clone(),
                        multibody_joints: self.multibody_joints.clone(),
                        ccd_solver: self.ccd_solver.clone(),
                        soft_bodies: self.soft_bodies.clone(),
                        entity_to_body: self.entity_to_body.clone(),
                        body_to_entity: self.body_to_entity.clone(),
                        entity_to_joint: self.entity_to_joint.clone(),
                    }
                }
                /// Capture at a completed fixed boundary. Pending motion inputs,
                /// teleports and native forces are included. Entity identities and
                /// descriptor values become restore preconditions, not game state
                /// that this API is permitted to reconstruct.
                pub fn capture_checkpoint(
                    &self,
                    bindings: &$bindings<'_>,
                ) -> Result<$checkpoint, Error> {
                    if !std::ptr::eq(self, bindings.physics) {
                        return Err(Error::WrongPhysicsResource);
                    }
                    if !self.mass_dirty.is_empty() {
                        return Err(Error::Unsynchronized("pending mass update"));
                    }
                    let ecs = EcsState::capture(bindings.world)?;
                    if self.entity_to_body.keys().any(|e| {
                        !ecs.bodies.contains_key(e)
                            || bindings.world.get::<Transform>(*e).is_none()
                            || bindings.world.is_excluded_from_game(*e)
                    }) || self.entity_to_collider.keys().any(|e| {
                        !ecs.colliders.contains_key(e) || bindings.world.is_excluded_from_game(*e)
                    }) || self.entity_to_joint.keys().any(|e| {
                        !ecs.joints.contains_key(e) || bindings.world.is_excluded_from_game(*e)
                    }) {
                        return Err(Error::Unsynchronized(
                            "native object no longer has an active ECS descriptor; run sync first",
                        ));
                    }
                    if ecs.body_handles.len() != self.entity_to_body.len()
                        || ecs
                            .body_handles
                            .iter()
                            .any(|(e, h)| self.entity_to_body.get(e) != Some(&h.0))
                        || ecs.collider_handles.len() != self.entity_to_collider.len()
                        || ecs
                            .collider_handles
                            .iter()
                            .any(|(e, h)| self.entity_to_collider.get(e) != Some(&h.0))
                        || ecs.joint_handles.len() != self.entity_to_joint.len()
                        || ecs
                            .joint_handles
                            .iter()
                            .any(|(e, h)| self.entity_to_joint.get(e) != Some(&h.0))
                    {
                        return Err(Error::Unsynchronized(
                            "sync systems and deferred handle publication must complete",
                        ));
                    }
                    Ok($checkpoint {
                        state: self.clone_checkpoint_state(),
                        ecs,
                        world: Arc::downgrade(bindings.world.identity()),
                    })
                }
                /// Game descriptors/identities must match the checkpoint. Native
                /// motion, solver caches, handles, inputs and interpolation rewind.
                /// Legacy physics Events are discarded without rewinding their IDs;
                /// replay consumers use last_step_events instead of frame cursors.
                pub fn restore_checkpoint(
                    &mut self,
                    checkpoint: &$checkpoint,
                    bindings: &mut $bindings<'_>,
                ) -> Result<(), Error> {
                    if !std::ptr::eq(self, bindings.physics)
                        || !Arc::ptr_eq(
                            &self.checkpoint_identity,
                            &checkpoint.state.checkpoint_identity,
                        )
                    {
                        return Err(Error::WrongPhysicsResource);
                    }
                    if !checkpoint
                        .world
                        .ptr_eq(&Arc::downgrade(bindings.world.identity()))
                    {
                        return Err(Error::WrongWorld);
                    }
                    checkpoint.ecs.validate(bindings.world, &checkpoint.state)?;
                    // Allocate/stage before touching either live side. No callbacks
                    // or fallible structural operations run during the commit.
                    let state = checkpoint.state.clone_checkpoint_state();
                    let ecs = checkpoint.ecs.clone();
                    bindings.world.advance_tick();
                    ecs.apply(bindings.world);
                    if let Some(events) = &mut bindings.collisions {
                        events.clear();
                    }
                    if let Some(events) = &mut bindings.forces {
                        events.clear();
                    }
                    *self = state;
                    Ok(())
                }
            }
        }
        pub use checkpoint_api::{$bindings, $checkpoint};
    };
}
pub(super) use physics_checkpoint;
