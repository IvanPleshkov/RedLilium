// The same ownership/lifecycle algorithm serves both dimensions and both runners.
macro_rules! collider_sync {
    ($physics:ident, $body:ident, $collider:ident, $owner:ident, $body_handle:ident, $collider_handle:ident, $exclusive:ident, $regular:ident) => {
        #[derive(Default)]
        struct SyncChanges {
            bodies: Vec<(crate::Entity, RigidBodyHandle)>,
            colliders: Vec<(crate::Entity, ColliderHandle)>,
            removed_bodies: Vec<crate::Entity>,
            removed_colliders: Vec<crate::Entity>,
            removed_joints: Vec<crate::Entity>,
        }

        fn sync_objects(
            world: &crate::World,
            physics: &mut $physics,
            bodies: Option<&crate::Ref<'_, $body>>,
            colliders: Option<&crate::Ref<'_, $collider>>,
            owners: Option<&crate::Ref<'_, $owner>>,
            transforms: Option<&crate::Ref<'_, crate::Transform>>,
            parents: Option<&crate::Ref<'_, crate::Parent>>,
        ) -> Result<SyncChanges, crate::SystemError> {
            // Validate the full batch before mutating native objects, including
            // descriptors currently waiting for their body to become available.
            for (idx, body) in bodies.into_iter().flat_map(|s| s.iter()) {
                if let Some(entity) = world.entity_at_index(idx) {
                    body.validate(entity)?;
                    if let Some(transform) = transforms.and_then(|s| s.get(idx)) {
                        super::validation::transform(
                            entity,
                            transform,
                            parents.and_then(|s| s.get(idx)).is_some(),
                        )?;
                    }
                }
            }
            for (idx, collider) in colliders.into_iter().flat_map(|s| s.iter()) {
                if let Some(entity) = world.entity_at_index(idx) {
                    collider.validate(entity)?;
                }
            }
            let mass_updates =
                prepare_mass_updates(world, physics, bodies, colliders, owners, transforms)?;
            let mut changes = SyncChanges::default();
            changes.removed_bodies = physics
                .entity_to_body
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_excluded_from_game(**e)
                        || bodies.and_then(|s| s.get(e.index())).is_none()
                        || transforms.and_then(|s| s.get(e.index())).is_none()
                })
                .copied()
                .collect();
            changes.removed_joints = remove_bodies(physics, &changes.removed_bodies);

            for (idx, desc) in bodies.into_iter().flat_map(|s| s.iter()) {
                let (Some(entity), Some(transform)) = (
                    world.entity_at_index(idx),
                    transforms.and_then(|s| s.get(idx)),
                ) else {
                    continue;
                };
                if let Some(handle) = physics.body_for_entity(entity) {
                    physics.apply_body_settings(handle, desc);
                } else {
                    let handle = physics.add_body(desc.to_rigid_body(transform));
                    physics.applied_bodies.insert(handle, desc.clone());
                    physics.entity_to_body.insert(entity, handle);
                    physics.body_to_entity.insert(handle, entity);
                    changes.bodies.push((entity, handle));
                }
            }

            // A changed owner recreates only this collider; geometry/pose edits
            // below preserve its identity. Full Entity identity prevents slot reuse.
            let stale: Vec<_> = physics
                .entity_to_collider
                .iter()
                .filter(|(entity, handle)| {
                    if !world.is_alive(**entity)
                        || world.is_excluded_from_game(**entity)
                        || colliders.and_then(|s| s.get(entity.index())).is_none()
                    {
                        return true;
                    }
                    let body_entity = owners
                        .and_then(|s| s.get(entity.index()))
                        .map_or(**entity, |o| o.body);
                    let expected = physics.body_for_entity(body_entity);
                    !physics
                        .colliders
                        .get(**handle)
                        .is_some_and(|c| expected.is_some() && c.parent() == expected)
                })
                .map(|(e, h)| (*e, *h))
                .collect();
            for (entity, handle) in stale {
                physics.remove_collider(handle);
                changes.removed_colliders.push(entity);
            }
            for (idx, desc) in colliders.into_iter().flat_map(|s| s.iter()) {
                let Some(entity) = world.entity_at_index(idx) else {
                    continue;
                };
                let body_entity = owners.and_then(|s| s.get(idx)).map_or(entity, |o| o.body);
                let Some(parent) = physics.body_for_entity(body_entity) else {
                    continue;
                };
                if let Some(handle) = physics.collider_for_entity(entity) {
                    physics.apply_collider_settings(handle, desc);
                } else {
                    let handle = physics.add_collider(desc.to_collider(), parent);
                    physics.applied_colliders.insert(handle, desc.clone());
                    physics.entity_to_collider.insert(entity, handle);
                    physics.collider_to_entity.insert(handle, entity);
                    changes.colliders.push((entity, handle));
                }
            }
            for (entity, properties) in mass_updates {
                if let Some(handle) = physics.body_for_entity(entity) {
                    physics.apply_mass_update(handle, properties);
                }
            }
            Ok(changes)
        }

        // Publication is immediate in the exclusive system and deferred in the
        // regular one. Native identity maps are available as soon as sync finishes.
        fn publish_objects(world: &mut crate::World, changes: SyncChanges) {
            remove_body_components(world, &changes.removed_bodies, &changes.removed_joints);
            for entity in changes.removed_colliders {
                if world.is_alive(entity) {
                    let _ = world.remove::<$collider_handle>(entity);
                }
            }
            for (entity, handle) in changes.bodies {
                let valid = world.is_alive(entity)
                    && !world.is_excluded_from_game(entity)
                    && world
                        .get::<$body>(entity)
                        .is_some_and(|d| d.validate(entity).is_ok())
                    && world.get::<crate::Transform>(entity).is_some_and(|t| {
                        super::validation::transform(
                            entity,
                            t,
                            world.get::<crate::Parent>(entity).is_some(),
                        )
                        .is_ok()
                    });
                let current = world.resource::<$physics>().body_for_entity(entity) == Some(handle);
                if !current {
                    continue;
                }
                if valid {
                    let _ = world.insert(entity, $body_handle(handle));
                } else {
                    let removed_joints =
                        remove_bodies(&mut world.resource_mut::<$physics>(), &[entity]);
                    remove_body_components(world, &[entity], &removed_joints);
                }
            }
            for (entity, handle) in changes.colliders {
                let valid = world.is_alive(entity)
                    && !world.is_excluded_from_game(entity)
                    && world
                        .get::<$collider>(entity)
                        .is_some_and(|d| d.validate(entity).is_ok());
                let owner = world.get::<$owner>(entity).map_or(entity, |o| o.body);
                let owner_valid = world.is_alive(owner)
                    && !world.is_excluded_from_game(owner)
                    && world
                        .get::<$body>(owner)
                        .is_some_and(|d| d.validate(owner).is_ok())
                    && world.get::<crate::Transform>(owner).is_some_and(|t| {
                        super::validation::transform(
                            owner,
                            t,
                            world.get::<crate::Parent>(owner).is_some(),
                        )
                        .is_ok()
                    });
                let attached = {
                    let physics = world.resource::<$physics>();
                    physics.body_for_entity(owner).is_some_and(|body| {
                        physics
                            .colliders
                            .get(handle)
                            .is_some_and(|c| c.parent() == Some(body))
                    })
                };
                let current =
                    world.resource::<$physics>().entity_to_collider.get(&entity) == Some(&handle);
                if !current {
                    continue;
                }
                if valid && owner_valid && attached {
                    let _ = world.insert(entity, $collider_handle(handle));
                } else {
                    world.resource_mut::<$physics>().remove_collider(handle);
                }
            }
            world.resource_mut::<$physics>().refresh_pending_mass();
        }

        /// Synchronizes bodies and their independently owned colliders.
        /// Bodies require RigidBody + Transform; colliders use an explicit owner
        /// or the body on the same entity. Missing owners leave colliders inactive.
        pub struct $exclusive;
        impl crate::ExclusiveSystem for $exclusive {
            type Result = ();
            fn run(&mut self, world: &mut crate::World) -> Result<(), crate::SystemError> {
                if !world.has_resource::<$physics>() {
                    world.insert_resource($physics::default());
                }
                let changes = {
                    let mut physics = world.resource_mut::<$physics>();
                    sync_objects(
                        world,
                        &mut physics,
                        world.try_read::<$body>().as_ref(),
                        world.try_read::<$collider>().as_ref(),
                        world.try_read::<$owner>().as_ref(),
                        world.try_read::<crate::Transform>().as_ref(),
                        world.try_read::<crate::Parent>().as_ref(),
                    )?
                };
                publish_objects(world, changes);
                Ok(())
            }
        }

        /// Regular sync. Native mappings update now; ECS handle components are
        /// published by deferred commands, with identity and prerequisite checks.
        pub struct $regular;
        impl crate::System for $regular {
            type Result = ();
            fn run<'a>(
                &'a self,
                ctx: &'a crate::SystemContext<'a>,
            ) -> Result<(), crate::SystemError> {
                let changes = ctx
                    .lock::<(
                        crate::ResMut<$physics>,
                        crate::OptionalRead<$body>,
                        crate::OptionalRead<$collider>,
                        crate::OptionalRead<$owner>,
                        crate::OptionalRead<crate::Transform>,
                        crate::OptionalRead<crate::Parent>,
                    )>()
                    .execute(
                        |(mut physics, bodies, colliders, owners, transforms, parents)| {
                            sync_objects(
                                ctx.raw_world(),
                                &mut physics,
                                bodies.as_ref(),
                                colliders.as_ref(),
                                owners.as_ref(),
                                transforms.as_ref(),
                                parents.as_ref(),
                            )
                        },
                    )?;
                if !changes.bodies.is_empty()
                    || !changes.colliders.is_empty()
                    || !changes.removed_bodies.is_empty()
                    || !changes.removed_colliders.is_empty()
                    || !changes.removed_joints.is_empty()
                {
                    ctx.commands(move |world| publish_objects(world, changes));
                }
                Ok(())
            }
        }
    };
}
pub(super) use collider_sync;
