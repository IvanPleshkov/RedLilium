// Shared mass planning: only changed owners are recalculated; validation precedes
// all native mutations. Density remains authored data even in explicit-mass mode.
macro_rules! mass_sync {
    ($physics:ident, $body:ident, $collider:ident, $owner:ident) => {
        pub(super) fn prepare_mass_updates(
            world: &crate::World,
            physics: &$physics,
            bodies: Option<&crate::Ref<'_, $body>>,
            colliders: Option<&crate::Ref<'_, $collider>>,
            owners: Option<&crate::Ref<'_, $owner>>,
            transforms: Option<&crate::Ref<'_, crate::Transform>>,
        ) -> Result<
            std::collections::HashMap<crate::Entity, Option<MassProperties>>,
            crate::SystemError,
        > {
            let active_body = |entity: crate::Entity| {
                world.is_alive(entity)
                    && !world.is_excluded_from_game(entity)
                    && bodies.and_then(|s| s.get(entity.index())).is_some()
                    && transforms.and_then(|s| s.get(entity.index())).is_some()
            };
            let mut affected = std::collections::HashSet::new();
            for &handle in &physics.mass_dirty {
                if let Some(entity) = physics.entity_for_body(handle) {
                    affected.insert(entity);
                }
            }
            for (idx, desc) in bodies.into_iter().flat_map(|s| s.iter()) {
                let Some(entity) = world.entity_at_index(idx) else {
                    continue;
                };
                let old = physics
                    .body_for_entity(entity)
                    .and_then(|h| physics.applied_bodies.get(&h));
                if old.is_none_or(|old| old.mass_properties != desc.mass_properties)
                    && active_body(entity)
                {
                    affected.insert(entity);
                }
            }
            for (entity, handle) in &physics.entity_to_collider {
                let current_owner = (world.is_alive(*entity)
                    && !world.is_excluded_from_game(*entity)
                    && colliders.and_then(|s| s.get(entity.index())).is_some())
                .then(|| {
                    owners
                        .and_then(|s| s.get(entity.index()))
                        .map_or(*entity, |o| o.body)
                });
                let old_owner = physics
                    .colliders
                    .get(*handle)
                    .and_then(|c| c.parent())
                    .and_then(|h| physics.entity_for_body(h));
                if current_owner != old_owner {
                    if let Some(old) = old_owner {
                        affected.insert(old);
                    }
                }
            }
            for (idx, desc) in colliders.into_iter().flat_map(|s| s.iter()) {
                let Some(entity) = world.entity_at_index(idx) else {
                    continue;
                };
                let owner = owners.and_then(|s| s.get(idx)).map_or(entity, |o| o.body);
                let current = physics.collider_for_entity(entity);
                let same_owner = current
                    .and_then(|h| physics.colliders[h].parent())
                    .and_then(|h| physics.entity_for_body(h))
                    == Some(owner);
                let old = current.and_then(|h| physics.applied_colliders.get(&h));
                if (!same_owner
                    || old.is_none_or(|old| {
                        old.shape != desc.shape
                            || old.local_pose != desc.local_pose
                            || old.density != desc.density
                    }))
                    && active_body(owner)
                {
                    affected.insert(owner);
                }
            }
            let mut plan = std::collections::HashMap::new();
            for entity in affected {
                if !world.is_alive(entity)
                    || world.is_excluded_from_game(entity)
                    || transforms.and_then(|s| s.get(entity.index())).is_none()
                {
                    continue;
                }
                if let Some(body) = bodies.and_then(|s| s.get(entity.index())) {
                    plan.insert(
                        entity,
                        body.mass_properties.map(|_| MassProperties::default()),
                    );
                }
            }
            if plan.is_empty() {
                return Ok(plan);
            }
            // A single pass over proposed descriptors, not one scan per body.
            for (idx, desc) in colliders.into_iter().flat_map(|s| s.iter()) {
                let Some(entity) = world.entity_at_index(idx) else {
                    continue;
                };
                let owner = owners.and_then(|s| s.get(idx)).map_or(entity, |o| o.body);
                let Some(Some(sum)) = plan.get_mut(&owner) else {
                    continue;
                };
                let settings = bodies
                    .and_then(|s| s.get(owner.index()))
                    .unwrap()
                    .mass_properties
                    .unwrap();
                if settings.needs_geometry() {
                    *sum += desc
                        .shape
                        .mass_properties_for_density(desc.density as Real)
                        .transform_by(&desc.local_pose.to_rapier());
                }
            }
            for (entity, computed) in &mut plan {
                if let Some(sum) = computed {
                    let settings = bodies
                        .and_then(|s| s.get(entity.index()))
                        .unwrap()
                        .mass_properties
                        .unwrap();
                    *sum = settings.resolve(*entity, *sum)?;
                }
            }
            Ok(plan)
        }
    };
}
pub(super) use mass_sync;

macro_rules! mass_world {
    ($physics:ident) => {
        impl $physics {
            pub(super) fn apply_mass_update(
                &mut self,
                handle: RigidBodyHandle,
                properties: Option<MassProperties>,
            ) {
                let Some(body) = self.bodies.get_mut(handle) else {
                    return;
                };
                // Keep the descriptor's density for future automatic mode and weighting.
                // Suppress only the backend collider contributions in explicit mode.
                for &collider in body.colliders() {
                    if let Some(desc) = self.applied_colliders.get(&collider) {
                        self.colliders[collider].set_density(if properties.is_some() {
                            0.0
                        } else {
                            desc.density as Real
                        });
                    }
                }
                if properties.is_some() || body.mass_properties().additional_local_mprops.is_some()
                {
                    body.set_additional_mass_properties(properties.unwrap_or_default(), true);
                }
                body.recompute_mass_properties_from_colliders(&self.colliders);
                body.wake_up(true);
                self.mass_dirty.remove(&handle);
            }

            // Earlier deferred commands can cancel a newly synchronized collider.
            // Re-evaluate affected explicit bodies from what actually survived.
            // An underdetermined model remains dirty, making Step fail until sync
            // receives valid descriptors instead of simulating with ghost mass.
            pub(super) fn refresh_pending_mass(&mut self) {
                if self.mass_dirty.is_empty() {
                    return;
                }
                let pending = std::mem::take(&mut self.mass_dirty);
                for handle in pending {
                    let Some(body) = self.bodies.get(handle) else {
                        continue;
                    };
                    let Some(entity) = self.body_to_entity.get(&handle).copied() else {
                        continue;
                    };
                    let Some(settings) = self
                        .applied_bodies
                        .get(&handle)
                        .and_then(|d| d.mass_properties)
                    else {
                        continue;
                    };
                    let mut sum = MassProperties::default();
                    if settings.needs_geometry() {
                        for &collider in body.colliders() {
                            if let Some(desc) = self.applied_colliders.get(&collider) {
                                sum += desc
                                    .shape
                                    .mass_properties_for_density(desc.density as Real)
                                    .transform_by(&desc.local_pose.to_rapier());
                            }
                        }
                    }
                    match settings.resolve(entity, sum) {
                        Ok(props) => self.apply_mass_update(handle, Some(props)),
                        Err(_) => {
                            self.mass_dirty.insert(handle);
                        }
                    }
                }
            }
        }
    };
}
pub(super) use mass_world;
