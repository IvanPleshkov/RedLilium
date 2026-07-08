mod actions;
mod batch_ops;
pub(crate) mod component_ops;
mod hooks;
mod inspector;
mod query_access;
mod resources;
mod snapshot;
#[cfg(test)]
mod tests;

use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};

use crate::component::Component;
use crate::entity::{Entities, Entity};
use crate::observer::Observers;
use crate::reactive::Triggers;
use crate::resource::Resources;
use crate::sparse_set::ComponentStorage;

/// Error returned by [`World`] operations that require a registered component
/// type or a living entity.
#[derive(Debug)]
pub enum WorldError {
    /// The component type has never been registered via
    /// [`World::register_component`] or inserted.
    ComponentNotRegistered {
        /// The name of the unregistered component type.
        type_name: &'static str,
    },
    /// The entity is not alive (already despawned or never existed).
    EntityNotAlive {
        /// The dead entity.
        entity: Entity,
    },
    /// A registration would map a `TypeId` that is already recorded under a
    /// **different** [`SourceId`](crate::SourceId) — the aliasing/collision
    /// case the ADR-020 type-identity amendment rejects loudly (e.g. a game
    /// cdylib registering a type whose `TypeId` clashes with an engine type
    /// from another source).
    TypeSourceConflict {
        /// The conflicting type's name.
        type_name: &'static str,
        /// The source already recorded for the `TypeId`.
        existing: crate::type_identity::SourceId,
        /// The source the current registration tried to stamp.
        incoming: crate::type_identity::SourceId,
    },
}

impl std::fmt::Display for WorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComponentNotRegistered { type_name } => write!(
                f,
                "Component type `{type_name}` has never been registered. Call register_component() first."
            ),
            Self::EntityNotAlive { entity } => {
                write!(f, "Entity {entity} is not alive")
            }
            Self::TypeSourceConflict {
                type_name,
                existing,
                incoming,
            } => write!(
                f,
                "Type `{type_name}` is already registered under source {existing:?}; \
                 refusing to re-register it under {incoming:?} (type-identity conflict)"
            ),
        }
    }
}

impl std::error::Error for WorldError {}

/// Convenience alias for backward compatibility.
pub type ComponentNotRegistered = WorldError;

// Re-export InspectResult for public API consumers.
pub use crate::sparse_set::InspectResult;

pub use actions::set_component_actions;

/// Type-erased serialize helper: reads `T` from the world and serializes it.
fn serialize_component_fn<T: Component>(
    world: &World,
    entity: Entity,
    ctx: &mut crate::serialize::SerializeContext<'_>,
) -> Result<Option<crate::serialize::SerializedComponent>, crate::serialize::SerializeError> {
    let Some(comp) = world.get::<T>(entity) else {
        return Ok(None);
    };
    match comp.serialize_component(ctx) {
        Ok(value) => Ok(Some(crate::serialize::SerializedComponent {
            type_name: T::NAME.to_owned(),
            schema_version: T::SCHEMA_VERSION,
            data: value,
        })),
        Err(crate::serialize::SerializeError::NotSerializable { .. }) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Type-erased deserialize helper: deserializes `T` and inserts it on an entity.
fn deserialize_component_fn<T: Component>(
    entity: Entity,
    data: &crate::serialize::Value,
    ctx: &mut crate::serialize::DeserializeContext<'_>,
) -> Result<(), crate::serialize::DeserializeError> {
    ctx.load_data(data)?;
    let comp = T::deserialize_component(ctx)?;
    ctx.world_mut().insert(entity, comp).map_err(|e| match e {
        WorldError::ComponentNotRegistered { type_name } => {
            crate::serialize::DeserializeError::UnknownComponent {
                type_name: type_name.to_string(),
            }
        }
        WorldError::EntityNotAlive { entity } => {
            crate::serialize::DeserializeError::UnknownComponent {
                type_name: format!("entity {entity} not alive"),
            }
        }
        // `insert` never surfaces a source conflict (registration, not insert,
        // is where the fail-fast lives); mapped for exhaustiveness only.
        WorldError::TypeSourceConflict { type_name, .. } => {
            crate::serialize::DeserializeError::UnknownComponent {
                type_name: type_name.to_string(),
            }
        }
    })?;
    Ok(())
}

/// An independent ECS world containing entities, components, and resources.
///
/// Each World is fully self-contained. Multiple worlds can coexist
/// in the same process, sharing no data between them.
///
/// A `World` is `Send`, but if main-thread resources are in use
/// ([`insert_main_thread_resource`](World::insert_main_thread_resource))
/// it must be dropped on the thread that owns them (debug-checked); see
/// that method's docs for the confinement contract.
///
/// # Example
///
/// ```
/// use redlilium_ecs::World;
///
/// struct Position { x: f32, y: f32 }
/// struct Velocity { x: f32, y: f32 }
///
/// let mut world = World::new();
/// world.register_component::<Position>();
/// world.register_component::<Velocity>();
///
/// let entity = world.spawn();
/// world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
/// world.insert(entity, Velocity { x: 1.0, y: 0.0 }).unwrap();
///
/// // Query components
/// let positions = world.read::<Position>().unwrap();
/// let velocities = world.read::<Velocity>().unwrap();
/// for (idx, pos) in positions.iter() {
///     if let Some(vel) = velocities.get(idx) {
///         println!("pos ({}, {}) vel ({}, {})", pos.x, pos.y, vel.x, vel.y);
///     }
/// }
/// ```
pub struct World {
    entities: Entities,
    components: HashMap<TypeId, crate::sync::RwLock<ComponentStorage>>,
    resources: Resources,
    /// Global tick counter for change detection. Atomic so runners can
    /// assign per-system ticks while worker threads hold `&World`.
    tick: std::sync::atomic::AtomicU64,
    /// Reverse index from component name to TypeId for name-based lookups.
    name_index: BTreeMap<&'static str, TypeId>,
    /// Deferred observer registry and pending triggers. `pub(crate)` so
    /// [`crate::observer::flush`] can re-borrow it through a raw world
    /// pointer instead of holding `&mut Observers` across handler calls.
    pub(crate) observers: Observers,
    /// Monomorphized swap functions for each registered `Triggers<M>` resource.
    trigger_swap_fns: Vec<fn(&mut World)>,
    /// Monomorphized update functions for each registered `Events<T>` resource.
    event_update_fns: Vec<fn(&mut World)>,
    /// Type-erased capture/restore hooks for resources that opt into
    /// whole-world snapshots, keyed by `SnapshotResource::NAME` (sorted for
    /// deterministic capture order).
    snapshot_resources: BTreeMap<&'static str, snapshot::SnapshotResourceEntry>,
    /// Origin (`SourceId`) recorded for each registered component/resource
    /// `TypeId`, per the ADR-020 type-identity amendment. Storage stays keyed
    /// by bare `TypeId`; this map carries the qualifying `source` as metadata,
    /// resolved at the boundaries (downcast guard, load-time fail-fast). In a
    /// single-process build every entry is [`SourceId::HOST`].
    type_sources: HashMap<TypeId, crate::type_identity::SourceId>,
    /// The source stamped onto registrations happening right now. Defaults to
    /// [`SourceId::HOST`]; a future loader scopes game-cdylib registrations by
    /// flipping it via [`with_registration_source`](World::with_registration_source).
    current_source: crate::type_identity::SourceId,
    /// Tracks which generations this world has registered types under (used at
    /// world drop to decrement counts in the shared registry). Maps generation_id
    /// to count of types registered under that generation.
    /// The authoritative registry lives in EngineContext, injected as a resource.
    worlds_generations: HashMap<u32, usize>,
    /// Registry of component migrations for handling schema changes during
    /// deserialization (Phase 3, R3). When a component fails to deserialize,
    /// a registered migration can transform the old-schema value into a
    /// new-schema one, or the component is gracefully skipped.
    pub(crate) migration_registry: crate::MigrationRegistry,
}

impl redlilium_core::abstract_editor::Editable for World {}

impl World {
    /// Creates a new empty world.
    pub fn new() -> Self {
        let mut resources = Resources::new();
        resources.insert(
            crate::commands::CommandBuffer::new(),
            crate::type_identity::SourceId::HOST,
        );
        let mut type_sources = HashMap::new();
        // The bootstrap `CommandBuffer` is inserted directly above (before
        // `Self` exists, so it bypasses `insert_resource`/`record_type_source`).
        // Record its `HOST` origin here so the downcast guard covers it too
        // rather than silently skipping it as an unknown-source entry.
        type_sources.insert(
            TypeId::of::<crate::commands::CommandBuffer>(),
            crate::type_identity::SourceId::HOST,
        );
        Self {
            entities: Entities::new(),
            components: HashMap::new(),
            resources,
            // Starts at 1 so setup-time writes (stamped with tick 1) are
            // visible to never-run systems, whose `last_run` is 0.
            tick: std::sync::atomic::AtomicU64::new(1),
            name_index: BTreeMap::new(),
            observers: Observers::new(),
            trigger_swap_fns: Vec::new(),
            event_update_fns: Vec::new(),
            snapshot_resources: BTreeMap::new(),
            type_sources,
            current_source: crate::type_identity::SourceId::HOST,
            worlds_generations: HashMap::new(),
            migration_registry: crate::MigrationRegistry::new(),
        }
    }

    // ---- Type identity (ADR-020 amendment) ----

    /// **Declares** the origin ([`SourceId`](crate::SourceId)) of a type at an
    /// explicit *registration* boundary (e.g. `register_component`), and fails
    /// fast on a source conflict.
    ///
    /// Registering a type is a claim that *this* source defines it. A *value
    /// insert* of an already-declared type is a different act — see
    /// [`resolve_type_source`](World::resolve_type_source), which respects the
    /// existing origin instead of conflicting. Keeping the two distinct is what
    /// lets a warm reload restore a snapshot (value inserts, possibly under a
    /// different scope than the original registration) without a spurious
    /// panic, while a genuine `TypeId` collision between two *registering*
    /// sources is still rejected loudly.
    ///
    /// The source used is the current registration source — normally
    /// [`SourceId::HOST`](crate::SourceId::HOST), or whatever a
    /// [`with_registration_source`](World::with_registration_source) scope set.
    /// Re-recording the *same* `(TypeId, source)` is idempotent (matching the
    /// idempotence of the `register_*` paths).
    ///
    /// # Supersession (R4)
    ///
    /// If `type_id` is already recorded under a **different** source:
    /// - If the existing source is a dead generation (no active registrations),
    ///   the new source takes over (supersession).
    /// - If the existing source is still alive, a [`WorldError::TypeSourceConflict`]
    ///   panic is raised.
    ///
    /// This allows snapshot restore after a game cdylib reload to re-stamp types
    /// under the new generation without falsely triggering a conflict when the
    /// old generation is no longer active.
    ///
    /// # Panics
    ///
    /// Panics with [`WorldError::TypeSourceConflict`] if `type_id` was already
    /// recorded under a **different** source that is still active — the
    /// aliasing/collision case the ADR mandates be rejected loudly.
    pub(crate) fn record_type_source(&mut self, type_id: TypeId, type_name: &'static str) {
        let incoming = self.current_source;

        // Check shared registry BEFORE taking mutable borrows.
        let existing_entry = self.type_sources.get(&type_id).copied();

        // Determine if we can proceed with registration.
        if let Some(existing) = existing_entry {
            if existing != incoming {
                // Check if the existing generation is still active.
                let existing_gen_id = existing.0;
                let is_active = self
                    .with_generation_registry(|reg| reg.is_generation_active(existing_gen_id))
                    .unwrap_or(true); // No shared registry = assume active (fail-safe)

                if is_active {
                    panic!(
                        "{}",
                        WorldError::TypeSourceConflict {
                            type_name,
                            existing,
                            incoming,
                        }
                    );
                }
                // Dead generation: supersession allowed, overwrite.
                self.type_sources.insert(type_id, incoming);
                // Record in shared registry.
                let incoming_gen_id = incoming.0;
                let _ = self.with_generation_registry_mut(|reg| {
                    reg.record_type_registration(type_id, incoming_gen_id);
                });
                // Track locally.
                *self.worlds_generations.entry(incoming_gen_id).or_insert(0) += 1;
            }
            // Else: same generation, idempotent, no action needed.
        } else {
            // New registration.
            self.type_sources.insert(type_id, incoming);
            let incoming_gen_id = incoming.0;
            let _ = self.with_generation_registry_mut(|reg| {
                reg.record_type_registration(type_id, incoming_gen_id);
            });
            // Track locally.
            *self.worlds_generations.entry(incoming_gen_id).or_insert(0) += 1;
        }
    }

    /// Resolves the [`SourceId`](crate::SourceId) to stamp on a *value insert*
    /// of `type_id`, declaring the current source only if the type has no
    /// recorded origin yet.
    ///
    /// Unlike [`record_type_source`](World::record_type_source), an insert is
    /// not a declaration: if the type already has an origin, that origin wins
    /// and the insert respects it (never conflicts). This is the path a
    /// snapshot restore takes — re-inserting a resource under whatever scope is
    /// active must not fight the origin declared when the type was first seen.
    /// A never-before-seen type is declared under the current source.
    ///
    /// The trade-off versus `record_type_source`: value inserts cannot detect a
    /// cross-source `TypeId` collision (the first inserter wins silently). That
    /// is acceptable — resources have no separate registration step, collisions
    /// are caught on the component path, and in the warm-reload model game data
    /// only ever crosses a generation boundary as plain serialized `Value`s,
    /// never as a live in-place downcast.
    pub(crate) fn resolve_type_source(
        &mut self,
        type_id: TypeId,
    ) -> crate::type_identity::SourceId {
        let current = self.current_source;
        *self.type_sources.entry(type_id).or_insert(current)
    }

    /// Runs `f` with the registration source temporarily set to `source`, so
    /// every component/resource registered inside is stamped with it.
    ///
    /// The previous source is restored afterward — **including if `f` panics**.
    /// This matters for the #45 loader: a failing plugin registration panics
    /// (e.g. [`record_type_source`](World::record_type_source)'s fail-fast
    /// fires *inside* this scope), and an editor that catches that panic to stay
    /// alive must not be left with `current_source` stuck on a dead generation —
    /// which would mis-stamp every subsequent host registration. The restore is
    /// therefore RAII, not a tail assignment.
    ///
    /// Today only tests call this with a non-`HOST` value; the engine's
    /// `register_std_components` and the like run under the default
    /// [`SourceId::HOST`](crate::SourceId::HOST).
    pub fn with_registration_source<R>(
        &mut self,
        source: crate::type_identity::SourceId,
        f: impl FnOnce(&mut World) -> R,
    ) -> R {
        /// Restores `current_source` on drop, so an unwinding `f` cannot leave
        /// the world stamping registrations with a stale generation.
        struct RestoreSource<'a> {
            world: &'a mut World,
            prev: crate::type_identity::SourceId,
        }
        impl Drop for RestoreSource<'_> {
            fn drop(&mut self) {
                self.world.current_source = self.prev;
            }
        }

        let prev = self.current_source;
        self.current_source = source;
        let scope = RestoreSource { world: self, prev };
        // Reborrow (not move) so `scope` stays intact to restore on drop.
        f(&mut *scope.world)
    }

    /// The [`QualifiedTypeId`](crate::QualifiedTypeId) for `T` if it has been
    /// registered (as a component or resource), resolving `source` via the
    /// registration metadata. Returns `None` for never-registered types.
    pub fn qualified_type_id_of<T: 'static>(&self) -> Option<crate::QualifiedTypeId> {
        self.type_sources
            .get(&TypeId::of::<T>())
            .map(|&source| crate::QualifiedTypeId::new(TypeId::of::<T>(), source))
    }

    /// The current registration source ([`SourceId::HOST`](crate::SourceId::HOST)
    /// outside a [`with_registration_source`](World::with_registration_source)
    /// scope).
    pub fn current_source(&self) -> crate::type_identity::SourceId {
        self.current_source
    }

    /// Resolves the recorded source for `T`, if registered. Used by the
    /// resource downcast guard to compare against an entry's stored source.
    pub(crate) fn resolved_source<T: 'static>(&self) -> Option<crate::type_identity::SourceId> {
        self.type_sources.get(&TypeId::of::<T>()).copied()
    }

    /// [`resolved_source`](World::resolved_source) by bare `TypeId`.
    pub(crate) fn resolved_source_by_id(
        &self,
        type_id: TypeId,
    ) -> Option<crate::type_identity::SourceId> {
        self.type_sources.get(&type_id).copied()
    }

    /// Test-only seam: overwrites the recorded source for `T` without the
    /// fail-fast, simulating a type surviving into a fresh generation (which in
    /// production only happens across a world-dropping reload). Lets a unit test
    /// force the live downcast-guard mismatch that the fail-fast otherwise makes
    /// unreachable.
    ///
    /// Gated on `debug_assertions` to match its only caller (the guard test,
    /// which the `debug_assert`-based guard makes a no-op in release) — so a
    /// release test build has neither the helper nor a dead-code warning for it.
    #[cfg(all(test, debug_assertions))]
    pub(crate) fn force_type_source_for_test<T: 'static>(
        &mut self,
        source: crate::type_identity::SourceId,
    ) {
        self.type_sources.insert(TypeId::of::<T>(), source);
    }

    // ---- Migration registry (Phase 3, R3) ----

    /// Registers a migration function for a component type.
    ///
    /// When deserialization of a component fails, if a migration is registered
    /// for that type, the migration function is called to attempt to transform
    /// the old-schema value into a new-schema one.
    ///
    /// The migration function receives:
    /// - `value: &Value` — the serialized component data
    /// - `from_schema_version: u32` — version from the snapshot
    /// - `to_schema_version: u32` — current component registration version
    ///
    /// Returns `Some(transformed)` if the migration was applied, `None` to skip.
    ///
    /// # Example
    ///
    /// ```ignore
    /// world.register_component_migration::<OldPosition>(
    ///     |value, from_schema, to_schema| {
    ///         if from_schema == 1 && to_schema == 2 {
    ///             // Transform old-v1 to new-v2 format
    ///             Some(Value::Object(map![
    ///                 "x".into() => value["x"].clone(),
    ///                 "y".into() => value["y"].clone(),
    ///             ]))
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// );
    /// ```
    pub fn register_component_migration(
        &mut self,
        type_name: &'static str,
        migration: crate::migration::ComponentMigrationFn,
    ) {
        self.migration_registry.register(type_name, migration);
    }

    // ---- Component lock helpers ----

    /// Returns a reference to the component storage without locking.
    ///
    /// # Safety
    ///
    /// Returns a mutable reference to the component storage.
    ///
    /// Uses `RwLock::get_mut()` which bypasses locking via the borrow checker
    /// (`&mut RwLock` guarantees no other references exist).
    fn storage_mut(&mut self, type_id: &TypeId) -> Option<&mut ComponentStorage> {
        self.components.get_mut(type_id).map(|lock| lock.get_mut())
    }

    // ---- Entity management ----

    /// Spawns a new entity and returns its ID.
    pub fn spawn(&mut self) -> Entity {
        let tick = self.current_tick();
        self.entities.allocate(tick)
    }

    /// Despawns an entity, removing all its components.
    ///
    /// Returns `true` if the entity was alive and is now despawned.
    /// Returns `false` if the entity was already dead.
    /// Records removals for [`removed`](World::removed) filter queries.
    ///
    /// `on_remove` hooks fire before removal (entity still alive, own
    /// component still readable) and exactly once per component; hook-less
    /// components stay readable for the whole hook phase (see #43 for
    /// ordering caveats between hooked components).
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }

        let index = entity.index();
        let tick = self.current_tick();

        // Pass 1: collect the components with on_remove hooks. Only the
        // TypeIds are snapshotted — presence is re-checked per hook, so a
        // hook that cascades a removal (e.g. removes another component of
        // this entity) does not cause that component's hooks to fire twice.
        let hooked: smallvec::SmallVec<[TypeId; 4]> = self
            .components
            .iter_mut()
            .filter_map(|(type_id, lock)| {
                let s = lock.get_mut();
                (!s.on_remove.is_empty() && s.contains_untyped(index)).then_some(*type_id)
            })
            .collect();

        // Pass 2: per hooked component — fire its hooks (re-validating
        // liveness and presence between calls), then remove it immediately
        // so a cascading remove() from a later hook cannot re-fire it (#43).
        for type_id in hooked {
            let hooks = match self.storage_mut(&type_id) {
                Some(s) => s.on_remove.clone(),
                None => continue,
            };
            if !self.fire_hooks_checked(entity, type_id, &hooks) {
                return true; // already fully despawned by a hook
            }
            let removed = self.storage_mut(&type_id).is_some_and(|s| {
                if s.remove_untyped(index) {
                    s.record_removal(index, tick);
                    true
                } else {
                    false
                }
            });
            if removed && let Some(trigger_key) = self.observers.remove_trigger_key(&type_id) {
                self.observers.push_trigger(trigger_key, entity);
            }
        }

        // Pass 3: Deallocate entity, remove all components, and queue
        // observer triggers in a single pass.
        self.entities.deallocate(entity);
        let components = &mut self.components;
        let observers = &mut self.observers;
        for (type_id, lock) in components.iter_mut() {
            let storage = lock.get_mut();
            if storage.remove_untyped(index) {
                storage.record_removal(index, tick);
                if let Some(trigger_key) = observers.remove_trigger_key(type_id) {
                    observers.push_trigger(trigger_key, entity);
                }
            }
        }

        true
    }

    /// Returns whether the entity is currently alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Returns the number of alive entities.
    pub fn entity_count(&self) -> u32 {
        self.entities.count()
    }

    /// Returns the alive entity at the given slot index, or `None` if the
    /// slot is empty or has been recycled.
    pub fn entity_at_index(&self, index: u32) -> Option<Entity> {
        self.entities.entity_at_index(index)
    }

    /// Iterates over all currently alive entity IDs.
    pub fn iter_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.entities.iter_alive()
    }
}

impl Drop for World {
    fn drop(&mut self) {
        // Decrement registration counts in the shared registry for each generation
        // this world registered types under. This marks generations as dead when
        // the last world referencing them is dropped (Phase 6 blocker fix).
        let generations_to_decrement: Vec<u32> = self.worlds_generations.keys().copied().collect();
        let _ = self.with_generation_registry_mut(|reg| {
            for generation_id in generations_to_decrement {
                reg.decrement_registration_count(generation_id);
            }
        });
    }
}

/// Swaps the trigger buffer for marker type `M`.
///
/// Used as a monomorphized function pointer stored in `World::trigger_swap_fns`.
fn swap_trigger_buffer<M: 'static>(world: &mut World) {
    world.resource_mut::<Triggers<M>>().swap();
}

/// Advances the event double buffer for event type `T`.
///
/// Used as a monomorphized function pointer stored in `World::event_update_fns`.
fn update_event_buffer<T: Send + Sync + 'static>(world: &mut World) {
    world.resource_mut::<crate::events::Events<T>>().update();
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}
