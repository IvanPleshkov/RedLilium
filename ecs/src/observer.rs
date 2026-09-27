use crate::SourceId;
use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, Location, catch_unwind};

use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// Trigger marker types (used as TypeId keys internally)
// ---------------------------------------------------------------------------

/// Observer trigger marker for first-time component addition.
///
/// Fires when a component is added to an entity that did not previously
/// have it. Does **not** fire on replacement of an existing component.
///
/// Used as a type parameter with observer and trigger APIs:
/// - `world.observe_add::<Health>(handler)` — register an observer
/// - `world.enable_add_triggers::<Health>()` — enable trigger buffer
/// - `Res<Triggers<OnAdd<Health>>>` — read triggered entities in systems
pub struct OnAdd<T: 'static>(PhantomData<T>);

/// Observer trigger marker for every component insertion.
///
/// Fires on both first-time addition and replacement of an existing value.
///
/// Used as a type parameter with observer and trigger APIs:
/// - `world.observe_insert::<Health>(handler)` — register an observer
/// - `world.enable_insert_triggers::<Health>()` — enable trigger buffer
/// - `Res<Triggers<OnInsert<Health>>>` — read triggered entities in systems
pub struct OnInsert<T: 'static>(PhantomData<T>);

/// Observer trigger marker for component removal (including despawn).
///
/// Fires when a component is removed from an entity, either explicitly
/// via `remove()` or implicitly via `despawn()`.
///
/// Used as a type parameter with observer and trigger APIs:
/// - `world.observe_remove::<Health>(handler)` — register an observer
/// - `world.enable_remove_triggers::<Health>()` — enable trigger buffer
/// - `Res<Triggers<OnRemove<Health>>>` — read triggered entities in systems
pub struct OnRemove<T: 'static>(PhantomData<T>);

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A deferred observer failure. Text metadata is owned and may outlive the
/// module that registered the handler. Partial world mutations are retained.
#[derive(Debug, Clone)]
pub enum ObserverError {
    /// One invocation panicked. Other handlers and triggers continue.
    Panicked {
        source: SourceId,
        trigger: String,
        entity: Entity,
        message: String,
        file: String,
        line: u32,
        column: u32,
    },
    /// The cascade still had work after the limit. Pending triggers were
    /// discarded; registrations remain available for newly generated events.
    CascadeLimitExceeded {
        iterations: u32,
        discarded_triggers: usize,
    },
}

impl std::fmt::Display for ObserverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Panicked {
                source,
                trigger,
                entity,
                message,
                file,
                line,
                column,
            } => write!(
                f,
                "observer for {trigger} on {entity:?} (source {source:?}, registered at {file}:{line}:{column}) panicked: {message}"
            ),
            Self::CascadeLimitExceeded {
                iterations,
                discarded_triggers,
            } => write!(
                f,
                "observer cascade exceeded {iterations} iterations; discarded {discarded_triggers} pending triggers"
            ),
        }
    }
}
impl std::error::Error for ObserverError {}

/// A type-erased observer whose panic boundary lives in its originating image.
type ObserverFn = Box<dyn Fn(&mut World, Entity) -> Result<(), ObserverError> + Send + Sync>;
type HandlerMap = HashMap<TypeId, Vec<(SourceId, ObserverFn)>>;

#[track_caller]
fn shield<Trigger: 'static>(
    source: SourceId,
    handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
) -> ObserverFn {
    let location = Location::caller();
    // Monomorphize the catch beside the guest handler, before type erasure.
    // Only owned reports cross the image boundary, never the panic payload.
    Box::new(move |world, entity| {
        catch_unwind(AssertUnwindSafe(|| {
            // Registrations created by a guest callback belong to that guest,
            // including when the callback subsequently panics.
            world.with_registration_source(source, |world| handler(world, entity));
        }))
        .map_err(|payload| ObserverError::Panicked {
            source,
            trigger: std::any::type_name::<Trigger>().to_owned(),
            entity,
            message: crate::system::panic_payload_to_string(&*payload),
            file: location.file().to_owned(),
            line: location.line(),
            column: location.column(),
        })
    })
}

/// A queued trigger waiting to fire its observers.
pub(crate) struct PendingTrigger {
    /// The TypeId of the trigger marker (e.g., `TypeId::of::<OnAdd<Health>>()`).
    observer_key: TypeId,
    /// The entity involved in the trigger.
    entity: Entity,
}

/// Registry of deferred observers and their pending triggers.
///
/// Stored inside [`World`]. Observers are registered during setup and
/// triggered during mutations (insert/remove/despawn). Pending triggers
/// are flushed by the runner after command application.
///
/// Each handler is stamped with the [`SourceId`](crate::type_identity::SourceId)
/// that was current at registration, so a game-module unload can purge the
/// closures whose code lives in the module's image (see
/// [`World::purge_source`](crate::World::purge_source)).
pub(crate) struct Observers {
    /// Observer handlers keyed by trigger marker TypeId, each stamped with
    /// its registration source.
    handlers: HandlerMap,
    flushing: bool,
    /// Queued triggers waiting to be flushed.
    pending: Vec<PendingTrigger>,
    /// Maps component `TypeId` → `OnRemove<T>` trigger `TypeId`.
    ///
    /// Needed by `despawn()` which iterates component storages without
    /// knowing the concrete component type parameter.
    remove_trigger_keys: HashMap<TypeId, TypeId>,
    /// Set of trigger TypeIds that have registered observers.
    ///
    /// Separate from `handlers` so that `push_trigger` works correctly
    /// during `flush()`, when `handlers` is temporarily taken out.
    registered_keys: HashSet<TypeId>,
}

impl Observers {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            flushing: false,
            pending: Vec::new(),
            remove_trigger_keys: HashMap::new(),
            registered_keys: HashSet::new(),
        }
    }

    /// Registers an observer for `OnAdd<T>`, stamped with its registration
    /// source (the world's `current_source` at call time).
    #[track_caller]
    pub fn add_on_add<T: 'static>(
        &mut self,
        source: crate::type_identity::SourceId,
        handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
    ) {
        let key = TypeId::of::<OnAdd<T>>();
        self.registered_keys.insert(key);
        self.handlers
            .entry(key)
            .or_default()
            .push((source, shield::<OnAdd<T>>(source, handler)));
    }

    /// Registers an observer for `OnInsert<T>`, stamped with its registration
    /// source.
    #[track_caller]
    pub fn add_on_insert<T: 'static>(
        &mut self,
        source: crate::type_identity::SourceId,
        handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
    ) {
        let key = TypeId::of::<OnInsert<T>>();
        self.registered_keys.insert(key);
        self.handlers
            .entry(key)
            .or_default()
            .push((source, shield::<OnInsert<T>>(source, handler)));
    }

    /// Registers an observer for `OnRemove<T>`, also recording the
    /// component→trigger mapping needed for untyped despawn iteration.
    #[track_caller]
    pub fn add_on_remove<T: 'static>(
        &mut self,
        source: crate::type_identity::SourceId,
        handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
    ) {
        let key = TypeId::of::<OnRemove<T>>();
        self.remove_trigger_keys.insert(TypeId::of::<T>(), key);
        self.registered_keys.insert(key);
        self.handlers
            .entry(key)
            .or_default()
            .push((source, shield::<OnRemove<T>>(source, handler)));
    }

    /// Drops every handler registered under `source` (a game-module unload:
    /// their closures' code lives in the module's image). Trigger keys whose
    /// handler list empties are unregistered entirely.
    pub fn purge_source(&mut self, source: crate::type_identity::SourceId) {
        let mut emptied: Vec<TypeId> = Vec::new();
        for (key, fns) in self.handlers.iter_mut() {
            fns.retain(|(s, _)| *s != source);
            if fns.is_empty() {
                emptied.push(*key);
            }
        }
        for key in emptied {
            self.handlers.remove(&key);
            self.registered_keys.remove(&key);
            self.remove_trigger_keys.retain(|_, v| *v != key);
        }
        // A future registration must not receive triggers queued for a source
        // that was completely removed. Keep events for surviving handlers.
        self.pending
            .retain(|trigger| self.registered_keys.contains(&trigger.observer_key));
    }

    /// Pushes a trigger for a known marker TypeId.
    ///
    /// Only pushes if observers are registered for this trigger type.
    /// Uses `registered_keys` (not `handlers`) so this works correctly
    /// during `flush()` when handlers are temporarily taken out.
    pub fn push_trigger(&mut self, observer_key: TypeId, entity: Entity) {
        if self.registered_keys.contains(&observer_key) {
            self.pending.push(PendingTrigger {
                observer_key,
                entity,
            });
        }
    }

    /// Pushes a typed trigger.
    ///
    /// Only pushes if observers exist for this trigger type.
    pub fn push_typed_trigger<Trigger: 'static>(&mut self, entity: Entity) {
        self.push_trigger(TypeId::of::<Trigger>(), entity);
    }

    /// Returns the OnRemove trigger TypeId for a component TypeId, if any.
    pub fn remove_trigger_key(&self, component_type_id: &TypeId) -> Option<TypeId> {
        self.remove_trigger_keys.get(component_type_id).copied()
    }

    pub(crate) fn is_flushing(&self) -> bool {
        self.flushing
    }

    /// Returns `true` if there are pending triggers.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// Owns the temporarily detached handlers and restores them on every exit.
/// The separate fields allow safe disjoint borrows of callbacks and the world.
struct FlushGuard<'a> {
    world: &'a mut World,
    handlers: HandlerMap,
}
impl FlushGuard<'_> {
    fn merge_registrations(&mut self) {
        let newly_added = std::mem::take(&mut self.world.observers.handlers);
        for (key, new_fns) in newly_added {
            self.handlers.entry(key).or_default().extend(new_fns);
        }
    }
}
impl Drop for FlushGuard<'_> {
    fn drop(&mut self) {
        self.merge_registrations();
        self.world.observers.handlers = std::mem::take(&mut self.handlers);
        self.world.observers.flushing = false;
    }
}

/// Processes at most 100 waves. Nested flushes leave queued triggers for the
/// active outer flush. New registrations participate starting with the next
/// wave. At the limit, pending triggers are discarded and reported.
pub(crate) fn flush(world: &mut World) -> Vec<ObserverError> {
    const MAX_ITERATIONS: u32 = 100;
    if world.observers.flushing || !world.observers.has_pending() {
        return Vec::new();
    }
    world.observers.flushing = true;
    let handlers = std::mem::take(&mut world.observers.handlers);
    let mut guard = FlushGuard { world, handlers };
    let mut errors = Vec::new();
    for _ in 0..MAX_ITERATIONS {
        let triggers = std::mem::take(&mut guard.world.observers.pending);
        if triggers.is_empty() {
            return errors;
        }
        for trigger in triggers {
            if let Some(fns) = guard.handlers.get(&trigger.observer_key) {
                for (_, handler) in fns {
                    if let Err(error) = handler(guard.world, trigger.entity) {
                        errors.push(error);
                    }
                }
            }
        }
        guard.merge_registrations();
    }
    let pending = &mut guard.world.observers.pending;
    if !pending.is_empty() {
        errors.push(ObserverError::CascadeLimitExceeded {
            iterations: MAX_ITERATIONS,
            discarded_triggers: pending.len(),
        });
        pending.clear();
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Debug, Clone, PartialEq)]
    struct Health(u32);

    #[derive(Debug, Clone, PartialEq)]
    struct Armor(u32);

    #[test]
    fn on_add_fires_on_insert() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_add::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        // Trigger is queued but not fired yet
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Flush fires the observer
        assert!(world.flush_observers().is_empty());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn on_add_does_not_fire_on_replace() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_add::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();
        world.insert(entity, Health(200)).unwrap(); // replace

        assert!(world.flush_observers().is_empty());
        // Only one OnAdd, not two
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn on_insert_fires_on_add_and_replace() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_insert::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap(); // add
        world.insert(entity, Health(200)).unwrap(); // replace

        assert!(world.flush_observers().is_empty());
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn on_remove_fires_on_remove() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_remove::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();
        let _ = world.remove::<Health>(entity);

        assert!(world.flush_observers().is_empty());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn on_remove_fires_on_despawn() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_remove::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();
        world.despawn(entity);

        assert!(world.flush_observers().is_empty());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn multiple_observers_per_trigger() {
        let counter = Arc::new(AtomicU32::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_add::<Health>(move |_world, _entity| {
            c1.fetch_add(1, Ordering::SeqCst);
        });
        world.observe_add::<Health>(move |_world, _entity| {
            c2.fetch_add(10, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        assert!(world.flush_observers().is_empty());
        assert_eq!(counter.load(Ordering::SeqCst), 11);
    }

    #[test]
    fn no_triggers_when_no_observers() {
        let mut world = World::new();
        world.register_component::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        // Should not panic, no pending triggers
        assert!(world.flush_observers().is_empty());
    }

    #[test]
    fn observer_can_read_component() {
        let value = Arc::new(AtomicU32::new(0));
        let value_clone = value.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_add::<Health>(move |world, entity| {
            let health = world.get::<Health>(entity).unwrap();
            value_clone.store(health.0, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(42)).unwrap();
        assert!(world.flush_observers().is_empty());

        assert_eq!(value.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn cascading_observers() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.register_component::<Armor>();

        // When Health is added, also add Armor
        world.observe_add::<Health>(|world, entity| {
            let _ = world.insert(entity, Armor(50));
        });

        // When Armor is added, increment counter
        world.observe_add::<Armor>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();
        assert!(world.flush_observers().is_empty());

        // Health observer added Armor, which triggered Armor observer
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(world.get::<Armor>(entity), Some(&Armor(50)));
    }

    #[test]
    fn infinite_cascade_returns_an_error() {
        let mut world = World::new();
        world.register_component::<Health>();

        // Observer that re-inserts the same component, causing infinite cascade
        world.observe_insert::<Health>(|world, entity| {
            let _ = world.insert(entity, Health(999));
        });

        let entity = world.spawn();
        world.insert(entity, Health(1)).unwrap();
        assert!(matches!(
            world.flush_observers().as_slice(),
            [ObserverError::CascadeLimitExceeded {
                iterations: 100,
                discarded_triggers: 1
            }]
        ));
        assert!(world.flush_observers().is_empty());
    }

    #[test]
    fn batch_insert_fires_triggers() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_add::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entities: Vec<_> = (0..5).map(|_| world.spawn()).collect();
        let components: Vec<_> = (0..5).map(|i| Health(i * 10)).collect();
        world
            .insert_batch(entities.iter().copied().zip(components))
            .unwrap();

        assert!(world.flush_observers().is_empty());
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn remove_batch_fires_triggers() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_remove::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entities: Vec<_> = (0..3).map(|_| world.spawn()).collect();
        for &e in &entities {
            world.insert(e, Health(100)).unwrap();
        }
        assert!(world.flush_observers().is_empty()); // flush any pending (none for remove)

        world.remove_batch::<Health>(&entities);
        assert!(world.flush_observers().is_empty());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn despawn_batch_fires_triggers() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.register_component::<Armor>();

        world.observe_remove::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        // No observer for Armor removal — should not trigger anything

        let entities: Vec<_> = (0..3).map(|_| world.spawn()).collect();
        for &e in &entities {
            world.insert(e, Health(100)).unwrap();
            world.insert(e, Armor(50)).unwrap();
        }

        world.despawn_batch(&entities);
        assert!(world.flush_observers().is_empty());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn flush_is_idempotent() {
        let mut world = World::new();
        world.register_component::<Health>();

        // Flush with nothing pending — should not panic
        assert!(world.flush_observers().is_empty());
        assert!(world.flush_observers().is_empty());
    }

    #[test]
    fn insert_fires_triggers() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_add::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        assert!(world.flush_observers().is_empty());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn different_component_types_independent() {
        let health_count = Arc::new(AtomicU32::new(0));
        let armor_count = Arc::new(AtomicU32::new(0));
        let hc = health_count.clone();
        let ac = armor_count.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.register_component::<Armor>();

        world.observe_add::<Health>(move |_world, _entity| {
            hc.fetch_add(1, Ordering::SeqCst);
        });
        world.observe_add::<Armor>(move |_world, _entity| {
            ac.fetch_add(1, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        assert!(world.flush_observers().is_empty());
        assert_eq!(health_count.load(Ordering::SeqCst), 1);
        assert_eq!(armor_count.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn panic_keeps_handlers_and_new_registrations_start_next_wave() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.insert_resource(Vec::<&'static str>::new());
        let first = world.spawn();
        let second = world.spawn();
        let line = line!() + 1;
        world.observe_insert::<Health>(move |world, entity| {
            world.resource_mut::<Vec<&str>>().push("old");
            if world.resource::<Vec<&str>>().len() == 1 {
                world.observe_insert::<Health>(|world, _| {
                    world.resource_mut::<Vec<&str>>().push("new");
                });
                world.insert(entity, Health(2)).unwrap();
                panic!("observer failure");
            }
        });
        world.insert(first, Health(1)).unwrap();
        world.insert(second, Health(1)).unwrap();
        let errors = world.flush_observers();
        assert_eq!(errors.len(), 1);
        match &errors[0] {
            ObserverError::Panicked {
                entity,
                message,
                file,
                line: actual_line,
                source,
                trigger,
                ..
            } => {
                assert_eq!(*entity, first);
                assert_eq!(message, "observer failure");
                assert_eq!(file, file!());
                assert_eq!(*actual_line, line);
                assert_eq!(*source, SourceId::HOST);
                assert!(trigger.contains("OnInsert<"));
            }
            error => panic!("wrong error: {error}"),
        }
        assert_eq!(
            &*world.resource::<Vec<&str>>(),
            &["old", "old", "old", "new"]
        );
        world.insert(first, Health(3)).unwrap();
        assert!(world.flush_observers().is_empty());
        assert_eq!(&world.resource::<Vec<&str>>()[4..], &["old", "new"]);
    }

    #[test]
    fn nested_flush_leaves_work_to_the_outer_cascade() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.insert_resource(0u32);
        world.observe_insert::<Health>(|world, entity| {
            *world.resource_mut::<u32>() += 1;
            let value = world.get::<Health>(entity).unwrap().0;
            if value < 3 {
                world.insert(entity, Health(value + 1)).unwrap();
                assert!(world.flush_observers().is_empty());
            }
        });
        let entity = world.spawn();
        world.insert(entity, Health(1)).unwrap();
        assert!(world.flush_observers().is_empty());
        assert_eq!(*world.resource::<u32>(), 3);
        assert_eq!(world.get::<Health>(entity).unwrap().0, 3);
    }

    #[test]
    fn cascade_limit_discards_pending_work_without_losing_registrations() {
        for limit in [100, 101] {
            let mut world = World::new();
            world.register_component::<Health>();
            world.insert_resource(0u32);
            world.observe_insert::<Health>(move |world, entity| {
                *world.resource_mut::<u32>() += 1;
                let value = world.get::<Health>(entity).unwrap().0;
                if value < limit {
                    world.insert(entity, Health(value + 1)).unwrap();
                }
            });
            let entity = world.spawn();
            world.insert(entity, Health(1)).unwrap();
            let errors = world.flush_observers();
            if limit == 100 {
                assert!(errors.is_empty());
            } else {
                assert!(matches!(
                    errors.as_slice(),
                    [ObserverError::CascadeLimitExceeded {
                        iterations: 100,
                        discarded_triggers: 1
                    }]
                ));
            }
            assert_eq!(*world.resource::<u32>(), 100);
            assert!(world.flush_observers().is_empty());
            assert_eq!(*world.resource::<u32>(), 100); // No replay next frame.
            world.insert(entity, Health(limit)).unwrap();
            assert!(world.flush_observers().is_empty());
            assert_eq!(*world.resource::<u32>(), 101);
        }
    }

    #[test]
    fn nested_registrations_inherit_source_and_are_purged_after_panic() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.register_component::<Armor>();
        let source = SourceId(7);
        let calls = Arc::new(AtomicU32::new(0));
        let tracked = calls.clone();
        world.with_registration_source(source, |world| {
            world.observe_insert::<Health>(move |world, entity| {
                let tracked = tracked.clone();
                world.observe_insert::<Armor>(move |_, _| {
                    tracked.fetch_add(1, Ordering::SeqCst);
                });
                world.insert(entity, Armor(2)).unwrap();
                panic!("after registration");
            });
        });
        let entity = world.spawn();
        world.insert(entity, Health(1)).unwrap();
        assert_eq!(world.flush_observers().len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(world.current_source(), SourceId::HOST);
        world.purge_source(source);
        world.insert(entity, Health(2)).unwrap();
        world.insert(entity, Armor(3)).unwrap();
        assert!(world.flush_observers().is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn purge_inside_callback_is_rejected_before_mutation() {
        let mut world = World::new();
        world.register_component::<Health>();
        let source = SourceId(7);
        world.with_registration_source(source, |world| {
            world.observe_insert::<Health>(move |world, _| world.purge_source(source));
        });
        let entity = world.spawn();
        for value in [1, 2] {
            world.insert(entity, Health(value)).unwrap();
            let errors = world.flush_observers();
            assert!(
                matches!(&errors[..], [ObserverError::Panicked { message, .. }] if message.contains("purge_source during observer flush"))
            );
        }
        world.purge_source(source);
        world.insert(entity, Health(3)).unwrap();
        assert!(world.flush_observers().is_empty());
    }
    #[test]
    fn purge_discards_orphaned_triggers_before_new_registration() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.with_registration_source(SourceId(9), |world| {
            world.observe_insert::<Health>(|_, _| panic!("purged"));
        });
        let entity = world.spawn();
        world.insert(entity, Health(1)).unwrap();
        world.purge_source(SourceId(9));
        let calls = Arc::new(AtomicU32::new(0));
        let tracked = calls.clone();
        world.observe_insert::<Health>(move |_, _| {
            tracked.fetch_add(1, Ordering::SeqCst);
        });
        assert!(world.flush_observers().is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        world.insert(entity, Health(2)).unwrap();
        assert!(world.flush_observers().is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
