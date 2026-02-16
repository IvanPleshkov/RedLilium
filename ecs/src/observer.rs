use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

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

/// A type-erased observer handler.
type ObserverFn = Box<dyn Fn(&mut World, Entity) + Send + Sync>;

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
pub(crate) struct Observers {
    /// Observer handlers keyed by trigger marker TypeId.
    handlers: HashMap<TypeId, Vec<ObserverFn>>,
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
            pending: Vec::new(),
            remove_trigger_keys: HashMap::new(),
            registered_keys: HashSet::new(),
        }
    }

    /// Registers an observer for `OnAdd<T>`.
    pub fn add_on_add<T: 'static>(
        &mut self,
        handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
    ) {
        let key = TypeId::of::<OnAdd<T>>();
        self.registered_keys.insert(key);
        self.handlers
            .entry(key)
            .or_default()
            .push(Box::new(handler));
    }

    /// Registers an observer for `OnInsert<T>`.
    pub fn add_on_insert<T: 'static>(
        &mut self,
        handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
    ) {
        let key = TypeId::of::<OnInsert<T>>();
        self.registered_keys.insert(key);
        self.handlers
            .entry(key)
            .or_default()
            .push(Box::new(handler));
    }

    /// Registers an observer for `OnRemove<T>`, also recording the
    /// component→trigger mapping needed for untyped despawn iteration.
    pub fn add_on_remove<T: 'static>(
        &mut self,
        handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
    ) {
        let key = TypeId::of::<OnRemove<T>>();
        self.remove_trigger_keys.insert(TypeId::of::<T>(), key);
        self.registered_keys.insert(key);
        self.handlers
            .entry(key)
            .or_default()
            .push(Box::new(handler));
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

    /// Returns `true` if there are pending triggers.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Drains and fires all pending triggers, supporting cascading.
    ///
    /// Observers that perform mutations (insert/remove/despawn) will queue
    /// new triggers. This method loops until no more triggers remain.
    ///
    /// # Panics
    ///
    /// Panics if cascading exceeds 100 iterations (likely infinite loop).
    pub fn flush(&mut self, world_ptr: *mut World) {
        const MAX_ITERATIONS: u32 = 100;

        for iteration in 0..MAX_ITERATIONS {
            let triggers = std::mem::take(&mut self.pending);
            if triggers.is_empty() {
                return;
            }

            // Take handlers out to release borrow on self, allowing
            // the handler closure to receive `&mut World` (which contains self).
            let handlers = std::mem::take(&mut self.handlers);

            for trigger in &triggers {
                if let Some(fns) = handlers.get(&trigger.observer_key) {
                    for f in fns {
                        // SAFETY: world_ptr points to the World that owns this Observers.
                        // We took `handlers` out of self, so the World can be mutably borrowed.
                        // The handler may push new triggers to self.pending (via World mutations),
                        // which is fine since we already took the current triggers.
                        unsafe {
                            f(&mut *world_ptr, trigger.entity);
                        }
                    }
                }
            }

            // Put handlers back, merging any newly registered observers
            // that were added during handler execution.
            let newly_added = std::mem::replace(&mut self.handlers, handlers);
            for (key, new_fns) in newly_added {
                self.handlers.entry(key).or_default().extend(new_fns);
            }

            if iteration == MAX_ITERATIONS - 1 {
                panic!(
                    "Observer cascade exceeded {MAX_ITERATIONS} iterations. \
                     This likely indicates an infinite loop where observers \
                     continuously trigger each other."
                );
            }
        }
    }
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
        world.flush_observers();
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

        world.flush_observers();
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

        world.flush_observers();
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
        world.remove::<Health>(entity);

        world.flush_observers();
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

        world.flush_observers();
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

        world.flush_observers();
        assert_eq!(counter.load(Ordering::SeqCst), 11);
    }

    #[test]
    fn no_triggers_when_no_observers() {
        let mut world = World::new();
        world.register_component::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        // Should not panic, no pending triggers
        world.flush_observers();
        assert!(!world.has_pending_observers());
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
        world.flush_observers();

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
        world.flush_observers();

        // Health observer added Armor, which triggered Armor observer
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(world.get::<Armor>(entity), Some(&Armor(50)));
    }

    #[test]
    #[should_panic(expected = "Observer cascade exceeded")]
    fn cascade_limit_panics() {
        let mut world = World::new();
        world.register_component::<Health>();

        // Observer that re-inserts the same component, causing infinite cascade
        world.observe_insert::<Health>(|world, entity| {
            let _ = world.insert(entity, Health(999));
        });

        let entity = world.spawn();
        world.insert(entity, Health(1)).unwrap();
        world.flush_observers();
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
        world.insert_batch(&entities, components).unwrap();

        world.flush_observers();
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
        world.flush_observers(); // flush any pending (none for remove)

        world.remove_batch::<Health>(&entities);
        world.flush_observers();
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
        world.flush_observers();
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn flush_is_idempotent() {
        let mut world = World::new();
        world.register_component::<Health>();

        // Flush with nothing pending — should not panic
        world.flush_observers();
        world.flush_observers();
    }

    #[test]
    fn insert_tracked_fires_triggers() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Health>();
        world.observe_add::<Health>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let entity = world.spawn();
        world.insert_tracked(entity, Health(100)).unwrap();

        world.flush_observers();
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

        world.flush_observers();
        assert_eq!(health_count.load(Ordering::SeqCst), 1);
        assert_eq!(armor_count.load(Ordering::SeqCst), 0);
    }
}
