use std::marker::PhantomData;

use crate::access_set::Res;
use crate::condition::Condition;
use crate::entity::Entity;
use crate::system::SystemError;
use crate::system_context::SystemContext;

/// Double-buffered list of triggered entities for a specific observer event.
///
/// Used with observer marker types ([`OnAdd<T>`](crate::OnAdd),
/// [`OnInsert<T>`](crate::OnInsert), [`OnRemove<T>`](crate::OnRemove))
/// to collect entities that triggered a component lifecycle event.
///
/// # Lifecycle
///
/// - During `flush_observers()`, internal observers push entities to the
///   `collecting` buffer.
/// - At the start of the next tick, the runner calls `update_triggers()`,
///   which swaps `collecting` → `readable`.
/// - Systems read `readable` via `Res<Triggers<OnAdd<Health>>>`.
///
/// This means systems always see **last tick's** triggered entities.
///
/// # Example
///
/// ```ignore
/// // Setup
/// world.register_component::<Health>();
/// world.enable_add_triggers::<Health>();
///
/// // In a system
/// ctx.lock::<(Res<Triggers<OnAdd<Health>>>,)>()
///     .execute(|(triggers,)| {
///         for &entity in triggers.iter() {
///             // entity had Health added last tick
///         }
///     });
/// ```
pub struct Triggers<M: 'static> {
    /// Entities readable by systems (populated from last tick's observer flush).
    readable: Vec<Entity>,
    /// Entities being collected during current tick's observer flush.
    collecting: Vec<Entity>,
    _marker: PhantomData<M>,
}

impl<M: 'static> Triggers<M> {
    /// Creates a new empty trigger buffer.
    pub fn new() -> Self {
        Self {
            readable: Vec::new(),
            collecting: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Iterates over triggered entities from the previous tick.
    pub fn iter(&self) -> impl Iterator<Item = &Entity> {
        self.readable.iter()
    }

    /// Returns triggered entities from the previous tick as a slice.
    pub fn entities(&self) -> &[Entity] {
        &self.readable
    }

    /// Returns `true` if no entities were triggered last tick.
    pub fn is_empty(&self) -> bool {
        self.readable.is_empty()
    }

    /// Returns the number of triggered entities from last tick.
    pub fn len(&self) -> usize {
        self.readable.len()
    }

    /// Pushes a triggered entity into the collecting buffer.
    ///
    /// Called by internal observer handlers during `flush_observers()`.
    pub(crate) fn push(&mut self, entity: Entity) {
        self.collecting.push(entity);
    }

    /// Swaps buffers: `collecting` becomes `readable`, old `readable` is cleared.
    ///
    /// Called by the runner at the start of each tick via `World::update_triggers()`.
    pub(crate) fn swap(&mut self) {
        self.readable.clear();
        std::mem::swap(&mut self.readable, &mut self.collecting);
    }
}

impl<M: 'static> Default for Triggers<M> {
    fn default() -> Self {
        Self::new()
    }
}

/// Condition system that gates downstream systems on non-empty [`Triggers<M>`].
///
/// Returns [`Condition::True`] when the trigger buffer has entities,
/// [`Condition::False`] when empty. Register as a condition and add an
/// edge to reactive systems that should only run when triggers fired.
///
/// # Example
///
/// ```ignore
/// container.add_condition(HasTriggers::<OnAdd<Health>>::new());
/// container.add(OnHealthAddedSystem);
/// container.add_edge::<HasTriggers<OnAdd<Health>>, OnHealthAddedSystem>().unwrap();
/// ```
pub struct HasTriggers<M: 'static>(PhantomData<M>);

impl<M: 'static> HasTriggers<M> {
    /// Creates a new `HasTriggers` condition.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<M: 'static> Default for HasTriggers<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: Send + Sync + 'static> crate::system::System for HasTriggers<M> {
    type Result = Condition<()>;

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Condition<()>, SystemError> {
        Ok(ctx.lock::<(Res<Triggers<M>>,)>().execute(|(triggers,)| {
            if triggers.is_empty() {
                Condition::False
            } else {
                Condition::True(())
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{OnAdd, OnInsert, OnRemove};
    use crate::world::World;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Debug, Clone, PartialEq)]
    struct Health(u32);

    #[test]
    fn basic_add_triggers() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_add_triggers::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        // Before flush: collecting has nothing, readable empty
        assert!(world.resource::<Triggers<OnAdd<Health>>>().is_empty());

        // Flush observers — observer pushes to collecting
        world.flush_observers();

        // Still empty in readable (haven't swapped yet)
        assert!(world.resource::<Triggers<OnAdd<Health>>>().is_empty());

        // Simulate next tick start: swap buffers
        world.update_triggers();

        // Now readable has the entity
        let triggers = world.resource::<Triggers<OnAdd<Health>>>();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers.entities()[0], entity);
    }

    #[test]
    fn multiple_entities_triggers() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_add_triggers::<Health>();

        let entities: Vec<_> = (0..5).map(|_| world.spawn()).collect();
        let components: Vec<_> = (0..5).map(|i| Health(i * 10)).collect();
        world.insert_batch(&entities, components).unwrap();

        world.flush_observers();
        world.update_triggers();

        let triggers = world.resource::<Triggers<OnAdd<Health>>>();
        assert_eq!(triggers.len(), 5);
        for (i, &e) in triggers.iter().enumerate() {
            assert_eq!(e, entities[i]);
        }
    }

    #[test]
    fn swap_clears_previous() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_add_triggers::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();
        world.flush_observers();
        world.update_triggers();

        assert_eq!(world.resource::<Triggers<OnAdd<Health>>>().len(), 1);

        // Second tick with no new mutations
        world.update_triggers();

        // Previous tick's triggers are gone
        assert!(world.resource::<Triggers<OnAdd<Health>>>().is_empty());
    }

    #[test]
    fn on_insert_triggers_on_add_and_replace() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_insert_triggers::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap(); // add
        world.insert(entity, Health(200)).unwrap(); // replace

        world.flush_observers();
        world.update_triggers();

        let triggers = world.resource::<Triggers<OnInsert<Health>>>();
        assert_eq!(triggers.len(), 2);
    }

    #[test]
    fn on_remove_triggers_on_remove() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_remove_triggers::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();
        world.remove::<Health>(entity);

        world.flush_observers();
        world.update_triggers();

        let triggers = world.resource::<Triggers<OnRemove<Health>>>();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers.entities()[0], entity);
    }

    #[test]
    fn on_remove_triggers_on_despawn() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_remove_triggers::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();
        world.despawn(entity);

        world.flush_observers();
        world.update_triggers();

        let triggers = world.resource::<Triggers<OnRemove<Health>>>();
        assert_eq!(triggers.len(), 1);
    }

    #[test]
    fn has_triggers_condition_true_when_non_empty() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_add_triggers::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();
        world.flush_observers();
        world.update_triggers();

        let condition = HasTriggers::<OnAdd<Health>>::new();
        let compute = crate::compute::ComputePool::new(crate::io_runtime::IoRuntime::new());
        let io = crate::io_runtime::IoRuntime::new();
        let result = crate::system::run_system_blocking(&condition, &world, &compute, &io).unwrap();
        assert!(result.is_true());
    }

    #[test]
    fn has_triggers_condition_false_when_empty() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_add_triggers::<Health>();

        // No mutations
        world.update_triggers();

        let condition = HasTriggers::<OnAdd<Health>>::new();
        let compute = crate::compute::ComputePool::new(crate::io_runtime::IoRuntime::new());
        let io = crate::io_runtime::IoRuntime::new();
        let result = crate::system::run_system_blocking(&condition, &world, &compute, &io).unwrap();
        assert!(result.is_false());
    }

    #[test]
    fn empty_when_no_mutations() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_add_triggers::<Health>();

        // Several ticks with no mutations
        for _ in 0..5 {
            world.flush_observers();
            world.update_triggers();
        }

        assert!(world.resource::<Triggers<OnAdd<Health>>>().is_empty());
    }

    #[test]
    fn integration_runner_test() {
        use crate::access_set::Res;
        use crate::systems_container::SystemsContainer;

        struct ReactiveSystem(Arc<AtomicU32>);
        impl crate::system::System for ReactiveSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                ctx.lock::<(Res<Triggers<OnAdd<Health>>>,)>()
                    .execute(|(triggers,)| {
                        self.0.fetch_add(triggers.len() as u32, Ordering::SeqCst);
                    });
                Ok(())
            }
        }

        let counter = Arc::new(AtomicU32::new(0));

        let mut world = World::new();
        world.register_component::<Health>();
        world.enable_add_triggers::<Health>();

        let mut container = SystemsContainer::new();
        container.add_condition(HasTriggers::<OnAdd<Health>>::new());
        container.add(ReactiveSystem(counter.clone()));
        container
            .add_edge::<HasTriggers<OnAdd<Health>>, ReactiveSystem>()
            .unwrap();

        let runner = crate::runner::EcsRunnerSingleThread::new();

        // Tick 1: no entities, reactive system should NOT run
        runner.run(&mut world, &container);
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Add entities (between ticks)
        let e1 = world.spawn();
        world.insert(e1, Health(100)).unwrap();
        let e2 = world.spawn();
        world.insert(e2, Health(200)).unwrap();
        world.flush_observers(); // simulate end-of-tick observer flush

        // Tick 2: reactive system should run and see 2 triggers
        runner.run(&mut world, &container);
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        // Tick 3: no new entities, reactive system should NOT run
        runner.run(&mut world, &container);
        assert_eq!(counter.load(Ordering::SeqCst), 2); // unchanged
    }
}
