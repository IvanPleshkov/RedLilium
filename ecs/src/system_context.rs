use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Mutex;

use crate::access_set::{AccessInfo, AccessSet, normalize_access_infos};
use crate::bundle::Bundle;
use crate::command_collector::{CommandCollector, SpawnBuilder};
use crate::compute::ComputePool;
use crate::entity::Entity;
use crate::io_runtime::IoRuntime;
use crate::lock_request::LockRequest;
use crate::main_thread_dispatcher::MainThreadDispatcher;
use crate::query_guard::QueryGuard;
use crate::system::System;
use crate::system_results_store::SystemResultsStore;
use crate::world::World;

// ---------------------------------------------------------------------------
// Held-lock tracking for same-system deadlock detection
// ---------------------------------------------------------------------------

/// Tracks whether a lock is held as read (with count) or write.
#[derive(Clone, Copy)]
enum HeldLock {
    Read(u32),
    Write,
}

/// RAII guard that unregisters held locks from a [`SystemContext`] when dropped.
///
/// Created by [`SystemContext::make_tracking`] and stored inside [`QueryGuard`]
/// or used as a scope guard in [`LockRequest::execute`].
pub(crate) struct LockTracking<'a> {
    infos: Vec<AccessInfo>,
    held_locks: &'a Mutex<HashMap<TypeId, HeldLock>>,
}

impl Drop for LockTracking<'_> {
    fn drop(&mut self) {
        let mut held = self.held_locks.lock().unwrap();
        for info in &self.infos {
            if info.is_write {
                held.remove(&info.type_id);
            } else {
                match held.get(&info.type_id).copied() {
                    Some(HeldLock::Read(1)) => {
                        held.remove(&info.type_id);
                    }
                    Some(HeldLock::Read(n)) => {
                        held.insert(info.type_id, HeldLock::Read(n - 1));
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Context passed to systems during execution.
///
/// Provides access to component locking, compute tasks, and deferred commands.
/// Systems receive a `&SystemContext` in their [`run`](crate::System::run) method.
///
/// # Component access
///
/// Use [`lock()`](SystemContext::lock) with a tuple of access types to
/// borrow components. The tuple specifies exactly which components are
/// needed and whether each is read or written:
///
/// ```ignore
/// ctx.lock::<(Write<Position>, Read<Velocity>)>()
///     .execute(|(mut positions, velocities)| {
///         for (idx, pos) in positions.iter_mut() {
///             if let Some(vel) = velocities.get(idx) {
///                 pos.x += vel.x;
///             }
///         }
///     });
/// ```
///
/// # Deferred commands
///
/// Use [`commands()`](SystemContext::commands) for structural changes
/// (spawn, despawn, insert) that require `&mut World`. Commands are
/// applied after all systems complete.
pub struct SystemContext<'a> {
    world: &'a World,
    compute: &'a ComputePool,
    io: &'a IoRuntime,
    commands: &'a CommandCollector,
    dispatcher: Option<&'a MainThreadDispatcher>,
    /// Per-system tracking of component locks currently held (via QueryGuard
    /// or lock().execute()). Used to detect same-system deadlocks.
    held_locks: Mutex<HashMap<TypeId, HeldLock>>,
    /// Storage for results produced by completed systems.
    system_results: Option<&'a SystemResultsStore>,
    /// Set of system TypeIds whose results this system may read
    /// (determined by the dependency graph — transitive ancestors).
    accessible_results: Option<&'a HashSet<TypeId>>,
}

impl<'a> SystemContext<'a> {
    /// Creates a new system context without a main-thread dispatcher.
    ///
    /// Used by the single-threaded runner where everything already
    /// runs on the main thread.
    pub(crate) fn new(
        world: &'a World,
        compute: &'a ComputePool,
        io: &'a IoRuntime,
        commands: &'a CommandCollector,
    ) -> Self {
        Self {
            world,
            compute,
            io,
            commands,
            dispatcher: None,
            held_locks: Mutex::new(HashMap::new()),
            system_results: None,
            accessible_results: None,
        }
    }

    /// Creates a new system context with a main-thread dispatcher.
    ///
    /// Used by the multi-threaded runner to enable main-thread resource
    /// access from worker threads.
    pub(crate) fn with_dispatcher(
        world: &'a World,
        compute: &'a ComputePool,
        io: &'a IoRuntime,
        commands: &'a CommandCollector,
        dispatcher: &'a MainThreadDispatcher,
    ) -> Self {
        Self {
            world,
            compute,
            io,
            commands,
            dispatcher: Some(dispatcher),
            held_locks: Mutex::new(HashMap::new()),
            system_results: None,
            accessible_results: None,
        }
    }

    /// Sets the system results store and accessible results set for this context.
    ///
    /// Called by runners before executing each system so that `system_result()`
    /// can look up predecessor results.
    pub(crate) fn with_system_results(
        mut self,
        store: &'a SystemResultsStore,
        accessible: &'a HashSet<TypeId>,
    ) -> Self {
        self.system_results = Some(store);
        self.accessible_results = Some(accessible);
        self
    }

    /// Returns the main-thread dispatcher, if one exists.
    pub(crate) fn dispatcher(&self) -> Option<&MainThreadDispatcher> {
        self.dispatcher
    }

    /// Checks if the given normalized access infos would conflict with
    /// component locks currently held by this system. Panics if a deadlock
    /// would occur.
    ///
    /// Conflict rules:
    /// - Write held + any new lock → deadlock
    /// - Read held + new write lock → deadlock
    /// - Read held + new read lock → OK
    pub(crate) fn check_held_locks(&self, sorted: &[AccessInfo]) {
        let held = self.held_locks.lock().unwrap();
        for info in sorted {
            if let Some(&state) = held.get(&info.type_id) {
                let conflict = match state {
                    HeldLock::Write => true,
                    HeldLock::Read(_) => info.is_write,
                };
                if conflict {
                    let type_name = self
                        .world
                        .component_type_name(info.type_id)
                        .unwrap_or("<resource>");
                    let held_mode = match state {
                        HeldLock::Write => "write",
                        HeldLock::Read(_) => "read",
                    };
                    let want_mode = if info.is_write { "write" } else { "read" };
                    // Drop the lock before panicking to avoid poison
                    drop(held);
                    panic!(
                        "ECS deadlock detected: component `{type_name}` is already locked \
                         for {held_mode}, but a new {want_mode} lock was requested. \
                         Drop the existing QueryGuard before acquiring new locks, or \
                         combine all needed accesses into a single query/lock call."
                    );
                }
            }
        }
    }

    /// Registers component locks as held. Called after successful lock acquisition.
    pub(crate) fn register_held_locks(&self, sorted: &[AccessInfo]) {
        let mut held = self.held_locks.lock().unwrap();
        for info in sorted {
            // Only track component TypeIds (resources self-lock via their own RwLock)
            if self.world.component_type_name(info.type_id).is_none() {
                continue;
            }
            if info.is_write {
                held.insert(info.type_id, HeldLock::Write);
            } else {
                match held.get(&info.type_id).copied() {
                    Some(HeldLock::Read(n)) => {
                        held.insert(info.type_id, HeldLock::Read(n + 1));
                    }
                    _ => {
                        held.insert(info.type_id, HeldLock::Read(1));
                    }
                }
            }
        }
    }

    /// Creates a [`LockTracking`] guard that will unregister the given
    /// component locks when dropped.
    pub(crate) fn make_tracking(&self, sorted: &[AccessInfo]) -> LockTracking<'_> {
        let component_infos: Vec<AccessInfo> = sorted
            .iter()
            .filter(|i| self.world.component_type_name(i.type_id).is_some())
            .copied()
            .collect();
        LockTracking {
            infos: component_infos,
            held_locks: &self.held_locks,
        }
    }

    /// Creates a lock request for the given access set.
    ///
    /// The type parameter `A` is a tuple of access types that specifies
    /// which components/resources to lock and whether each is read or written.
    ///
    /// Call `.execute()` on the returned [`LockRequest`] to run a closure
    /// with the locked data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.lock::<(Write<Position>, Read<Velocity>)>()
    ///     .execute(|(mut pos, vel)| {
    ///         // use pos and vel
    ///     });
    /// ```
    pub fn lock<A: AccessSet>(&self) -> LockRequest<'_, A> {
        LockRequest {
            ctx: self,
            _marker: PhantomData,
        }
    }

    /// Acquires locks for the given access set and returns a guard holding
    /// the locked data.
    ///
    /// Unlike [`lock().execute()`](LockRequest::execute), this does not
    /// require a closure — the data is returned directly and can be used
    /// in normal control flow. Locks are held until the returned
    /// [`QueryGuard`] is dropped.
    ///
    /// Locks are acquired in TypeId-sorted order to prevent deadlocks,
    /// identical to the `lock().execute()` path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut q = ctx.query::<(Write<Position>, Read<Velocity>)>();
    /// let (positions, velocities) = &mut q.items;
    /// for (idx, pos) in positions.iter_mut() {
    ///     if let Some(vel) = velocities.get(idx) {
    ///         pos.x += vel.x;
    ///     }
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the access set contains `MainThreadRes` or `MainThreadResMut`.
    /// Use `lock().execute()` for main-thread resources.
    pub fn query<A: AccessSet>(&self) -> QueryGuard<'_, A> {
        if A::needs_main_thread() {
            panic!("query() does not support main-thread resources; use lock().execute() instead");
        }

        let infos = A::access_infos();
        let sorted = normalize_access_infos(&infos);
        self.check_held_locks(&sorted);

        let guards = self.world.acquire_sorted(&infos);

        self.register_held_locks(&sorted);
        let tracking = self.make_tracking(&sorted);
        let items = A::fetch_unlocked(self.world);
        QueryGuard::new_tracked(guards, items, tracking)
    }

    /// Returns the result produced by a predecessor system.
    ///
    /// The type parameter `S` identifies the producer system. This method
    /// returns `&S::Result` — the value that system returned from its
    /// [`run()`](System::run) method.
    ///
    /// # Access rules
    ///
    /// A system may only read results from systems that are **guaranteed**
    /// to have completed before it starts. This is determined by the
    /// dependency graph: if there is a transitive edge from `S` to the
    /// current system, the result is accessible.
    ///
    /// # Panics
    ///
    /// - If no results store is available (called outside a runner).
    /// - If system `S` is not an ancestor of the current system.
    /// - If the result has not been stored yet (should not happen with
    ///   correct dependency edges).
    pub fn system_result<S: System>(&self) -> &S::Result {
        let type_id = TypeId::of::<S>();

        let accessible = self
            .accessible_results
            .expect("system_result() called outside a runner — no results available");
        assert!(
            accessible.contains(&type_id),
            "System `{}` result is not accessible from this system — \
             add a dependency edge to guarantee it completes first",
            std::any::type_name::<S>()
        );

        let store = self
            .system_results
            .expect("system_result() called outside a runner — no results store");
        store
            .get::<S::Result>(type_id)
            .expect("system result not yet available — dependency graph error")
    }

    /// Returns a reference to the compute pool for spawning background tasks.
    pub fn compute(&self) -> &ComputePool {
        self.compute
    }

    /// Returns a reference to the IO runtime for spawning async IO tasks.
    ///
    /// Compute tasks receive an [`EcsComputeContext`](crate::EcsComputeContext)
    /// that provides IO access automatically:
    /// ```ignore
    /// ctx.compute().spawn(Priority::Low, |cctx| async move {
    ///     let data = cctx.io().run(async { fetch().await }).await;
    ///     process(data)
    /// });
    /// ```
    pub fn io(&self) -> &IoRuntime {
        self.io
    }

    /// Pushes a deferred command to be applied after all systems complete.
    ///
    /// Commands receive `&mut World` and can perform structural changes
    /// like spawning, despawning, and inserting components.
    pub fn commands(&self, cmd: impl FnOnce(&mut World) + Send + 'static) {
        self.commands.push(cmd);
    }

    /// Queues an entity despawn to be applied after all systems complete.
    pub fn despawn(&self, entity: Entity) {
        self.commands.despawn(entity);
    }

    /// Queues a component insertion to be applied after all systems complete.
    ///
    /// # Panics
    ///
    /// Panics when applied if the component type has not been registered.
    pub fn insert<T: Send + Sync + 'static>(&self, entity: Entity, component: T) {
        self.commands.insert(entity, component);
    }

    /// Queues a component removal to be applied after all systems complete.
    pub fn remove<T: Send + Sync + 'static>(&self, entity: Entity) {
        self.commands.remove::<T>(entity);
    }

    /// Begins building a spawn command with components.
    ///
    /// The entity is spawned and all components inserted when
    /// [`build()`](SpawnBuilder::build) is called.
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.spawn_entity()
    ///     .with(Transform::IDENTITY)
    ///     .with(Visibility::VISIBLE)
    ///     .build();
    /// ```
    pub fn spawn_entity(&self) -> SpawnBuilder<'_> {
        self.commands.spawn_entity()
    }

    /// Queues a bundle of components to be inserted on an entity.
    ///
    /// # Panics
    ///
    /// Panics when applied if any component type has not been registered.
    pub fn insert_bundle(&self, entity: Entity, bundle: impl Bundle) {
        self.commands.insert_bundle(entity, bundle);
    }

    /// Queues spawning a new entity with a bundle of components.
    ///
    /// # Panics
    ///
    /// Panics when applied if any component type has not been registered.
    pub fn spawn_with(&self, bundle: impl Bundle) {
        self.commands.spawn_with(bundle);
    }

    /// Returns a reference to the world.
    pub(crate) fn world(&self) -> &'a World {
        self.world
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io_runtime::IoRuntime;

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    fn apply(commands: &CommandCollector, world: &mut World) {
        for cmd in commands.drain() {
            cmd(world);
        }
    }

    #[test]
    fn context_provides_compute() {
        let world = World::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &io, &commands);
        assert_eq!(ctx.compute().pending_count(), 0);
    }

    #[test]
    fn commands_are_collected() {
        let world = World::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        ctx.commands(|w| {
            w.insert_resource(42u32);
        });

        let drained = commands.drain();
        assert_eq!(drained.len(), 1);
    }

    #[test]
    fn ctx_despawn() {
        let mut world = World::new();
        let entity = world.spawn();

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        let commands = CommandCollector::new();
        {
            let ctx = SystemContext::new(&world, &compute, &io, &commands);
            ctx.despawn(entity);
        }
        apply(&commands, &mut world);

        assert!(!world.is_alive(entity));
    }

    #[test]
    fn ctx_insert() {
        let mut world = World::new();
        world.register_component::<Position>();
        let entity = world.spawn();

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        let commands = CommandCollector::new();
        {
            let ctx = SystemContext::new(&world, &compute, &io, &commands);
            ctx.insert(entity, Position { x: 3.0, y: 4.0 });
        }
        apply(&commands, &mut world);

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 3.0, y: 4.0 })
        );
    }

    #[test]
    fn ctx_remove() {
        let mut world = World::new();
        world.register_component::<Health>();
        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        let commands = CommandCollector::new();
        {
            let ctx = SystemContext::new(&world, &compute, &io, &commands);
            ctx.remove::<Health>(entity);
        }
        apply(&commands, &mut world);

        assert!(world.get::<Health>(entity).is_none());
    }

    #[test]
    fn ctx_spawn_entity() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        let commands = CommandCollector::new();
        {
            let ctx = SystemContext::new(&world, &compute, &io, &commands);
            ctx.spawn_entity()
                .with(Position { x: 1.0, y: 2.0 })
                .with(Health(50))
                .build();
        }
        apply(&commands, &mut world);

        assert_eq!(world.entity_count(), 1);
        let entity = world.iter_entities().next().unwrap();
        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 1.0, y: 2.0 })
        );
        assert_eq!(world.get::<Health>(entity), Some(&Health(50)));
    }

    // -----------------------------------------------------------------------
    // Deadlock detection tests
    // -----------------------------------------------------------------------

    use crate::access_set::{Read, Write};

    struct Velocity {
        _x: f32,
    }

    fn make_ctx(_world: &World) -> (ComputePool, IoRuntime, CommandCollector) {
        (
            ComputePool::new(IoRuntime::new()),
            IoRuntime::new(),
            CommandCollector::new(),
        )
    }

    #[test]
    #[should_panic(expected = "ECS deadlock detected")]
    fn deadlock_write_then_write() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.spawn();

        let (compute, io, commands) = make_ctx(&world);
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let _q1 = ctx.query::<(Write<Position>,)>();
        let _q2 = ctx.query::<(Write<Position>,)>(); // should panic
    }

    #[test]
    #[should_panic(expected = "ECS deadlock detected")]
    fn deadlock_write_then_read() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.spawn();

        let (compute, io, commands) = make_ctx(&world);
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let _q1 = ctx.query::<(Write<Position>,)>();
        let _q2 = ctx.query::<(Read<Position>,)>(); // should panic
    }

    #[test]
    #[should_panic(expected = "ECS deadlock detected")]
    fn deadlock_read_then_write() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.spawn();

        let (compute, io, commands) = make_ctx(&world);
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let _q1 = ctx.query::<(Read<Position>,)>();
        let _q2 = ctx.query::<(Write<Position>,)>(); // should panic
    }

    #[test]
    fn no_deadlock_read_then_read() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0, y: 2.0 }).unwrap();

        let (compute, io, commands) = make_ctx(&world);
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let q1 = ctx.query::<(Read<Position>,)>();
        let q2 = ctx.query::<(Read<Position>,)>(); // should NOT panic
        let (pos1,) = &q1.items;
        let (pos2,) = &q2.items;
        assert_eq!(pos1.len(), pos2.len());
    }

    #[test]
    fn deadlock_check_clears_after_drop() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0, y: 2.0 }).unwrap();

        let (compute, io, commands) = make_ctx(&world);
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        // Write lock, then drop
        {
            let _q1 = ctx.query::<(Write<Position>,)>();
        }
        // Now a new write lock should succeed
        let q2 = ctx.query::<(Write<Position>,)>();
        let (positions,) = &q2.items;
        assert_eq!(positions.len(), 1);
    }

    #[test]
    #[should_panic(expected = "ECS deadlock detected")]
    fn deadlock_query_then_execute() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0, y: 2.0 }).unwrap();

        let (compute, io, commands) = make_ctx(&world);
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let _q = ctx.query::<(Write<Position>,)>();
        // lock().execute also uses tracking — should detect conflict
        let req = ctx.lock::<(Write<Position>,)>();
        req.execute(|_| {}); // should panic
    }

    #[test]
    #[should_panic(expected = "ECS deadlock detected")]
    fn deadlock_partial_overlap() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0, y: 2.0 }).unwrap();
        world.insert(e, Velocity { _x: 3.0 }).unwrap();

        let (compute, io, commands) = make_ctx(&world);
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let _q1 = ctx.query::<(Write<Position>,)>();
        // Overlaps on Position (write held + read wanted)
        let _q2 = ctx.query::<(Read<Position>, Read<Velocity>)>(); // should panic
    }

    #[test]
    fn no_deadlock_disjoint_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0, y: 2.0 }).unwrap();
        world.insert(e, Velocity { _x: 3.0 }).unwrap();

        let (compute, io, commands) = make_ctx(&world);
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let _q1 = ctx.query::<(Write<Position>,)>();
        let _q2 = ctx.query::<(Write<Velocity>,)>(); // different component, should NOT panic
    }
}
