use std::marker::PhantomData;

use crate::access_set::AccessSet;
use crate::bundle::Bundle;
use crate::command_collector::{CommandCollector, SpawnBuilder};
use crate::compute::ComputePool;
use crate::entity::Entity;
use crate::io_runtime::IoRuntime;
use crate::lock_request::LockRequest;
use crate::main_thread_dispatcher::MainThreadDispatcher;
use crate::query_guard::QueryGuard;
use crate::world::World;

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
///     }).await;
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
        }
    }

    /// Returns the main-thread dispatcher, if one exists.
    pub(crate) fn dispatcher(&self) -> Option<&MainThreadDispatcher> {
        self.dispatcher
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
    ///     }).await;
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
    /// let mut q = ctx.query::<(Write<Position>, Read<Velocity>)>().await;
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
    pub async fn query<A: AccessSet>(&self) -> QueryGuard<'_, A> {
        if A::needs_main_thread() {
            panic!("query() does not support main-thread resources; use lock().execute() instead");
        }

        let infos = A::access_infos();
        let guards = loop {
            redlilium_core::profile_scope!("ecs: query acquire");
            if let Some(guards) = self.world.try_acquire_sorted(&infos) {
                break guards;
            }
            // Locks contended — yield to let the executor run compute tasks
            let mut yielded = false;
            std::future::poll_fn(|_| {
                if yielded {
                    std::task::Poll::Ready(())
                } else {
                    yielded = true;
                    std::task::Poll::Pending
                }
            })
            .await;
        };
        let items = A::fetch_unlocked(self.world);
        QueryGuard::new(guards, items)
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
}
