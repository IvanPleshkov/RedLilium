use std::marker::PhantomData;

use crate::access_set::AccessSet;
use crate::command_collector::{CommandCollector, SpawnBuilder};
use crate::compute::ComputePool;
use crate::entity::Entity;
use crate::io_runtime::IoRuntime;
use crate::lock_request::LockRequest;
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
}

impl<'a> SystemContext<'a> {
    /// Creates a new system context.
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
    ///     }).await;
    /// ```
    pub fn lock<A: AccessSet>(&self) -> LockRequest<'_, A> {
        LockRequest {
            ctx: self,
            _marker: PhantomData,
        }
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
