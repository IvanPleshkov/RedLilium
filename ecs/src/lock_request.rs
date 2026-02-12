use std::marker::PhantomData;

use crate::access_set::AccessSet;
use crate::system_context::SystemContext;

/// A pending lock request for a set of component/resource accesses.
///
/// Created by [`SystemContext::lock()`]. Call [`execute()`](LockRequest::execute)
/// to run a closure with the locked data.
///
/// The `execute()` method is `async` to support future optimization to
/// true async lock acquisition. Currently it completes synchronously.
///
/// # Example
///
/// ```ignore
/// use redlilium_ecs::{Read, Write};
///
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
/// # Lock ordering
///
/// Locks are acquired in TypeId-sorted order via `World::acquire_sorted`
/// to prevent deadlocks. The closure is synchronous (`FnOnce`), preventing
/// locks from being held across await points.
pub struct LockRequest<'a, A: AccessSet> {
    pub(crate) ctx: &'a SystemContext<'a>,
    pub(crate) _marker: PhantomData<A>,
}

impl<'a, A: AccessSet> LockRequest<'a, A> {
    /// Executes a closure with the locked component/resource data.
    ///
    /// Acquires per-storage RwLocks in TypeId-sorted order to prevent
    /// deadlocks, fetches data without per-fetch locking, then calls
    /// the closure. All locks are released when the closure returns.
    ///
    /// The closure is synchronous (`FnOnce`) to prevent holding locks
    /// across await points.
    pub async fn execute<R, F>(self, f: F) -> R
    where
        F: FnOnce(A::Item<'_>) -> R,
    {
        self.execute_inner(f)
    }

    fn execute_inner<R, F>(&self, f: F) -> R
    where
        F: FnOnce(A::Item<'_>) -> R,
    {
        // Acquire per-storage RwLocks in TypeId-sorted order (deadlock prevention)
        let _guards = {
            redlilium_core::profile_scope!("ecs: lock acquire");
            self.ctx.world().acquire_sorted(&A::access_infos())
        };

        // Fetch typed data without per-fetch locking (locks already held)
        let items = A::fetch_unlocked(self.ctx.world());

        // Run closure (guards drop after closure returns)
        f(items)
    }
}

#[cfg(test)]
mod tests {
    use crate::access_set::{Read, Write};
    use crate::command_collector::CommandCollector;
    use crate::compute::ComputePool;
    use crate::io_runtime::IoRuntime;
    use crate::system_context::SystemContext;
    use crate::world::World;

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }

    #[test]
    fn execute_reads_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        let compute = ComputePool::new(IoRuntime::new());
        let io = crate::io_runtime::IoRuntime::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        // Use pollster-style blocking since execute is async
        let request = ctx.lock::<(Read<Position>,)>();
        let count = request.execute_inner(|(positions,)| positions.len());
        assert_eq!(count, 1);
    }

    #[test]
    fn execute_writes_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();

        let compute = ComputePool::new(IoRuntime::new());
        let io = crate::io_runtime::IoRuntime::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let request = ctx.lock::<(Write<Position>,)>();
        request.execute_inner(|(mut positions,)| {
            for (_, pos) in positions.iter_mut() {
                pos.x = 99.0;
            }
        });

        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn execute_multiple_accesses() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();
        world.insert(e, Velocity { x: 5.0 }).unwrap();

        let compute = ComputePool::new(IoRuntime::new());
        let io = crate::io_runtime::IoRuntime::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let request = ctx.lock::<(Write<Position>, Read<Velocity>)>();
        request.execute_inner(|(mut positions, velocities)| {
            for (idx, pos) in positions.iter_mut() {
                if let Some(vel) = velocities.get(idx) {
                    pos.x += vel.x;
                }
            }
        });

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[test]
    fn execute_returns_value() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        let compute = ComputePool::new(IoRuntime::new());
        let io = crate::io_runtime::IoRuntime::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        let request = ctx.lock::<(Read<Position>,)>();
        let sum =
            request.execute_inner(|(positions,)| positions.iter().map(|(_, p)| p.x).sum::<f32>());
        assert_eq!(sum, 42.0);
    }
}
