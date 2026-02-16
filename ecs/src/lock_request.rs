use std::marker::PhantomData;
use std::sync::mpsc;

use crate::access_set::{AccessSet, normalize_access_infos};
use crate::main_thread_dispatcher::MainThreadWork;
use crate::system_context::SystemContext;

/// A pending lock request for a set of component/resource accesses.
///
/// Created by [`SystemContext::lock()`]. Call [`execute()`](LockRequest::execute)
/// to run a closure with the locked data.
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
///     });
/// ```
///
/// # Lock ordering
///
/// Locks are acquired in TypeId-sorted order via `World::acquire_sorted`
/// to prevent deadlocks. The closure is synchronous (`FnOnce`), ensuring
/// locks are released deterministically when the closure returns.
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
    /// If the access set contains any main-thread resources
    /// ([`MainThreadRes`](crate::MainThreadRes) or
    /// [`MainThreadResMut`](crate::MainThreadResMut)), the entire closure
    /// is transparently dispatched to the main thread via the
    /// [`MainThreadDispatcher`](crate::main_thread_dispatcher::MainThreadDispatcher).
    ///
    /// # Panics
    ///
    /// Panics if any requested component lock is already held by this
    /// system (same-system deadlock detection via [`SystemContext`]).
    pub fn execute<R, F>(self, f: F) -> R
    where
        F: FnOnce(A::Item<'_>) -> R + Send,
        R: Send,
    {
        // Check + register tracking before acquiring actual locks.
        let sorted = normalize_access_infos(&A::access_infos());
        self.ctx.check_held_locks(&sorted);
        self.ctx.record_access(&sorted);
        self.ctx.register_held_locks(&sorted);
        let _tracking = self.ctx.make_tracking(&sorted);

        if A::needs_main_thread() {
            self.run_on_main_thread(f)
        } else {
            self.run_local(f)
        }
        // _tracking drops here → unregisters held locks
    }

    /// Fast path: runs directly on the calling thread (no tracking).
    fn run_local<R, F>(&self, f: F) -> R
    where
        F: FnOnce(A::Item<'_>) -> R,
    {
        let _guards = {
            redlilium_core::profile_scope!("ecs: lock acquire");
            self.ctx.world().acquire_sorted(&A::access_infos())
        };
        let items = A::fetch_unlocked(self.ctx.world());
        f(items)
    }

    /// Slow path: dispatches the closure to the main thread (no tracking).
    fn run_on_main_thread<R, F>(&self, f: F) -> R
    where
        F: FnOnce(A::Item<'_>) -> R + Send,
        R: Send,
    {
        match self.ctx.dispatcher() {
            Some(dispatcher) => {
                let world = self.ctx.world();
                let (result_tx, result_rx) = mpsc::sync_channel::<R>(1);

                let work: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
                    let _guards = {
                        redlilium_core::profile_scope!("ecs: lock acquire (main-thread)");
                        world.acquire_sorted(&A::access_infos())
                    };
                    let items = A::fetch_unlocked(world);
                    let result = f(items);
                    let _ = result_tx.send(result);
                });

                // SAFETY: The closure captures `&'a World` and `F` which live for
                // the duration of `std::thread::scope` in the runner. The main
                // thread executes this closure within the same scope, so all
                // captured references are valid at the time of execution. The
                // closure is consumed (FnOnce) before the scope exits.
                let work: MainThreadWork = unsafe {
                    std::mem::transmute::<Box<dyn FnOnce() + Send + '_>, MainThreadWork>(work)
                };

                dispatcher.send_work(work);
                result_rx
                    .recv()
                    .expect("Main thread did not send result back")
            }
            None => {
                // Single-threaded: no dispatcher, already on main thread
                self.run_local(f)
            }
        }
    }

    /// Acquires locks and iterates over matching entities in parallel.
    ///
    /// Convenience method that combines lock acquisition with parallel
    /// per-entity iteration. The access set `A` must implement
    /// [`ForEachAccess`](crate::ForEachAccess).
    ///
    /// On WASM, falls back to sequential iteration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.lock::<(Write<Position>, Read<Velocity>)>()
    ///     .par_for_each(|(pos, vel): (&mut Position, &Velocity)| {
    ///         pos.x += vel.x;
    ///     });
    /// ```
    pub fn par_for_each<F>(self, f: F)
    where
        A: crate::function_system::ForEachAccess,
        for<'w> A::Item<'w>: Sync,
        F: for<'w> Fn(<A as crate::function_system::ForEachAccess>::EachItem<'w>) + Send + Sync,
    {
        self.execute(|items| {
            A::run_par_for_each(&items, &f);
        });
    }

    /// Like [`par_for_each`](Self::par_for_each), but with explicit
    /// parallelism configuration.
    pub fn par_for_each_with<F>(self, config: crate::par_for_each::ParConfig, f: F)
    where
        A: crate::function_system::ForEachAccess,
        for<'w> A::Item<'w>: Sync,
        F: for<'w> Fn(<A as crate::function_system::ForEachAccess>::EachItem<'w>) + Send + Sync,
    {
        self.execute(|items| {
            A::run_par_for_each_with(&items, &config, &f);
        });
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
        let count = request.execute(|(positions,)| positions.len());
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
        request.execute(|(mut positions,)| {
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
        request.execute(|(mut positions, velocities)| {
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
        let sum = request.execute(|(positions,)| positions.iter().map(|(_, p)| p.x).sum::<f32>());
        assert_eq!(sum, 42.0);
    }

    #[test]
    fn par_for_each_via_lock_request() {
        let mut world = World::new();
        world.register_component::<Position>();

        for _ in 0..100 {
            let e = world.spawn();
            world.insert(e, Position { x: 1.0 }).unwrap();
        }

        let compute = ComputePool::new(IoRuntime::new());
        let io = crate::io_runtime::IoRuntime::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        ctx.lock::<(Write<Position>,)>()
            .par_for_each(|(pos,): (&mut Position,)| {
                pos.x = 42.0;
            });

        for e in world.iter_entities() {
            assert_eq!(world.get::<Position>(e).unwrap().x, 42.0);
        }
    }

    #[test]
    fn par_for_each_two_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        for _ in 0..100 {
            let e = world.spawn();
            world.insert(e, Position { x: 0.0 }).unwrap();
            world.insert(e, Velocity { x: 5.0 }).unwrap();
        }

        let compute = ComputePool::new(IoRuntime::new());
        let io = crate::io_runtime::IoRuntime::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &io, &commands);

        ctx.lock::<(Write<Position>, Read<Velocity>)>()
            .par_for_each(|(pos, vel): (&mut Position, &Velocity)| {
                pos.x += vel.x;
            });

        for e in world.iter_entities() {
            assert_eq!(world.get::<Position>(e).unwrap().x, 5.0);
        }
    }
}
