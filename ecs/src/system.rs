use std::any::TypeId;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::access::Access;
use crate::compute::{ComputePool, noop_waker};
use crate::query_access::QueryAccess;
use crate::system_future::SystemFuture;

use crate::world::World;

/// A system that processes entities and components in the world.
///
/// All systems are async. Systems that don't need `.await` simply fetch once
/// inside a [`scope()`](QueryAccess::scope) call and return.
///
/// # Compile-time safety
///
/// Systems receive a [`QueryAccess`] token instead of direct `&World` access.
/// Component borrows are confined to `scope()` closures, ensuring guards
/// cannot be held across `.await` points at compile time.
///
/// # Example — simple system
///
/// ```ignore
/// struct MovementSystem;
///
/// impl System for MovementSystem {
///     fn run(&self, access: QueryAccess<'_>) -> SystemFuture<'_> {
///         SystemFuture::new(async move {
///             access.scope(|world| {
///                 let mut positions = world.write::<Position>();
///                 let velocities = world.read::<Velocity>();
///                 for (idx, pos) in positions.iter_mut() {
///                     if let Some(vel) = velocities.get(idx) {
///                         pos.x += vel.x;
///                     }
///                 }
///             });
///         })
///     }
///
///     fn access(&self) -> Access {
///         let mut access = Access::new();
///         access.add_write::<Position>();
///         access.add_read::<Velocity>();
///         access
///     }
/// }
/// ```
///
/// # Example — two-phase async system
///
/// ```ignore
/// struct PathfindSystem;
///
/// impl System for PathfindSystem {
///     fn run(&self, access: QueryAccess<'_>) -> SystemFuture<'_> {
///         SystemFuture::new(async move {
///             // Phase 1: extract data (guards scoped)
///             let graph = access.scope(|world| {
///                 let nav = world.read::<NavMesh>();
///                 nav.iter().next().map(|(_, n)| n.clone())
///             });
///
///             // Safe to .await — no guards held
///             let mut handle = access.compute().spawn(Priority::Low, async move {
///                 compute_paths(graph)
///             });
///             let paths = (&mut handle).await;
///
///             // Phase 2: apply results
///             if let Some(paths) = paths {
///                 access.scope(|world| {
///                     let mut agents = world.write::<Agent>();
///                     for (idx, agent) in agents.iter_mut() {
///                         if let Some(path) = paths.get(&idx) {
///                             agent.path = path.clone();
///                         }
///                     }
///                 });
///             }
///         })
///     }
///
///     fn access(&self) -> Access {
///         let mut a = Access::new();
///         a.add_read::<NavMesh>();
///         a.add_write::<Agent>();
///         a
///     }
/// }
/// ```
pub trait System: Send + Sync + 'static {
    /// Execute the system with scoped access to the world.
    ///
    /// Use [`QueryAccess::scope`] to borrow components. Guards are
    /// automatically dropped when the scope closure returns, making it
    /// safe to `.await` between scopes.
    fn run<'a>(&'a self, access: QueryAccess<'a>) -> SystemFuture<'a>;

    /// Returns the access descriptor for this system.
    ///
    /// Called once during [`Schedule::add`](crate::Schedule::add) and cached.
    /// The scheduler uses this to detect conflicts between systems.
    fn access(&self) -> Access;
}

/// Runs a system synchronously to completion.
///
/// Creates a [`QueryAccess`], calls [`System::run`], and polls
/// the returned future to completion. Drives the [`ComputePool`]
/// between polls so spawned compute tasks make progress.
///
/// Useful for tests and one-off system invocations outside a schedule.
pub fn run_system_blocking(system: &dyn System, world: &World, compute: &ComputePool) {
    let access = QueryAccess::new(world, compute);
    let future = system.run(access);
    poll_system_future_to_completion(future, compute);
}

/// Polls a system future to completion, driving compute tasks between polls.
pub(crate) fn poll_system_future_to_completion(future: SystemFuture<'_>, compute: &ComputePool) {
    let mut future = future;
    // Safety: future is on the stack and won't be moved after this point.
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    let waker = noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(()) => break,
            std::task::Poll::Pending => {
                compute.tick_all();
                std::thread::yield_now();
            }
        }
    }
}

/// Typed wrapper for system output, stored as a World resource.
///
/// Keyed by the system type `S` for uniqueness — multiple systems can
/// produce different `T` values without collision.
///
/// # Example
///
/// ```ignore
/// // Producer system returns PhysicsResult:
/// #[system]
/// impl PhysicsSystem {
///     async fn run(&self, access: QueryAccess<'_>) -> PhysicsResult { ... }
///     fn access(&self) -> Access { ... }
/// }
///
/// // Consumer system reads the result:
/// access.scope(|world| {
///     let result = world.resource::<SystemResult<PhysicsSystem, PhysicsResult>>();
///     // use result.value
/// });
/// ```
pub struct SystemResult<S: 'static, T: Send + Sync + 'static> {
    /// The system's output value.
    pub value: T,
    _marker: PhantomData<fn() -> S>,
}

impl<S: 'static, T: Send + Sync + 'static> SystemResult<S, T> {
    /// Creates a new system result.
    pub fn new(value: T) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }
}

/// Internal wrapper holding a registered system plus scheduler metadata.
pub(crate) struct StoredSystem {
    /// The system instance, type-erased.
    pub system: Box<dyn System>,
    /// TypeId of the concrete system struct.
    pub type_id: TypeId,
    /// Human-readable type name for debug/error messages.
    pub type_name: &'static str,
    /// Cached access descriptor (populated at add time).
    pub access: Access,
    /// TypeIds of systems that must run before this one.
    pub after: Vec<TypeId>,
    /// TypeIds of systems that must run after this one.
    pub before: Vec<TypeId>,
}

/// A reference returned by [`Schedule::add`](crate::Schedule::add) to
/// configure ordering constraints.
///
/// # Example
///
/// ```ignore
/// schedule.add(UpdateCameraMatrices)
///     .after::<UpdateGlobalTransforms>();
/// ```
pub struct SystemRef<'a> {
    stored: &'a mut StoredSystem,
}

impl<'a> SystemRef<'a> {
    pub(crate) fn new(stored: &'a mut StoredSystem) -> Self {
        Self { stored }
    }

    /// Declares that this system must run after `S`.
    ///
    /// `S` can be any registered system type.
    /// Panics at [`Schedule::build`](crate::Schedule::build) if `S` is not registered.
    pub fn after<S: 'static>(self) -> Self {
        self.stored.after.push(TypeId::of::<S>());
        self
    }

    /// Declares that this system must run before `S`.
    ///
    /// `S` can be any registered system type.
    /// Panics at [`Schedule::build`](crate::Schedule::build) if `S` is not registered.
    pub fn before<S: 'static>(self) -> Self {
        self.stored.before.push(TypeId::of::<S>());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CompA;
    struct CompB;

    struct EmptySystem;
    impl System for EmptySystem {
        fn run<'a>(&'a self, _access: QueryAccess<'a>) -> SystemFuture<'a> {
            SystemFuture::new(async {})
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    struct ReadSystem;
    impl System for ReadSystem {
        fn run<'a>(&'a self, _access: QueryAccess<'a>) -> SystemFuture<'a> {
            SystemFuture::new(async {})
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_read::<CompA>();
            a.add_read::<CompB>();
            a
        }
    }

    #[test]
    fn stored_system_captures_type_info() {
        let sys = EmptySystem;
        let access = sys.access();
        let stored = StoredSystem {
            system: Box::new(sys),
            type_id: TypeId::of::<EmptySystem>(),
            type_name: std::any::type_name::<EmptySystem>(),
            access,
            after: Vec::new(),
            before: Vec::new(),
        };
        assert_eq!(stored.type_id, TypeId::of::<EmptySystem>());
        assert!(stored.type_name.contains("EmptySystem"));
    }

    #[test]
    fn system_ref_collects_ordering() {
        let sys = EmptySystem;
        let access = sys.access();
        let mut stored = StoredSystem {
            system: Box::new(sys),
            type_id: TypeId::of::<EmptySystem>(),
            type_name: std::any::type_name::<EmptySystem>(),
            access,
            after: Vec::new(),
            before: Vec::new(),
        };
        SystemRef::new(&mut stored).after::<ReadSystem>();
        assert_eq!(stored.after.len(), 1);
        assert_eq!(stored.after[0], TypeId::of::<ReadSystem>());
    }

    #[test]
    fn read_only_access() {
        let sys = ReadSystem;
        assert!(sys.access().is_read_only());
    }

    #[test]
    fn system_runs_blocking() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct FlagSystem(std::sync::Arc<AtomicBool>);
        impl System for FlagSystem {
            fn run<'a>(&'a self, _access: QueryAccess<'a>) -> SystemFuture<'a> {
                SystemFuture::new(async move {
                    self.0.store(true, Ordering::Relaxed);
                })
            }
            fn access(&self) -> Access {
                Access::new()
            }
        }

        let flag = std::sync::Arc::new(AtomicBool::new(false));
        let sys = FlagSystem(flag.clone());
        let world = World::new();
        let compute = ComputePool::new();
        run_system_blocking(&sys, &world, &compute);
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn run_blocking_drives_compute() {
        use crate::Priority;

        struct ComputeSystem(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl System for ComputeSystem {
            fn run<'a>(&'a self, access: QueryAccess<'a>) -> SystemFuture<'a> {
                let slot = self.0.clone();
                SystemFuture::new(async move {
                    let mut handle = access.compute().spawn(Priority::Low, async { 99u32 });
                    let result = (&mut handle).await;
                    *slot.lock().unwrap() = result;
                })
            }
            fn access(&self) -> Access {
                Access::new()
            }
        }

        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        let sys = ComputeSystem(result.clone());
        let world = World::new();
        let compute = ComputePool::new();
        run_system_blocking(&sys, &world, &compute);
        assert_eq!(*result.lock().unwrap(), Some(99));
    }
}
