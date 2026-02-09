use std::any::TypeId;

use crate::access::Access;
use crate::compute::ComputePool;
use crate::world::World;

/// Context passed to systems during execution.
///
/// Provides access to the [`World`] for entity/component queries
/// and to the [`ComputePool`] for spawning async background tasks.
///
/// # Example
///
/// ```ignore
/// fn run(&self, ctx: &SystemContext) {
///     let positions = ctx.world().read::<Position>();
///     let handle = ctx.compute().spawn(Priority::Low, async { heavy_work() });
/// }
/// ```
#[derive(Clone, Copy)]
pub struct SystemContext<'a> {
    world: &'a World,
    compute: &'a ComputePool,
}

impl<'a> SystemContext<'a> {
    /// Creates a new system context.
    pub fn new(world: &'a World, compute: &'a ComputePool) -> Self {
        Self { world, compute }
    }

    /// Returns a reference to the world.
    pub fn world(&self) -> &World {
        self.world
    }

    /// Returns a reference to the compute pool for spawning async tasks.
    pub fn compute(&self) -> &ComputePool {
        self.compute
    }
}

/// A system that processes entities and components in the world.
///
/// Systems are structs implementing this trait. The struct's `TypeId` serves
/// as the unique identifier for ordering constraints and duplicate detection.
///
/// # Parallel safety
///
/// `run` takes `&self` to allow parallel execution within a schedule stage.
/// Systems that need mutable internal state should use interior mutability
/// (`AtomicU64`, `Mutex`, etc.).
///
/// # Example
///
/// ```ignore
/// struct MovementSystem;
///
/// impl System for MovementSystem {
///     fn run(&self, ctx: &SystemContext) {
///         let mut positions = ctx.world().write::<Position>();
///         let velocities = ctx.world().read::<Velocity>();
///         for (idx, pos) in positions.iter_mut() {
///             if let Some(vel) = velocities.get(idx) {
///                 pos.x += vel.x;
///             }
///         }
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
pub trait System: Send + Sync + 'static {
    /// Execute the system with access to the world and compute pool.
    fn run(&self, ctx: &SystemContext);

    /// Returns the access descriptor for this system.
    ///
    /// Called once during [`Schedule::add`](crate::Schedule::add) and cached
    /// internally. The scheduler uses this to detect conflicts between systems.
    fn access(&self) -> Access;
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

impl StoredSystem {
    /// Executes the system function.
    pub fn run(&self, ctx: &SystemContext) {
        self.system.run(ctx);
    }
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
    pub fn after<S: System>(self) -> Self {
        self.stored.after.push(TypeId::of::<S>());
        self
    }

    /// Declares that this system must run before `S`.
    pub fn before<S: System>(self) -> Self {
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
        fn run(&self, _ctx: &SystemContext) {}
        fn access(&self) -> Access {
            Access::new()
        }
    }

    struct ReadSystem;
    impl System for ReadSystem {
        fn run(&self, _ctx: &SystemContext) {}
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
        let stored = StoredSystem {
            type_id: TypeId::of::<EmptySystem>(),
            type_name: std::any::type_name::<EmptySystem>(),
            access: sys.access(),
            after: Vec::new(),
            before: Vec::new(),
            system: Box::new(sys),
        };
        assert_eq!(stored.type_id, TypeId::of::<EmptySystem>());
        assert!(stored.type_name.contains("EmptySystem"));
    }

    #[test]
    fn system_ref_collects_ordering() {
        let sys = EmptySystem;
        let mut stored = StoredSystem {
            type_id: TypeId::of::<EmptySystem>(),
            type_name: std::any::type_name::<EmptySystem>(),
            access: sys.access(),
            after: Vec::new(),
            before: Vec::new(),
            system: Box::new(sys),
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
    fn system_runs() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct FlagSystem(std::sync::Arc<AtomicBool>);
        impl System for FlagSystem {
            fn run(&self, _ctx: &SystemContext) {
                self.0.store(true, Ordering::Relaxed);
            }
            fn access(&self) -> Access {
                Access::new()
            }
        }

        let flag = std::sync::Arc::new(AtomicBool::new(false));
        let sys = FlagSystem(flag.clone());
        let world = World::new();
        let compute = ComputePool::new();
        let ctx = SystemContext::new(&world, &compute);
        sys.run(&ctx);
        assert!(flag.load(Ordering::Relaxed));
    }
}
