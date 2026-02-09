use crate::compute::ComputePool;
use crate::world::World;

/// A token granting scoped access to the [`World`] for borrowing components.
///
/// Systems receive `QueryAccess` and use [`scope()`](QueryAccess::scope) to
/// access component data. All borrow guards are confined to the closure and
/// automatically dropped when it returns, making it safe to `.await` afterward.
///
/// # Compile-time safety
///
/// The `scope()` closure receives `&'w World` with a fresh lifetime `'w`.
/// Any [`Ref`](crate::Ref) / [`RefMut`](crate::RefMut) guards are bounded by
/// `'w` and **cannot** be returned from the closure (the higher-ranked lifetime
/// bound prevents it). This guarantees guards cannot be held across `.await`.
///
/// # Example
///
/// ```ignore
/// SystemFuture::new(async move {
///     // Phase 1: read data (guards scoped)
///     let sum = access.scope(|world| {
///         let positions = world.read::<Position>();
///         positions.iter().map(|(_, p)| p.x).sum::<f32>()
///     });
///     // Guards dropped here — safe to .await
///
///     let mut handle = access.compute().spawn(Priority::Low, async move { sum * 2.0 });
///     let result = (&mut handle).await;
///
///     // Phase 2: write results
///     access.scope(|world| {
///         let mut res = world.resource_mut::<f32>();
///         *res = result.unwrap();
///     });
/// })
/// ```
pub struct QueryAccess<'a> {
    world: &'a World,
    compute: &'a ComputePool,
}

impl<'a> QueryAccess<'a> {
    /// Creates a new `QueryAccess` token. Called by the scheduler.
    pub fn new(world: &'a World, compute: &'a ComputePool) -> Self {
        Self { world, compute }
    }

    /// Executes a closure with access to the [`World`] for borrowing components.
    ///
    /// All borrow guards ([`Ref`](crate::Ref), [`RefMut`](crate::RefMut),
    /// [`ResourceRef`](crate::ResourceRef), [`ResourceRefMut`](crate::ResourceRefMut),
    /// [`ContainsChecker`](crate::ContainsChecker)) are scoped to the closure
    /// and automatically dropped when it returns.
    ///
    /// The closure can return a value, allowing data to be extracted for use
    /// after the scope (and potentially across `.await` points).
    pub fn scope<R>(&self, f: impl FnOnce(&World) -> R) -> R {
        f(self.world)
    }

    /// Returns a reference to the [`ComputePool`] for spawning async tasks.
    ///
    /// Does not require `scope()` because compute pool access does not
    /// involve World borrow guards.
    pub fn compute(&self) -> &ComputePool {
        self.compute
    }

    /// Returns the current world tick for change detection.
    pub fn current_tick(&self) -> u64 {
        self.world.current_tick()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    struct Position {
        x: f32,
    }
    impl crate::component::Component for Position {
        fn component_name(&self) -> &'static str {
            "Position"
        }
        fn field_infos(&self) -> &'static [crate::component::FieldInfo] {
            &[]
        }
        fn field(&self, _name: &str) -> Option<&dyn Any> {
            None
        }
        fn field_mut(&mut self, _name: &str) -> Option<&mut dyn Any> {
            None
        }
    }

    #[test]
    fn scope_returns_extracted_data() {
        let mut world = World::new();
        let compute = ComputePool::new();

        let e = world.spawn();
        world.insert(e, Position { x: 42.0 });

        let access = QueryAccess::new(&world, &compute);
        let sum = access.scope(|world| {
            let positions = world.read::<Position>();
            positions.iter().map(|(_, p)| p.x).sum::<f32>()
        });

        assert_eq!(sum, 42.0);
    }

    #[test]
    fn multiple_scopes_work() {
        let mut world = World::new();
        let compute = ComputePool::new();

        let e = world.spawn();
        world.insert(e, Position { x: 10.0 });

        let access = QueryAccess::new(&world, &compute);

        // First scope: read
        let val = access.scope(|world| {
            let positions = world.read::<Position>();
            positions.iter().next().map(|(_, p)| p.x).unwrap()
        });

        // Second scope: write
        access.scope(|world| {
            let mut positions = world.write::<Position>();
            for (_, pos) in positions.iter_mut() {
                pos.x += val;
            }
        });

        // Verify
        let result = access.scope(|world| {
            let positions = world.read::<Position>();
            positions.iter().next().map(|(_, p)| p.x).unwrap()
        });
        assert_eq!(result, 20.0);
    }

    #[test]
    fn compute_accessible_outside_scope() {
        let world = World::new();
        let compute = ComputePool::new();
        let access = QueryAccess::new(&world, &compute);

        // Can access compute pool without scope
        let handle = access
            .compute()
            .spawn(crate::Priority::Low, async { 42u32 });
        compute.tick();
        assert_eq!(handle.try_recv(), Some(42));
    }
}
