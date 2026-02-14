use crate::access_set::AccessSet;
use crate::sparse_set::LockGuard;
use crate::system_context::LockTracking;

/// A guard holding component/resource locks and their fetched data.
///
/// Created by [`SystemContext::query()`](crate::SystemContext::query).
/// Locks are acquired in TypeId-sorted order (same as `lock().execute()`)
/// to prevent deadlocks. Locks are held until the guard is dropped.
///
/// Unlike `lock().execute()`, which runs a synchronous closure with the
/// locked data, `QueryGuard` lets you access the data directly without
/// a closure — enabling normal control flow, `?` operators, and multiple
/// statements with the locks held.
///
/// Access fetched data via the public `items` field:
///
/// ```ignore
/// // Read-only:
/// let q = ctx.query::<(Read<Position>, Read<Velocity>)>().await;
/// let (positions, velocities) = &q.items;
///
/// // With writes:
/// let mut q = ctx.query::<(Write<Position>, Read<Velocity>)>().await;
/// let (positions, velocities) = &mut q.items;
/// for (idx, pos) in positions.iter_mut() {
///     if let Some(vel) = velocities.get(idx) {
///         pos.x += vel.x;
///     }
/// }
/// // locks released when `q` goes out of scope
/// ```
///
/// # Limitations
///
/// Main-thread resources ([`MainThreadRes`](crate::MainThreadRes),
/// [`MainThreadResMut`](crate::MainThreadResMut)) are not supported.
/// Use `lock().execute()` for those.
pub struct QueryGuard<'a, A: AccessSet> {
    _guards: Vec<LockGuard<'a>>,
    /// The fetched component/resource data. Destructure this to access
    /// individual storages.
    pub items: A::Item<'a>,
    /// Deadlock tracking — unregisters held locks when this guard is dropped.
    /// `None` when created outside of a SystemContext (e.g. in tests).
    _tracking: Option<LockTracking<'a>>,
}

impl<'a, A: AccessSet> QueryGuard<'a, A> {
    #[cfg(test)]
    pub(crate) fn new(guards: Vec<LockGuard<'a>>, items: A::Item<'a>) -> Self {
        Self {
            _guards: guards,
            items,
            _tracking: None,
        }
    }

    pub(crate) fn new_tracked(
        guards: Vec<LockGuard<'a>>,
        items: A::Item<'a>,
        tracking: LockTracking<'a>,
    ) -> Self {
        Self {
            _guards: guards,
            items,
            _tracking: Some(tracking),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_set::{AccessSet, Read, Res, Write};
    use crate::world::World;

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }

    /// Helper: constructs a QueryGuard directly from a World (same logic as
    /// `SystemContext::query()` but callable from sync tests).
    fn query<'w, A: AccessSet>(world: &'w World) -> QueryGuard<'w, A> {
        let guards = world.acquire_sorted(&A::access_infos());
        let items = A::fetch_unlocked(world);
        QueryGuard::new(guards, items)
    }

    #[test]
    fn query_reads_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        let q = query::<(Read<Position>,)>(&world);
        let (positions,) = &q.items;
        assert_eq!(positions.len(), 1);
        assert_eq!(positions.iter().next().unwrap().1.x, 42.0);
    }

    #[test]
    fn query_writes_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();

        {
            let mut q = query::<(Write<Position>,)>(&world);
            let (positions,) = &mut q.items;
            for (_, pos) in positions.iter_mut() {
                pos.x = 99.0;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn query_multiple_accesses() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();
        world.insert(e, Velocity { x: 5.0 }).unwrap();

        {
            let mut q = query::<(Write<Position>, Read<Velocity>)>(&world);
            let (positions, velocities) = &mut q.items;
            for (idx, pos) in positions.iter_mut() {
                if let Some(vel) = velocities.get(idx) {
                    pos.x += vel.x;
                }
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[test]
    fn query_with_resources() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.insert_resource(2.0f32);
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();

        {
            let mut q = query::<(Write<Position>, Res<f32>)>(&world);
            let (positions, factor) = &mut q.items;
            let f = **factor;
            for (_, pos) in positions.iter_mut() {
                pos.x *= f;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 20.0);
    }

    #[test]
    fn query_locks_released_on_drop() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        // First query: read
        {
            let q = query::<(Read<Position>,)>(&world);
            let (positions,) = &q.items;
            assert_eq!(positions.len(), 1);
        }
        // Guard dropped — now we can acquire a write lock
        {
            let mut q = query::<(Write<Position>,)>(&world);
            let (positions,) = &mut q.items;
            for (_, pos) in positions.iter_mut() {
                pos.x = 42.0;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 42.0);
    }

    #[test]
    fn query_returns_value_from_get() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        let q = query::<(Read<Position>,)>(&world);
        let (positions,) = &q.items;
        let sum: f32 = positions.iter().map(|(_, p)| p.x).sum();
        assert_eq!(sum, 42.0);
    }
}
