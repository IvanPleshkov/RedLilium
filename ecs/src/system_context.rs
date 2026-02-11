use std::marker::PhantomData;

use crate::access_set::AccessSet;
use crate::command_collector::CommandCollector;
use crate::compute::ComputePool;
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
    commands: &'a CommandCollector,
}

impl<'a> SystemContext<'a> {
    /// Creates a new system context.
    pub(crate) fn new(
        world: &'a World,
        compute: &'a ComputePool,
        commands: &'a CommandCollector,
    ) -> Self {
        Self {
            world,
            compute,
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

    /// Pushes a deferred command to be applied after all systems complete.
    ///
    /// Commands receive `&mut World` and can perform structural changes
    /// like spawning, despawning, and inserting components.
    pub fn commands(&self, cmd: impl FnOnce(&mut World) + Send + 'static) {
        self.commands.push(cmd);
    }

    /// Returns a reference to the world.
    pub(crate) fn world(&self) -> &'a World {
        self.world
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_provides_compute() {
        let world = World::new();
        let compute = ComputePool::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &commands);
        assert_eq!(ctx.compute().pending_count(), 0);
    }

    #[test]
    fn commands_are_collected() {
        let world = World::new();
        let compute = ComputePool::new();
        let commands = CommandCollector::new();
        let ctx = SystemContext::new(&world, &compute, &commands);

        ctx.commands(|world| {
            world.insert_resource(42u32);
        });

        let drained = commands.drain();
        assert_eq!(drained.len(), 1);
    }
}
