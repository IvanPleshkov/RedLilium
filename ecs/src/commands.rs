use std::sync::Mutex;

use crate::world::World;

/// A boxed command closure that mutates the world.
type Command = Box<dyn FnOnce(&mut World) + Send>;

/// A boxed insert closure that inserts a component into a specific entity.
type InsertFn = Box<dyn FnOnce(&mut World, crate::entity::Entity) + Send>;

/// A thread-safe buffer for deferred world mutations.
///
/// Systems receive `&World` (immutable), so structural changes like spawning,
/// despawning, or inserting components cannot be done directly. Instead,
/// systems queue commands into a `CommandBuffer`, which is applied after
/// the schedule finishes running.
///
/// The buffer uses an internal `Mutex` so multiple parallel systems can
/// queue commands concurrently via a shared `resource::<CommandBuffer>()` borrow.
///
/// # Example
///
/// ```ignore
/// fn spawner_system(world: &World) {
///     let commands = world.resource::<CommandBuffer>();
///     commands.spawn_entity()
///         .with(Transform::IDENTITY)
///         .with(Visibility::VISIBLE)
///         .build();
/// }
///
/// // After schedule.run():
/// world.apply_commands();
/// ```
pub struct CommandBuffer {
    commands: Mutex<Vec<Command>>,
}

impl CommandBuffer {
    /// Creates a new empty command buffer.
    pub fn new() -> Self {
        Self {
            commands: Mutex::new(Vec::new()),
        }
    }

    /// Queues a raw command closure.
    ///
    /// The closure will receive `&mut World` when `apply_commands` is called.
    pub fn push(&self, cmd: impl FnOnce(&mut World) + Send + 'static) {
        self.commands.lock().unwrap().push(Box::new(cmd));
    }

    /// Queues an entity despawn.
    pub fn despawn(&self, entity: crate::entity::Entity) {
        self.push(move |world| {
            world.despawn(entity);
        });
    }

    /// Queues a component insertion on an entity.
    pub fn insert<T: Send + Sync + 'static>(&self, entity: crate::entity::Entity, component: T) {
        self.push(move |world| {
            world.insert(entity, component);
        });
    }

    /// Queues a component removal from an entity.
    pub fn remove<T: Send + Sync + 'static>(&self, entity: crate::entity::Entity) {
        self.push(move |world| {
            world.remove::<T>(entity);
        });
    }

    /// Begins building a spawn command that creates an entity with components.
    ///
    /// # Example
    ///
    /// ```ignore
    /// commands.spawn_entity()
    ///     .with(Transform::IDENTITY)
    ///     .with(Visibility::VISIBLE)
    ///     .build();
    /// ```
    pub fn spawn_entity(&self) -> SpawnBuilder<'_> {
        SpawnBuilder {
            buffer: self,
            inserts: Vec::new(),
        }
    }

    /// Drains all queued commands, returning them.
    ///
    /// After this call, the buffer is empty and ready for new commands.
    pub fn drain(&self) -> Vec<Command> {
        std::mem::take(&mut *self.commands.lock().unwrap())
    }

    /// Returns the number of queued commands.
    pub fn len(&self) -> usize {
        self.commands.lock().unwrap().len()
    }

    /// Returns whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.lock().unwrap().is_empty()
    }
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: CommandBuffer uses a Mutex for interior mutability,
// making it safe to share across threads.
unsafe impl Sync for CommandBuffer {}

/// Builder for spawning an entity with multiple components.
///
/// Created by [`CommandBuffer::spawn_entity`]. The entity is spawned
/// and all components are inserted in a single command when [`build`](SpawnBuilder::build)
/// is called.
pub struct SpawnBuilder<'a> {
    buffer: &'a CommandBuffer,
    inserts: Vec<InsertFn>,
}

impl<'a> SpawnBuilder<'a> {
    /// Adds a component to the entity being built.
    pub fn with<T: Send + Sync + 'static>(mut self, component: T) -> Self {
        self.inserts.push(Box::new(move |world, entity| {
            world.insert(entity, component);
        }));
        self
    }

    /// Finalizes the builder, queuing the spawn command.
    pub fn build(self) {
        let inserts = self.inserts;
        self.buffer.push(move |world| {
            let entity = world.spawn();
            for insert_fn in inserts {
                insert_fn(world, entity);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Velocity {
        x: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    #[test]
    fn new_buffer_is_empty() {
        let buffer = CommandBuffer::new();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn push_increments_len() {
        let buffer = CommandBuffer::new();
        buffer.push(|_| {});
        assert_eq!(buffer.len(), 1);
        buffer.push(|_| {});
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn drain_returns_and_clears() {
        let buffer = CommandBuffer::new();
        buffer.push(|_| {});
        buffer.push(|_| {});

        let cmds = buffer.drain();
        assert_eq!(cmds.len(), 2);
        assert!(buffer.is_empty());
    }

    #[test]
    fn despawn_command() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 2.0 });

        let buffer = CommandBuffer::new();
        buffer.despawn(entity);

        let cmds = buffer.drain();
        for cmd in cmds {
            cmd(&mut world);
        }

        assert!(!world.is_alive(entity));
    }

    #[test]
    fn insert_command() {
        let mut world = World::new();
        let entity = world.spawn();

        let buffer = CommandBuffer::new();
        buffer.insert(entity, Position { x: 5.0, y: 10.0 });

        let cmds = buffer.drain();
        for cmd in cmds {
            cmd(&mut world);
        }

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 5.0, y: 10.0 })
        );
    }

    #[test]
    fn remove_command() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Health(100));

        let buffer = CommandBuffer::new();
        buffer.remove::<Health>(entity);

        let cmds = buffer.drain();
        for cmd in cmds {
            cmd(&mut world);
        }

        assert!(world.get::<Health>(entity).is_none());
    }

    #[test]
    fn spawn_entity_builder() {
        let mut world = World::new();
        let buffer = CommandBuffer::new();

        buffer
            .spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .with(Velocity { x: 3.0 })
            .build();

        assert_eq!(buffer.len(), 1);

        let cmds = buffer.drain();
        for cmd in cmds {
            cmd(&mut world);
        }

        assert_eq!(world.entity_count(), 1);

        let entity = world.iter_entities().next().unwrap();
        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 1.0, y: 2.0 })
        );
        assert_eq!(world.get::<Velocity>(entity), Some(&Velocity { x: 3.0 }));
    }

    #[test]
    fn multiple_spawn_commands() {
        let mut world = World::new();
        let buffer = CommandBuffer::new();

        for i in 0..5 {
            buffer
                .spawn_entity()
                .with(Position {
                    x: i as f32,
                    y: 0.0,
                })
                .build();
        }

        let cmds = buffer.drain();
        for cmd in cmds {
            cmd(&mut world);
        }

        assert_eq!(world.entity_count(), 5);
    }

    #[test]
    fn commands_execute_in_order() {
        let mut world = World::new();
        let entity = world.spawn();

        let buffer = CommandBuffer::new();
        buffer.insert(entity, Health(100));
        buffer.push(move |world| {
            let h = world.get_mut::<Health>(entity).unwrap();
            h.0 += 50;
        });

        let cmds = buffer.drain();
        for cmd in cmds {
            cmd(&mut world);
        }

        assert_eq!(world.get::<Health>(entity), Some(&Health(150)));
    }

    #[test]
    fn custom_command_spawns_with_reference() {
        let mut world = World::new();

        let buffer = CommandBuffer::new();
        // Demonstrate the pattern for when you need the spawned Entity
        buffer.push(|world| {
            let parent = world.spawn();
            world.insert(parent, Position { x: 0.0, y: 0.0 });

            let child = world.spawn();
            world.insert(child, Position { x: 1.0, y: 1.0 });
            // Could set parent-child relationship here
        });

        let cmds = buffer.drain();
        for cmd in cmds {
            cmd(&mut world);
        }

        assert_eq!(world.entity_count(), 2);
    }

    #[test]
    fn concurrent_access_via_mutex() {
        let buffer = CommandBuffer::new();

        // Simulate concurrent pushes (sequential here, but tests the Mutex path)
        std::thread::scope(|s| {
            let b = &buffer;
            s.spawn(move || {
                for _ in 0..100 {
                    b.push(|_| {});
                }
            });
            s.spawn(move || {
                for _ in 0..100 {
                    b.push(|_| {});
                }
            });
        });

        assert_eq!(buffer.len(), 200);
    }
}
