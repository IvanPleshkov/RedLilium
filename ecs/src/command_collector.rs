use std::sync::Mutex;

use crate::entity::Entity;
use crate::world::World;

/// A boxed deferred command that mutates the World.
type Command = Box<dyn FnOnce(&mut World) + Send>;

/// A boxed insert closure that inserts a component into a specific entity.
type InsertFn = Box<dyn FnOnce(&mut World, Entity) + Send>;

/// A thread-safe collector for deferred world commands.
///
/// Systems push commands during execution via [`SystemContext::commands`](crate::SystemContext::commands).
/// After all systems complete, the runner drains the collector and applies
/// commands to `&mut World`.
///
/// Multiple systems can push commands concurrently in multi-threaded mode.
pub struct CommandCollector {
    commands: Mutex<Vec<Command>>,
}

impl CommandCollector {
    /// Creates a new empty command collector.
    pub fn new() -> Self {
        Self {
            commands: Mutex::new(Vec::new()),
        }
    }

    /// Pushes a deferred command.
    ///
    /// The command will receive `&mut World` when applied after all systems complete.
    pub fn push(&self, cmd: impl FnOnce(&mut World) + Send + 'static) {
        self.commands.lock().unwrap().push(Box::new(cmd));
    }

    /// Queues an entity despawn.
    pub fn despawn(&self, entity: Entity) {
        self.push(move |world| {
            world.despawn(entity);
        });
    }

    /// Queues a component insertion on an entity.
    ///
    /// # Panics
    ///
    /// Panics when applied if the component type has not been registered.
    pub fn insert<T: Send + Sync + 'static>(&self, entity: Entity, component: T) {
        self.push(move |world| {
            world
                .insert(entity, component)
                .expect("Component not registered");
        });
    }

    /// Queues a component removal from an entity.
    pub fn remove<T: Send + Sync + 'static>(&self, entity: Entity) {
        self.push(move |world| {
            world.remove::<T>(entity);
        });
    }

    /// Begins building a spawn command that creates an entity with components.
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
        SpawnBuilder {
            collector: self,
            inserts: Vec::new(),
        }
    }

    /// Drains all collected commands, returning them in push order.
    pub fn drain(&self) -> Vec<Command> {
        let mut commands = self.commands.lock().unwrap();
        std::mem::take(&mut *commands)
    }
}

/// Builder for spawning an entity with multiple components via [`CommandCollector`].
///
/// Created by [`CommandCollector::spawn_entity`]. The entity is spawned
/// and all components are inserted in a single command when [`build`](SpawnBuilder::build)
/// is called.
pub struct SpawnBuilder<'a> {
    collector: &'a CommandCollector,
    inserts: Vec<InsertFn>,
}

impl<'a> SpawnBuilder<'a> {
    /// Adds a component to the entity being built.
    ///
    /// # Panics
    ///
    /// Panics when applied if the component type has not been registered.
    pub fn with<T: Send + Sync + 'static>(mut self, component: T) -> Self {
        self.inserts.push(Box::new(move |world, entity| {
            world
                .insert(entity, component)
                .expect("Component not registered");
        }));
        self
    }

    /// Finalizes the builder, queuing the spawn command.
    pub fn build(self) {
        let inserts = self.inserts;
        self.collector.push(move |world| {
            let entity = world.spawn();
            for insert_fn in inserts {
                insert_fn(world, entity);
            }
        });
    }
}

impl Default for CommandCollector {
    fn default() -> Self {
        Self::new()
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
    struct Health(u32);

    fn apply(collector: &CommandCollector, world: &mut World) {
        for cmd in collector.drain() {
            cmd(world);
        }
    }

    #[test]
    fn push_and_drain() {
        let collector = CommandCollector::new();
        collector.push(|world| {
            world.insert_resource(42u32);
        });
        collector.push(|world| {
            world.insert_resource("hello");
        });

        let commands = collector.drain();
        assert_eq!(commands.len(), 2);

        // Drain again should be empty
        let commands = collector.drain();
        assert!(commands.is_empty());
    }

    #[test]
    fn commands_apply_to_world() {
        let collector = CommandCollector::new();
        collector.push(|world| {
            world.insert_resource(42u32);
        });

        let mut world = World::new();
        apply(&collector, &mut world);

        let val = world.resource::<u32>();
        assert_eq!(*val, 42);
    }

    #[test]
    fn despawn_command() {
        let mut world = World::new();
        let entity = world.spawn();
        assert!(world.is_alive(entity));

        let collector = CommandCollector::new();
        collector.despawn(entity);
        apply(&collector, &mut world);

        assert!(!world.is_alive(entity));
    }

    #[test]
    fn insert_command() {
        let mut world = World::new();
        world.register_component::<Position>();
        let entity = world.spawn();

        let collector = CommandCollector::new();
        collector.insert(entity, Position { x: 1.0, y: 2.0 });
        apply(&collector, &mut world);

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 1.0, y: 2.0 })
        );
    }

    #[test]
    fn remove_command() {
        let mut world = World::new();
        world.register_component::<Health>();
        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        let collector = CommandCollector::new();
        collector.remove::<Health>(entity);
        apply(&collector, &mut world);

        assert!(world.get::<Health>(entity).is_none());
    }

    #[test]
    fn spawn_entity_builder() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();

        let collector = CommandCollector::new();
        collector
            .spawn_entity()
            .with(Position { x: 5.0, y: 10.0 })
            .with(Health(100))
            .build();
        apply(&collector, &mut world);

        assert_eq!(world.entity_count(), 1);

        let entity = world.iter_entities().next().unwrap();
        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 5.0, y: 10.0 })
        );
        assert_eq!(world.get::<Health>(entity), Some(&Health(100)));
    }

    #[test]
    fn commands_execute_in_order() {
        let mut world = World::new();
        world.register_component::<Health>();
        let entity = world.spawn();

        let collector = CommandCollector::new();
        collector.insert(entity, Health(100));
        collector.push(move |world| {
            world.get_mut::<Health>(entity).unwrap().0 += 50;
        });
        apply(&collector, &mut world);

        assert_eq!(world.get::<Health>(entity), Some(&Health(150)));
    }
}
