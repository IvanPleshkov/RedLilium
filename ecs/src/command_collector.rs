use std::sync::Mutex;

use crate::world::World;

/// A boxed deferred command that mutates the World.
type Command = Box<dyn FnOnce(&mut World) + Send>;

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
    /// The command will be applied to `&mut World` after all systems complete.
    pub fn push(&self, cmd: impl FnOnce(&mut World) + Send + 'static) {
        self.commands.lock().unwrap().push(Box::new(cmd));
    }

    /// Drains all collected commands, returning them in push order.
    pub fn drain(&self) -> Vec<Command> {
        let mut commands = self.commands.lock().unwrap();
        std::mem::take(&mut *commands)
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
        for cmd in collector.drain() {
            cmd(&mut world);
        }

        let val = world.resource::<u32>();
        assert_eq!(*val, 42);
    }
}
