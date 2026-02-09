use crate::access::Access;
use crate::world::World;

/// A system function that operates on the world.
///
/// Systems are the primary way to process entities and components.
/// Each system declares its component access and runs once per
/// schedule execution.
pub trait System: 'static {
    /// Execute the system with access to the world.
    fn run(&self, world: &World);

    /// Returns a human-readable name for this system.
    fn name(&self) -> &str;

    /// Returns the access descriptor for this system.
    fn access(&self) -> &Access;
}

/// A system built from a closure or function pointer.
pub(crate) struct FnSystem {
    pub name: String,
    pub access: Access,
    pub after: Vec<String>,
    pub before: Vec<String>,
    func: Box<dyn Fn(&World) + Send + Sync>,
}

impl FnSystem {
    /// Executes the system function.
    pub fn run(&self, world: &World) {
        (self.func)(world);
    }
}

impl System for FnSystem {
    fn run(&self, world: &World) {
        self.run(world);
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn access(&self) -> &Access {
        &self.access
    }
}

/// Builder for constructing and registering systems with access declarations.
///
/// Created by [`Schedule::add_system`](crate::Schedule::add_system).
///
/// # Example
///
/// ```ignore
/// schedule.add_system("physics")
///     .writes::<Transform>()
///     .reads::<RigidBody>()
///     .after("input")
///     .run(physics_system);
/// ```
pub struct SystemBuilder<'a> {
    systems: &'a mut Vec<FnSystem>,
    name: String,
    access: Access,
    after: Vec<String>,
    before: Vec<String>,
}

impl<'a> SystemBuilder<'a> {
    /// Creates a new system builder that will push into the given systems vec.
    pub(crate) fn new(systems: &'a mut Vec<FnSystem>, name: String) -> Self {
        Self {
            systems,
            name,
            access: Access::new(),
            after: Vec::new(),
            before: Vec::new(),
        }
    }

    /// Declares that this system reads component type T.
    pub fn reads<T: 'static>(mut self) -> Self {
        self.access.add_read::<T>();
        self
    }

    /// Declares that this system writes component type T.
    pub fn writes<T: 'static>(mut self) -> Self {
        self.access.add_write::<T>();
        self
    }

    /// Declares that this system reads resource type T.
    pub fn reads_resource<T: 'static>(mut self) -> Self {
        self.access.add_resource_read::<T>();
        self
    }

    /// Declares that this system writes resource type T.
    pub fn writes_resource<T: 'static>(mut self) -> Self {
        self.access.add_resource_write::<T>();
        self
    }

    /// Declares that this system must run after the named system.
    pub fn after(mut self, name: &str) -> Self {
        self.after.push(name.to_string());
        self
    }

    /// Declares that this system must run before the named system.
    pub fn before(mut self, name: &str) -> Self {
        self.before.push(name.to_string());
        self
    }

    /// Completes registration with the given system function.
    ///
    /// The function receives `&World` and accesses components through
    /// runtime-borrow-checked queries.
    pub fn run(self, func: impl Fn(&World) + Send + Sync + 'static) {
        self.systems.push(FnSystem {
            name: self.name,
            access: self.access,
            after: self.after,
            before: self.before,
            func: Box::new(func),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Position;
    struct Velocity;

    #[test]
    fn builder_builds_system() {
        let mut systems = Vec::new();

        SystemBuilder::new(&mut systems, "test".to_string())
            .reads::<Position>()
            .writes::<Velocity>()
            .after("other")
            .run(|_world| {});

        assert_eq!(systems.len(), 1);
        assert_eq!(systems[0].name, "test");
        assert_eq!(systems[0].after, vec!["other"]);
        assert!(!systems[0].access.is_read_only());
    }

    #[test]
    fn system_runs() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let ran = std::sync::Arc::new(AtomicBool::new(false));
        let ran_clone = ran.clone();

        let mut systems = Vec::new();
        SystemBuilder::new(&mut systems, "test".to_string()).run(move |_world| {
            ran_clone.store(true, Ordering::Relaxed);
        });

        let world = World::new();
        systems[0].run(&world);
        assert!(ran.load(Ordering::Relaxed));
    }

    #[test]
    fn read_only_access() {
        let mut systems = Vec::new();
        SystemBuilder::new(&mut systems, "reader".to_string())
            .reads::<Position>()
            .reads::<Velocity>()
            .run(|_| {});

        assert!(systems[0].access.is_read_only());
    }
}
