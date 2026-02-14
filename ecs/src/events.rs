/// Double-buffered event queue for typed inter-system communication.
///
/// Events are sent during one frame and can be read during the same frame
/// and the next. After two `update()` calls, events are dropped.
///
/// # Usage pattern
///
/// 1. Register the event type: `world.add_event::<MyEvent>()`
/// 2. Add an update system at the start of the schedule (or call `update()` manually)
/// 3. Systems send events via `resource_mut::<Events<MyEvent>>().send(...)`
/// 4. Systems read events via `resource::<Events<MyEvent>>().iter()`
///
/// # Double buffering
///
/// - `current`: events sent this frame
/// - `previous`: events from last frame (still readable)
/// - `update()`: clears previous, swaps current → previous
///
/// This ensures events survive for at least one full frame after being sent,
/// so systems that run before the sender can still read last frame's events.
pub struct Events<T: Send + Sync + 'static> {
    current: Vec<T>,
    previous: Vec<T>,
}

impl<T: Send + Sync + 'static> Events<T> {
    /// Creates a new empty event queue.
    pub fn new() -> Self {
        Self {
            current: Vec::new(),
            previous: Vec::new(),
        }
    }

    /// Sends an event, adding it to the current frame's buffer.
    pub fn send(&mut self, event: T) {
        self.current.push(event);
    }

    /// Iterates over events from both buffers (previous then current).
    ///
    /// This returns all events that haven't been cleared yet — both
    /// from the current frame and the previous frame.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.previous.iter().chain(self.current.iter())
    }

    /// Iterates over only current-frame events.
    pub fn iter_current(&self) -> impl Iterator<Item = &T> {
        self.current.iter()
    }

    /// Advances the double buffer: clears previous, swaps current → previous.
    ///
    /// Call this at the start of each frame (typically via
    /// [`EventUpdateSystem`]).
    pub fn update(&mut self) {
        self.previous.clear();
        std::mem::swap(&mut self.current, &mut self.previous);
    }

    /// Returns true if no events exist in either buffer.
    pub fn is_empty(&self) -> bool {
        self.current.is_empty() && self.previous.is_empty()
    }

    /// Returns the total event count across both buffers.
    pub fn len(&self) -> usize {
        self.current.len() + self.previous.len()
    }

    /// Clears all events from both buffers.
    pub fn clear(&mut self) {
        self.current.clear();
        self.previous.clear();
    }
}

impl<T: Send + Sync + 'static> Default for Events<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A system that advances the [`Events<T>`] double buffer.
///
/// Register at the start of your schedule so events from the previous
/// frame are cleared and the current buffer becomes readable as
/// "previous" for systems that run later.
///
/// # Example
///
/// ```ignore
/// schedule.add(EventUpdateSystem::<CollisionEvent>::new())
///     .before::<PhysicsSystem>();
/// ```
pub struct EventUpdateSystem<T: Send + Sync + 'static> {
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: Send + Sync + 'static> EventUpdateSystem<T> {
    /// Creates a new event update system for event type `T`.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Send + Sync + 'static> Default for EventUpdateSystem<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + Sync + 'static> crate::system::System for EventUpdateSystem<T> {
    type Result = ();
    async fn run<'a>(&'a self, ctx: &'a crate::system_context::SystemContext<'a>) {
        ctx.lock::<(crate::access_set::ResMut<Events<T>>,)>()
            .execute(|(mut events,)| {
                events.update();
            })
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    #[derive(Debug, PartialEq, Clone)]
    struct TestEvent {
        value: u32,
    }

    #[derive(Debug, PartialEq)]
    struct OtherEvent(String);

    #[test]
    fn new_events_empty() {
        let events = Events::<TestEvent>::new();
        assert!(events.is_empty());
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn send_and_iter_current() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent { value: 1 });
        events.send(TestEvent { value: 2 });

        assert_eq!(events.len(), 2);
        assert!(!events.is_empty());

        let values: Vec<u32> = events.iter_current().map(|e| e.value).collect();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn iter_includes_both_buffers() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent { value: 1 });
        events.update(); // 1 moves to previous
        events.send(TestEvent { value: 2 });

        let values: Vec<u32> = events.iter().map(|e| e.value).collect();
        assert_eq!(values, vec![1, 2]);

        // iter_current only has 2
        let current: Vec<u32> = events.iter_current().map(|e| e.value).collect();
        assert_eq!(current, vec![2]);
    }

    #[test]
    fn update_clears_old_events() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent { value: 1 });
        events.update(); // 1 → previous
        events.send(TestEvent { value: 2 });
        events.update(); // 2 → previous, 1 cleared

        // Only event 2 remains (in previous)
        let values: Vec<u32> = events.iter().map(|e| e.value).collect();
        assert_eq!(values, vec![2]);
    }

    #[test]
    fn double_update_clears_all() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent { value: 1 });
        events.update();
        events.update();

        assert!(events.is_empty());
    }

    #[test]
    fn clear_removes_everything() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent { value: 1 });
        events.update();
        events.send(TestEvent { value: 2 });
        events.clear();

        assert!(events.is_empty());
        assert_eq!(events.len(), 0);
    }

    #[test]
    fn event_update_system_works() {
        let mut world = World::new();
        world.add_event::<TestEvent>();

        // Send an event
        {
            let mut events = world.resource_mut::<Events<TestEvent>>();
            events.send(TestEvent { value: 42 });
        }

        // Run update system
        use crate::compute::ComputePool;
        use crate::io_runtime::IoRuntime;
        use crate::system::run_system_blocking;
        let update = EventUpdateSystem::<TestEvent>::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&update, &world, &compute, &io);

        // Event should be in previous now
        let events = world.resource::<Events<TestEvent>>();
        let values: Vec<u32> = events.iter().map(|e| e.value).collect();
        assert_eq!(values, vec![42]);

        // Current should be empty
        let current: Vec<u32> = events.iter_current().map(|e| e.value).collect();
        assert!(current.is_empty());
    }

    #[test]
    fn world_add_event() {
        let mut world = World::new();
        world.add_event::<TestEvent>();
        assert!(world.has_resource::<Events<TestEvent>>());
    }

    #[test]
    fn multiple_event_types() {
        let mut world = World::new();
        world.add_event::<TestEvent>();
        world.add_event::<OtherEvent>();

        {
            let mut test_events = world.resource_mut::<Events<TestEvent>>();
            test_events.send(TestEvent { value: 1 });
        }
        {
            let mut other_events = world.resource_mut::<Events<OtherEvent>>();
            other_events.send(OtherEvent("hello".to_string()));
        }

        let test_events = world.resource::<Events<TestEvent>>();
        assert_eq!(test_events.len(), 1);

        let other_events = world.resource::<Events<OtherEvent>>();
        assert_eq!(other_events.len(), 1);
    }

    #[test]
    fn frame_lifecycle() {
        let mut events = Events::<TestEvent>::new();

        // Frame 1: send events
        events.send(TestEvent { value: 1 });
        events.send(TestEvent { value: 2 });
        assert_eq!(events.len(), 2);

        // Frame 2: update, send more
        events.update();
        events.send(TestEvent { value: 3 });
        // All three visible
        assert_eq!(events.len(), 3);

        // Frame 3: update again — frame 1 events gone
        events.update();
        let values: Vec<u32> = events.iter().map(|e| e.value).collect();
        assert_eq!(values, vec![3]);

        // Frame 4: update — frame 2 events gone
        events.update();
        assert!(events.is_empty());
    }
}
