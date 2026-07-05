use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// Double-buffered event queue with per-reader cursors.
///
/// Every event gets a sequential id from a per-queue counter that never
/// resets. Readers track their position with an [`EventCursor`], so each
/// reader sees every event **exactly once**, regardless of whether it is
/// ordered before or after the sender.
///
/// # Usage pattern
///
/// 1. Register the event type: `world.add_event::<MyEvent>()`
/// 2. Systems send events via `ResMut<Events<MyEvent>>` + `send(...)`
/// 3. Each reader system owns an [`EventCursor<MyEvent>`] field and reads
///    via `events.read(&self.cursor)`
///
/// ```ignore
/// struct DamageSystem {
///     cursor: EventCursor<Damage>,
/// }
///
/// impl System for DamageSystem {
///     type Result = ();
///     fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
///         ctx.lock::<(Res<Events<Damage>>,)>().execute(|(events,)| {
///             for damage in events.read(&self.cursor) {
///                 // seen exactly once, even across frames
///             }
///         });
///         Ok(())
///     }
/// }
/// ```
///
/// # Buffering and lifetime
///
/// Events live for two [`update`](Self::update) calls (the current and the
/// previous frame), which bounds memory. `update()` is driven automatically:
/// [`World::add_event`](crate::World::add_event) registers the queue and
/// [`Schedules::run_frame`](crate::Schedules::run_frame) advances all
/// registered queues once at the start of each frame (via
/// [`World::update_events`](crate::World::update_events) — call that
/// manually when running containers without `run_frame`).
///
/// A reader that stalls for more than a frame (e.g. its schedule did not
/// run) skips the dropped events; the loss is recorded on its cursor
/// ([`EventCursor::missed`]).
pub struct Events<T: Send + Sync + 'static> {
    /// Events sent this frame.
    current: Vec<T>,
    /// Events from the previous frame (still readable by lagging readers).
    previous: Vec<T>,
    /// Id of the first event in `previous`. Ids grow monotonically and
    /// never reset, so cursor positions stay meaningful across frames.
    oldest_id: u64,
}

impl<T: Send + Sync + 'static> Events<T> {
    /// Creates a new empty event queue.
    pub fn new() -> Self {
        Self {
            current: Vec::new(),
            previous: Vec::new(),
            oldest_id: 0,
        }
    }

    /// Id of the first event in `current`.
    fn current_start(&self) -> u64 {
        self.oldest_id + self.previous.len() as u64
    }

    /// Id that the next sent event will receive.
    fn next_id(&self) -> u64 {
        self.current_start() + self.current.len() as u64
    }

    /// Sends an event, adding it to the current frame's buffer.
    pub fn send(&mut self, event: T) {
        self.current.push(event);
    }

    /// Reads all events this cursor has not seen yet, oldest first.
    ///
    /// The cursor advances as the iterator is consumed, so an early
    /// `break` resumes where it left off on the next `read` call. Events
    /// already dropped from the buffer (the reader lagged more than a
    /// frame) are skipped and recorded via [`EventCursor::missed`].
    pub fn read<'a>(&'a self, cursor: &'a EventCursor<T>) -> EventIter<'a, T> {
        let mut next = cursor.next.load(Ordering::Relaxed);
        if next < self.oldest_id {
            cursor
                .missed
                .fetch_add(self.oldest_id - next, Ordering::Relaxed);
            next = self.oldest_id;
            // Store immediately so an unconsumed iterator does not count
            // the same loss again on the next `read` call.
            cursor.next.store(next, Ordering::Relaxed);
        }
        EventIter {
            events: self,
            cursor,
            next,
        }
    }

    /// Fetches an event by id. `None` once `id` runs past `current`.
    ///
    /// Callers guarantee `id >= oldest_id` (an [`EventIter`] holds `&self`,
    /// so the buffer cannot advance under it).
    fn get_by_id(&self, id: u64) -> Option<&T> {
        debug_assert!(id >= self.oldest_id);
        let current_start = self.current_start();
        if id < current_start {
            self.previous.get((id - self.oldest_id) as usize)
        } else {
            self.current.get((id - current_start) as usize)
        }
    }

    /// Advances the double buffer: drops `previous`, moves `current` into
    /// `previous`.
    ///
    /// Driven once per frame by [`World::update_events`](crate::World::update_events)
    /// (which `Schedules::run_frame` calls at the start of each frame).
    pub fn update(&mut self) {
        self.oldest_id += self.previous.len() as u64;
        self.previous.clear();
        std::mem::swap(&mut self.current, &mut self.previous);
    }

    /// Returns true if no events are buffered.
    ///
    /// This is about buffer occupancy, not about any particular reader's
    /// position — use [`read`](Self::read) for consumption.
    pub fn is_empty(&self) -> bool {
        self.current.is_empty() && self.previous.is_empty()
    }

    /// Returns the number of buffered events (both frames).
    pub fn len(&self) -> usize {
        self.current.len() + self.previous.len()
    }

    /// Drops all buffered events.
    ///
    /// Ids keep advancing: unread events count as missed for readers that
    /// had not consumed them yet.
    pub fn clear(&mut self) {
        self.oldest_id = self.next_id();
        self.previous.clear();
        self.current.clear();
    }
}

impl<T: Send + Sync + 'static> Default for Events<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A reader's position in an [`Events<T>`] queue.
///
/// Owned by the reader system (typically a struct field) and passed to
/// [`Events::read`]. Interior mutability (atomics) lets the cursor advance
/// from the `&self` context of [`System::run`](crate::System::run); a
/// system runs at most once at a time, so `Relaxed` ordering suffices
/// (cross-thread visibility comes from the runner's lock/join edges).
///
/// A fresh cursor starts at the beginning of the queue: it reads whatever
/// is still buffered and counts anything older as missed.
pub struct EventCursor<T: Send + Sync + 'static> {
    /// Id of the next unread event.
    next: AtomicU64,
    /// Total number of events dropped from the buffer before this reader
    /// consumed them.
    missed: AtomicU64,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Send + Sync + 'static> EventCursor<T> {
    /// Creates a cursor positioned at the start of the queue.
    pub fn new() -> Self {
        Self {
            next: AtomicU64::new(0),
            missed: AtomicU64::new(0),
            _marker: PhantomData,
        }
    }

    /// Total number of events this reader lost by lagging more than a
    /// frame behind (they were dropped from the buffer unread).
    ///
    /// Monotonically increasing; diff it across frames to detect fresh
    /// losses worth a warning.
    pub fn missed(&self) -> u64 {
        self.missed.load(Ordering::Relaxed)
    }
}

impl<T: Send + Sync + 'static> Default for EventCursor<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over unread events, created by [`Events::read`].
///
/// Advances the [`EventCursor`] one event at a time as it is consumed.
pub struct EventIter<'a, T: Send + Sync + 'static> {
    events: &'a Events<T>,
    cursor: &'a EventCursor<T>,
    next: u64,
}

impl<'a, T: Send + Sync + 'static> Iterator for EventIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let item = self.events.get_by_id(self.next)?;
        self.next += 1;
        self.cursor.next.store(self.next, Ordering::Relaxed);
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.events.next_id() - self.next) as usize;
        (remaining, Some(remaining))
    }
}

impl<T: Send + Sync + 'static> ExactSizeIterator for EventIter<'_, T> {}

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

    fn values(events: &Events<TestEvent>, cursor: &EventCursor<TestEvent>) -> Vec<u32> {
        events.read(cursor).map(|e| e.value).collect()
    }

    #[test]
    fn new_events_empty() {
        let events = Events::<TestEvent>::new();
        assert!(events.is_empty());
        assert_eq!(events.len(), 0);
        let cursor = EventCursor::new();
        assert_eq!(values(&events, &cursor), Vec::<u32>::new());
    }

    #[test]
    fn read_sees_events_exactly_once() {
        let mut events = Events::<TestEvent>::new();
        let cursor = EventCursor::new();

        events.send(TestEvent { value: 1 });
        events.send(TestEvent { value: 2 });

        assert_eq!(values(&events, &cursor), vec![1, 2]);
        // Second read: nothing new.
        assert_eq!(values(&events, &cursor), Vec::<u32>::new());

        // Same events still buffered next frame — but already consumed.
        events.update();
        assert_eq!(values(&events, &cursor), Vec::<u32>::new());
        assert_eq!(cursor.missed(), 0);
    }

    #[test]
    fn reader_ordered_before_sender_sees_events_next_frame() {
        let mut events = Events::<TestEvent>::new();
        let cursor = EventCursor::new();

        // Frame 1: reader runs first (nothing yet), sender sends after.
        assert_eq!(values(&events, &cursor), Vec::<u32>::new());
        events.send(TestEvent { value: 1 });

        // Frame 2: buffer advances, reader picks the event up exactly once.
        events.update();
        assert_eq!(values(&events, &cursor), vec![1]);

        // Frame 3: gone, no double delivery.
        events.update();
        assert_eq!(values(&events, &cursor), Vec::<u32>::new());
        assert_eq!(cursor.missed(), 0);
    }

    #[test]
    fn independent_readers_each_see_everything() {
        let mut events = Events::<TestEvent>::new();
        let fast = EventCursor::new();
        let slow = EventCursor::new();

        events.send(TestEvent { value: 1 });
        assert_eq!(values(&events, &fast), vec![1]);

        events.update();
        events.send(TestEvent { value: 2 });
        assert_eq!(values(&events, &fast), vec![2]);
        // The slow reader skipped a frame but both events are buffered.
        assert_eq!(values(&events, &slow), vec![1, 2]);
    }

    #[test]
    fn lagging_reader_skips_and_counts_missed() {
        let mut events = Events::<TestEvent>::new();
        let cursor = EventCursor::new();

        events.send(TestEvent { value: 1 });
        events.send(TestEvent { value: 2 });
        events.update();
        events.send(TestEvent { value: 3 });
        events.update(); // 1 and 2 dropped unread
        events.send(TestEvent { value: 4 });

        assert_eq!(values(&events, &cursor), vec![3, 4]);
        assert_eq!(cursor.missed(), 2);

        // The loss is not double-counted by later reads.
        assert_eq!(values(&events, &cursor), Vec::<u32>::new());
        assert_eq!(cursor.missed(), 2);
    }

    #[test]
    fn missed_not_double_counted_without_consuming() {
        let mut events = Events::<TestEvent>::new();
        let cursor = EventCursor::new();

        events.send(TestEvent { value: 1 });
        events.update();
        events.update(); // dropped unread

        // Create the iterator but consume nothing.
        let _ = events.read(&cursor);
        assert_eq!(cursor.missed(), 1);
        let _ = events.read(&cursor);
        assert_eq!(cursor.missed(), 1);
    }

    #[test]
    fn early_break_resumes() {
        let mut events = Events::<TestEvent>::new();
        let cursor = EventCursor::new();

        events.send(TestEvent { value: 1 });
        events.send(TestEvent { value: 2 });
        events.send(TestEvent { value: 3 });

        let first: Vec<u32> = events.read(&cursor).take(1).map(|e| e.value).collect();
        assert_eq!(first, vec![1]);
        assert_eq!(values(&events, &cursor), vec![2, 3]);
    }

    #[test]
    fn read_spanning_both_buffers() {
        let mut events = Events::<TestEvent>::new();
        let cursor = EventCursor::new();

        events.send(TestEvent { value: 1 });
        events.update();
        events.send(TestEvent { value: 2 });

        let iter = events.read(&cursor);
        assert_eq!(iter.len(), 2); // ExactSizeIterator
        assert_eq!(iter.map(|e| e.value).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn clear_drops_unread_as_missed() {
        let mut events = Events::<TestEvent>::new();
        let cursor = EventCursor::new();

        events.send(TestEvent { value: 1 });
        events.update();
        events.send(TestEvent { value: 2 });
        events.clear();

        assert!(events.is_empty());
        assert_eq!(values(&events, &cursor), Vec::<u32>::new());
        assert_eq!(cursor.missed(), 2);
    }

    #[test]
    fn update_bounds_memory() {
        let mut events = Events::<TestEvent>::new();
        events.send(TestEvent { value: 1 });
        assert_eq!(events.len(), 1);
        events.update();
        assert_eq!(events.len(), 1);
        events.update();
        assert!(events.is_empty());
    }

    #[test]
    fn world_add_event() {
        let mut world = World::new();
        world.add_event::<TestEvent>();
        assert!(world.has_resource::<Events<TestEvent>>());
    }

    #[test]
    fn add_event_is_idempotent() {
        let mut world = World::new();
        world.add_event::<TestEvent>();
        world.add_event::<TestEvent>(); // must NOT register a second update fn

        let cursor = EventCursor::new();
        world
            .resource_mut::<Events<TestEvent>>()
            .send(TestEvent { value: 7 });

        // One frame boundary. With a duplicate update fn the buffer would
        // advance twice and drop the event unread.
        world.update_events();

        let events = world.resource::<Events<TestEvent>>();
        assert_eq!(values(&events, &cursor), vec![7]);
        assert_eq!(cursor.missed(), 0);
    }

    #[test]
    fn update_events_advances_all_queues() {
        let mut world = World::new();
        world.add_event::<TestEvent>();
        world.add_event::<OtherEvent>();

        world
            .resource_mut::<Events<TestEvent>>()
            .send(TestEvent { value: 1 });
        world
            .resource_mut::<Events<OtherEvent>>()
            .send(OtherEvent("hello".to_string()));

        world.update_events();
        world.update_events();

        assert!(world.resource::<Events<TestEvent>>().is_empty());
        assert!(world.resource::<Events<OtherEvent>>().is_empty());
    }
}
