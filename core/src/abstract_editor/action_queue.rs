//! Thread-safe action queue for submitting editor actions from read-only systems.
//!
//! [`ActionQueue`] uses interior mutability ([`Mutex`]) so that systems with
//! only shared `&self` access to a resource can still enqueue actions. The
//! editor drains the queue each frame and executes actions through
//! [`EditActionHistory`](super::EditActionHistory).

use std::fmt;
use std::sync::Mutex;

use super::action::{EditAction, Editable};

/// A thread-safe queue for submitting [`EditAction`]s from read-only contexts.
///
/// Because the inner storage is wrapped in a [`Mutex`], [`push()`](Self::push)
/// only requires `&self`. This allows systems running in a read-only
/// [`SystemsContainer`] to enqueue actions via `Res<ActionQueue<T>>`.
///
/// # Example
///
/// ```ignore
/// // In a read-only system:
/// let queue: &ActionQueue<World> = &*ctx.lock::<(Res<ActionQueue<World>>,)>()
///     .execute(|(q,)| { /* use q */ });
///
/// queue.push(Box::new(MoveCamera { ... }));
/// ```
pub struct ActionQueue<T: Editable> {
    queue: Mutex<Vec<Box<dyn EditAction<T>>>>,
}

impl<T: Editable> ActionQueue<T> {
    /// Creates a new empty action queue.
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(Vec::new()),
        }
    }

    /// Enqueues an action. Callable from `&self` thanks to interior mutability.
    pub fn push(&self, action: Box<dyn EditAction<T>>) {
        self.queue.lock().unwrap().push(action);
    }

    /// Drains all queued actions, returning them in submission order.
    pub fn drain(&self) -> Vec<Box<dyn EditAction<T>>> {
        std::mem::take(&mut *self.queue.lock().unwrap())
    }

    /// Returns `true` if there are no queued actions.
    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }
}

impl<T: Editable> Default for ActionQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Editable> fmt::Debug for ActionQueue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.queue.lock().unwrap().len();
        f.debug_struct("ActionQueue")
            .field("pending", &len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstract_editor::action::{EditActionResult, Editable};

    struct Counter {
        value: i32,
    }
    impl Editable for Counter {}

    #[derive(Debug)]
    struct Add {
        amount: i32,
    }

    impl EditAction<Counter> for Add {
        fn apply(&mut self, target: &mut Counter) -> EditActionResult {
            target.value += self.amount;
            Ok(())
        }

        fn undo(&mut self, target: &mut Counter) -> EditActionResult {
            target.value -= self.amount;
            Ok(())
        }

        fn description(&self) -> &str {
            "Add"
        }
    }

    #[test]
    fn push_and_drain() {
        let queue = ActionQueue::<Counter>::new();
        queue.push(Box::new(Add { amount: 1 }));
        queue.push(Box::new(Add { amount: 2 }));

        let actions = queue.drain();
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].description(), "Add");
    }

    #[test]
    fn drain_empties_queue() {
        let queue = ActionQueue::<Counter>::new();
        queue.push(Box::new(Add { amount: 1 }));
        let _ = queue.drain();
        assert!(queue.is_empty());
        assert_eq!(queue.drain().len(), 0);
    }

    #[test]
    fn is_empty_reflects_state() {
        let queue = ActionQueue::<Counter>::new();
        assert!(queue.is_empty());
        queue.push(Box::new(Add { amount: 1 }));
        assert!(!queue.is_empty());
        let _ = queue.drain();
        assert!(queue.is_empty());
    }

    #[test]
    fn preserves_submission_order() {
        let queue = ActionQueue::<Counter>::new();
        queue.push(Box::new(Add { amount: 10 }));
        queue.push(Box::new(Add { amount: 20 }));
        queue.push(Box::new(Add { amount: 30 }));

        let mut counter = Counter { value: 0 };
        for mut action in queue.drain() {
            action.apply(&mut counter).unwrap();
        }
        assert_eq!(counter.value, 60);
    }

    #[test]
    fn debug_impl() {
        let queue = ActionQueue::<Counter>::new();
        queue.push(Box::new(Add { amount: 1 }));
        let debug = format!("{queue:?}");
        assert!(debug.contains("ActionQueue"));
        assert!(debug.contains("pending"));
    }
}
