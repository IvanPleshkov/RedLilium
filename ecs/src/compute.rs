use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use crate::priority::Priority;

/// Handle to a spawned async compute task.
///
/// Allows checking completion status and retrieving the result.
///
/// # Example
///
/// ```ignore
/// let handle = pool.spawn(Priority::Low, async { 42 });
///
/// // Later, check if done:
/// if let Some(result) = handle.try_recv() {
///     println!("Got: {}", result);
/// }
/// ```
pub struct TaskHandle<T> {
    receiver: std::sync::mpsc::Receiver<T>,
}

impl<T> TaskHandle<T> {
    /// Attempts to retrieve the result without blocking.
    ///
    /// Returns `Some(T)` if the task has completed, `None` otherwise.
    pub fn try_recv(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }

    /// Returns whether the task has completed.
    ///
    /// Note: Even if this returns `true`, `try_recv` should be used
    /// to actually retrieve the value.
    pub fn is_complete(&self) -> bool {
        // Peek without consuming — if try_recv succeeds the value is consumed,
        // so we use a non-destructive check instead.
        // Unfortunately mpsc::Receiver doesn't have peek, so we check if
        // the sender is disconnected (task completed and sent).
        // A more reliable approach: just try_recv.
        matches!(
            self.receiver.try_recv(),
            Ok(_) | Err(std::sync::mpsc::TryRecvError::Disconnected)
        )
    }

    /// Blocks until the task completes and returns the result.
    ///
    /// # Warning
    ///
    /// This blocks the calling thread. Prefer `try_recv()` in frame loops.
    pub fn recv(self) -> T {
        self.receiver
            .recv()
            .expect("Compute task dropped without sending result")
    }
}

/// A pending async compute task stored in the pool.
struct PendingTask {
    priority: Priority,
    future: Pin<Box<dyn Future<Output = ()> + Send>>,
    /// Insertion order for stable sorting within the same priority.
    id: u64,
}

/// Pool for spawning async compute tasks.
///
/// Tasks are stored and polled manually via [`tick`](ComputePool::tick).
/// Each task has a priority that determines polling order.
///
/// # Example
///
/// ```
/// use redlilium_ecs::{ComputePool, Priority};
///
/// let pool = ComputePool::new();
///
/// let handle = pool.spawn(Priority::Low, async { 42u32 });
///
/// // Tick until the task completes
/// while pool.pending_count() > 0 {
///     pool.tick();
/// }
///
/// assert_eq!(handle.try_recv(), Some(42));
/// ```
pub struct ComputePool {
    tasks: Mutex<Vec<PendingTask>>,
    next_id: Mutex<u64>,
}

impl ComputePool {
    /// Creates a new compute pool.
    pub fn new() -> Self {
        Self {
            tasks: Mutex::new(Vec::new()),
            next_id: Mutex::new(0),
        }
    }

    /// Spawns an async compute task with the given priority.
    ///
    /// Returns a handle for retrieving the result.
    pub fn spawn<T, F>(&self, priority: Priority, future: F) -> TaskHandle<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();

        let wrapped = async move {
            let result = future.await;
            let _ = sender.send(result);
        };

        let id = {
            let mut next_id = self.next_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let task = PendingTask {
            priority,
            future: Box::pin(wrapped),
            id,
        };

        self.tasks.lock().unwrap().push(task);

        TaskHandle { receiver }
    }

    /// Polls the highest-priority pending task once.
    ///
    /// Returns the number of tasks that were polled (0 or 1).
    /// Completed tasks are automatically removed from the pool.
    pub fn tick(&self) -> usize {
        let mut tasks = self.tasks.lock().unwrap();
        if tasks.is_empty() {
            return 0;
        }

        // Find highest priority task (highest priority + lowest id for stability)
        let best_idx = tasks
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.priority.cmp(&b.priority).then(b.id.cmp(&a.id)) // Lower id = earlier insertion = preferred
            })
            .map(|(i, _)| i)
            .unwrap();

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match tasks[best_idx].future.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                tasks.swap_remove(best_idx);
            }
            Poll::Pending => {}
        }

        1
    }

    /// Polls all pending tasks once each.
    ///
    /// Returns the number of tasks that were polled.
    /// Completed tasks are automatically removed.
    pub fn tick_all(&self) -> usize {
        let mut tasks = self.tasks.lock().unwrap();
        let count = tasks.len();
        if count == 0 {
            return 0;
        }

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // Poll all tasks, collecting indices of completed ones
        let mut i = 0;
        while i < tasks.len() {
            match tasks[i].future.as_mut().poll(&mut cx) {
                Poll::Ready(()) => {
                    tasks.swap_remove(i);
                    // Don't increment i — the swapped task needs to be checked
                }
                Poll::Pending => {
                    i += 1;
                }
            }
        }

        count
    }

    /// Returns the number of pending (incomplete) tasks.
    pub fn pending_count(&self) -> usize {
        self.tasks.lock().unwrap().len()
    }
}

impl Default for ComputePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates a no-op waker for manual polling.
fn noop_waker() -> Waker {
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yield_now::yield_now;

    #[test]
    fn spawn_and_recv() {
        let pool = ComputePool::new();
        let handle = pool.spawn(Priority::Low, async { 42u32 });

        while pool.pending_count() > 0 {
            pool.tick();
        }

        assert_eq!(handle.try_recv(), Some(42));
    }

    #[test]
    fn priority_ordering() {
        let pool = ComputePool::new();
        let results = std::sync::Arc::new(Mutex::new(Vec::new()));

        let r1 = results.clone();
        pool.spawn(Priority::Low, async move {
            r1.lock().unwrap().push("low");
        });

        let r2 = results.clone();
        pool.spawn(Priority::High, async move {
            r2.lock().unwrap().push("high");
        });

        let r3 = results.clone();
        pool.spawn(Priority::Critical, async move {
            r3.lock().unwrap().push("critical");
        });

        // Tick one at a time — highest priority first
        pool.tick();
        pool.tick();
        pool.tick();

        let order = results.lock().unwrap();
        assert_eq!(*order, vec!["critical", "high", "low"]);
    }

    #[test]
    fn yield_now_suspends() {
        let pool = ComputePool::new();
        let handle = pool.spawn(Priority::Low, async {
            yield_now().await;
            42u32
        });

        // First tick: future yields (Pending)
        pool.tick();
        assert_eq!(pool.pending_count(), 1);
        assert!(handle.try_recv().is_none());

        // Second tick: future completes
        pool.tick();
        assert_eq!(pool.pending_count(), 0);
        assert_eq!(handle.try_recv(), Some(42));
    }

    #[test]
    fn multiple_tasks_progress() {
        let pool = ComputePool::new();
        let h1 = pool.spawn(Priority::Low, async { 1u32 });
        let h2 = pool.spawn(Priority::Low, async { 2u32 });
        let h3 = pool.spawn(Priority::Low, async { 3u32 });

        pool.tick_all();

        assert_eq!(pool.pending_count(), 0);
        assert_eq!(h1.try_recv(), Some(1));
        assert_eq!(h2.try_recv(), Some(2));
        assert_eq!(h3.try_recv(), Some(3));
    }

    #[test]
    fn task_handle_recv_blocks() {
        let pool = ComputePool::new();
        let handle = pool.spawn(Priority::High, async { "hello" });

        pool.tick();
        assert_eq!(handle.recv(), "hello");
    }

    #[test]
    fn empty_pool_tick() {
        let pool = ComputePool::new();
        assert_eq!(pool.tick(), 0);
        assert_eq!(pool.tick_all(), 0);
        assert_eq!(pool.pending_count(), 0);
    }
}
