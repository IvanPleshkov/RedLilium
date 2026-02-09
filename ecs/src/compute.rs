use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::Duration;

use crate::priority::Priority;

/// Shared state between a [`TaskHandle`] and the pool's internal future wrapper.
///
/// Tracks completion and cancellation flags atomically so the handle can
/// query status without consuming the result value.
struct TaskState {
    completed: AtomicBool,
    cancelled: AtomicBool,
}

impl TaskState {
    fn new() -> Self {
        Self {
            completed: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
        }
    }
}

/// Handle to a spawned async compute task.
///
/// Allows checking completion status, retrieving the result, cancelling,
/// and waiting with a timeout.
///
/// # Example
///
/// ```ignore
/// let handle = pool.spawn(Priority::Low, async { 42 });
///
/// // Non-destructive completion check:
/// if handle.is_done() {
///     let result = handle.try_recv();
/// }
///
/// // Or wait with timeout:
/// if let Some(val) = handle.recv_timeout(Duration::from_millis(100)) {
///     println!("Got: {}", val);
/// }
///
/// // Cancel a long-running task:
/// handle.cancel();
/// ```
pub struct TaskHandle<T> {
    receiver: std::sync::mpsc::Receiver<T>,
    state: Arc<TaskState>,
}

impl<T> TaskHandle<T> {
    /// Attempts to retrieve the result without blocking.
    ///
    /// Returns `Some(T)` if the task has completed, `None` otherwise.
    /// This consumes the value — subsequent calls return `None`.
    pub fn try_recv(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }

    /// Returns whether the task has completed (non-destructive).
    ///
    /// Does not consume the result value. Use `try_recv()` or `recv()`
    /// to actually retrieve it.
    pub fn is_done(&self) -> bool {
        self.state.completed.load(Ordering::Acquire)
    }

    /// Returns whether the task has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Requests cancellation of the task.
    ///
    /// The pool will drop the task's future on its next tick instead of
    /// polling it. If the task has already completed, this has no effect.
    pub fn cancel(&self) {
        self.state.cancelled.store(true, Ordering::Release);
    }

    /// Blocks until the task completes and returns the result.
    ///
    /// Returns `None` if the task was cancelled or the sender was dropped.
    ///
    /// # Warning
    ///
    /// This blocks the calling thread. Prefer `try_recv()` in frame loops.
    pub fn recv(self) -> Option<T> {
        self.receiver.recv().ok()
    }

    /// Waits up to `timeout` for the task to complete.
    ///
    /// Returns `Some(T)` if the result arrives within the deadline,
    /// `None` if the timeout expires or the task was cancelled.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<T> {
        self.receiver.recv_timeout(timeout).ok()
    }
}

/// A pending async compute task stored in the pool.
struct PendingTask {
    priority: Priority,
    future: Pin<Box<dyn Future<Output = ()> + Send>>,
    /// Insertion order for stable sorting within the same priority.
    id: u64,
    /// Shared state for completion/cancellation tracking.
    state: Arc<TaskState>,
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
        let state = Arc::new(TaskState::new());
        let task_state = state.clone();

        let wrapped = async move {
            let result = future.await;
            let _ = sender.send(result);
            task_state.completed.store(true, Ordering::Release);
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
            state: state.clone(),
        };

        self.tasks.lock().unwrap().push(task);

        TaskHandle { receiver, state }
    }

    /// Polls the highest-priority pending task once.
    ///
    /// Returns the number of tasks that were polled (0 or 1).
    /// Completed and cancelled tasks are automatically removed from the pool.
    pub fn tick(&self) -> usize {
        let mut tasks = self.tasks.lock().unwrap();

        // Remove cancelled tasks first
        tasks.retain(|t| !t.state.cancelled.load(Ordering::Acquire));

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
    /// Completed and cancelled tasks are automatically removed.
    pub fn tick_all(&self) -> usize {
        let mut tasks = self.tasks.lock().unwrap();

        // Remove cancelled tasks first
        tasks.retain(|t| !t.state.cancelled.load(Ordering::Acquire));

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
        assert!(!handle.is_done());

        // Second tick: future completes
        pool.tick();
        assert_eq!(pool.pending_count(), 0);
        assert!(handle.is_done());
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
        assert_eq!(handle.recv(), Some("hello"));
    }

    #[test]
    fn empty_pool_tick() {
        let pool = ComputePool::new();
        assert_eq!(pool.tick(), 0);
        assert_eq!(pool.tick_all(), 0);
        assert_eq!(pool.pending_count(), 0);
    }

    #[test]
    fn is_done_non_destructive() {
        let pool = ComputePool::new();
        let handle = pool.spawn(Priority::Low, async { 99u32 });

        assert!(!handle.is_done());
        pool.tick();
        assert!(handle.is_done());
        // Value is still available after checking is_done
        assert_eq!(handle.try_recv(), Some(99));
    }

    #[test]
    fn cancel_prevents_execution() {
        let pool = ComputePool::new();
        let handle = pool.spawn(Priority::Low, async {
            yield_now().await;
            42u32
        });

        // First tick: task yields
        pool.tick();
        assert!(!handle.is_done());

        // Cancel before it can complete
        handle.cancel();
        assert!(handle.is_cancelled());

        // Next tick should drop the cancelled task
        pool.tick();
        assert_eq!(pool.pending_count(), 0);
        assert!(!handle.is_done());
    }

    #[test]
    fn recv_timeout_returns_none_on_timeout() {
        let pool = ComputePool::new();
        let handle = pool.spawn(Priority::Low, async {
            yield_now().await;
            42u32
        });

        // Task hasn't been ticked yet
        assert_eq!(handle.recv_timeout(Duration::from_millis(1)), None);

        // Complete the task and retrieve
        pool.tick();
        pool.tick();
        assert_eq!(handle.recv_timeout(Duration::from_millis(100)), Some(42));
    }

    #[test]
    fn cancel_already_completed_is_harmless() {
        let pool = ComputePool::new();
        let handle = pool.spawn(Priority::Low, async { 10u32 });

        pool.tick();
        assert!(handle.is_done());

        // Cancelling after completion is fine
        handle.cancel();
        assert_eq!(handle.try_recv(), Some(10));
    }
}
