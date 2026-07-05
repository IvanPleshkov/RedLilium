use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::{Duration, Instant};

use redlilium_core::compute::{CancellationToken, Priority, reset_yield_timer};

use crate::compute::EcsComputeContext;
use crate::compute::IoRuntime;

/// Shared state between a [`TaskHandle`] and the pool's internal future wrapper.
///
/// Tracks completion via an atomic flag and cancellation via a
/// [`CancellationToken`] that is also wired into the task's
/// [`EcsComputeContext`] so that [`checkpoint()`](redlilium_core::compute::ComputeContext::checkpoint)
/// can detect cancellation at yield points.
struct TaskState {
    completed: AtomicBool,
    token: CancellationToken,
    /// Panic message if the task panicked during polling.
    panicked: Mutex<Option<String>>,
}

impl TaskState {
    fn new() -> Self {
        Self {
            completed: AtomicBool::new(false),
            token: CancellationToken::new(),
            panicked: Mutex::new(None),
        }
    }

    fn set_panicked(&self, msg: String) {
        *self.panicked.lock() = Some(msg);
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
/// let handle = pool.spawn(Priority::Low, |_ctx| async { 42 });
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
        self.state.token.is_cancelled()
    }

    /// Returns whether the task panicked during execution.
    ///
    /// If true, [`try_recv()`](TaskHandle::try_recv) will return `None`
    /// and [`panic_message()`](TaskHandle::panic_message) contains the
    /// panic payload.
    pub fn is_panicked(&self) -> bool {
        self.state.panicked.lock().is_some()
    }

    /// Returns the panic message if the task panicked, `None` otherwise.
    pub fn panic_message(&self) -> Option<String> {
        self.state.panicked.lock().clone()
    }

    /// Requests cancellation of the task.
    ///
    /// The task will observe the cancellation at its next
    /// [`checkpoint()`](redlilium_core::compute::ComputeContext::checkpoint)
    /// and can stop early. The pool also drops cancelled tasks that haven't
    /// reached a checkpoint on its next tick.
    ///
    /// If the task has already completed, this has no effect.
    pub fn cancel(&self) {
        self.state.token.cancel();
    }

    /// Blocks until the task completes and returns the result.
    ///
    /// Returns `None` if the task was cancelled or the sender was dropped.
    ///
    /// # Warning
    ///
    /// This blocks the calling thread **without driving the pool**: if
    /// nothing else ticks the pool (e.g. inside a system on the
    /// single-threaded runner), the task never progresses and this hangs
    /// forever. Use [`ComputePool::block_on`] to wait while driving the
    /// pool, or `try_recv()` in frame loops.
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

impl<T> Future for TaskHandle<T> {
    type Output = Option<T>;

    /// Polls the task for completion.
    ///
    /// Returns `Poll::Ready(Some(T))` if the task has completed,
    /// `Poll::Ready(None)` if the task was cancelled or the sender dropped,
    /// `Poll::Pending` if the task is still running.
    ///
    /// Designed for manual polling with a noop waker — the scheduler
    /// drives progress via [`ComputePool::tick_all`] between polls.
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<T>> {
        match self.receiver.try_recv() {
            Ok(val) => Poll::Ready(Some(val)),
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                if self.state.token.is_cancelled() {
                    Poll::Ready(None)
                } else {
                    Poll::Pending
                }
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Poll::Ready(None),
        }
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
    /// The fairness round this task was last polled in (see [`TaskQueue`]).
    last_polled_round: u64,
}

/// The pool's task list plus the fairness round counter.
///
/// [`ComputePool::tick`] picks the highest-priority task *not yet polled in
/// the current round*; once every task has been polled the round advances.
/// Without rounds, a high-priority task that waits on a low-priority one
/// would be re-polled forever and the low-priority task would never run
/// (priority livelock in the `while pending_count() > 0 {{ tick() }}` loop).
struct TaskQueue {
    tasks: Vec<PendingTask>,
    round: u64,
}

/// Pool for spawning async compute tasks.
///
/// Tasks are stored and polled manually via [`tick`](ComputePool::tick).
/// Each task has a priority that determines polling order.
///
/// # Example
///
/// ```ignore
/// use redlilium_ecs::{ComputePool, Priority, IoRuntime};
///
/// let io = IoRuntime::new();
/// let pool = ComputePool::new(io);
///
/// let handle = pool.spawn(Priority::Low, |_ctx| async { 42u32 });
///
/// // Tick until the task completes
/// while pool.pending_count() > 0 {
///     pool.tick();
/// }
///
/// assert_eq!(handle.try_recv(), Some(42));
/// ```
pub struct ComputePool {
    queue: Mutex<TaskQueue>,
    next_id: Mutex<u64>,
    io: IoRuntime,
}

impl ComputePool {
    /// Creates a new compute pool with the given IO runtime.
    ///
    /// Resets the per-thread yield timer so the first [`yield_now`](crate::yield_now)
    /// in any task spawned on this pool will always suspend.
    pub fn new(io: IoRuntime) -> Self {
        reset_yield_timer();
        Self {
            queue: Mutex::new(TaskQueue {
                tasks: Vec::new(),
                round: 1,
            }),
            next_id: Mutex::new(0),
            io,
        }
    }

    /// Spawns an async compute task with the given priority.
    ///
    /// The closure receives an [`EcsComputeContext`] that provides
    /// cooperative yielding and IO access.
    ///
    /// Returns a handle for retrieving the result.
    pub fn spawn<T, F, Fut>(&self, priority: Priority, f: F) -> TaskHandle<T>
    where
        T: Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
        F: FnOnce(EcsComputeContext) -> Fut + Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();
        let state = Arc::new(TaskState::new());
        let task_state = state.clone();

        let ctx = EcsComputeContext::new(self.io.clone(), state.token.clone());
        let future = f(ctx);

        let wrapped = async move {
            let result = future.await;
            let _ = sender.send(result);
            task_state.completed.store(true, Ordering::Release);
        };

        let id = {
            let mut next_id = self.next_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let task = PendingTask {
            priority,
            future: Box::pin(wrapped),
            id,
            state: state.clone(),
            last_polled_round: 0,
        };

        self.queue.lock().tasks.push(task);

        TaskHandle { receiver, state }
    }

    /// Polls a task once, catching panics.
    ///
    /// Returns `true` if the task should be removed from the pool
    /// (completed, cancelled after grace poll, or panicked).
    fn poll_task_guarded(task: &mut PendingTask, cx: &mut Context<'_>) -> bool {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            task.future.as_mut().poll(cx)
        })) {
            Ok(Poll::Ready(())) => true,
            Ok(Poll::Pending) if task.state.token.is_cancelled() => true,
            Ok(Poll::Pending) => false,
            Err(payload) => {
                let msg = crate::system::panic_payload_to_string(&*payload);
                task.state.set_panicked(msg);
                task.state.completed.store(true, Ordering::Release);
                true
            }
        }
    }

    /// Extracts the next task to poll, honoring priority and fairness.
    ///
    /// Picks the highest-priority task not yet polled in the current round;
    /// when every task has been polled, the round advances and all tasks
    /// become eligible again. The task is removed from the queue so it can
    /// be polled without holding the lock — the caller must push it back if
    /// it is still pending.
    fn extract_next(&self) -> Option<PendingTask> {
        let mut q = self.queue.lock();
        if q.tasks.is_empty() {
            return None;
        }
        let round = q.round;
        let pick = |tasks: &[PendingTask], round: u64| {
            tasks
                .iter()
                .enumerate()
                .filter(|(_, t)| t.last_polled_round < round)
                .max_by(|(_, a), (_, b)| a.priority.cmp(&b.priority).then(b.id.cmp(&a.id)))
                .map(|(i, _)| i)
        };
        let idx = match pick(&q.tasks, round) {
            Some(i) => i,
            None => {
                // Every task was polled this round — start the next one.
                q.round += 1;
                let round = q.round;
                pick(&q.tasks, round).expect("non-empty queue must yield a task")
            }
        };
        let round = q.round;
        let mut task = q.tasks.swap_remove(idx);
        task.last_polled_round = round;
        Some(task)
    }

    /// Puts a still-pending task back into the queue.
    fn requeue(&self, task: PendingTask) {
        self.queue.lock().tasks.push(task);
    }

    /// Polls one pending task, chosen by priority with round-robin fairness.
    ///
    /// Returns the number of tasks that were polled (0 or 1).
    /// Completed and cancelled tasks are automatically removed from the pool.
    ///
    /// The future is polled with the queue lock **released**, so tasks may
    /// re-enter the pool (spawn, `pending_count`, nested `block_on`) and
    /// futures may lock world storages without deadlocking the pool.
    ///
    /// Cancelled tasks are polled one final time so that
    /// [`checkpoint()`](redlilium_core::compute::ComputeContext::checkpoint)
    /// can return [`Cancelled`](redlilium_core::compute::Cancelled) and the
    /// task body can clean up. If the task is still `Pending` after that
    /// poll it is dropped.
    pub fn tick(&self) -> usize {
        redlilium_core::profile_scope!("ecs: compute tick");
        let Some(mut task) = self.extract_next() else {
            return 0;
        };

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        if !Self::poll_task_guarded(&mut task, &mut cx) {
            self.requeue(task);
        }

        1
    }

    /// Polls all pending tasks once each.
    ///
    /// Returns the number of tasks that were polled.
    /// Completed and cancelled tasks are automatically removed.
    /// Cancelled tasks receive one final poll so checkpoints can fire.
    ///
    /// The batch is taken out of the queue and polled with the lock
    /// **released** (tasks extracted here are invisible to
    /// [`pending_count`](Self::pending_count) until re-queued).
    pub fn tick_all(&self) -> usize {
        redlilium_core::profile_scope!("ecs: compute tick_all");
        let mut batch = {
            let mut q = self.queue.lock();
            std::mem::take(&mut q.tasks)
        };
        let count = batch.len();
        if count == 0 {
            return 0;
        }

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        batch.retain_mut(|task| !Self::poll_task_guarded(task, &mut cx));

        let mut q = self.queue.lock();
        q.tasks.append(&mut batch);
        redlilium_core::profile_plot!("ecs: compute pending", q.tasks.len() as f64);

        count
    }

    /// Polls pending tasks until the time budget is exceeded.
    ///
    /// Each task is polled at most once. The budget is checked between polls,
    /// so a single non-yielding task may exceed the budget — but no further
    /// tasks will be polled after that.
    ///
    /// Returns the number of tasks that were polled. Polling happens with
    /// the queue lock **released** (see [`tick_all`](Self::tick_all)).
    pub fn tick_with_budget(&self, budget: Duration) -> usize {
        redlilium_core::profile_scope!("ecs: compute tick_budget");
        let start = Instant::now();
        let mut batch = {
            let mut q = self.queue.lock();
            std::mem::take(&mut q.tasks)
        };
        if batch.is_empty() {
            return 0;
        }

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut polled = 0;
        let mut i = 0;

        while i < batch.len() {
            if polled > 0 && start.elapsed() >= budget {
                break;
            }

            if Self::poll_task_guarded(&mut batch[i], &mut cx) {
                batch.swap_remove(i);
            } else {
                i += 1;
            }
            polled += 1;
        }

        let mut q = self.queue.lock();
        q.tasks.append(&mut batch);
        redlilium_core::profile_plot!("ecs: compute pending", q.tasks.len() as f64);

        polled
    }

    /// Polls one task outside the lock — alias of [`tick`](Self::tick).
    ///
    /// Historically `tick` polled under the queue lock and this method was
    /// the lock-free variant for parallel draining; `tick` now uses the
    /// same extract-poll-requeue pattern, so both are equivalent.
    pub fn tick_extract(&self) -> usize {
        self.tick()
    }

    /// Blocks the calling thread until the task completes, driving the
    /// compute pool between attempts.
    ///
    /// Returns `Some(T)` if the task completes, `None` if cancelled.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut handle = pool.spawn(Priority::Low, |_ctx| async { 42u32 });
    /// let result = pool.block_on(&mut handle); // Some(42)
    /// ```
    pub fn block_on<T>(&self, handle: &mut TaskHandle<T>) -> Option<T> {
        use std::sync::mpsc::TryRecvError;
        // Counts consecutive rounds in which no compute task made progress —
        // i.e. we're waiting purely on external IO or another thread. Used to
        // back off so this loop doesn't peg a core at 100%.
        let mut idle_rounds: u32 = 0;
        loop {
            match handle.receiver.try_recv() {
                Ok(val) => return Some(val),
                // The task future is gone (removed after completion, its
                // final cancellation poll, or a panic) and no value was
                // sent — there is nothing left to wait for. Checking the
                // channel rather than `is_cancelled()` avoids the race
                // where a cancelled task still completes with a result
                // between the two checks.
                Err(TryRecvError::Disconnected) => return None,
                Err(TryRecvError::Empty) => {}
            }
            let progressed = self.tick_all();
            #[cfg(not(target_arch = "wasm32"))]
            {
                if progressed > 0 {
                    idle_rounds = 0;
                    std::thread::yield_now();
                } else {
                    // Nothing to drive locally; escalate from cheap yields to a
                    // short sleep so we stop busy-spinning while waiting on IO.
                    idle_rounds = idle_rounds.saturating_add(1);
                    if idle_rounds < 32 {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                idle_rounds = if progressed > 0 { 0 } else { idle_rounds + 1 };
                // On wasm there are no threads and IO futures only resolve
                // after control returns to the browser event loop — which a
                // blocking loop never does. If local compute makes no
                // progress, spinning would hang the tab forever; fail loudly
                // instead. (A couple of grace rounds let a just-completed
                // task's result propagate.)
                if idle_rounds > 2 {
                    match handle.receiver.try_recv() {
                        Ok(val) => return Some(val),
                        Err(TryRecvError::Disconnected) => return None,
                        Err(TryRecvError::Empty) => panic!(
                            "ComputePool::block_on would deadlock on wasm: the task is \
                             waiting on IO (or another task) that can only progress after \
                             yielding to the browser event loop. Await the TaskHandle from \
                             an async context instead of blocking."
                        ),
                    }
                }
            }
        }
    }

    /// Returns the number of pending (incomplete) tasks.
    ///
    /// Tasks currently extracted for polling by a concurrent `tick*` call
    /// are not counted until they are re-queued.
    pub fn pending_count(&self) -> usize {
        self.queue.lock().tasks.len()
    }
}

/// Creates a no-op waker for manual polling.
pub(crate) fn noop_waker() -> Waker {
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
    use redlilium_core::compute::{ComputeContext, yield_now};

    fn test_pool() -> ComputePool {
        ComputePool::new(IoRuntime::new())
    }

    #[test]
    fn high_priority_waiting_on_low_does_not_livelock() {
        // Issue #18 (B6): tick() used to re-poll the highest-priority task
        // forever; a High task waiting on a Low task's side effect never let
        // the Low task run in the documented `while pending { tick() }` loop.
        use std::sync::atomic::AtomicBool;
        let pool = test_pool();

        let flag = Arc::new(AtomicBool::new(false));
        let flag_low = flag.clone();
        let _low = pool.spawn(Priority::Low, move |_ctx| async move {
            flag_low.store(true, Ordering::SeqCst);
        });
        let flag_high = flag.clone();
        let high = pool.spawn(Priority::High, move |_ctx| async move {
            while !flag_high.load(Ordering::SeqCst) {
                yield_now().await;
            }
            7u32
        });

        let mut ticks = 0;
        while pool.pending_count() > 0 {
            pool.tick();
            ticks += 1;
            assert!(ticks < 100, "priority livelock: Low task never polled");
        }
        assert_eq!(high.try_recv(), Some(7));
    }

    #[test]
    fn task_can_reenter_the_pool() {
        // Issue #18 (B4): polling used to happen while holding the task-queue
        // mutex, so any reentrant pool call from inside a task (spawn,
        // pending_count, nested block_on) deadlocked.
        let pool = Arc::new(ComputePool::new(IoRuntime::new()));

        let pool2 = pool.clone();
        let handle = pool.spawn(Priority::High, move |_ctx| async move {
            let inner = pool2.spawn(Priority::High, |_ctx| async { 5u32 });
            // Reentrant queries must not deadlock either.
            let n = pool2.pending_count();
            // The inner task keeps running after its handle is dropped;
            // the outer task only proves reentrancy doesn't deadlock.
            drop(inner);
            n
        });

        let mut ticks = 0;
        while pool.pending_count() > 0 {
            pool.tick();
            ticks += 1;
            assert!(ticks < 100, "reentrant spawn deadlocked or never finished");
        }
        assert!(handle.try_recv().is_some(), "outer task completed");
    }

    #[test]
    fn block_on_returns_result_of_cancelled_but_completed_task() {
        // Issue #18 (D): block_on used to check is_cancelled() after
        // try_recv() and could drop a result that a cancelled task still
        // produced. Now it gives up only when the task future is gone
        // without having sent a value.
        let pool = test_pool();
        let mut handle = pool.spawn(Priority::High, |_ctx| async { 11u32 });
        handle.cancel();
        // The task has no checkpoints, so its final poll completes it and
        // sends the value despite the cancellation request.
        assert_eq!(pool.block_on(&mut handle), Some(11));
    }

    #[test]
    fn spawn_and_recv() {
        let pool = test_pool();
        let handle = pool.spawn(Priority::Low, |_ctx| async { 42u32 });

        while pool.pending_count() > 0 {
            pool.tick();
        }

        assert_eq!(handle.try_recv(), Some(42));
    }

    #[test]
    fn priority_ordering() {
        let pool = test_pool();
        let results = std::sync::Arc::new(Mutex::new(Vec::new()));

        let r1 = results.clone();
        pool.spawn(Priority::Low, |_ctx| async move {
            r1.lock().push("low");
        });

        let r2 = results.clone();
        pool.spawn(Priority::High, |_ctx| async move {
            r2.lock().push("high");
        });

        let r3 = results.clone();
        pool.spawn(Priority::Critical, |_ctx| async move {
            r3.lock().push("critical");
        });

        // Tick one at a time — highest priority first
        pool.tick();
        pool.tick();
        pool.tick();

        let order = results.lock();
        assert_eq!(*order, vec!["critical", "high", "low"]);
    }

    #[test]
    fn yield_now_suspends() {
        let pool = test_pool();
        let handle = pool.spawn(Priority::Low, |_ctx| async {
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
        let pool = test_pool();
        let h1 = pool.spawn(Priority::Low, |_ctx| async { 1u32 });
        let h2 = pool.spawn(Priority::Low, |_ctx| async { 2u32 });
        let h3 = pool.spawn(Priority::Low, |_ctx| async { 3u32 });

        pool.tick_all();

        assert_eq!(pool.pending_count(), 0);
        assert_eq!(h1.try_recv(), Some(1));
        assert_eq!(h2.try_recv(), Some(2));
        assert_eq!(h3.try_recv(), Some(3));
    }

    #[test]
    fn task_handle_recv_blocks() {
        let pool = test_pool();
        let handle = pool.spawn(Priority::High, |_ctx| async { "hello" });

        pool.tick();
        assert_eq!(handle.recv(), Some("hello"));
    }

    #[test]
    fn empty_pool_tick() {
        let pool = test_pool();
        assert_eq!(pool.tick(), 0);
        assert_eq!(pool.tick_all(), 0);
        assert_eq!(pool.pending_count(), 0);
    }

    #[test]
    fn is_done_non_destructive() {
        let pool = test_pool();
        let handle = pool.spawn(Priority::Low, |_ctx| async { 99u32 });

        assert!(!handle.is_done());
        pool.tick();
        assert!(handle.is_done());
        // Value is still available after checking is_done
        assert_eq!(handle.try_recv(), Some(99));
    }

    #[test]
    fn cancel_drops_non_checkpoint_task() {
        let pool = test_pool();

        // Task that yields but never calls checkpoint — not cancellation-aware.
        // On the grace poll it may finish, but at minimum it is removed from the pool.
        let handle = pool.spawn(Priority::Low, |_ctx| async {
            yield_now().await;
            yield_now().await; // two yields so the grace poll re-enters a yield → Pending
            42u32
        });

        // First tick: first yield suspends
        pool.tick();
        assert!(!handle.is_done());

        handle.cancel();
        assert!(handle.is_cancelled());

        // Grace poll: resumes first yield, hits second yield → Pending + cancelled → dropped
        pool.tick();
        assert_eq!(pool.pending_count(), 0);
    }

    #[test]
    fn recv_timeout_returns_none_on_timeout() {
        let pool = test_pool();
        let handle = pool.spawn(Priority::Low, |_ctx| async {
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
        let pool = test_pool();
        let handle = pool.spawn(Priority::Low, |_ctx| async { 10u32 });

        pool.tick();
        assert!(handle.is_done());

        // Cancelling after completion is fine
        handle.cancel();
        assert_eq!(handle.try_recv(), Some(10));
    }

    #[test]
    fn task_handle_future_pending_then_ready() {
        let pool = test_pool();
        let mut handle = pool.spawn(Priority::Low, |_ctx| async { 42u32 });

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // Before ticking: should be Pending
        assert!(Pin::new(&mut handle).poll(&mut cx).is_pending());

        // Tick the compute pool
        pool.tick();

        // After ticking: should be Ready
        match Pin::new(&mut handle).poll(&mut cx) {
            Poll::Ready(Some(42)) => {}
            other => panic!("Expected Ready(Some(42)), got {other:?}"),
        }
    }

    #[test]
    fn task_handle_future_cancelled() {
        let pool = test_pool();
        let mut handle = pool.spawn(Priority::Low, |_ctx| async {
            yield_now().await;
            42u32
        });

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        pool.tick(); // first poll: yields
        handle.cancel();

        // Next poll should return Ready(None) due to cancellation
        match Pin::new(&mut handle).poll(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("Expected Ready(None), got {other:?}"),
        }
    }

    #[test]
    fn task_handle_future_with_yield() {
        let pool = test_pool();
        let mut handle = pool.spawn(Priority::Low, |_ctx| async {
            yield_now().await;
            99u32
        });

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // First tick: task yields
        pool.tick();
        assert!(Pin::new(&mut handle).poll(&mut cx).is_pending());

        // Second tick: task completes
        pool.tick();
        match Pin::new(&mut handle).poll(&mut cx) {
            Poll::Ready(Some(99)) => {}
            other => panic!("Expected Ready(Some(99)), got {other:?}"),
        }
    }

    #[test]
    fn tick_with_budget_completes_fast_tasks() {
        let pool = test_pool();
        let h1 = pool.spawn(Priority::Low, |_ctx| async { 1u32 });
        let h2 = pool.spawn(Priority::Low, |_ctx| async { 2u32 });

        // Generous budget — both tasks should complete
        let polled = pool.tick_with_budget(Duration::from_secs(1));
        assert_eq!(polled, 2);
        assert_eq!(pool.pending_count(), 0);
        assert_eq!(h1.try_recv(), Some(1));
        assert_eq!(h2.try_recv(), Some(2));
    }

    #[test]
    fn tick_with_budget_stops_after_budget() {
        let pool = test_pool();

        // Spawn several tasks that yield so each poll returns Pending
        for _ in 0..10 {
            pool.spawn(Priority::Low, |_ctx| async {
                yield_now().await;
            });
        }

        // Zero budget — polls at most one task (first is always polled)
        let polled = pool.tick_with_budget(Duration::ZERO);
        assert_eq!(polled, 1);
        // 10 tasks still pending (the one we polled yielded)
        assert_eq!(pool.pending_count(), 10);
    }

    #[test]
    fn tick_with_budget_empty_pool() {
        let pool = test_pool();
        assert_eq!(pool.tick_with_budget(Duration::from_secs(1)), 0);
    }

    #[test]
    fn checkpoint_stops_on_cancel() {
        let pool = test_pool();
        let handle = pool.spawn(Priority::Low, |ctx| async move {
            let mut count = 0u32;
            loop {
                count += 1;
                if ctx.checkpoint().await.is_err() {
                    return count;
                }
            }
        });

        // Tick a few times — task keeps yielding and looping
        pool.tick();
        pool.tick();
        assert!(pool.pending_count() > 0);

        // Cancel and tick — task should observe Cancelled and return
        handle.cancel();
        pool.tick();
        assert_eq!(pool.pending_count(), 0);

        let count = handle.try_recv().unwrap();
        assert!(count > 0, "task ran at least once before being cancelled");
    }

    #[test]
    fn checkpoint_with_question_mark() {
        let pool = test_pool();
        let handle = pool.spawn(Priority::Low, |ctx| async move {
            let mut sum = 0u32;
            for i in 0..100 {
                sum += i;
                ctx.checkpoint().await.ok()?;
            }
            Some(sum)
        });

        // Let it run to completion without cancelling
        while pool.pending_count() > 0 {
            pool.tick();
        }

        // 0+1+2+...+99 = 4950
        assert_eq!(handle.try_recv(), Some(Some(4950)));
    }

    #[test]
    fn panicking_task_caught_and_reported() {
        let pool = test_pool();
        let handle = pool.spawn(Priority::Low, |_ctx| async {
            panic!("compute boom");
        });

        pool.tick();

        assert_eq!(pool.pending_count(), 0);
        assert!(handle.is_done());
        assert!(handle.is_panicked());
        assert_eq!(handle.panic_message().unwrap(), "compute boom");
        // Sender was dropped — no result
        assert!(handle.try_recv().is_none());
    }

    #[test]
    fn panicking_task_does_not_affect_others() {
        let pool = test_pool();

        pool.spawn(Priority::High, |_ctx| async {
            panic!("bad task");
        });
        let good = pool.spawn(Priority::Low, |_ctx| async { 42u32 });

        // Tick twice: first polls the high-priority (panics), second polls the low-priority
        pool.tick();
        pool.tick();

        assert_eq!(pool.pending_count(), 0);
        assert_eq!(good.try_recv(), Some(42));
    }

    #[test]
    fn checkpoint_with_question_mark_cancelled() {
        let pool = test_pool();
        let handle = pool.spawn(Priority::Low, |ctx| async move {
            let mut sum = 0u32;
            for i in 0..100 {
                sum += i;
                ctx.checkpoint().await.ok()?;
            }
            Some(sum)
        });

        // Tick once so the task starts, then cancel
        pool.tick();
        handle.cancel();
        pool.tick();

        assert_eq!(pool.pending_count(), 0);
        // Task returned None via `?` propagation
        assert_eq!(handle.try_recv(), Some(None));
    }
}
