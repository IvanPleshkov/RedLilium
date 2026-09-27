use crate::system::parallel_executor::{BackgroundNotifier, BackgroundWork};
use redlilium_core::compute::{CancellationRegistration, IoHandle};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::task::Wake;

use crate::sync::Mutex;
use std::task::{Context, Poll, Waker};
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
    /// Panic message if the task panicked during polling or destruction.
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
    receiver: IoHandle<T>,
    state: Arc<TaskState>,
}

impl<T> TaskHandle<T> {
    /// Attempts to retrieve the result without blocking.
    ///
    /// Returns `Some(T)` if the task has completed, `None` otherwise.
    /// This consumes the value — subsequent calls return `None`.
    pub fn try_recv(&self) -> Option<T> {
        self.receiver.try_recv()
    }

    /// Returns whether the task future has been destroyed after completion,
    /// cancellation, panic, or pool drop (non-destructive). A result may arrive earlier.
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

    /// Returns a clone of the task's cancellation token.
    ///
    /// Useful for managing task lifecycle through generation-scoped token collections
    /// (e.g., [`PlayTasks`](crate::PlayTasks)) where tokens need to be cancelled in bulk.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.state.token.clone()
    }

    /// Returns whether the task panicked during polling or future destruction.
    ///
    /// [`panic_message()`](TaskHandle::panic_message) contains the owned panic
    /// message. A result received before a destruction failure remains valid.
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
        self.receiver.recv()
    }

    /// Waits up to `timeout` for the task to complete.
    ///
    /// Returns `Some(T)` if the result arrives within the deadline,
    /// `None` if the timeout expires or the task was cancelled.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<T> {
        self.receiver.recv_timeout(timeout)
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
    /// Completion or sender destruction wakes the awaiting task. Cancellation
    /// requests cleanup; the handle resolves when the task produces a result
    /// or releases its sender after the final poll.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
        self.receiver.poll_recv(cx)
    }
}

/// A timeout leaves the pool usable, but does not authorize module unload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuiesceTimeout {
    pub remaining_tasks: usize,
}
impl std::fmt::Display for QuiesceTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "compute quiescence timed out with {} active tasks",
            self.remaining_tasks
        )
    }
}
impl std::error::Error for QuiesceTimeout {}

// These generic methods execute in the future's originating image. Both
// polling and destruction must consume panic payloads before crossing back
// into the host; catching a guest unwind in the host is too late.
trait TaskFuture: Send {
    fn poll_guarded(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Result<Poll<()>, String>;
    fn dispose(self: Pin<Box<Self>>) -> Result<(), String>;
}
impl<F: Future<Output = ()> + Send> TaskFuture for F {
    fn poll_guarded(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Result<Poll<()>, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.poll(cx)))
            .map_err(|payload| crate::system::panic_payload_to_string(&*payload))
    }
    fn dispose(self: Pin<Box<Self>>) -> Result<(), String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(self)))
            .map_err(|payload| crate::system::panic_payload_to_string(&*payload))
    }
}

// One counter allocation per pool, one Arc clone per task. The lease is
// acquired before constructing/publishing a task and released after its
// erased future and task metadata have been destroyed, including on unwind.
struct TaskLifetime(Arc<Progress>);
impl TaskLifetime {
    fn new(counter: &Arc<Progress>) -> Self {
        counter.active_tasks.fetch_add(1, Ordering::AcqRel);
        Self(counter.clone())
    }
}
impl Drop for TaskLifetime {
    fn drop(&mut self) {
        self.0.active_tasks.fetch_sub(1, Ordering::AcqRel);
        self.0.notify();
    }
}

/// A pending async compute task stored in the pool.
struct PendingTask {
    priority: Priority,
    future: Option<Pin<Box<dyn TaskFuture>>>,
    /// Insertion order for stable sorting within the same priority.
    id: u64,
    /// Shared state for completion/cancellation tracking.
    state: Arc<TaskState>,
    /// The fairness round this task was last polled in (see [`TaskQueue`]).
    last_polled_round: u64,
    wake: Arc<TaskWake>,
    _cancel: CancellationRegistration,
    // Must be last: quiescence includes destruction of all preceding fields.
    _lifetime: TaskLifetime,
}

impl Drop for PendingTask {
    fn drop(&mut self) {
        self.wake.status.store(COMPLETE, Ordering::Release);
        if let Some(future) = self.future.take()
            && let Err(message) = future.dispose()
        {
            let mut error = self.state.panicked.lock();
            if let Some(previous) = error.as_mut() {
                previous.push_str("; during task destruction: ");
                previous.push_str(&message);
            } else {
                *error = Some(message);
            }
        }
        self.state.completed.store(true, Ordering::Release);
    }
}

// A wake while polling is remembered, but never permits concurrent polling.
const IDLE: u8 = 0;
const QUEUED: u8 = 1;
const RUNNING: u8 = 2;
const NOTIFIED: u8 = 3;
const COMPLETE: u8 = 4;
struct TaskWake {
    owner: Weak<Inner>,
    id: u64,
    status: AtomicU8,
}
impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        loop {
            let state = self.status.load(Ordering::Acquire);
            let next = match state {
                IDLE => QUEUED,
                RUNNING => NOTIFIED,
                _ => return,
            };
            if self
                .status
                .compare_exchange(state, next, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            if next == QUEUED {
                if let Some(owner) = self.owner.upgrade() {
                    {
                        let mut queue = owner.queue.lock();
                        if queue.closed {
                            return;
                        }
                        queue.ready.push(self.id);
                        owner.ready.store(queue.ready.len(), Ordering::Release);
                    }
                    owner.notify();
                }
            }
            return;
        }
    }
}
// Capture these constructors in the pool's image. External IO may retain a
// task waker after cancellation; its vtable must not belong to the guest.
#[inline(never)]
fn task_waker(task: Arc<TaskWake>) -> Waker {
    Waker::from(task)
}
#[inline(never)]
fn progress_waker(progress: Arc<Progress>) -> Waker {
    Waker::from(progress)
}
#[derive(Default)]
struct Progress {
    active_tasks: AtomicUsize,
    epoch: std::sync::Mutex<u64>,
    changed: std::sync::Condvar,
}
impl Progress {
    fn epoch(&self) -> u64 {
        *self.epoch.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn notify(&self) {
        *self.epoch.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        self.changed.notify_all();
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self, epoch: u64, timeout: Option<Duration>) {
        let guard = self.epoch.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(timeout) = timeout {
            drop(
                self.changed
                    .wait_timeout_while(guard, timeout, |value| *value == epoch)
                    .unwrap_or_else(|e| e.into_inner()),
            );
        } else {
            drop(
                self.changed
                    .wait_while(guard, |value| *value == epoch)
                    .unwrap_or_else(|e| e.into_inner()),
            );
        }
    }
}
impl Wake for Progress {
    fn wake(self: Arc<Self>) {
        self.notify();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.notify();
    }
}
struct TaskQueue {
    tasks: HashMap<u64, PendingTask>,
    ready: Vec<u64>,
    round: u64,
    next_id: u64,
    closed: bool,
}
struct Inner {
    queue: Mutex<TaskQueue>,
    ready: AtomicUsize,
    progress: Arc<Progress>,
    notifier: Mutex<Option<BackgroundNotifier>>,
    make_waker: fn(Arc<TaskWake>) -> Waker,
    make_progress_waker: fn(Arc<Progress>) -> Waker,
}
impl Inner {
    fn notify(&self) {
        self.progress.notify();
        let notifier = self.notifier.lock().clone();
        if let Some(notifier) = notifier {
            notifier.notify();
        }
    }
    fn extract(&self) -> Option<PendingTask> {
        let mut queue = self.queue.lock();
        if queue.ready.is_empty() {
            return None;
        }
        let pick = |q: &TaskQueue| {
            q.ready
                .iter()
                .enumerate()
                .filter_map(|(i, id)| {
                    q.tasks
                        .get(id)
                        .filter(|t| t.last_polled_round < q.round)
                        .map(|t| (i, t))
                })
                .max_by(|(_, a), (_, b)| a.priority.cmp(&b.priority).then(b.id.cmp(&a.id)))
                .map(|(i, _)| i)
        };
        let index = pick(&queue).unwrap_or_else(|| {
            queue.round += 1;
            pick(&queue).expect("ready task exists")
        });
        let id = queue.ready.swap_remove(index);
        let mut task = queue.tasks.remove(&id).expect("ready task");
        task.last_polled_round = queue.round;
        task.wake.status.store(RUNNING, Ordering::Release);
        self.ready.store(queue.ready.len(), Ordering::Release);
        Some(task)
    }
    fn requeue(&self, task: PendingTask, polled: bool) {
        let mut queue = self.queue.lock();
        if queue.closed {
            drop(queue);
            drop(task);
            self.progress.notify();
            return;
        }
        let id = task.id;
        let wake = task.wake.clone();
        queue.tasks.insert(id, task);
        let ready = !polled
            || wake
                .status
                .compare_exchange(RUNNING, IDLE, Ordering::AcqRel, Ordering::Acquire)
                .is_err();
        if ready {
            wake.status.store(QUEUED, Ordering::Release);
            queue.ready.push(id);
            self.ready.store(queue.ready.len(), Ordering::Release);
        }
        drop(queue);
        if ready {
            self.notify();
        }
    }
    fn poll_task(&self, mut task: PendingTask) {
        let waker = (self.make_waker)(task.wake.clone());
        let mut cx = Context::from_waker(&waker);
        let finished = match task
            .future
            .as_mut()
            .expect("live task")
            .as_mut()
            .poll_guarded(&mut cx)
        {
            Ok(Poll::Ready(())) => true,
            Ok(Poll::Pending) => task.state.token.is_cancelled(),
            Err(message) => {
                task.state.set_panicked(message);
                true
            }
        };
        if finished {
            drop(task);
            self.progress.notify();
        } else {
            self.requeue(task, true);
        }
    }
    fn tick(&self) -> usize {
        if let Some(task) = self.extract() {
            self.poll_task(task);
            1
        } else {
            0
        }
    }
    fn batch(&self, budget: Option<Duration>) -> usize {
        let mut batch = {
            let mut q = self.queue.lock();
            let ids = std::mem::take(&mut q.ready);
            let mut batch: Vec<_> = ids
                .into_iter()
                .map(|id| {
                    let task = q.tasks.remove(&id).expect("ready task");
                    task.wake.status.store(RUNNING, Ordering::Release);
                    task
                })
                .collect();
            self.ready.store(0, Ordering::Release);
            batch.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.id.cmp(&b.id)));
            batch
        }
        .into_iter();
        let start = budget.map(|_| Instant::now());
        let mut polled = 0;
        while let Some(task) = batch.next() {
            if polled > 0
                && budget.is_some_and(|budget| start.as_ref().unwrap().elapsed() >= budget)
            {
                self.requeue(task, false);
                for task in batch {
                    self.requeue(task, false);
                }
                break;
            }
            self.poll_task(task);
            polled += 1;
        }
        polled
    }
}
impl BackgroundWork for Inner {
    fn ready_count(&self) -> usize {
        self.ready.load(Ordering::Acquire)
    }
    fn active_count(&self) -> usize {
        self.progress.active_tasks.load(Ordering::Acquire)
    }
    fn poll_one(&self) {
        self.tick();
    }
}

/// Wake-driven compute tasks. `new` is manually driven; `with_executor`
/// registers background work on explicitly shared native CPU workers.
/// Construct shared pools in the host: the constructor image must remain
/// mapped until all retained task wakers have been released.
pub struct ComputePool {
    inner: Arc<Inner>,
    io: IoRuntime,
    // Keep worker ownership outside Inner: worker polling never owns the pool
    // it is executing on. Wakers and the source registry use weak references.
    _executor: Option<crate::ParallelExecutor>,
}
impl ComputePool {
    pub fn new(io: IoRuntime) -> Self {
        reset_yield_timer();
        Self {
            inner: Arc::new(Inner {
                queue: Mutex::new(TaskQueue {
                    tasks: HashMap::new(),
                    ready: Vec::new(),
                    round: 1,
                    next_id: 0,
                    closed: false,
                }),
                ready: AtomicUsize::new(0),
                progress: Arc::new(Progress::default()),
                notifier: Mutex::new(None),
                make_waker: task_waker,
                make_progress_waker: progress_waker,
            }),
            io,
            _executor: None,
        }
    }
    /// Native tasks continue between frames on the supplied executor. WASM
    /// retains cooperative manual ticking and creates no CPU threads.
    pub fn with_executor(io: IoRuntime, executor: crate::ParallelExecutor) -> Self {
        let mut pool = Self::new(io);
        let source: Arc<dyn BackgroundWork> = pool.inner.clone();
        *pool.inner.notifier.lock() = Some(executor.register_background(&source));
        pool._executor = Some(executor);
        pool
    }
    pub fn spawn<T, F, Fut>(&self, priority: Priority, f: F) -> TaskHandle<T>
    where
        T: Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
        F: FnOnce(EcsComputeContext) -> Fut + Send + 'static,
    {
        let lifetime = TaskLifetime::new(&self.inner.progress);
        let (sender, receiver) = IoHandle::channel();
        let state = Arc::new(TaskState::new());
        let ctx = EcsComputeContext::new(self.io.clone(), state.token.clone());
        let future = f(ctx);
        let wrapped = async move {
            let result = future.await;
            let _ = sender.send(result);
        };
        let id = {
            let mut q = self.inner.queue.lock();
            let id = q.next_id;
            q.next_id += 1;
            id
        };
        let wake = Arc::new(TaskWake {
            owner: Arc::downgrade(&self.inner),
            id,
            status: AtomicU8::new(RUNNING),
        });
        let cancel = state
            .token
            .on_cancel(&(self.inner.make_waker)(wake.clone()));
        let task = PendingTask {
            priority,
            future: Some(Box::pin(wrapped)),
            id,
            state: state.clone(),
            last_polled_round: 0,
            wake,
            _cancel: cancel,
            _lifetime: lifetime,
        };
        self.inner.requeue(task, false);
        TaskHandle { receiver, state }
    }
    /// Poll one ready task; sleeping tasks are not polled again until woken.
    pub fn tick(&self) -> usize {
        self.inner.tick()
    }
    /// Poll the current ready batch once, in priority order.
    pub fn tick_all(&self) -> usize {
        self.inner.batch(None)
    }
    /// Cooperative budget, checked between polls. A single poll is not preempted.
    pub fn tick_with_budget(&self, budget: Duration) -> usize {
        self.inner.batch(Some(budget))
    }
    pub fn tick_extract(&self) -> usize {
        self.tick()
    }
    /// Wait while helping ready compute work. Release ECS guards before calling:
    /// this can execute other compute tasks, but never steals systems/queries.
    pub fn block_on<T>(&self, handle: &mut TaskHandle<T>) -> Option<T> {
        let waker = (self.inner.make_progress_waker)(self.inner.progress.clone());
        let mut cx = Context::from_waker(&waker);
        loop {
            let epoch = self.inner.progress.epoch();
            if let Poll::Ready(value) = handle.receiver.poll_recv(&mut cx) {
                return value;
            }
            if self.tick() != 0 {
                continue;
            }
            #[cfg(not(target_arch = "wasm32"))]
            self.inner.progress.wait(epoch, None);
            #[cfg(target_arch = "wasm32")]
            {
                let _ = epoch;
                panic!(
                    "ComputePool::block_on would deadlock on wasm: await the TaskHandle to let the browser process IO"
                );
            }
        }
    }
    /// Queued tasks, both ready and sleeping; excludes concurrently polled tasks.
    pub fn pending_count(&self) -> usize {
        self.inner.queue.lock().tasks.len()
    }
    pub fn active_count(&self) -> usize {
        self.inner.active_count()
    }
    /// Stop producers first. Success includes future destruction, but callers
    /// must separately release guest results and join workers/TLS before unload.
    pub fn quiesce(&self, timeout: Duration) -> Result<(), QuiesceTimeout> {
        let start = Instant::now();
        loop {
            #[cfg(not(target_arch = "wasm32"))]
            let epoch = self.inner.progress.epoch();
            let remaining_tasks = self.active_count();
            if remaining_tasks == 0 {
                return Ok(());
            }
            if start.elapsed() >= timeout {
                return Err(QuiesceTimeout { remaining_tasks });
            }
            if self.tick() != 0 {
                continue;
            }
            #[cfg(not(target_arch = "wasm32"))]
            self.inner
                .progress
                .wait(epoch, Some(timeout.saturating_sub(start.elapsed())));
        }
    }
}
impl Drop for ComputePool {
    fn drop(&mut self) {
        let tasks = {
            let mut queue = self.inner.queue.lock();
            queue.closed = true;
            queue.ready.clear();
            self.inner.ready.store(0, Ordering::Release);
            std::mem::take(&mut queue.tasks)
        };
        drop(tasks);
        if let Some(notifier) = self.inner.notifier.lock().take() {
            notifier.unregister();
        }
        self.inner.progress.notify();
    }
}

#[cfg(test)]
pub(crate) fn noop_waker() -> Waker {
    Waker::noop().clone()
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

        // Await waits for cleanup; the final grace poll can still return a result.
        assert!(Pin::new(&mut handle).poll(&mut cx).is_pending());
        pool.tick();
        assert_eq!(Pin::new(&mut handle).poll(&mut cx), Poll::Ready(Some(42)));
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
