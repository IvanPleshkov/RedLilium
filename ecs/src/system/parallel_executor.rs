//! Shared workers for systems, borrowed queries, and owned compute tasks.
//! Ready systems are queued by their runner. Query helpers use only idle
//! workers; a waiting query never steals unrelated systems or compute work.

use std::sync::Arc;

/// An owned compute queue registered with the shared CPU workers.
pub(crate) trait BackgroundWork: Send + Sync {
    fn ready_count(&self) -> usize;
    fn active_count(&self) -> usize;
    fn poll_one(&self);
}

#[derive(Clone)]
pub(crate) struct BackgroundNotifier {
    #[cfg(not(target_arch = "wasm32"))]
    pool: std::sync::Weak<native::Pool>,
    #[cfg(not(target_arch = "wasm32"))]
    source: u64,
}
impl BackgroundNotifier {
    pub(crate) fn unregister(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(pool) = self.pool.upgrade() {
            pool.unregister_background(self.source);
        }
    }
    pub(crate) fn notify(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(pool) = self.pool.upgrade() {
            pool.notify_background();
        }
    }
}

/// Shared CPU workers. Clones refer to the same pool, including across worlds.
///
/// The configured count excludes calling/coordinating threads and IO workers.
/// Startup is lazy. Before unloading a guest, stop its producers, drain compute,
/// and call [`shutdown_workers`](Self::shutdown_workers) to run worker TLS
/// destructors. Dropping one world does not stop a pool shared with a runner.
/// The image that constructed the pool must outlive all its handles.
#[derive(Clone)]
pub struct ParallelExecutor {
    #[cfg(not(target_arch = "wasm32"))]
    native: Arc<native::Pool>,
    #[cfg(target_arch = "wasm32")]
    identity: Arc<()>,
}

/// Workers cannot be stopped while borrowed scopes or registered compute tasks are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorBusy;
impl std::fmt::Display for ExecutorBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("executor has active scopes/compute tasks or is stopping workers")
    }
}
impl std::error::Error for ExecutorBusy {}

impl ParallelExecutor {
    /// Maximum number of background CPU workers. Zero is treated as one.
    /// The main/coordinating thread is additional. WASM creates no workers.
    pub fn new(worker_threads: usize) -> Self {
        let _ = worker_threads;
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            native: Arc::new(native::Pool::new(worker_threads.max(1))),
            #[cfg(target_arch = "wasm32")]
            identity: Arc::new(()),
        }
    }

    pub fn worker_threads(&self) -> usize {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native.capacity
        }
        #[cfg(target_arch = "wasm32")]
        {
            0
        }
    }

    /// Query participant limit, including an external caller. A query running
    /// on one of these workers can use only the other idle workers as helpers.
    pub fn parallelism(&self) -> usize {
        self.worker_threads().saturating_add(1)
    }

    pub fn shares_workers_with(&self, other: &Self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Arc::ptr_eq(&self.native, &other.native)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Arc::ptr_eq(&self.identity, &other.identity)
        }
    }

    /// Join all workers, including their TLS destructors. Fails without stopping
    /// anything if a borrowed scope or registered compute task is active. Admissions are rejected during
    /// shutdown; later calls may lazily restart workers. The caller must keep
    /// guest producers stopped until unloading completes. Quiesce all attached
    /// compute pools first, including sleeping tasks.
    pub fn shutdown_workers(&self) -> Result<(), ExecutorBusy> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native.shutdown()
        }
        #[cfg(target_arch = "wasm32")]
        {
            Ok(())
        }
    }

    pub(crate) fn register_background(
        &self,
        source: &Arc<dyn BackgroundWork>,
    ) -> BackgroundNotifier {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let id = self.native.register_background(source);
            BackgroundNotifier {
                source: id,
                pool: Arc::downgrade(&self.native),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = source;
            BackgroundNotifier {}
        }
    }

    pub(crate) fn shutdown_if_unique(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        if Arc::strong_count(&self.native) == 1 {
            self.shutdown_workers()
                .expect("unique executor cannot have an external active scope");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn prepare_systems(&self) -> Result<native::Lease<'_>, String> {
        self.native.prepare_run(true)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn prepare_queries(&self) -> Result<native::Lease<'_>, String> {
        self.native.prepare_run(false)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn scope<'env, R>(
        &'env self,
        f: impl for<'scope> FnOnce(&'scope native::TaskScope<'scope, 'env>) -> R,
    ) -> R {
        self.native.scope(f)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn run(&self, participants: usize, work: impl Fn() + Sync) {
        self.native.run(participants, &work);
    }
}
impl Default for ParallelExecutor {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self::new(cpus.saturating_sub(1).max(1))
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{BackgroundWork, ExecutorBusy};
    use crate::main_thread_dispatcher::RunnerEvent;
    use std::any::Any;
    use std::marker::PhantomData;
    use std::sync::Weak;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::Sender;
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    use std::thread::JoinHandle;

    fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }
    type Panic = Box<dyn Any + Send>;
    #[derive(Default)]
    struct Completion {
        remaining: usize,
        panics: Vec<Panic>,
    }
    #[derive(Default)]
    struct Done {
        state: Mutex<Completion>,
        wake: Condvar,
    }
    impl Done {
        fn wait(&self) {
            let mut state = lock(&self.state);
            while state.remaining != 0 {
                state = self.wake.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        }
        fn complete(&self, panic: Option<Panic>) {
            let mut state = lock(&self.state);
            if let Some(panic) = panic {
                state.panics.push(panic);
            }
            state.remaining -= 1;
            self.wake.notify_all();
        }
        fn propagate(&self) {
            self.wait();
            if let Some(panic) = lock(&self.state).panics.pop() {
                std::panic::resume_unwind(panic);
            }
        }
    }
    struct Drain<'a>(&'a Done);
    impl Drop for Drain<'_> {
        fn drop(&mut self) {
            self.0.wait();
        }
    }

    #[derive(Clone, Copy)]
    struct Job {
        data: *mut (),
        run: unsafe fn(*mut ()),
        finish: unsafe fn(*mut ()),
    }
    // SAFETY: job constructors constrain captures to Send/shared Sync and a
    // scope waits before their borrowed environment is released, even on unwind.
    unsafe impl Send for Job {}
    #[derive(Default)]
    struct SlotState {
        job: Option<Job>,
        busy: bool,
        background: bool,
    }
    #[derive(Default)]
    struct Slot {
        state: Mutex<SlotState>,
    }
    struct Worker {
        slot: Arc<Slot>,
        thread: JoinHandle<()>,
    }
    type Spawn = fn(Arc<Shared>, Arc<Slot>, usize) -> std::io::Result<JoinHandle<()>>;
    #[inline(never)]
    fn spawn_worker(
        shared: Arc<Shared>,
        slot: Arc<Slot>,
        index: usize,
    ) -> std::io::Result<JoinHandle<()>> {
        std::thread::Builder::new()
            .name(format!("ecs-worker-{index}"))
            .spawn(move || {
                enum Work {
                    Borrowed(Job),
                    Background(Arc<dyn BackgroundWork>),
                }
                loop {
                    let work = {
                        let mut state = lock(&shared.state);
                        loop {
                            if state.stopping {
                                return;
                            }
                            if let Some(job) = lock(&slot.state).job.take() {
                                break Work::Borrowed(job);
                            }
                            if state.system_demand == 0 {
                                let len = state.background.len();
                                let mut source = None;
                                for offset in 0..len {
                                    let index = (state.next_background + offset) % len;
                                    if let Some(candidate) = state.background[index].1.upgrade()
                                        && candidate.ready_count() != 0
                                    {
                                        source = Some(candidate);
                                        state.next_background = (index + 1) % len;
                                        break;
                                    }
                                }
                                if let Some(source) = source {
                                    {
                                        let mut slot = lock(&slot.state);
                                        slot.busy = true;
                                        slot.background = true;
                                    }
                                    state.background_running += 1;
                                    break Work::Background(source);
                                }
                            }
                            state = shared.wake.wait(state).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    match work {
                        Work::Borrowed(job) => {
                            // SAFETY: the submitting scope retains data through finish.
                            unsafe {
                                (job.run)(job.data);
                            }
                            {
                                let state = lock(&shared.state);
                                lock(&slot.state).busy = false;
                                state.notify_available();
                            }
                            shared.wake.notify_all();
                            unsafe {
                                (job.finish)(job.data);
                            }
                        }
                        Work::Background(source) => {
                            // Future polling/destruction have their own in-image shields.
                            source.poll_one();
                            drop(source);
                            {
                                let mut state = lock(&shared.state);
                                lock(&slot.state).busy = false;
                                state.background_running -= 1;
                                state.notify_available();
                            }
                            shared.wake.notify_all();
                        }
                    }
                }
            })
    }
    #[derive(Default)]
    struct PoolState {
        workers: Vec<Worker>,
        scopes: usize,
        stopping: bool,
        coordinators: Vec<std::thread::ThreadId>,
        background: Vec<(u64, Weak<dyn BackgroundWork>)>,
        next_source: u64,
        next_listener: u64,
        listeners: Vec<(u64, Sender<RunnerEvent>)>,
        background_running: usize,
        next_background: usize,
        system_demand: usize,
    }
    impl PoolState {
        fn notify_available(&self) {
            if self.system_demand != 0 {
                for (_, sender) in &self.listeners {
                    let _ = sender.send(RunnerEvent::WorkerAvailable);
                }
            }
        }
    }
    #[derive(Default)]
    struct Shared {
        state: Mutex<PoolState>,
        wake: Condvar,
    }
    pub(super) struct Pool {
        pub(super) capacity: usize,
        shared: Arc<Shared>,
        spawn: Spawn,
    }
    pub(crate) struct Lease<'a> {
        pool: &'a Pool,
        coordinator: Option<std::thread::ThreadId>,
    }
    impl Drop for Lease<'_> {
        fn drop(&mut self) {
            let mut state = lock(&self.pool.shared.state);
            state.scopes -= 1;
            if let Some(id) = self.coordinator {
                state.coordinators.retain(|other| *other != id);
            }
        }
    }

    impl Pool {
        pub(super) fn new(capacity: usize) -> Self {
            Self {
                capacity,
                shared: Arc::new(Shared::default()),
                spawn: spawn_worker,
            }
        }
        fn enter(&self) -> Lease<'_> {
            let mut state = lock(&self.shared.state);
            assert!(!state.stopping, "executor is stopping workers");
            state.scopes += 1;
            Lease {
                pool: self,
                coordinator: None,
            }
        }
        fn start(&self, state: &mut PoolState) -> std::io::Result<()> {
            let slot = Arc::new(Slot::default());
            let thread = (self.spawn)(self.shared.clone(), slot.clone(), state.workers.len())?;
            state.workers.push(Worker { slot, thread });
            Ok(())
        }
        pub(super) fn prepare_run(&self, needs_worker: bool) -> Result<Lease<'_>, String> {
            let mut state = lock(&self.shared.state);
            if state.stopping {
                return Err(ExecutorBusy.to_string());
            }
            let id = std::thread::current().id();
            if state.coordinators.contains(&id)
                || state.workers.iter().any(|w| w.thread.thread().id() == id)
            {
                return Err(
                    "a runner cannot re-enter its executor from a coordinator or worker".into(),
                );
            }
            if needs_worker && state.workers.is_empty() {
                self.start(&mut state).map_err(|e| e.to_string())?;
            }
            state.scopes += 1;
            state.coordinators.push(id);
            Ok(Lease {
                pool: self,
                coordinator: Some(id),
            })
        }
        // Only idle slots accept jobs. A waiting query cannot enqueue work
        // behind its own blocked system and cannot steal unrelated systems.
        fn submit(&self, make: impl FnOnce() -> Job) -> bool {
            let mut pool = lock(&self.shared.state);
            let idle = pool.workers.iter().position(|w| !lock(&w.slot.state).busy);
            let index = if let Some(index) = idle {
                index
            } else {
                if pool.workers.len() == self.capacity {
                    return false;
                }
                if let Err(error) = self.start(&mut pool) {
                    drop(pool);
                    log::warn!("worker startup failed: {error}");
                    return false;
                }
                pool.workers.len() - 1
            };
            let slot = &pool.workers[index].slot;
            let mut state = lock(&slot.state);
            state.job = Some(make());
            state.busy = true;
            state.background = false;
            self.shared.wake.notify_all();
            true
        }
        pub(super) fn register_background(&self, source: &Arc<dyn BackgroundWork>) -> u64 {
            let mut state = lock(&self.shared.state);
            let id = state.next_source;
            state.next_source += 1;
            state.background.push((id, Arc::downgrade(source)));
            id
        }
        pub(super) fn unregister_background(&self, id: u64) {
            let removed = {
                let mut state = lock(&self.shared.state);
                state
                    .background
                    .iter()
                    .position(|(key, _)| *key == id)
                    .map(|index| state.background.swap_remove(index))
            };
            drop(removed);
            self.shared.wake.notify_all();
        }
        pub(super) fn notify_background(&self) {
            let mut state = lock(&self.shared.state);
            if state.stopping {
                return;
            }
            let (ready, active) = state
                .background
                .iter()
                .filter_map(|(_, source)| source.upgrade())
                .fold((0, 0), |(ready, active), source| {
                    (ready + source.ready_count(), active + source.active_count())
                });
            let (busy, borrowed) = state
                .workers
                .iter()
                .fold((0, 0), |(busy, borrowed), worker| {
                    let slot = lock(&worker.slot.state);
                    (
                        busy + usize::from(slot.busy),
                        borrowed + usize::from(slot.busy && !slot.background),
                    )
                });
            // A self-wake can publish the next poll before the previous worker
            // releases its slot. Do not count that as a second compute task.
            let desired = (busy + ready).min(borrowed + active).min(self.capacity);
            while state.workers.len() < desired {
                if let Err(error) = self.start(&mut state) {
                    drop(state);
                    log::warn!(
                        "compute worker startup failed: {error}; manual ticking remains available"
                    );
                    return;
                }
            }
            self.shared.wake.notify_all();
        }
        pub(super) fn shutdown(&self) -> Result<(), ExecutorBusy> {
            self.stop(false)
        }
        fn stop(&self, dropping: bool) -> Result<(), ExecutorBusy> {
            let workers = {
                let mut state = lock(&self.shared.state);
                if !dropping
                    && (state.scopes != 0
                        || state.stopping
                        || state
                            .background
                            .iter()
                            .filter_map(|(_, source)| source.upgrade())
                            .any(|s| s.active_count() != 0))
                {
                    return Err(ExecutorBusy);
                }
                state.stopping = true;
                self.shared.wake.notify_all();
                std::mem::take(&mut state.workers)
            };
            let current = std::thread::current().id();
            for worker in workers {
                if worker.thread.thread().id() != current {
                    let _ = worker.thread.join();
                }
            }
            if !dropping {
                lock(&self.shared.state).stopping = false;
            }
            Ok(())
        }
        pub(super) fn scope<'env, R>(
            &'env self,
            f: impl for<'scope> FnOnce(&'scope TaskScope<'scope, 'env>) -> R,
        ) -> R {
            let _lease = self.enter();
            let demand = SystemDemand {
                pool: self,
                ready: AtomicBool::new(false),
                listener: Mutex::new(None),
            };
            let scope = TaskScope {
                pool: self,
                done: Arc::new(Done::default()),
                marker: PhantomData,
                systems_ready: &demand,
            };
            let _drain = Drain(&scope.done);
            let result = f(&scope);
            scope.done.propagate();
            result
        }
        pub(super) fn run<F: Fn() + Sync>(&self, participants: usize, work: &F) {
            let _lease = self.enter();
            let helpers = participants.max(1).saturating_sub(1).min(self.capacity);
            if helpers == 0 {
                work();
                return;
            }
            let scope = QueryScope {
                work,
                done: Done::default(),
            };
            let _drain = Drain(&scope.done);
            for _ in 0..helpers {
                if !self.submit(|| {
                    lock(&scope.done.state).remaining += 1;
                    Job {
                        data: (&scope as *const QueryScope<'_, F>).cast_mut().cast(),
                        run: query_run::<F>,
                        finish: query_finish::<F>,
                    }
                }) {
                    break;
                }
            }
            let caller = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
            scope.done.wait();
            if let Err(panic) = caller {
                std::panic::resume_unwind(panic);
            }
            scope.done.propagate();
        }
    }
    impl Drop for Pool {
        fn drop(&mut self) {
            let _ = self.stop(true);
        }
    }

    struct QueryScope<'a, F> {
        work: &'a F,
        done: Done,
    }
    unsafe fn query_run<F: Fn() + Sync>(data: *mut ()) {
        let scope = unsafe { &*data.cast::<QueryScope<'_, F>>() };
        if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(scope.work)) {
            lock(&scope.done.state).panics.push(panic);
        }
    }
    unsafe fn query_finish<F>(data: *mut ()) {
        let scope = unsafe { &*data.cast::<QueryScope<'_, F>>() };
        scope.done.complete(None);
    }

    /// Invariant scope lifetime prevents jobs from outliving local captures.
    pub(crate) struct TaskScope<'scope, 'env: 'scope> {
        pool: &'env Pool,
        done: Arc<Done>,
        marker: PhantomData<(&'scope mut &'scope (), &'env mut &'env ())>,
        systems_ready: &'scope SystemDemand<'env>,
    }
    struct Task<F, C> {
        work: Option<F>,
        complete: Option<C>,
        panic: Option<Panic>,
        done: Arc<Done>,
    }
    impl<'scope, 'env> TaskScope<'scope, 'env> {
        pub(crate) fn notify_available(&self, sender: Sender<RunnerEvent>) {
            let mut state = lock(&self.pool.shared.state);
            let id = state.next_listener;
            state.next_listener += 1;
            assert!(lock(&self.systems_ready.listener).replace(id).is_none());
            state.listeners.push((id, sender));
        }
        pub(crate) fn set_systems_ready(&self, ready: bool) {
            self.systems_ready.set(ready);
        }
        pub(crate) fn try_spawn<F, C>(&'scope self, work: F, complete: C) -> bool
        where
            F: FnOnce() + Send + 'scope,
            C: FnOnce() + Send + 'scope,
        {
            self.pool.submit(|| {
                let task = Box::new(Task {
                    work: Some(work),
                    complete: Some(complete),
                    panic: None,
                    done: self.done.clone(),
                });
                lock(&self.done.state).remaining += 1;
                Job {
                    data: Box::into_raw(task).cast(),
                    run: task_run::<F, C>,
                    finish: task_finish::<F, C>,
                }
            })
        }
    }
    struct SystemDemand<'a> {
        pool: &'a Pool,
        ready: AtomicBool,
        listener: Mutex<Option<u64>>,
    }
    impl SystemDemand<'_> {
        fn set(&self, ready: bool) {
            let mut state = lock(&self.pool.shared.state);
            if self.ready.swap(ready, Ordering::Relaxed) != ready {
                if ready {
                    state.system_demand += 1;
                } else {
                    state.system_demand -= 1;
                }
                self.pool.shared.wake.notify_all();
            }
        }
    }
    impl Drop for SystemDemand<'_> {
        fn drop(&mut self) {
            self.set(false);
            let id = lock(&self.listener).take();
            let sender = {
                let mut state = lock(&self.pool.shared.state);
                id.and_then(|id| state.listeners.iter().position(|(key, _)| *key == id))
                    .map(|index| state.listeners.swap_remove(index))
            };
            drop(sender);
        }
    }
    unsafe fn task_run<F: FnOnce(), C>(data: *mut ()) {
        let task = unsafe { &mut *data.cast::<Task<F, C>>() };
        task.panic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(task.work.take().unwrap())).err();
    }
    unsafe fn task_finish<F, C: FnOnce()>(data: *mut ()) {
        let mut task = unsafe { Box::from_raw(data.cast::<Task<F, C>>()) };
        let complete =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(task.complete.take().unwrap()));
        let panic = task.panic.take().or_else(|| complete.err());
        // Drop all borrowed captures before publishing scope completion.
        let done = task.done.clone();
        drop(task);
        done.complete(panic);
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[test]
        fn startup_failure_is_reported_and_queries_fall_back_inline() {
            let mut pool = Pool::new(2);
            pool.spawn = |_, _, _| Err(std::io::Error::other("unavailable"));
            assert!(pool.prepare_run(true).is_err());
            let calls = AtomicUsize::new(0);
            pool.run(3, &|| {
                calls.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert!(pool.shutdown().is_ok());
        }

        #[test]
        fn coordinator_unwind_drains_borrowed_jobs_and_releases_admission() {
            let pool = Pool::new(1);
            let finished = AtomicUsize::new(0);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.scope(|scope| {
                    assert!(scope.try_spawn(
                        || {
                            std::thread::sleep(std::time::Duration::from_millis(10));
                            finished.fetch_add(1, Ordering::Relaxed);
                        },
                        || {}
                    ));
                    panic!("coordinator failure");
                });
            }));
            assert!(result.is_err());
            assert_eq!(finished.load(Ordering::Relaxed), 1);
            assert!(pool.shutdown().is_ok());
        }
    }
}
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, Mutex, mpsc};
    use std::thread;
    use std::time::Duration;

    // A failure in nested scheduling must fail the test, not hang the suite.
    fn with_timeout(f: impl FnOnce() + Send + 'static) {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            tx.send(result).unwrap();
        });
        if let Err(panic) = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("executor stalled")
        {
            std::panic::resume_unwind(panic);
        }
    }

    #[test]
    fn coordinator_unwind_releases_queued_main_thread_requests() {
        with_timeout(|| {
            let executor = ParallelExecutor::new(1);
            let (sent_tx, sent_rx) = mpsc::channel();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                executor.scope(|scope| {
                    let (tx, rx) = mpsc::channel();
                    scope.notify_available(tx.clone());
                    assert!(scope.try_spawn(
                        move || {
                            let (answer_tx, answer_rx) = mpsc::channel::<()>();
                            tx.send(
                                crate::main_thread_dispatcher::RunnerEvent::MainThreadRequest(
                                    Box::new(move || {
                                        let _ = answer_tx.send(());
                                    }),
                                ),
                            )
                            .unwrap();
                            sent_tx.send(()).unwrap();
                            assert!(answer_rx.recv().is_err());
                        },
                        || {}
                    ));
                    sent_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    drop(rx);
                    panic!("coordinator failed");
                });
            }));
            assert!(result.is_err());
            executor.shutdown_workers().unwrap();
        });
    }

    #[test]
    fn ready_system_takes_the_next_worker_before_background_compute() {
        with_timeout(|| {
            let executor = ParallelExecutor::new(1);
            let compute =
                crate::ComputePool::with_executor(crate::IoRuntime::new(), executor.clone());
            let order = Arc::new(Mutex::new(Vec::new()));
            let (entered_tx, entered_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let first = compute.spawn(crate::Priority::Low, move |_| async move {
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            });
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let recorded = order.clone();
            let second = compute.spawn(crate::Priority::Critical, move |_| async move {
                recorded.lock().unwrap().push("compute");
            });
            executor.scope(|scope| {
                scope.set_systems_ready(true);
                release_tx.send(()).unwrap();
                let deadline = std::time::Instant::now() + Duration::from_secs(5);
                while !scope.try_spawn(
                    || {
                        order.lock().unwrap().push("system");
                    },
                    || {},
                ) {
                    assert!(std::time::Instant::now() < deadline);
                    thread::yield_now();
                }
                scope.set_systems_ready(false);
            });
            assert_eq!(first.recv_timeout(Duration::from_secs(5)), Some(()));
            assert_eq!(second.recv_timeout(Duration::from_secs(5)), Some(()));
            assert_eq!(*order.lock().unwrap(), ["system", "compute"]);
            compute.quiesce(Duration::from_secs(1)).unwrap();
            executor.shutdown_workers().unwrap();
        });
    }

    #[test]
    fn reuses_workers_and_includes_caller() {
        with_timeout(|| {
            let executor = ParallelExecutor::new(3);
            let caller = thread::current().id();
            let ids = Mutex::new(HashSet::new());
            let all_started = Barrier::new(4);
            executor.run(usize::MAX, || {
                ids.lock().unwrap().insert(thread::current().id());
                all_started.wait();
            });
            let original = ids.lock().unwrap().clone();
            assert_eq!(original.len(), 4);
            assert!(original.contains(&caller));
            for _ in 0..100 {
                executor.run(4, || {
                    assert!(original.contains(&thread::current().id()));
                });
            }
        });
    }

    #[test]
    fn zero_and_one_use_only_the_caller() {
        let caller = thread::current().id();
        for capacity in [0, 1, 4] {
            let executor = ParallelExecutor::new(capacity);
            for participants in [0, 1] {
                let calls = AtomicUsize::new(0);
                executor.run(participants, || {
                    assert_eq!(thread::current().id(), caller);
                    calls.fetch_add(1, Ordering::Relaxed);
                });
                assert_eq!(calls.load(Ordering::Relaxed), 1);
            }
        }
    }

    #[test]
    fn nested_calls_finish_when_all_workers_are_busy() {
        with_timeout(|| {
            let executor = ParallelExecutor::new(3);
            let all_started = Barrier::new(4);
            let calls = AtomicUsize::new(0);
            executor.run(4, || {
                all_started.wait();
                executor.run(4, || {
                    executor.run(4, || {
                        calls.fetch_add(1, Ordering::Relaxed);
                    });
                });
            });
            assert!(calls.load(Ordering::Relaxed) >= 4);
        });
    }

    #[test]
    fn concurrent_call_uses_caller_when_worker_is_occupied() {
        with_timeout(|| {
            let executor = Arc::new(ParallelExecutor::new(1));
            let occupied = Arc::new(Barrier::new(3));
            let release = Arc::new(Barrier::new(3));
            let first = {
                let (executor, occupied, release) =
                    (executor.clone(), occupied.clone(), release.clone());
                thread::spawn(move || {
                    executor.run(2, || {
                        occupied.wait();
                        release.wait();
                    })
                })
            };
            occupied.wait();
            let caller = thread::current().id();
            let calls = AtomicUsize::new(0);
            executor.run(2, || {
                assert_eq!(thread::current().id(), caller);
                calls.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            release.wait();
            first.join().unwrap();
        });
    }

    #[test]
    fn panic_drains_borrowed_jobs_and_executor_remains_usable() {
        with_timeout(|| {
            for panic_in_caller in [false, true] {
                let executor = ParallelExecutor::new(1);
                let caller = thread::current().id();
                let started = Barrier::new(2);
                let finished = AtomicUsize::new(0);
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    executor.run(2, || {
                        started.wait();
                        if (thread::current().id() == caller) == panic_in_caller {
                            panic!("query failure");
                        }
                        thread::sleep(Duration::from_millis(10));
                        finished.fetch_add(1, Ordering::Relaxed);
                    });
                }));
                assert!(panic.is_err());
                assert_eq!(finished.load(Ordering::Relaxed), 1);
                executor.run(2, || {
                    finished.fetch_add(1, Ordering::Relaxed);
                });
                assert!(finished.load(Ordering::Relaxed) >= 2);
            }
        });
    }
}
