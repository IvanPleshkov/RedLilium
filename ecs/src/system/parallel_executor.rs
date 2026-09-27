//! Shared workers for systems and synchronous borrowed queries.
//! Ready systems are queued by their runner. Query helpers use only idle
//! workers; a waiting query never steals unrelated systems or compute work.

use std::sync::Arc;

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

/// Workers cannot be stopped while borrowed execution scopes are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorBusy;
impl std::fmt::Display for ExecutorBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("executor has active scopes or is stopping workers")
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
    /// anything if a borrowed scope is active. Admissions are rejected during
    /// shutdown; later calls may lazily restart workers. The caller must keep
    /// guest producers stopped until unloading completes.
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
    use super::ExecutorBusy;
    use std::any::Any;
    use std::marker::PhantomData;
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
        stop: bool,
    }
    #[derive(Default)]
    struct Slot {
        state: Mutex<SlotState>,
        wake: Condvar,
    }
    struct Worker {
        slot: Arc<Slot>,
        thread: JoinHandle<()>,
    }
    type Spawn = fn(Arc<Slot>, usize) -> std::io::Result<JoinHandle<()>>;
    // Store this entry point at construction: lazy startup from a guest must
    // keep the persistent loop in the constructor's image.
    #[inline(never)]
    fn spawn_worker(slot: Arc<Slot>, index: usize) -> std::io::Result<JoinHandle<()>> {
        std::thread::Builder::new()
            .name(format!("ecs-worker-{index}"))
            .spawn(move || {
                loop {
                    let job = {
                        let mut state = lock(&slot.state);
                        while state.job.is_none() && !state.stop {
                            state = slot.wake.wait(state).unwrap_or_else(|e| e.into_inner());
                        }
                        if state.stop {
                            break;
                        }
                        state.job.take().expect("notified with a job")
                    };
                    // SAFETY: the submitting scope retains the data until finish.
                    unsafe {
                        (job.run)(job.data);
                    }
                    lock(&slot.state).busy = false;
                    // Publish completion after releasing the slot, so a coordinator
                    // can immediately dispatch its next ready system to this worker.
                    unsafe {
                        (job.finish)(job.data);
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
    }
    pub(super) struct Pool {
        pub(super) capacity: usize,
        state: Mutex<PoolState>,
        spawn: Spawn,
    }
    pub(crate) struct Lease<'a> {
        pool: &'a Pool,
        coordinator: Option<std::thread::ThreadId>,
    }
    impl Drop for Lease<'_> {
        fn drop(&mut self) {
            let mut state = lock(&self.pool.state);
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
                state: Mutex::new(PoolState::default()),
                spawn: spawn_worker,
            }
        }
        fn enter(&self) -> Lease<'_> {
            let mut state = lock(&self.state);
            assert!(!state.stopping, "executor is stopping workers");
            state.scopes += 1;
            Lease {
                pool: self,
                coordinator: None,
            }
        }
        fn start(&self, state: &mut PoolState) -> std::io::Result<()> {
            let slot = Arc::new(Slot::default());
            let thread = (self.spawn)(slot.clone(), state.workers.len())?;
            state.workers.push(Worker { slot, thread });
            Ok(())
        }
        pub(super) fn prepare_run(&self, needs_worker: bool) -> Result<Lease<'_>, String> {
            let mut state = lock(&self.state);
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
            let mut pool = lock(&self.state);
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
            slot.wake.notify_one();
            true
        }
        pub(super) fn shutdown(&self) -> Result<(), ExecutorBusy> {
            let workers = {
                let mut state = lock(&self.state);
                if state.scopes != 0 || state.stopping {
                    return Err(ExecutorBusy);
                }
                state.stopping = true;
                std::mem::take(&mut state.workers)
            };
            for worker in &workers {
                lock(&worker.slot.state).stop = true;
                worker.slot.wake.notify_one();
            }
            for worker in workers {
                let _ = worker.thread.join();
            }
            lock(&self.state).stopping = false;
            Ok(())
        }
        pub(super) fn scope<'env, R>(
            &'env self,
            f: impl for<'scope> FnOnce(&'scope TaskScope<'scope, 'env>) -> R,
        ) -> R {
            let _lease = self.enter();
            let scope = TaskScope {
                pool: self,
                done: Arc::new(Done::default()),
                marker: PhantomData,
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
            self.shutdown().expect("last pool owner has no scopes");
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
    }
    struct Task<F, C> {
        work: Option<F>,
        complete: Option<C>,
        panic: Option<Panic>,
        done: Arc<Done>,
    }
    impl<'scope, 'env> TaskScope<'scope, 'env> {
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
            pool.spawn = |_, _| Err(std::io::Error::other("unavailable"));
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
