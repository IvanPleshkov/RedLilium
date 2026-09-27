//! Reusable workers for synchronous, borrowed parallel queries.
//!
//! Jobs go only to idle workers. Callers also execute work and never wait for
//! queued jobs to start, so nested calls cannot starve a fixed-size pool. This
//! executor never steals arbitrary systems/compute tasks while a query holds
//! component locks. Native synchronization uses std primitives across dylibs.

/// Reusable execution capacity for parallel queries.
///
/// A world owns one executor, shared by all its queries. `num_threads` includes
/// the calling thread; workers are started lazily and joined when the executor
/// is dropped. On wasm execution is sequential. The executor must be destroyed
/// while the image that constructed it is still loaded. Standalone executors
/// that run guest callbacks must also be dropped before unloading that guest,
/// so worker TLS destructors cannot outlive its code.
pub struct ParallelExecutor {
    #[cfg(not(target_arch = "wasm32"))]
    native: native::Pool,
}

impl ParallelExecutor {
    /// Set the maximum parallelism of one call, including its caller.
    /// Zero is treated as one. Workers are created only when needed.
    pub fn new(num_threads: usize) -> Self {
        let _ = num_threads;
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            native: native::Pool::new(num_threads.max(1)),
        }
    }

    /// Maximum number of participants in a call, including the caller.
    pub fn parallelism(&self) -> usize {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.native.parallelism
        }
        #[cfg(target_arch = "wasm32")]
        {
            1
        }
    }

    /// Stop workers and run their TLS destructors before unloading game code.
    /// Exclusive access guarantees no borrowed query jobs are still running.
    pub(crate) fn shutdown(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.native.shutdown();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn run(&self, participants: usize, work: impl Fn() + Sync) {
        self.native.run(participants, &work);
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let parallelism = std::thread::available_parallelism().map_or(1, |n| n.get());
        #[cfg(target_arch = "wasm32")]
        let parallelism = 1;
        Self::new(parallelism)
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::any::Any;
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    use std::thread::JoinHandle;

    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(|e| e.into_inner())
    }

    struct Completion {
        remaining: usize,
        panics: Vec<Box<dyn Any + Send>>,
    }
    struct Scope<'a, F> {
        work: &'a F,
        completion: Mutex<Completion>,
        done: Condvar,
    }
    impl<F> Scope<'_, F> {
        fn wait(&self) {
            let mut state = lock(&self.completion);
            while state.remaining != 0 {
                state = self.done.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        }
    }
    // Keep cleanup separate: Drop must not create an exclusive reference to
    // Scope while workers may still be borrowing it during dispatch unwind.
    struct Drain<'scope, 'work, F>(&'scope Scope<'work, F>);
    impl<F> Drop for Drain<'_, '_, F> {
        fn drop(&mut self) {
            self.0.wait();
        }
    }

    #[derive(Clone, Copy)]
    struct Job {
        data: *const (),
        invoke: unsafe fn(*const ()),
    }
    // SAFETY: only shared references to F: Sync cross threads; Drain
    // waits for all invocations before either the scope or F can be destroyed.
    unsafe impl Send for Job {}

    unsafe fn invoke<F: Fn() + Sync>(data: *const ()) {
        // SAFETY: dispatch increments remaining before publishing this pointer;
        // the scope owner waits for completion even if it unwinds.
        let scope = unsafe { &*data.cast::<Scope<'_, F>>() };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(scope.work));
        let mut state = lock(&scope.completion);
        if let Err(panic) = result {
            state.panics.push(panic);
        }
        state.remaining -= 1;
        scope.done.notify_all();
        // No access to scope after releasing this mutex: the caller may return.
    }

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

    // Keep the spawn entry point in the constructor's image even if a game
    // dylib triggers lazy startup. The worker loop must not live in that dylib.
    type Spawn = fn(Arc<Slot>, usize) -> std::io::Result<JoinHandle<()>>;
    #[inline(never)]
    fn spawn_worker(slot: Arc<Slot>, index: usize) -> std::io::Result<JoinHandle<()>> {
        std::thread::Builder::new()
            .name(format!("ecs-query-{index}"))
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
                    // SAFETY: the submitting scope retains all captured borrows.
                    unsafe { (job.invoke)(job.data) };
                    lock(&slot.state).busy = false;
                }
            })
    }

    pub(super) struct Pool {
        pub(super) parallelism: usize,
        workers: Mutex<Vec<Worker>>,
        spawn: Spawn,
    }
    impl Pool {
        pub(super) fn new(parallelism: usize) -> Self {
            Self {
                parallelism,
                workers: Mutex::new(Vec::new()),
                spawn: spawn_worker,
            }
        }

        pub(super) fn run<F: Fn() + Sync>(&self, participants: usize, work: &F) {
            let helpers = participants.max(1).min(self.parallelism) - 1;
            if helpers == 0 {
                work();
                return;
            }
            let scope = Scope {
                work,
                completion: Mutex::new(Completion {
                    remaining: 0,
                    panics: Vec::new(),
                }),
                done: Condvar::new(),
            };
            let _drain = Drain(&scope);
            let job = Job {
                data: (&scope as *const Scope<'_, F>).cast(),
                invoke: invoke::<F>,
            };
            {
                let mut workers = lock(&self.workers);
                // Grow lazily, never beyond the executor's capacity. Startup
                // failure leaves a smaller working pool; the caller still runs.
                while workers.len() < helpers {
                    let slot = Arc::new(Slot::default());
                    match (self.spawn)(slot.clone(), workers.len()) {
                        Ok(thread) => workers.push(Worker { slot, thread }),
                        Err(error) => {
                            log::warn!("query worker startup failed: {error}");
                            break;
                        }
                    }
                }
                let mut submitted = 0;
                for worker in workers.iter() {
                    if submitted == helpers {
                        break;
                    }
                    let mut state = lock(&worker.slot.state);
                    if state.busy {
                        continue;
                    }
                    lock(&scope.completion).remaining += 1;
                    state.busy = true;
                    state.job = Some(job);
                    worker.slot.wake.notify_one();
                    submitted += 1;
                }
            }
            // Caller participates; panic never skips draining the borrowed jobs.
            let caller = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
            scope.wait();
            let mut panics = std::mem::take(&mut lock(&scope.completion).panics);
            if let Err(panic) = caller {
                std::panic::resume_unwind(panic);
            }
            if let Some(panic) = panics.pop() {
                std::panic::resume_unwind(panic);
            }
        }

        pub(super) fn shutdown(&mut self) {
            let workers = self.workers.get_mut().unwrap_or_else(|e| e.into_inner());
            for worker in workers.iter() {
                lock(&worker.slot.state).stop = true;
                worker.slot.wake.notify_one();
            }
            for worker in workers.drain(..) {
                let _ = worker.thread.join();
            }
        }
    }
    impl Drop for Pool {
        fn drop(&mut self) {
            self.shutdown();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[test]
        fn startup_failure_still_runs_the_caller() {
            let mut pool = Pool::new(4);
            pool.spawn = |_, _| Err(std::io::Error::other("worker unavailable"));
            let calls = AtomicUsize::new(0);
            pool.run(4, &|| {
                calls.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert!(lock(&pool.workers).is_empty());
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
            let executor = ParallelExecutor::new(4);
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
            let executor = ParallelExecutor::new(4);
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
            let executor = Arc::new(ParallelExecutor::new(2));
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
                let executor = ParallelExecutor::new(2);
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
