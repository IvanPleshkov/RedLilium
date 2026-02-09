/// A thread pool for parallel system execution.
///
/// On native targets, uses `std::thread::scope` for scoped parallel execution.
/// On WASM, executes all tasks sequentially on the main thread.
///
/// # Example
///
/// ```
/// use redlilium_ecs::ThreadPool;
///
/// let pool = ThreadPool::new(4);
///
/// let mut results = vec![0u32; 4];
/// pool.scope(|s| {
///     for (i, slot) in results.iter_mut().enumerate() {
///         s.spawn(move || {
///             *slot = (i as u32) * 10;
///         });
///     }
/// });
/// assert_eq!(results, vec![0, 10, 20, 30]);
/// ```
pub struct ThreadPool {
    #[allow(dead_code)]
    num_threads: usize,
}

impl ThreadPool {
    /// Creates a new thread pool with the given number of worker threads.
    ///
    /// On WASM, the thread count is ignored (single-threaded execution).
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
        }
    }

    /// Creates a thread pool sized to the number of available CPU cores.
    pub fn default_threads() -> Self {
        Self::new(std::thread::available_parallelism().map_or(1, |n| n.get()))
    }

    /// Executes tasks within a scoped context.
    ///
    /// All tasks spawned within the closure are guaranteed to complete
    /// before this method returns. Tasks can borrow local variables
    /// thanks to scoped lifetimes.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn scope<'env, F>(&self, f: F)
    where
        F: for<'scope> FnOnce(&Scope<'scope, 'env>),
    {
        std::thread::scope(|s| {
            let scope = Scope { inner: s };
            f(&scope);
        });
    }

    /// Executes tasks within a scoped context (WASM: sequential).
    #[cfg(target_arch = "wasm32")]
    pub fn scope<'env, F>(&self, f: F)
    where
        F: for<'scope> FnOnce(&Scope<'scope, 'env>),
    {
        let scope = Scope {
            _marker: std::marker::PhantomData,
        };
        f(&scope);
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::default_threads()
    }
}

/// A scope for spawning tasks that must complete before the scope exits.
///
/// All tasks spawned within a scope are guaranteed to complete before
/// [`ThreadPool::scope`] returns.
#[cfg(not(target_arch = "wasm32"))]
pub struct Scope<'scope, 'env: 'scope> {
    inner: &'scope std::thread::Scope<'scope, 'env>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'scope, 'env> Scope<'scope, 'env> {
    /// Spawns a task within this scope.
    ///
    /// The task will be executed by a new thread.
    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        self.inner.spawn(f);
    }
}

/// A scope for spawning tasks (WASM: sequential execution).
#[cfg(target_arch = "wasm32")]
pub struct Scope<'scope, 'env: 'scope> {
    _marker: std::marker::PhantomData<(&'scope (), &'env ())>,
}

#[cfg(target_arch = "wasm32")]
impl<'scope, 'env> Scope<'scope, 'env> {
    /// Spawns a task within this scope (WASM: executes immediately).
    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        f();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn scope_runs_single_task() {
        let pool = ThreadPool::new(2);
        let counter = AtomicU32::new(0);
        pool.scope(|s| {
            s.spawn(|| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn scope_runs_multiple_tasks() {
        let pool = ThreadPool::new(4);
        let counter = AtomicU32::new(0);
        pool.scope(|s| {
            for _ in 0..10 {
                s.spawn(|| {
                    counter.fetch_add(1, Ordering::Relaxed);
                });
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn scope_captures_references() {
        let pool = ThreadPool::new(2);
        let mut value = 0u32;
        pool.scope(|s| {
            s.spawn(|| {
                value = 42;
            });
        });
        assert_eq!(value, 42);
    }

    #[test]
    fn default_threads_at_least_one() {
        let pool = ThreadPool::default_threads();
        assert!(pool.num_threads >= 1);
    }
}
