use std::future::Future;
use std::sync::Arc;

use crate::io_handle::IoHandle;

/// Runtime for spawning real async IO operations.
///
/// Bridges the ECS custom executor (noop wakers, manual polling) with a
/// real async runtime that can drive IO futures:
/// - **Native**: tokio multi-thread runtime with 1 worker thread
/// - **WASM**: `wasm_bindgen_futures::spawn_local`
///
/// Owned by the ECS runner, accessible via [`SystemContext::io()`](crate::SystemContext::io).
/// Clone is cheap (Arc-wrapped) — capture in compute tasks to do IO from background work.
///
/// # Example
///
/// ```ignore
/// // From a system:
/// let handle = ctx.io().run(async {
///     tokio::fs::read_to_string("config.json").await
/// });
/// let config = handle.await;
///
/// // From a compute task (clone the runtime):
/// let io = ctx.io().clone();
/// ctx.compute().spawn(Priority::Low, async move {
///     let data = io.run(async { fetch_data().await }).await;
///     process(data)
/// });
/// ```
#[derive(Clone)]
pub struct IoRuntime {
    inner: Arc<IoRuntimeInner>,
}

#[cfg(not(target_arch = "wasm32"))]
struct IoRuntimeInner {
    runtime: tokio::runtime::Runtime,
}

#[cfg(target_arch = "wasm32")]
struct IoRuntimeInner {}

impl IoRuntime {
    /// Creates a new IO runtime.
    ///
    /// On native, this starts a tokio runtime with one worker thread
    /// dedicated to driving IO futures. On WASM, this is a lightweight
    /// handle (no actual runtime — `spawn_local` is global).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("Failed to create tokio IO runtime");

        Self {
            inner: Arc::new(IoRuntimeInner { runtime }),
        }
    }

    /// Creates a new IO runtime (WASM variant).
    #[cfg(target_arch = "wasm32")]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(IoRuntimeInner {}),
        }
    }

    /// Spawns an async IO future on the real runtime.
    ///
    /// Returns an [`IoHandle`] that can be `.await`ed or polled via
    /// `try_recv()`. The handle works with the ECS noop waker — it
    /// checks a channel internally.
    ///
    /// # Platform behavior
    ///
    /// - **Native**: Future runs on tokio's worker thread with real wakers
    ///   and IO reactor. Results are available within the same frame.
    /// - **WASM**: Future runs via `spawn_local` on the browser event loop.
    ///   Results arrive after control returns to the browser (next frame).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run<T, F>(&self, future: F) -> IoHandle<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();

        self.inner.runtime.spawn(async move {
            let result = future.await;
            let _ = sender.send(result);
        });

        IoHandle::new(receiver)
    }

    /// Spawns an async IO future on the browser event loop (WASM variant).
    #[cfg(target_arch = "wasm32")]
    pub fn run<T, F>(&self, future: F) -> IoHandle<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();

        wasm_bindgen_futures::spawn_local(async move {
            let result = future.await;
            let _ = sender.send(result);
        });

        IoHandle::new(receiver)
    }
}

impl Default for IoRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;
    use std::task::{Context, Poll};

    use std::pin::Pin;

    use crate::compute::noop_waker;

    #[test]
    fn creation_and_clone() {
        let io = IoRuntime::new();
        let _io2 = io.clone();
    }

    #[test]
    fn run_simple_task() {
        let io = IoRuntime::new();
        let handle = io.run(async { 42u32 });
        assert_eq!(handle.recv(), Some(42));
    }

    #[test]
    fn run_with_tokio_sleep() {
        let io = IoRuntime::new();
        let handle = io.run(async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            99u32
        });

        // Block until ready
        assert_eq!(handle.recv(), Some(99));
    }

    #[test]
    fn handle_as_future_with_noop_waker() {
        let io = IoRuntime::new();
        let mut handle = io.run(async { 77u32 });

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // Poll until ready (IO runs on tokio thread)
        loop {
            match Pin::new(&mut handle).poll(&mut cx) {
                Poll::Ready(Some(val)) => {
                    assert_eq!(val, 77);
                    break;
                }
                Poll::Ready(None) => panic!("IO task dropped without result"),
                Poll::Pending => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
    }

    #[test]
    fn multiple_concurrent_tasks() {
        let io = IoRuntime::new();
        let h1 = io.run(async { 1u32 });
        let h2 = io.run(async { 2u32 });
        let h3 = io.run(async { 3u32 });

        assert_eq!(h1.recv(), Some(1));
        assert_eq!(h2.recv(), Some(2));
        assert_eq!(h3.recv(), Some(3));
    }

    #[test]
    fn clone_used_in_closure() {
        let io = IoRuntime::new();
        let io_clone = io.clone();

        let handle = std::thread::spawn(move || {
            let h = io_clone.run(async { "from_thread" });
            h.recv()
        });

        assert_eq!(handle.join().unwrap(), Some("from_thread"));
    }
}
