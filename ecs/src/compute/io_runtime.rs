use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use redlilium_core::compute::{IoHandle, IoRunner, IoSender};

/// IO drain failed; the originating module must remain loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoShutdownError {
    /// Includes polling and future destruction, even without a result handle.
    TasksRemaining { remaining_tasks: usize },
    /// Tokio workers, blocking operations, or thread-local destructors remain.
    WorkersRemaining,
}
impl std::fmt::Display for IoShutdownError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TasksRemaining { remaining_tasks } => write!(
                f,
                "IO quiescence timeout with {remaining_tasks} active tasks"
            ),
            Self::WorkersRemaining => write!(f, "IO worker shutdown timeout"),
        }
    }
}
impl std::error::Error for IoShutdownError {}

/// An independently owned IO lifetime domain. Clones share its tasks and workers.
///
/// Each ECS runner owns one domain. Use a separate runtime for host work that
/// must survive a guest reload. Construct reloadable runtimes in the host image:
/// submission captures host code so retained Tokio wakers never use guest vtables.
/// Results retained by callers must still be released before guest unload.
#[derive(Clone)]
pub struct IoRuntime {
    inner: Arc<IoRuntimeInner>,
}
struct IoRuntimeInner {
    progress: Arc<Progress>,
    submit: fn(&IoRuntimeInner, TrackedTask),
    #[cfg(not(target_arch = "wasm32"))]
    native: Mutex<Native>,
}
#[derive(Default)]
struct Progress {
    active: Mutex<usize>,
    changed: Condvar,
}
struct Lifetime(Arc<Progress>);
impl Lifetime {
    fn new(progress: &Arc<Progress>) -> Self {
        *progress.active.lock().unwrap() += 1;
        Self(progress.clone())
    }
}
impl Drop for Lifetime {
    fn drop(&mut self) {
        *self.0.active.lock().unwrap() -= 1;
        self.0.changed.notify_all();
    }
}

#[cfg(not(target_arch = "wasm32"))]
type RuntimeContext = Option<tokio::runtime::Handle>;
#[cfg(target_arch = "wasm32")]
type RuntimeContext = ();

// Catch and destroy panic payloads inside the future's originating image.
trait IoFuture: Send {
    fn poll_guarded(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        runtime: &RuntimeContext,
    ) -> Result<Poll<()>, String>;
    fn dispose(self: Pin<Box<Self>>, runtime: &RuntimeContext) -> Result<(), String>;
}
// Keep the user future as a field rather than inside an async-await wrapper:
// an unwind from poll must reach our boundary before dropping that future.
// Otherwise a second panic in its destructor would abort during unwinding.
struct SendFuture<F, T> {
    future: F,
    sender: Option<IoSender<T>>,
}
impl<F: Future<Output = T> + Send, T: Send> IoFuture for SendFuture<F, T> {
    fn poll_guarded(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        runtime: &RuntimeContext,
    ) -> Result<Poll<()>, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // A statically linked guest has its own Tokio context TLS. Enter
            // the host handle here, inside the guest's generic implementation.
            #[cfg(not(target_arch = "wasm32"))]
            let _enter = runtime.as_ref().map(tokio::runtime::Handle::enter);
            #[cfg(target_arch = "wasm32")]
            let _ = runtime;
            // SAFETY: the enclosing allocation stays pinned until disposal.
            // We never move `future`; only the unpinned sender is taken.
            let this = unsafe { self.get_unchecked_mut() };
            let future = unsafe { Pin::new_unchecked(&mut this.future) };
            match future.poll(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(result) => {
                    let _ = this
                        .sender
                        .take()
                        .expect("polled after completion")
                        .send(result);
                    Poll::Ready(())
                }
            }
        }))
        .map_err(|payload| crate::system::panic_payload_to_string(&*payload))
    }
    fn dispose(self: Pin<Box<Self>>, runtime: &RuntimeContext) -> Result<(), String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[cfg(not(target_arch = "wasm32"))]
            let _enter = runtime.as_ref().map(tokio::runtime::Handle::enter);
            #[cfg(target_arch = "wasm32")]
            let _ = runtime;
            drop(self);
        }))
        .map_err(|payload| crate::system::panic_payload_to_string(&*payload))
    }
}

struct TrackedTask {
    future: Option<Pin<Box<dyn IoFuture>>>,
    runtime: RuntimeContext,
    // Last: the counter includes destruction of the future and its sender.
    _lifetime: Lifetime,
}
impl Future for TrackedTask {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = &mut *self;
        match this
            .future
            .as_mut()
            .expect("polled after completion")
            .as_mut()
            .poll_guarded(cx, &this.runtime)
        {
            Ok(Poll::Pending) => return Poll::Pending,
            Ok(Poll::Ready(())) => {}
            Err(message) => log::error!("IO task panicked: {message}"),
        }
        self.dispose();
        Poll::Ready(())
    }
}
impl TrackedTask {
    fn dispose(&mut self) {
        if let Some(future) = self.future.take()
            && let Err(message) = future.dispose(&self.runtime)
        {
            log::error!("IO future destruction panicked: {message}");
        }
    }
}
impl Drop for TrackedTask {
    fn drop(&mut self) {
        self.dispose();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct Native {
    runtime: Option<tokio::runtime::Runtime>,
    joining: Option<Arc<Joining>>,
}
#[cfg(not(target_arch = "wasm32"))]
struct Joining {
    done: Mutex<bool>,
    changed: Condvar,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}
#[cfg(not(target_arch = "wasm32"))]
impl Drop for IoRuntimeInner {
    fn drop(&mut self) {
        // Drop is not an unload barrier, and may run inside an IO future.
        if let Some(runtime) = self.native.get_mut().unwrap().runtime.take() {
            runtime.shutdown_background();
        }
    }
}

// Do not monomorphize Tokio's task/waker or the browser callback in a guest.
// The function pointer is captured when the host creates the runtime.
#[inline(never)]
fn submit(inner: &IoRuntimeInner, task: TrackedTask) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut native = inner.native.lock().unwrap();
        if native.joining.is_some() {
            drop(native);
            log::warn!("IO submission rejected while workers are shutting down");
            drop(task);
            return;
        }
        let runtime = native.runtime.get_or_insert_with(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .expect("Failed to create tokio IO runtime")
        });
        let mut task = task;
        task.runtime = Some(runtime.handle().clone());
        runtime.spawn(task);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = inner;
        wasm_bindgen_futures::spawn_local(task);
    }
}

impl IoRuntime {
    /// Creates an independent domain; native workers start on first submission.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(IoRuntimeInner {
                progress: Arc::new(Progress::default()),
                submit,
                #[cfg(not(target_arch = "wasm32"))]
                native: Mutex::new(Native::default()),
            }),
        }
    }

    /// Counts tasks until their futures and result senders have been destroyed.
    pub fn active_count(&self) -> usize {
        *self.inner.progress.active.lock().unwrap()
    }

    /// Wait for tracked IO without cancellation. Stop external producers first.
    /// A timeout leaves tasks running and does not permit unloading their code.
    /// On WASM this only checks: synchronous waits cannot drive the event loop.
    pub fn quiesce(&self, timeout: Duration) -> Result<(), IoShutdownError> {
        let active = self.inner.progress.active.lock().unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        let active = self
            .inner
            .progress
            .changed
            .wait_timeout_while(active, timeout, |active| *active != 0)
            .unwrap()
            .0;
        #[cfg(target_arch = "wasm32")]
        let _ = timeout;
        if *active == 0 {
            Ok(())
        } else {
            Err(IoShutdownError::TasksRemaining {
                remaining_tasks: *active,
            })
        }
    }

    pub(crate) fn is_stopped(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let native = self.inner.native.lock().unwrap();
            if native.runtime.is_some() || native.joining.is_some() {
                return false;
            }
        }
        self.active_count() == 0
    }

    /// Drain IO, then join native workers, blocking operations and their TLS.
    /// Call from the host outside IO tasks, with producers stopped through reload.
    /// Uses one deadline. After a worker timeout, submissions return disconnected
    /// handles until a successful retry. After success, workers restart lazily.
    /// Detached work on other runtimes/threads is outside this domain.
    pub fn shutdown(&self, timeout: Duration) -> Result<(), IoShutdownError> {
        #[cfg(not(target_arch = "wasm32"))]
        let start = Instant::now();
        self.quiesce(timeout)?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let joining = {
                let mut native = self.inner.native.lock().unwrap();
                let remaining_tasks = self.active_count();
                if remaining_tasks != 0 {
                    return Err(IoShutdownError::TasksRemaining { remaining_tasks });
                }
                if let Some(joining) = &native.joining {
                    joining.clone()
                } else if let Some(runtime) = native.runtime.take() {
                    let joining = Arc::new(Joining {
                        done: Mutex::new(false),
                        changed: Condvar::new(),
                        thread: Mutex::new(None),
                    });
                    let done = joining.clone();
                    *joining.thread.lock().unwrap() = Some(std::thread::spawn(move || {
                        // Unlike shutdown_timeout, dropping waits for blocking
                        // operations too. This helper may outlive the timeout;
                        // the host must keep the guest mapped until retry succeeds.
                        drop(runtime);
                        *done.done.lock().unwrap() = true;
                        done.changed.notify_all();
                    }));
                    native.joining = Some(joining.clone());
                    joining
                } else {
                    return Ok(());
                }
            };
            // No admission lock held: worker TLS may release runtime clones.
            let done = joining.done.lock().unwrap();
            let done = joining
                .changed
                .wait_timeout_while(done, timeout.saturating_sub(start.elapsed()), |done| !*done)
                .unwrap()
                .0;
            if !*done {
                return Err(IoShutdownError::WorkersRemaining);
            }
            drop(done);
            if let Some(thread) = joining.thread.lock().unwrap().take() {
                thread.join().expect("IO shutdown helper panicked");
            }
            let mut native = self.inner.native.lock().unwrap();
            if native
                .joining
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &joining))
            {
                native.joining = None;
            }
        }
        Ok(())
    }
}
impl IoRunner for IoRuntime {
    /// Submission is tracked independently of the result handle. Dropping the
    /// handle does not cancel IO. A panic disconnects it and is logged; panic
    /// payloads and future destructors are handled inside the originating image.
    /// A published result may precede destruction; use quiescence before unload.
    /// Raw Tokio task spawning bypasses this lifetime and panic boundary.
    fn run<T, F>(&self, future: F) -> IoHandle<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let lifetime = Lifetime::new(&self.inner.progress);
        let (sender, handle) = IoHandle::channel();
        let future = Box::pin(SendFuture {
            future,
            sender: Some(sender),
        });
        (self.inner.submit)(
            &self.inner,
            TrackedTask {
                future: Some(future),
                runtime: RuntimeContext::default(),
                _lifetime: lifetime,
            },
        );
        handle
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

    use crate::compute::pool::noop_waker;

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
