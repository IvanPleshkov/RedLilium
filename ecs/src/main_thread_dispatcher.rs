use std::sync::mpsc;

/// Type-erased work closure sent to the main thread.
///
/// Uses `'static` bound to avoid invariance issues with `mpsc::Sender`.
/// The actual closures may capture shorter-lived references — the caller
/// uses `unsafe` transmute to erase the lifetime, which is safe because
/// `std::thread::scope` guarantees the closure is consumed before the
/// captured references expire.
pub(crate) type MainThreadWork = Box<dyn FnOnce() + Send>;

/// Events processed by the multi-threaded runner's main loop.
///
/// Unifies system completion signals and main-thread dispatch requests
/// into a single channel, so `recv_timeout` wakes on either event type.
pub(crate) enum RunnerEvent {
    /// A system at the given index has finished execution.
    SystemCompleted(usize),
    /// A worker requests that a closure be executed on the main thread.
    MainThreadRequest(MainThreadWork),
}

/// Handle used by workers to dispatch closures to the main thread.
///
/// Stored in [`SystemContext`](crate::SystemContext) and accessed by
/// [`LockRequest::execute()`](crate::LockRequest) when the access set
/// contains main-thread resources.
pub(crate) struct MainThreadDispatcher {
    sender: mpsc::Sender<RunnerEvent>,
}

impl MainThreadDispatcher {
    /// Creates a new dispatcher backed by the given event channel sender.
    pub fn new(sender: mpsc::Sender<RunnerEvent>) -> Self {
        Self { sender }
    }

    /// Sends a work closure to the main thread and blocks until the result
    /// is available.
    ///
    /// The caller is responsible for ensuring the work closure is valid
    /// (i.e., all captured references are alive) at the time of execution.
    /// This is guaranteed by the runner's `std::thread::scope`.
    pub fn send_work(&self, work: MainThreadWork) {
        self.sender
            .send(RunnerEvent::MainThreadRequest(work))
            .expect("Main thread receiver disconnected");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_runs_on_receiver_thread() {
        let (tx, rx) = mpsc::channel::<RunnerEvent>();
        let dispatcher = MainThreadDispatcher::new(tx);

        // Spawn a "worker" that dispatches to "main thread"
        let handle = std::thread::spawn(move || {
            let (result_tx, result_rx) = mpsc::sync_channel::<u32>(1);
            dispatcher.send_work(Box::new(move || {
                let _ = result_tx.send(42u32);
            }));
            result_rx.recv().unwrap()
        });

        // "Main thread" services the request
        match rx.recv().unwrap() {
            RunnerEvent::MainThreadRequest(work) => work(),
            _ => panic!("Expected MainThreadRequest"),
        }

        assert_eq!(handle.join().unwrap(), 42);
    }

    #[test]
    fn send_completed_event_via_channel() {
        let (tx, rx) = mpsc::channel::<RunnerEvent>();
        let _ = tx.send(RunnerEvent::SystemCompleted(5));

        match rx.recv().unwrap() {
            RunnerEvent::SystemCompleted(idx) => assert_eq!(idx, 5),
            _ => panic!("Expected SystemCompleted"),
        }
    }
}
