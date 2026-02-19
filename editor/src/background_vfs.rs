use std::sync::mpsc;

use redlilium_vfs::{Vfs, VfsError};

/// Opaque identifier for an in-flight VFS request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VfsRequestId(u64);

/// Result of a completed background VFS operation.
pub enum VfsResult {
    ListDir(Result<Vec<String>, VfsError>),
    Write(Result<(), VfsError>),
}

/// Non-blocking VFS dispatcher for the editor UI.
///
/// VFS futures from network providers (e.g. SFTP) are truly async and
/// cannot be polled with `poll_now()`. This helper spawns them on a
/// background tokio runtime and delivers results via a channel that
/// the UI thread can poll each frame.
pub struct BackgroundVfs {
    runtime: tokio::runtime::Runtime,
    result_tx: mpsc::Sender<(VfsRequestId, VfsResult)>,
    result_rx: mpsc::Receiver<(VfsRequestId, VfsResult)>,
    next_id: u64,
}

impl BackgroundVfs {
    pub fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("failed to create BackgroundVfs tokio runtime");

        let (tx, rx) = mpsc::channel();

        Self {
            runtime,
            result_tx: tx,
            result_rx: rx,
            next_id: 0,
        }
    }

    /// Dispatch an async `list_dir` request. Returns an ID to match the result.
    pub fn list_dir(&mut self, vfs: &Vfs, path: &str) -> VfsRequestId {
        let id = VfsRequestId(self.next_id);
        self.next_id += 1;

        let future = vfs.list_dir(path);
        let tx = self.result_tx.clone();

        self.runtime.spawn(async move {
            let result = future.await;
            let _ = tx.send((id, VfsResult::ListDir(result)));
        });

        id
    }

    /// Dispatch an async `write` request. Returns an ID to match the result.
    pub fn write(&mut self, vfs: &Vfs, path: &str, data: Vec<u8>) -> VfsRequestId {
        let id = VfsRequestId(self.next_id);
        self.next_id += 1;

        let future = vfs.write(path, data);
        let tx = self.result_tx.clone();

        self.runtime.spawn(async move {
            let result = future.await;
            let _ = tx.send((id, VfsResult::Write(result)));
        });

        id
    }

    /// Drain all completed results available this frame.
    pub fn poll_results(&self) -> Vec<(VfsRequestId, VfsResult)> {
        let mut results = Vec::new();
        while let Ok(item) = self.result_rx.try_recv() {
            results.push(item);
        }
        results
    }
}
