//! Cooperative async compute primitives.
//!
//! This module provides the building blocks for cooperative async compute:
//!
//! - [`Priority`] — Task priority levels (Critical, High, Low)
//! - [`IoHandle`] — Channel-based future for IO task results
//! - [`YieldNow`] — Cooperative yielding future
//! - [`IoRunner`] — Trait for spawning real async IO operations
//! - [`ComputeContext`] — Trait combining yield + IO for compute tasks
//! - [`ComputeMutex`] — Async-aware mutex for cooperative executors
//! - [`ComputeRwLock`] — Async-aware reader-writer lock for cooperative executors

mod cancellation;
mod io_handle;
mod mutex;
mod priority;
mod rwlock;
mod yield_now;

pub use cancellation::{CancellationToken, Cancelled, Checkpoint};
pub use io_handle::IoHandle;
pub use mutex::{ComputeMutex, ComputeMutexGuard, ComputeMutexLock};
pub use priority::Priority;
pub use rwlock::{
    ComputeReadGuard, ComputeRwLock, ComputeRwLockRead, ComputeRwLockWrite, ComputeWriteGuard,
};
pub use yield_now::{YieldNow, reset_yield_timer, set_yield_interval, yield_now};

use std::future::Future;

/// Trait for spawning real async IO operations.
///
/// Bridges a cooperative executor (noop wakers, manual polling) with a real
/// async runtime that can drive IO futures. Results are delivered via
/// [`IoHandle`], which works with noop wakers by checking a channel.
///
/// # Implementors
///
/// - `IoRuntime` in the ECS crate (tokio on native, wasm-bindgen on web)
///
/// Not object-safe due to generic methods — that is intentional. There is
/// typically one concrete implementation per application.
pub trait IoRunner: Clone + Send + Sync + 'static {
    /// Spawns an async IO future on the real runtime.
    ///
    /// Returns an [`IoHandle`] that can be `.await`ed or polled via
    /// `try_recv()`. The handle works with noop wakers — it checks
    /// a channel internally.
    fn run<T, F>(&self, future: F) -> IoHandle<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static;
}

/// Context provided to async compute tasks for cooperative yielding and IO.
///
/// Compute tasks (pathfinding, terrain generation, etc.) receive a
/// `ComputeContext` that provides:
/// - Cooperative yielding via [`yield_now()`](ComputeContext::yield_now)
/// - IO access via [`io()`](ComputeContext::io)
///
/// # Usage from standalone libraries
///
/// Libraries that depend only on `redlilium-core` can be generic over
/// `C: ComputeContext`:
///
/// ```ignore
/// async fn find_path<C: ComputeContext>(ctx: &C, graph: &NavMesh) -> Path {
///     for chunk in graph.chunks() {
///         process(chunk);
///         ctx.yield_now().await;
///     }
///     result
/// }
/// ```
///
/// # Usage from ECS systems
///
/// ```ignore
/// ctx.compute().spawn(Priority::Low, |cctx| async move {
///     cctx.io().run(async { load_data().await }).await;
///     cctx.yield_now().await;
///     heavy_computation()
/// });
/// ```
pub trait ComputeContext: Clone + Send + Sync + 'static {
    /// The IO runner type used by this context.
    type Io: IoRunner;

    /// Yields control back to the executor, allowing other tasks to run.
    ///
    /// Calls can be liberal — the runtime only actually suspends when
    /// enough wall-clock time has elapsed since the last real yield.
    fn yield_now(&self) -> YieldNow;

    /// Yields control and checks for cancellation.
    ///
    /// Like [`yield_now()`](Self::yield_now), but returns `Err(Cancelled)`
    /// if the task has been cancelled. Use with `?` to stop cooperative
    /// tasks early at yield points.
    ///
    /// The default implementation never cancels (no token attached).
    /// Implementations backed by a task pool should override this to
    /// wire in the real cancellation token.
    ///
    /// # Example
    ///
    /// ```ignore
    /// async fn work<C: ComputeContext>(ctx: &C) -> Result<u32, Cancelled> {
    ///     for i in 0..1000 {
    ///         ctx.checkpoint().await?;
    ///     }
    ///     Ok(42)
    /// }
    /// ```
    fn checkpoint(&self) -> Checkpoint {
        Checkpoint::yield_only()
    }

    /// Returns a reference to the IO runner for spawning async IO tasks.
    fn io(&self) -> &Self::Io;
}
