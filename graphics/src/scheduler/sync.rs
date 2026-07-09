//! GPU synchronization primitives.
//!
//! This module provides synchronization types for coordinating work
//! between the CPU and GPU, and between different GPU operations.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::backend::GpuFence;
use crate::instance::GraphicsInstance;

/// Status of a fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceStatus {
    /// The fence has not yet been signaled.
    Unsignaled,
    /// The fence has been signaled (GPU work complete).
    Signaled,
}

/// Internal fence implementation.
enum FenceInner {
    /// CPU-only fence for testing without GPU.
    Dummy { signaled: Arc<AtomicBool> },
    /// GPU-backed fence for real async rendering.
    /// The fence is boxed to reduce enum size (GpuFence is large due to backend variants).
    Gpu {
        fence: Box<GpuFence>,
        instance: Arc<GraphicsInstance>,
    },
}

impl std::fmt::Debug for FenceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy { signaled } => f
                .debug_struct("Dummy")
                .field("signaled", &signaled.load(Ordering::Relaxed))
                .finish(),
            Self::Gpu { fence, .. } => f.debug_struct("Gpu").field("fence", fence).finish(),
        }
    }
}

/// CPU-GPU synchronization primitive.
///
/// Fences allow the CPU to wait for GPU work to complete.
/// Used to synchronize frame boundaries and ensure resources
/// are safe to reuse.
///
/// # Async Behavior
///
/// When backed by a real GPU fence, `wait()` blocks until the GPU
/// signals completion. This enables true async rendering where the
/// CPU can continue building subsequent frames while the GPU works.
///
/// # Example
///
/// ```ignore
/// let fence = schedule.take_fence();
///
/// // Do other CPU work while GPU executes...
///
/// // Before reusing frame resources, wait for GPU:
/// fence.wait();
/// assert_eq!(fence.status(), FenceStatus::Signaled);
/// ```
pub struct Fence {
    inner: FenceInner,
}

impl std::fmt::Debug for Fence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fence").field("inner", &self.inner).finish()
    }
}

impl Fence {
    /// Create a new CPU-only fence in the unsignaled state (for testing).
    pub(crate) fn new_unsignaled() -> Self {
        Self {
            inner: FenceInner::Dummy {
                signaled: Arc::new(AtomicBool::new(false)),
            },
        }
    }

    /// Create a new CPU-only fence in the signaled state.
    ///
    /// Used as the frame fence when a submit failed and no GPU work is in
    /// flight (the slot is trivially safe to recycle), and in tests.
    pub(crate) fn new_signaled() -> Self {
        Self {
            inner: FenceInner::Dummy {
                signaled: Arc::new(AtomicBool::new(true)),
            },
        }
    }

    /// Create a new GPU-backed fence.
    ///
    /// The fence is created in the signaled state initially (ready for first use).
    /// When passed to `execute_graph`, the GPU will signal it upon completion.
    pub(crate) fn new_gpu(
        instance: Arc<GraphicsInstance>,
    ) -> Result<Self, crate::error::GraphicsError> {
        let fence = Box::new(instance.backend().create_fence(true)?); // Start signaled
        Ok(Self {
            inner: FenceInner::Gpu { fence, instance },
        })
    }

    /// Get the underlying GpuFence (if GPU-backed).
    ///
    /// Returns `None` for CPU-only fences.
    pub(crate) fn gpu_fence(&self) -> Option<&GpuFence> {
        match &self.inner {
            FenceInner::Dummy { .. } => None,
            FenceInner::Gpu { fence, .. } => Some(fence),
        }
    }

    /// Check the current status of the fence.
    pub fn status(&self) -> FenceStatus {
        if self.is_signaled() {
            FenceStatus::Signaled
        } else {
            FenceStatus::Unsignaled
        }
    }

    /// Check if the fence is signaled (non-blocking).
    pub fn is_signaled(&self) -> bool {
        match &self.inner {
            FenceInner::Dummy { signaled } => signaled.load(Ordering::Acquire),
            FenceInner::Gpu { fence, instance } => instance.backend().is_fence_signaled(fence),
        }
    }

    /// Wait for the fence to be signaled (blocking, bounded).
    ///
    /// This blocks the calling thread until the GPU signals the fence.
    /// Returns immediately if already signaled. GPU-backed waits are bounded
    /// by a backend-internal timeout (10 s); timeout and device loss are
    /// returned as errors — in that case the GPU may still be using the
    /// resources guarded by this fence, so they must not be recycled.
    pub fn wait(&self) -> Result<(), crate::error::GraphicsError> {
        match &self.inner {
            FenceInner::Dummy { signaled } => {
                while !signaled.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                Ok(())
            }
            FenceInner::Gpu { fence, instance } => instance.backend().wait_fence(fence),
        }
    }

    /// Wait for the fence with a timeout.
    ///
    /// Returns `Ok(true)` if the fence was signaled, `Ok(false)` if the
    /// timeout elapsed, and an error on device loss or wait failure.
    pub fn wait_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> Result<bool, crate::error::GraphicsError> {
        match &self.inner {
            FenceInner::Dummy { signaled } => {
                let start = web_time::Instant::now();
                while !signaled.load(Ordering::Acquire) {
                    if start.elapsed() >= timeout {
                        return Ok(false);
                    }
                    std::hint::spin_loop();
                }
                Ok(true)
            }
            FenceInner::Gpu { fence, instance } => {
                // Use proper backend wait with timeout instead of polling
                instance.backend().wait_fence_timeout(fence, timeout)
            }
        }
    }

    /// Reset the fence to unsignaled state.
    ///
    /// Must only be called when no GPU work is pending on this fence.
    pub fn reset(&self) {
        match &self.inner {
            FenceInner::Dummy { signaled } => {
                signaled.store(false, Ordering::Release);
            }
            FenceInner::Gpu { .. } => {
                // GPU fences are reset automatically when passed to execute_graph
                // No manual reset needed - the backend handles this
            }
        }
    }

    /// Signal the fence (for testing mode).
    ///
    /// For GPU-backed fences, the GPU signals automatically when work completes.
    #[cfg(test)]
    pub(crate) fn signal(&self) {
        match &self.inner {
            FenceInner::Dummy { signaled } => {
                signaled.store(true, Ordering::Release);
            }
            FenceInner::Gpu { .. } => {
                // GPU signals the fence automatically - this is a no-op
            }
        }
    }
}

impl Clone for Fence {
    fn clone(&self) -> Self {
        match &self.inner {
            FenceInner::Dummy { signaled } => Self {
                inner: FenceInner::Dummy {
                    signaled: Arc::clone(signaled),
                },
            },
            FenceInner::Gpu { .. } => {
                // GPU fences cannot be cloned because they represent unique GPU state.
                // Cloning would create a fence with incorrect signaled status, leading
                // to synchronization bugs (waiting on a fence that's already signaled
                // or never signals). Panic to catch this programming error early.
                panic!(
                    "GPU fences cannot be cloned. Each GPU fence represents unique GPU state. \
                     If you need to share fence state, use Arc<Fence> instead."
                );
            }
        }
    }
}

impl Default for Fence {
    fn default() -> Self {
        Self::new_unsignaled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fence_unsignaled() {
        let fence = Fence::new_unsignaled();
        assert_eq!(fence.status(), FenceStatus::Unsignaled);
        assert!(!fence.is_signaled());
    }

    #[test]
    fn test_fence_signaled() {
        let fence = Fence::new_signaled();
        assert_eq!(fence.status(), FenceStatus::Signaled);
        assert!(fence.is_signaled());
    }

    #[test]
    fn test_fence_signal_and_wait() {
        let fence = Fence::new_unsignaled();

        // Simulate GPU signaling from another thread
        let fence_clone = fence.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            fence_clone.signal();
        });

        fence.wait().unwrap();
        assert!(fence.is_signaled());
    }

    #[test]
    fn test_fence_wait_timeout() {
        let fence = Fence::new_unsignaled();

        // Should timeout since nothing signals it
        let result = fence.wait_timeout(std::time::Duration::from_millis(10));
        assert!(!result.unwrap());
        assert!(!fence.is_signaled());
    }

    #[test]
    fn test_fence_reset() {
        let fence = Fence::new_signaled();
        assert!(fence.is_signaled());

        fence.reset();
        assert!(!fence.is_signaled());
    }

    #[test]
    fn test_fence_clone_shares_state() {
        let fence1 = Fence::new_unsignaled();
        let fence2 = fence1.clone();

        assert!(!fence1.is_signaled());
        assert!(!fence2.is_signaled());

        fence1.signal();

        assert!(fence1.is_signaled());
        assert!(fence2.is_signaled());
    }
}
