//! GPU synchronization primitives.
//!
//! This module provides synchronization types for coordinating work
//! between the CPU and GPU, and between different GPU operations.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::backend::GpuFence;
use crate::instance::GraphicsInstance;

/// GPU semaphore for synchronizing operations within a frame.
///
/// Semaphores are used for GPU-GPU synchronization:
/// - One operation signals the semaphore when complete
/// - Another operation waits on the semaphore before starting
///
/// Unlike fences, semaphores cannot be waited on from the CPU.
#[derive(Debug)]
pub struct Semaphore {
    /// Unique identifier for debugging.
    id: u64,
    // TODO: Actual GPU semaphore handle would go here
    // e.g., vk::Semaphore for Vulkan, ID3D12Fence for D3D12
}

impl Semaphore {
    /// Create a new semaphore with the given ID.
    pub(crate) fn new(id: u64) -> Self {
        Self { id }
    }

    /// Get the semaphore's unique ID (for debugging).
    pub fn id(&self) -> u64 {
        self.id
    }
}

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

    /// Create a new CPU-only fence in the signaled state (for testing).
    #[allow(dead_code)]
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
    pub(crate) fn new_gpu(instance: Arc<GraphicsInstance>) -> Self {
        let fence = Box::new(instance.backend().create_fence(true)); // Start signaled
        Self {
            inner: FenceInner::Gpu { fence, instance },
        }
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

    /// Wait for the fence to be signaled (blocking).
    ///
    /// This blocks the calling thread until the GPU signals the fence.
    /// Returns immediately if already signaled.
    pub fn wait(&self) {
        match &self.inner {
            FenceInner::Dummy { signaled } => {
                while !signaled.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
            }
            FenceInner::Gpu { fence, instance } => {
                instance.backend().wait_fence(fence);
            }
        }
    }

    /// Wait for the fence with a timeout.
    ///
    /// Returns `true` if the fence was signaled, `false` if timeout elapsed.
    pub fn wait_timeout(&self, timeout: std::time::Duration) -> bool {
        match &self.inner {
            FenceInner::Dummy { signaled } => {
                let start = std::time::Instant::now();
                while !signaled.load(Ordering::Acquire) {
                    if start.elapsed() >= timeout {
                        return false;
                    }
                    std::hint::spin_loop();
                }
                true
            }
            FenceInner::Gpu { fence, instance } => {
                // For GPU fences, we poll with short sleeps until timeout
                let start = std::time::Instant::now();
                while !instance.backend().is_fence_signaled(fence) {
                    if start.elapsed() >= timeout {
                        return false;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                true
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

    /// Signal the fence (for CPU-only/testing mode).
    ///
    /// For GPU-backed fences, the GPU signals automatically when work completes.
    pub(crate) fn signal(&self) {
        match &self.inner {
            FenceInner::Dummy { signaled } => {
                signaled.store(true, Ordering::Release);
            }
            FenceInner::Gpu { .. } => {
                // GPU signals the fence automatically - this is a no-op
                log::trace!("signal() called on GPU fence - GPU will signal automatically");
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
                // GPU fences cannot be cloned - create a new dummy one
                // This maintains API compatibility but logs a warning
                log::warn!("Cloning GPU fence creates a dummy fence - use with caution");
                Self::new_signaled()
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
    fn test_semaphore_id() {
        let sem = Semaphore::new(42);
        assert_eq!(sem.id(), 42);
    }

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

        fence.wait();
        assert!(fence.is_signaled());
    }

    #[test]
    fn test_fence_wait_timeout() {
        let fence = Fence::new_unsignaled();

        // Should timeout since nothing signals it
        let result = fence.wait_timeout(std::time::Duration::from_millis(10));
        assert!(!result);
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
