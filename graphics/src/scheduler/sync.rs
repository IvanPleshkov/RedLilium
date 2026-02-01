//! GPU synchronization primitives.
//!
//! This module provides synchronization types for coordinating work
//! between the CPU and GPU, and between different GPU operations.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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

/// CPU-GPU synchronization primitive.
///
/// Fences allow the CPU to wait for GPU work to complete.
/// Used to synchronize frame boundaries and ensure resources
/// are safe to reuse.
///
/// # Example
///
/// ```ignore
/// let fence = schedule.submit_and_present(graph, &[deps], swapchain);
///
/// // Later, before reusing frame resources:
/// fence.wait();
/// assert_eq!(fence.status(), FenceStatus::Signaled);
/// ```
#[derive(Debug)]
pub struct Fence {
    /// Whether the fence has been signaled.
    signaled: Arc<AtomicBool>,
    // TODO: Actual GPU fence handle would go here
    // e.g., vk::Fence for Vulkan, ID3D12Fence for D3D12
}

impl Fence {
    /// Create a new fence in the unsignaled state.
    pub(crate) fn new_unsignaled() -> Self {
        Self {
            signaled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a new fence in the signaled state.
    #[allow(dead_code)]
    pub(crate) fn new_signaled() -> Self {
        Self {
            signaled: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Check the current status of the fence.
    pub fn status(&self) -> FenceStatus {
        if self.signaled.load(Ordering::Acquire) {
            FenceStatus::Signaled
        } else {
            FenceStatus::Unsignaled
        }
    }

    /// Check if the fence is signaled (non-blocking).
    pub fn is_signaled(&self) -> bool {
        self.status() == FenceStatus::Signaled
    }

    /// Wait for the fence to be signaled (blocking).
    ///
    /// This blocks the calling thread until the GPU signals the fence.
    /// Returns immediately if already signaled.
    pub fn wait(&self) {
        // TODO: Actual GPU fence wait would happen here
        // For now, just spin (in real impl this would call vkWaitForFences etc.)
        while !self.signaled.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }

    /// Wait for the fence with a timeout.
    ///
    /// Returns `true` if the fence was signaled, `false` if timeout elapsed.
    pub fn wait_timeout(&self, timeout: std::time::Duration) -> bool {
        let start = std::time::Instant::now();
        while !self.signaled.load(Ordering::Acquire) {
            if start.elapsed() >= timeout {
                return false;
            }
            std::hint::spin_loop();
        }
        true
    }

    /// Reset the fence to unsignaled state.
    ///
    /// Must only be called when no GPU work is pending on this fence.
    pub fn reset(&self) {
        self.signaled.store(false, Ordering::Release);
    }

    /// Signal the fence.
    ///
    /// In real GPU backends, the GPU signals the fence when work completes.
    /// For the dummy backend (no actual GPU), call this immediately after
    /// "submitting" work to simulate completion.
    pub(crate) fn signal(&self) {
        self.signaled.store(true, Ordering::Release);
    }
}

impl Clone for Fence {
    fn clone(&self) -> Self {
        Self {
            signaled: Arc::clone(&self.signaled),
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
