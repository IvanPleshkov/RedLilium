//! Deferred destruction system for Vulkan resources.
//!
//! GPU commands are executed asynchronously - when you submit work to the GPU,
//! the CPU continues while the GPU processes commands 1-3 frames behind.
//! This means we can't destroy resources immediately when their Rust references
//! are dropped, as the GPU may still be using them.
//!
//! This module provides a deferred destruction queue that holds resources until
//! the GPU has finished using them (tracked via frame fences).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     DeferredDestructor                          │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │                   Frame-indexed queues                     │  │
//! │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │  │
//! │  │  │ Frame 0  │  │ Frame 1  │  │ Frame 2  │  ...            │  │
//! │  │  │ pending  │  │ pending  │  │ pending  │                 │  │
//! │  │  └──────────┘  └──────────┘  └──────────┘                 │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! On Drop(Resource):
//!   1. Don't call vkDestroy*
//!   2. Send resource handle to DeferredDestructor
//!   3. DeferredDestructor stores it in current frame's pending list
//!
//! On frame boundary (after fence wait):
//!   1. Destroy all resources in oldest frame's pending list
//!   2. Rotate frame index
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};

use ash::vk;
use gpu_allocator::vulkan::Allocation;
use parking_lot::Mutex;

/// Maximum number of frames in flight for deferred destruction.
/// Resources are held for this many frames before being destroyed.
pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// A Vulkan resource pending destruction.
///
/// This enum holds the raw Vulkan handles and associated allocations
/// that need to be destroyed after the GPU finishes using them.
pub enum DeferredResource {
    /// A buffer with its associated memory allocation.
    Buffer {
        device: ash::Device,
        buffer: vk::Buffer,
        allocation: Option<Allocation>,
    },
    /// A texture (image + view) with its associated memory allocation.
    Texture {
        device: ash::Device,
        image: vk::Image,
        view: vk::ImageView,
        allocation: Option<Allocation>,
    },
    /// A sampler resource.
    Sampler {
        device: ash::Device,
        sampler: vk::Sampler,
    },
    /// A fence resource.
    Fence {
        device: ash::Device,
        fence: vk::Fence,
    },
    /// A semaphore resource.
    Semaphore {
        device: ash::Device,
        semaphore: vk::Semaphore,
    },
    /// Command buffers to be freed.
    CommandBuffers {
        device: ash::Device,
        command_pool: vk::CommandPool,
        buffers: Vec<vk::CommandBuffer>,
    },
}

// SAFETY: DeferredResource only contains Vulkan handles which are thread-safe.
// The ash::Device is also thread-safe (it's just a wrapper around raw pointers).
unsafe impl Send for DeferredResource {}
unsafe impl Sync for DeferredResource {}

impl DeferredResource {
    /// Destroy the resource immediately.
    ///
    /// # Safety
    ///
    /// The caller must ensure the GPU is no longer using this resource.
    pub unsafe fn destroy(self, allocator: &Mutex<gpu_allocator::vulkan::Allocator>) {
        match self {
            DeferredResource::Buffer {
                device,
                buffer,
                allocation,
            } => {
                // Free the allocation first if present
                if let Some(alloc) = allocation
                    && let Err(e) = allocator.lock().free(alloc)
                {
                    log::error!("Failed to free buffer allocation: {}", e);
                }
                unsafe { device.destroy_buffer(buffer, None) };
            }
            DeferredResource::Texture {
                device,
                image,
                view,
                allocation,
            } => {
                // Free the allocation first if present
                if let Some(alloc) = allocation
                    && let Err(e) = allocator.lock().free(alloc)
                {
                    log::error!("Failed to free texture allocation: {}", e);
                }
                unsafe {
                    device.destroy_image_view(view, None);
                    device.destroy_image(image, None);
                }
            }
            DeferredResource::Sampler { device, sampler } => {
                unsafe { device.destroy_sampler(sampler, None) };
            }
            DeferredResource::Fence { device, fence } => {
                unsafe { device.destroy_fence(fence, None) };
            }
            DeferredResource::Semaphore { device, semaphore } => {
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            DeferredResource::CommandBuffers {
                device,
                command_pool,
                buffers,
            } => {
                unsafe { device.free_command_buffers(command_pool, &buffers) };
            }
        }
    }

    /// Destroy the resource immediately without freeing allocations.
    /// Used when we don't have access to the allocator (e.g., for fences/semaphores).
    ///
    /// # Safety
    ///
    /// The caller must ensure the GPU is no longer using this resource.
    pub unsafe fn destroy_without_allocator(self) {
        match self {
            DeferredResource::Buffer {
                device,
                buffer,
                allocation,
            } => {
                // Drop allocation without freeing - will be cleaned up when allocator is dropped
                drop(allocation);
                unsafe { device.destroy_buffer(buffer, None) };
            }
            DeferredResource::Texture {
                device,
                image,
                view,
                allocation,
            } => {
                // Drop allocation without freeing
                drop(allocation);
                unsafe {
                    device.destroy_image_view(view, None);
                    device.destroy_image(image, None);
                }
            }
            DeferredResource::Sampler { device, sampler } => {
                unsafe { device.destroy_sampler(sampler, None) };
            }
            DeferredResource::Fence { device, fence } => {
                unsafe { device.destroy_fence(fence, None) };
            }
            DeferredResource::Semaphore { device, semaphore } => {
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            DeferredResource::CommandBuffers {
                device,
                command_pool,
                buffers,
            } => {
                unsafe { device.free_command_buffers(command_pool, &buffers) };
            }
        }
    }
}

/// Manages deferred destruction of Vulkan resources.
///
/// Resources are queued for destruction and only actually destroyed
/// after enough frames have passed to ensure the GPU is done with them.
pub struct DeferredDestructor {
    /// Per-frame queues of resources pending destruction.
    /// Index is frame_index % MAX_FRAMES_IN_FLIGHT.
    frame_queues: [Mutex<Vec<DeferredResource>>; MAX_FRAMES_IN_FLIGHT],

    /// Current frame index (monotonically increasing).
    current_frame: AtomicUsize,

    /// Reference to the allocator for freeing memory.
    allocator: Mutex<Option<std::sync::Weak<Mutex<gpu_allocator::vulkan::Allocator>>>>,
}

impl std::fmt::Debug for DeferredDestructor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeferredDestructor")
            .field("current_frame", &self.current_frame.load(Ordering::Relaxed))
            .field(
                "pending_count",
                &self
                    .frame_queues
                    .iter()
                    .map(|q| q.lock().len())
                    .sum::<usize>(),
            )
            .finish()
    }
}

impl DeferredDestructor {
    /// Create a new deferred destructor.
    pub fn new() -> Self {
        Self {
            frame_queues: Default::default(),
            current_frame: AtomicUsize::new(0),
            allocator: Mutex::new(None),
        }
    }

    /// Set the allocator reference.
    /// This must be called after the allocator is created but before any resources are queued.
    pub fn set_allocator(
        &self,
        allocator: std::sync::Weak<Mutex<gpu_allocator::vulkan::Allocator>>,
    ) {
        *self.allocator.lock() = Some(allocator);
    }

    /// Queue a resource for deferred destruction.
    ///
    /// The resource will be destroyed after `MAX_FRAMES_IN_FLIGHT` frames
    /// have completed, ensuring the GPU is done with it.
    pub fn queue(&self, resource: DeferredResource) {
        let frame = self.current_frame.load(Ordering::Relaxed);
        let queue_index = frame % MAX_FRAMES_IN_FLIGHT;
        self.frame_queues[queue_index].lock().push(resource);
    }

    /// Advance to the next frame, destroying resources from the oldest frame.
    ///
    /// This should be called at frame boundaries, after the fence for the
    /// oldest in-flight frame has been signaled (meaning the GPU is done
    /// with all resources used in that frame).
    ///
    /// # Safety
    ///
    /// The caller must ensure that the GPU has finished executing all commands
    /// from `MAX_FRAMES_IN_FLIGHT` frames ago. This is typically done by waiting
    /// on a frame fence.
    pub unsafe fn advance_frame(&self) {
        let current = self.current_frame.fetch_add(1, Ordering::SeqCst);

        // After enough frames have passed, we can safely destroy resources
        // from the oldest frame's queue.
        if current >= MAX_FRAMES_IN_FLIGHT {
            let oldest_queue_index = (current + 1) % MAX_FRAMES_IN_FLIGHT;
            let resources: Vec<_> = self.frame_queues[oldest_queue_index]
                .lock()
                .drain(..)
                .collect();

            if !resources.is_empty() {
                // Try to get the allocator
                let allocator_guard = self.allocator.lock();
                if let Some(weak_alloc) = allocator_guard.as_ref() {
                    if let Some(allocator) = weak_alloc.upgrade() {
                        for resource in resources {
                            // SAFETY: Caller guarantees GPU is done with these resources
                            unsafe { resource.destroy(&allocator) };
                        }
                    } else {
                        // Allocator was dropped, destroy without freeing allocations
                        for resource in resources {
                            // SAFETY: Caller guarantees GPU is done with these resources
                            unsafe { resource.destroy_without_allocator() };
                        }
                    }
                } else {
                    // No allocator set, destroy without freeing allocations
                    for resource in resources {
                        // SAFETY: Caller guarantees GPU is done with these resources
                        unsafe { resource.destroy_without_allocator() };
                    }
                }
            }
        }
    }

    /// Flush all pending resources immediately.
    ///
    /// This destroys all queued resources regardless of frame timing.
    /// Should only be called during shutdown when the device is idle.
    ///
    /// # Safety
    ///
    /// The caller must ensure the GPU is completely idle (e.g., after
    /// calling vkDeviceWaitIdle).
    pub unsafe fn flush_all(&self) {
        let allocator_guard = self.allocator.lock();
        let allocator = allocator_guard.as_ref().and_then(|w| w.upgrade());

        let mut total = 0;
        for queue in &self.frame_queues {
            let resources: Vec<_> = queue.lock().drain(..).collect();
            total += resources.len();

            if let Some(ref alloc) = allocator {
                for resource in resources {
                    // SAFETY: Caller guarantees GPU is idle
                    unsafe { resource.destroy(alloc) };
                }
            } else {
                for resource in resources {
                    // SAFETY: Caller guarantees GPU is idle
                    unsafe { resource.destroy_without_allocator() };
                }
            }
        }

        let _ = total; // suppress unused warning
    }

    /// Get the number of resources currently pending destruction.
    pub fn pending_count(&self) -> usize {
        self.frame_queues.iter().map(|q| q.lock().len()).sum()
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> usize {
        self.current_frame.load(Ordering::Relaxed)
    }
}

impl Default for DeferredDestructor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deferred_destructor_frame_cycling() {
        let destructor = DeferredDestructor::new();

        // Initial state
        assert_eq!(destructor.current_frame(), 0);
        assert_eq!(destructor.pending_count(), 0);

        // Advance frames
        for i in 0..MAX_FRAMES_IN_FLIGHT * 2 {
            unsafe { destructor.advance_frame() };
            assert_eq!(destructor.current_frame(), i + 1);
        }
    }
}
