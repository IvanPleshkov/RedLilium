//! Vulkan synchronization primitives (fences, semaphores).
//!
//! This module provides utilities for GPU synchronization.
//! Most synchronization is handled directly in mod.rs, but this module
//! can be extended for more complex synchronization needs.

// Currently, fence and semaphore creation is handled inline in mod.rs.
// This module is reserved for future synchronization utilities such as:
// - Timeline semaphores
// - Event synchronization
// - Memory barriers helpers
