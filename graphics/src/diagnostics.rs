//! Backend diagnostics for applications and integration tests.
//!
//! These helpers report validation output without exposing native GPU objects.

/// Vulkan validation-layer diagnostics (requires `vulkan-backend`).
///
/// Enable validation through [`crate::InstanceParameters`]. Counters are local
/// to the calling thread; zero does not imply validation layers are installed.
#[cfg(feature = "vulkan-backend")]
pub mod vulkan {
    pub use crate::backend::vulkan::{reset_validation_error_count, validation_error_count};
}
