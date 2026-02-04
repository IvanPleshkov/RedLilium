//! Profiling support via Tracy.
//!
//! This module re-exports CPU profiling from [`redlilium_core::profiling`] and adds
//! GPU-specific profiling support for graphics backends.
//!
//! # Enabling Profiling
//!
//! Add the `profiling` feature to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! redlilium-graphics = { version = "0.1", features = ["profiling"] }
//! ```
//!
//! Or enable it when running:
//!
//! ```bash
//! cargo run --features profiling
//! ```
//!
//! # CPU Profiling
//!
//! See [`redlilium_core::profiling`] for CPU profiling macros:
//!
//! ```ignore
//! use redlilium_graphics::profiling::{profile_scope, profile_function, frame_mark};
//!
//! fn expensive_operation() {
//!     profile_function!();  // Profiles entire function
//!
//!     {
//!         profile_scope!("inner_work");  // Profiles this scope
//!         // ... do work ...
//!     }
//! }
//! ```
//!
//! # GPU Profiling (Advanced)
//!
//! Tracy supports GPU profiling via timestamp queries. This module provides
//! [`GpuProfileContext`] for GPU timeline profiling:
//!
//! ```ignore
//! use redlilium_graphics::profiling::GpuProfileContext;
//!
//! // Create during device initialization
//! let gpu_ctx = GpuProfileContext::new_vulkan("Main Queue", initial_timestamp, timestamp_period);
//!
//! // Record zones around GPU work (requires manual timestamp query management)
//! ```
//!
//! GPU profiling requires backend-specific integration with timestamp queries.
//! See the Tracy documentation for details on timestamp synchronization.

// Re-export everything from core profiling module
pub use redlilium_core::profiling::*;

// Re-export GPU-specific tracy-client types when profiling is enabled
#[cfg(feature = "profiling")]
pub use tracy_client::{GpuContext, GpuContextType, GpuSpan};

/// GPU profiling context wrapper.
///
/// This provides a higher-level interface for Tracy's GPU profiling capabilities.
/// GPU profiling requires creating timestamp queries and uploading results.
///
/// # Vulkan Integration
///
/// For Vulkan, you need to:
/// 1. Create a timestamp query pool
/// 2. Insert timestamp queries around GPU work
/// 3. Read back timestamps after work completes
/// 4. Upload timestamps to Tracy
///
/// # Example (Conceptual)
///
/// ```ignore
/// // Create GPU context during device initialization
/// let gpu_ctx = GpuProfileContext::new_vulkan("Main GPU Queue", 0, 1.0);
///
/// // In command buffer recording:
/// let span = gpu_ctx.begin_zone("render_pass");
/// // ... record render pass commands ...
/// // span is automatically ended when dropped
///
/// // After GPU work completes, upload timestamps
/// gpu_ctx.upload_timestamp(query_id, timestamp);
/// ```
#[cfg(feature = "profiling")]
pub struct GpuProfileContext {
    context: GpuContext,
    next_query_id: u16,
}

#[cfg(feature = "profiling")]
impl GpuProfileContext {
    /// Create a new GPU profiling context for Vulkan.
    ///
    /// # Arguments
    ///
    /// * `name` - Name shown in Tracy (e.g., "Vulkan Graphics Queue")
    /// * `gpu_timestamp` - Initial GPU timestamp value
    /// * `timestamp_period_ns` - GPU timestamp period in nanoseconds (from device properties)
    pub fn new_vulkan(name: &str, gpu_timestamp: i64, timestamp_period_ns: f32) -> Option<Self> {
        let client = Client::running()?;
        let context = client
            .new_gpu_context(
                Some(name),
                GpuContextType::Vulkan,
                gpu_timestamp,
                timestamp_period_ns,
            )
            .ok()?;

        Some(Self {
            context,
            next_query_id: 0,
        })
    }

    /// Create a new GPU profiling context for other GPU APIs.
    ///
    /// # Arguments
    ///
    /// * `name` - Name shown in Tracy
    /// * `context_type` - Type of GPU context (OpenGL, Direct3D, etc.)
    /// * `gpu_timestamp` - Initial GPU timestamp value
    /// * `timestamp_period_ns` - GPU timestamp period in nanoseconds
    pub fn new(
        name: &str,
        context_type: GpuContextType,
        gpu_timestamp: i64,
        timestamp_period_ns: f32,
    ) -> Option<Self> {
        let client = Client::running()?;
        let context = client
            .new_gpu_context(Some(name), context_type, gpu_timestamp, timestamp_period_ns)
            .ok()?;

        Some(Self {
            context,
            next_query_id: 0,
        })
    }

    /// Allocate a query ID for a new GPU zone.
    ///
    /// Returns the query ID to use for timestamp queries.
    pub fn alloc_query_id(&mut self) -> u16 {
        let id = self.next_query_id;
        self.next_query_id = self.next_query_id.wrapping_add(1);
        id
    }

    /// Begin a GPU zone and allocate a span.
    ///
    /// Call this before writing the start timestamp query.
    /// Returns a span that will be ended when dropped.
    pub fn begin_zone(&mut self, name: &str) -> Option<GpuSpan> {
        self.context.span_alloc(name, "", "", 0).ok()
    }

    /// Begin a manual GPU span with a specific query ID.
    ///
    /// Use this for more control over timestamp query management.
    /// Call `end_span` when the GPU work is complete.
    pub fn begin_manual_zone(&mut self, name: &str, query_id: u16) {
        let _ = self.context.begin_span_alloc(name, "", "", 0, query_id);
    }

    /// End a manual GPU span.
    pub fn end_manual_zone(&mut self, query_id: u16) {
        let _ = self.context.end_span(query_id);
    }

    /// Upload a GPU timestamp for a specific query ID.
    ///
    /// Call this after reading back timestamp query results.
    ///
    /// # Arguments
    ///
    /// * `query_id` - The query ID for the timestamp
    /// * `timestamp` - GPU timestamp value
    pub fn upload_timestamp(&self, query_id: u16, timestamp: i64) {
        self.context.upload_gpu_timestamp(query_id, timestamp);
    }

    /// Synchronize GPU time with Tracy.
    ///
    /// Call this periodically (e.g., once per frame) to keep Tracy's
    /// GPU timeline synchronized with the actual GPU timestamps.
    pub fn sync_time(&self, gpu_timestamp: i64) {
        self.context.sync_gpu_time(gpu_timestamp);
    }

    /// Get the underlying Tracy GPU context.
    pub fn inner(&self) -> &GpuContext {
        &self.context
    }
}

/// Dummy GPU profile context when profiling is disabled.
#[cfg(not(feature = "profiling"))]
pub struct GpuProfileContext;

#[cfg(not(feature = "profiling"))]
impl GpuProfileContext {
    pub fn new_vulkan(_name: &str, _gpu_timestamp: i64, _timestamp_period_ns: f32) -> Option<Self> {
        None
    }

    pub fn new(
        _name: &str,
        _context_type: (),
        _gpu_timestamp: i64,
        _timestamp_period_ns: f32,
    ) -> Option<Self> {
        None
    }

    pub fn alloc_query_id(&mut self) -> u16 {
        0
    }

    pub fn begin_zone(&mut self, _name: &str) -> Option<()> {
        None
    }

    pub fn begin_manual_zone(&mut self, _name: &str, _query_id: u16) {}

    pub fn end_manual_zone(&mut self, _query_id: u16) {}

    pub fn upload_timestamp(&self, _query_id: u16, _timestamp: i64) {}

    pub fn sync_time(&self, _gpu_timestamp: i64) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macros_compile() {
        // These should compile regardless of profiling feature
        frame_mark!();
        profile_scope!("test_scope");
        profile_function!();
        profile_plot!("test_value", 42.0);
        set_thread_name!("test_thread");
    }

    #[test]
    fn test_gpu_context_creation() {
        // Should return None when profiling is disabled or Tracy not running
        let ctx = GpuProfileContext::new_vulkan("test", 0, 1.0);
        // Context may or may not be created depending on feature flag and Tracy state
        let _ = ctx;
    }
}
