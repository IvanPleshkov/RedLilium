//! Profiling support via Tracy.
//!
//! This module provides optional profiling instrumentation using the [Tracy profiler](https://github.com/wolfpld/tracy).
//! Profiling is enabled via the `profiling` Cargo feature.
//!
//! # Enabling Profiling
//!
//! Add the `profiling` feature to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! redlilium-core = { version = "0.1", features = ["profiling"] }
//! ```
//!
//! Or enable it when running:
//!
//! ```bash
//! cargo run --features profiling
//! ```
//!
//! # Connecting Tracy
//!
//! 1. Download Tracy from <https://github.com/wolfpld/tracy/releases>
//! 2. Run your application with profiling enabled
//! 3. Connect Tracy to your running application
//!
//! # CPU Profiling
//!
//! Use the provided macros to instrument your code:
//!
//! ```ignore
//! use redlilium_core::profiling::{profile_scope, profile_function};
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
//! # Frame Marking
//!
//! Mark frame boundaries for frame-time analysis:
//!
//! ```ignore
//! use redlilium_core::profiling::frame_mark;
//!
//! loop {
//!     // ... render frame ...
//!     frame_mark!();  // Signal end of frame
//! }
//! ```
//!
//! # Memory Profiling
//!
//! Track memory allocations and plot memory usage:
//!
//! ```ignore
//! use redlilium_core::profiling::{profile_alloc, profile_free, profile_memory_stats};
//!
//! // Track individual allocations
//! let ptr = allocate_memory(size);
//! profile_alloc!(ptr, size);
//!
//! // When freeing
//! profile_free!(ptr);
//! free_memory(ptr);
//!
//! // Plot current memory usage (call periodically)
//! profile_memory_stats!();
//! ```
//!
//! # Performance
//!
//! When profiling is disabled (the default), all macros compile to no-ops with
//! zero runtime overhead.

// Re-export tracy-client types when profiling is enabled
#[cfg(feature = "profiling")]
pub use tracy_client::{
    self, Client, PlotName, ProfiledAllocator, Span, frame_mark as tracy_frame_mark,
    plot as tracy_plot, span,
};

/// Create a profiled global allocator that tracks all memory allocations in Tracy.
///
/// This macro creates a `#[global_allocator]` static that wraps the system allocator
/// with Tracy's profiling capabilities. Memory allocations and deallocations will
/// appear in Tracy's memory view with callstack information.
///
/// # Arguments
///
/// * `$name` - The name of the static allocator variable
/// * `$callstack_depth` - Number of callstack frames to capture (0 = no callstack, higher = more detail but slower)
///
/// # Example
///
/// ```ignore
/// use redlilium_core::profiling::create_profiled_allocator;
///
/// // Create a profiled allocator with 32 frames of callstack
/// create_profiled_allocator!(GLOBAL_ALLOCATOR, 32);
/// ```
///
/// # Performance Note
///
/// Non-zero callstack depth adds overhead to every allocation. Use 0 for minimal
/// overhead, or 16-32 for detailed allocation tracking during debugging.
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! create_profiled_allocator {
    ($name:ident, $callstack_depth:expr) => {
        #[global_allocator]
        static $name: $crate::profiling::ProfiledAllocator<std::alloc::System> =
            $crate::profiling::ProfiledAllocator::new(std::alloc::System, $callstack_depth);
    };
}

/// Create a profiled allocator (no-op when profiling disabled - uses system allocator).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! create_profiled_allocator {
    ($name:ident, $callstack_depth:expr) => {
        // When profiling is disabled, don't override the default allocator
    };
}

/// Mark the end of a frame for Tracy's frame analysis.
///
/// This should be called once per frame, typically at the end of your render loop.
/// Tracy will use these markers to calculate frame times and display frame boundaries.
///
/// # Example
///
/// ```ignore
/// loop {
///     // Process input
///     // Update game state
///     // Render frame
///     frame_mark!();
/// }
/// ```
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! frame_mark {
    () => {
        $crate::profiling::tracy_frame_mark()
    };
}

/// Mark the end of a frame (no-op when profiling disabled).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! frame_mark {
    () => {};
}

/// Create a profiling span for the current scope.
///
/// The span automatically ends when the scope exits.
///
/// # Example
///
/// ```ignore
/// fn process_data() {
///     {
///         profile_scope!("parse_json");
///         // JSON parsing code...
///     }
///
///     {
///         profile_scope!("validate_data");
///         // Validation code...
///     }
/// }
/// ```
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! profile_scope {
    ($name:expr) => {
        let _profile_span = $crate::profiling::span!($name);
    };
}

/// Create a profiling span (no-op when profiling disabled).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! profile_scope {
    ($name:expr) => {};
}

/// Create a profiling span for the entire function.
///
/// Place this at the start of a function to profile its entire execution.
///
/// # Example
///
/// ```ignore
/// fn expensive_computation() {
///     profile_function!();
///     // Function body...
/// }
/// ```
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! profile_function {
    () => {
        // Use function!() for automatic function name, or construct from module path
        let _profile_span = $crate::profiling::span!();
    };
}

/// Create a profiling span for function (no-op when profiling disabled).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! profile_function {
    () => {};
}

/// Plot a value over time in Tracy.
///
/// This is useful for tracking metrics like frame time, memory usage, etc.
///
/// # Example
///
/// ```ignore
/// let frame_time_ms = frame_time.as_secs_f64() * 1000.0;
/// profile_plot!("frame_time_ms", frame_time_ms);
/// ```
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! profile_plot {
    ($name:expr, $value:expr) => {
        $crate::profiling::tracy_plot!($name, $value as f64)
    };
}

/// Plot a value (no-op when profiling disabled).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! profile_plot {
    ($name:expr, $value:expr) => {
        let _ = $value; // Avoid unused warnings
    };
}

/// Set the name of the current thread for Tracy.
///
/// This helps identify threads in the profiler.
///
/// # Example
///
/// ```ignore
/// std::thread::spawn(|| {
///     set_thread_name!("Worker Thread");
///     // Thread work...
/// });
/// ```
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! set_thread_name {
    ($name:expr) => {
        $crate::profiling::tracy_client::set_thread_name!($name)
    };
}

/// Set thread name (no-op when profiling disabled).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! set_thread_name {
    ($name:expr) => {};
}

/// Send a message to Tracy's message log.
///
/// Messages appear in Tracy's "Messages" view and can be used for important
/// events, state changes, or debugging information.
///
/// # Example
///
/// ```ignore
/// profile_message!("Loading scene: forest.scene");
/// profile_message!("Shader compilation complete");
/// ```
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! profile_message {
    ($msg:expr) => {
        if let Some(client) = $crate::profiling::Client::running() {
            client.message($msg, 0);
        }
    };
    ($msg:expr, $callstack_depth:expr) => {
        if let Some(client) = $crate::profiling::Client::running() {
            client.message($msg, $callstack_depth);
        }
    };
}

/// Send a message (no-op when profiling disabled).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! profile_message {
    ($msg:expr) => {};
    ($msg:expr, $callstack_depth:expr) => {};
}

/// Report current memory usage statistics to Tracy as plots.
///
/// This reports the current heap memory usage using the system's allocation info.
/// Call this periodically (e.g., once per frame) to track memory over time.
///
/// # Example
///
/// ```ignore
/// // In your main loop
/// loop {
///     // ... frame work ...
///     profile_memory_stats!();
///     frame_mark!();
/// }
/// ```
#[macro_export]
#[cfg(all(feature = "profiling", target_os = "windows"))]
macro_rules! profile_memory_stats {
    () => {{
        // On Windows, use GetProcessMemoryInfo for accurate heap stats
        use std::mem::MaybeUninit;
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        #[link(name = "psapi")]
        unsafe extern "system" {
            fn GetProcessMemoryInfo(
                process: *mut std::ffi::c_void,
                counters: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
        }
        let mut counters = MaybeUninit::<ProcessMemoryCounters>::uninit();
        unsafe {
            let process = GetCurrentProcess();
            if GetProcessMemoryInfo(
                process,
                counters.as_mut_ptr(),
                std::mem::size_of::<ProcessMemoryCounters>() as u32,
            ) != 0
            {
                let counters = counters.assume_init();
                let working_set_mb = counters.working_set_size as f64 / (1024.0 * 1024.0);
                let private_mb = counters.pagefile_usage as f64 / (1024.0 * 1024.0);
                $crate::profiling::tracy_plot!("Memory: Working Set (MB)", working_set_mb);
                $crate::profiling::tracy_plot!("Memory: Private (MB)", private_mb);
            }
        }
    }};
}

/// Report memory stats (Linux/macOS implementation).
#[macro_export]
#[cfg(all(feature = "profiling", not(target_os = "windows")))]
macro_rules! profile_memory_stats {
    () => {{
        // On Unix-like systems, read from /proc/self/statm or use mallinfo
        // For simplicity, we'll use a rough estimate from the allocator stats
        // A more accurate implementation could read /proc/self/status
        if let Some(client) = $crate::profiling::Client::running() {
            // Plot a placeholder - users can implement custom memory tracking
            $crate::profiling::tracy_plot!("Memory: Tracked", 0.0f64);
        }
    }};
}

/// Report memory stats (no-op when profiling disabled).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! profile_memory_stats {
    () => {};
}

/// Mark a named frame for multi-threaded frame analysis.
///
/// Use this for secondary frame markers (e.g., physics thread, audio thread).
///
/// # Example
///
/// ```ignore
/// // In physics thread
/// loop {
///     // ... physics work ...
///     profile_frame_mark_named!("Physics");
/// }
/// ```
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! profile_frame_mark_named {
    ($name:expr) => {
        $crate::profiling::tracy_client::secondary_frame_mark!($name)
    };
}

/// Named frame mark (no-op when profiling disabled).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! profile_frame_mark_named {
    ($name:expr) => {};
}

/// Create a profiling span with a runtime-determined name.
///
/// Unlike [`profile_scope!`] which requires a string literal, this macro
/// accepts any `&str` expression. It uses `tracy_client::Client::span_alloc`
/// which heap-allocates the span name. Prefer [`profile_scope!`] for static
/// names.
///
/// # Example
///
/// ```ignore
/// let system_name = "MovementSystem";
/// profile_scope_dynamic!(system_name);
/// // ... profiled work ...
/// ```
#[macro_export]
#[cfg(feature = "profiling")]
macro_rules! profile_scope_dynamic {
    ($name:expr) => {
        let _profile_span = $crate::profiling::Client::running()
            .map(|c| c.span_alloc(Some($name), "", file!(), line!(), 0));
    };
}

/// Create a profiling span with a dynamic name (no-op when profiling disabled).
#[macro_export]
#[cfg(not(feature = "profiling"))]
macro_rules! profile_scope_dynamic {
    ($name:expr) => {
        let _ = $name;
    };
}

// Re-export macros at module level
pub use create_profiled_allocator;
pub use frame_mark;
pub use profile_frame_mark_named;
pub use profile_function;
pub use profile_memory_stats;
pub use profile_message;
pub use profile_plot;
pub use profile_scope;
pub use profile_scope_dynamic;
pub use set_thread_name;

#[cfg(test)]
mod tests {
    #[test]
    fn test_macros_compile() {
        // These should compile regardless of profiling feature
        frame_mark!();
        profile_scope!("test_scope");
        profile_scope_dynamic!("dynamic_scope");
        profile_function!();
        profile_plot!("test_value", 42.0);
        set_thread_name!("test_thread");
        profile_message!("test message");
        profile_memory_stats!();
        profile_frame_mark_named!("test_frame");
    }
}
