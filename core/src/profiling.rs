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
//! # Performance
//!
//! When profiling is disabled (the default), all macros compile to no-ops with
//! zero runtime overhead.

// Re-export tracy-client types when profiling is enabled
#[cfg(feature = "profiling")]
pub use tracy_client::{
    self, Client, PlotName, Span, frame_mark as tracy_frame_mark, plot as tracy_plot, span,
};

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

// Re-export macros at module level
pub use frame_mark;
pub use profile_function;
pub use profile_plot;
pub use profile_scope;
pub use set_thread_name;

#[cfg(test)]
mod tests {
    #[test]
    fn test_macros_compile() {
        // These should compile regardless of profiling feature
        frame_mark!();
        profile_scope!("test_scope");
        profile_function!();
        profile_plot!("test_value", 42.0);
        set_thread_name!("test_thread");
    }
}
