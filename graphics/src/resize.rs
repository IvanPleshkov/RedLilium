//! Window resize management with debouncing and rendering strategies.
//!
//! This module provides [`ResizeManager`] for handling window resize events
//! smoothly, avoiding the performance issues that come from recreating the
//! swapchain on every resize event.
//!
//! # The Problem
//!
//! During window drag-resize, the OS sends many resize events rapidly:
//!
//! ```text
//! Time:   0ms   16ms  32ms  48ms  ...  500ms (user stops dragging)
//! Events:  R     R     R     R    ...    R
//! Sizes:  800   820   850   900  ...   1200
//! ```
//!
//! Recreating the swapchain on every event causes:
//! - 30+ swapchain recreations during a 500ms drag
//! - GPU stalls on each recreation
//! - Extremely choppy resize experience
//!
//! # Solution: Deferred Resize
//!
//! Buffer resize events and only apply when the user stops resizing:
//!
//! ```text
//! Events:  R  R  R  R  R  R  ... R [50ms quiet]
//!          └──────────────────────┘      │
//!            (events buffered)           ▼
//!                              Single swapchain resize
//! ```
//!
//! # Rendering During Resize
//!
//! While resize is pending, the manager provides a [`ResizeStrategy`] for
//! what to render:
//!
//! - [`Stretch`](ResizeStrategy::Stretch) - Render at old size, OS stretches
//! - [`IntermediateTarget`](ResizeStrategy::IntermediateTarget) - Render to fixed-size texture
//! - [`DynamicResolution`](ResizeStrategy::DynamicResolution) - Render at reduced resolution
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::resize::{ResizeManager, ResizeStrategy};
//!
//! let mut resize_manager = ResizeManager::new(
//!     (1920, 1080),
//!     50,  // 50ms debounce
//!     ResizeStrategy::DynamicResolution { scale_during_resize: 0.5 },
//! );
//!
//! // In event loop:
//! match event {
//!     WindowEvent::Resized(size) => {
//!         resize_manager.on_resize_event(size.width, size.height);
//!     }
//!     _ => {}
//! }
//!
//! // Each frame:
//! if let Some((width, height)) = resize_manager.update() {
//!     pipeline.wait_current_slot();
//!     surface.resize(width, height);
//! }
//!
//! let render_size = resize_manager.render_size();
//! // ... render at render_size ...
//! ```

use std::time::{Duration, Instant};

/// Strategy for rendering during active resize.
///
/// When the user is actively resizing the window, the swapchain may be
/// a different size than the window. This enum determines how to handle
/// that mismatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeStrategy {
    /// Render at current swapchain size, let OS compositor stretch.
    ///
    /// This is the simplest approach. The rendered image may appear
    /// slightly blurry or distorted during resize, but returns to
    /// full quality once resize completes.
    ///
    /// # When to Use
    ///
    /// - Simplest implementation
    /// - Acceptable for most applications
    /// - When you don't have upscaling infrastructure
    Stretch,

    /// Render to a fixed-size intermediate target, then blit to swapchain.
    ///
    /// The scene is always rendered at the specified resolution, then
    /// copied/scaled to whatever the current swapchain size is. This
    /// provides consistent quality during resize.
    ///
    /// # Fields
    ///
    /// * `width` - Fixed render target width
    /// * `height` - Fixed render target height
    ///
    /// # When to Use
    ///
    /// - When you want consistent render quality
    /// - For applications with a "native" resolution
    /// - When upscaling is acceptable
    IntermediateTarget {
        /// Fixed render width.
        width: u32,
        /// Fixed render height.
        height: u32,
    },

    /// Render at reduced resolution during resize, full resolution after.
    ///
    /// During active resize, the scene is rendered at a fraction of the
    /// target resolution, then upscaled. Once resize completes, rendering
    /// returns to full resolution.
    ///
    /// # Fields
    ///
    /// * `scale_during_resize` - Scale factor during resize (0.0 to 1.0).
    ///   For example, 0.5 means render at 50% resolution.
    ///
    /// # When to Use
    ///
    /// - Best user experience (smooth resize + full quality)
    /// - When you have upscaling (FSR, DLSS, or bilinear)
    /// - For performance-sensitive applications
    DynamicResolution {
        /// Scale factor during resize (0.0 to 1.0).
        scale_during_resize: f32,
    },
}

impl Default for ResizeStrategy {
    /// Default strategy is [`Stretch`](ResizeStrategy::Stretch).
    fn default() -> Self {
        Self::Stretch
    }
}

/// Resize event information returned by [`ResizeManager::update`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeEvent {
    /// New width in pixels.
    pub width: u32,
    /// New height in pixels.
    pub height: u32,
    /// Previous width in pixels.
    pub previous_width: u32,
    /// Previous height in pixels.
    pub previous_height: u32,
}

/// Manages window resize events with debouncing.
///
/// `ResizeManager` buffers resize events and only signals that the swapchain
/// should be recreated after a configurable quiet period. This prevents
/// excessive swapchain recreation during drag-resize operations.
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────┐
/// │                     ResizeManager                               │
/// ├─────────────────────────────────────────────────────────────────┤
/// │  on_resize_event()  ──►  [Pending Size + Timestamp]             │
/// │                                    │                            │
/// │                          [Debounce Timer]                       │
/// │                                    │                            │
/// │  update()  ◄───────────  (if quiet period elapsed)              │
/// │     │                                                           │
/// │     └──►  Returns ResizeEvent to apply                          │
/// │                                                                 │
/// │  render_size()  ──►  Returns size based on ResizeStrategy       │
/// └─────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Example
///
/// ```
/// use redlilium_graphics::resize::{ResizeManager, ResizeStrategy};
/// use std::time::Duration;
///
/// let mut manager = ResizeManager::new(
///     (1920, 1080),
///     50,
///     ResizeStrategy::Stretch,
/// );
///
/// // Simulate resize events
/// manager.on_resize_event(1024, 768);
/// assert!(manager.is_resizing());
///
/// // Before debounce period, no resize event
/// assert!(manager.update().is_none());
/// ```
#[derive(Debug)]
pub struct ResizeManager {
    /// Pending resize size (buffered, not yet applied).
    pending_size: Option<(u32, u32)>,

    /// Time of last resize event.
    last_event_time: Instant,

    /// How long to wait after last event before applying resize.
    debounce_duration: Duration,

    /// Current confirmed swapchain size.
    current_size: (u32, u32),

    /// Rendering strategy during resize.
    strategy: ResizeStrategy,

    /// Whether we're in active resize mode.
    is_actively_resizing: bool,

    /// Minimum size (to prevent zero-size swapchains).
    min_size: (u32, u32),
}

impl ResizeManager {
    /// Create a new resize manager.
    ///
    /// # Arguments
    ///
    /// * `initial_size` - Initial window/swapchain size.
    /// * `debounce_ms` - Milliseconds to wait after last resize event
    ///   before applying the resize. Recommended values:
    ///   - Desktop app: 50-100ms
    ///   - Game: 100-150ms
    ///   - Editor/Tool: 30-50ms
    /// * `strategy` - How to handle rendering during resize.
    ///
    /// # Example
    ///
    /// ```
    /// use redlilium_graphics::resize::{ResizeManager, ResizeStrategy};
    ///
    /// let manager = ResizeManager::new(
    ///     (1920, 1080),
    ///     50,
    ///     ResizeStrategy::DynamicResolution { scale_during_resize: 0.5 },
    /// );
    /// ```
    pub fn new(initial_size: (u32, u32), debounce_ms: u64, strategy: ResizeStrategy) -> Self {
        Self {
            pending_size: None,
            last_event_time: Instant::now(),
            debounce_duration: Duration::from_millis(debounce_ms),
            current_size: initial_size,
            strategy,
            is_actively_resizing: false,
            min_size: (1, 1),
        }
    }

    /// Set the minimum allowed size.
    ///
    /// Resize events smaller than this will be clamped. This prevents
    /// zero-size swapchains which are invalid on most graphics APIs.
    ///
    /// Default is (1, 1).
    pub fn set_min_size(&mut self, min_width: u32, min_height: u32) {
        self.min_size = (min_width.max(1), min_height.max(1));
    }

    /// Set the debounce duration.
    pub fn set_debounce(&mut self, debounce_ms: u64) {
        self.debounce_duration = Duration::from_millis(debounce_ms);
    }

    /// Set the resize strategy.
    pub fn set_strategy(&mut self, strategy: ResizeStrategy) {
        self.strategy = strategy;
    }

    /// Get the current resize strategy.
    pub fn strategy(&self) -> ResizeStrategy {
        self.strategy
    }

    /// Handle an OS window resize event.
    ///
    /// Call this whenever you receive a resize event from the windowing system.
    /// The resize will be buffered and applied after the debounce period.
    ///
    /// # Arguments
    ///
    /// * `width` - New window width in pixels.
    /// * `height` - New window height in pixels.
    ///
    /// # Example
    ///
    /// ```ignore
    /// match event {
    ///     WindowEvent::Resized(size) => {
    ///         resize_manager.on_resize_event(size.width, size.height);
    ///     }
    ///     _ => {}
    /// }
    /// ```
    pub fn on_resize_event(&mut self, width: u32, height: u32) {
        // Clamp to minimum size
        let width = width.max(self.min_size.0);
        let height = height.max(self.min_size.1);
        let new_size = (width, height);

        // Ignore if same as pending or current
        if Some(new_size) == self.pending_size || new_size == self.current_size {
            return;
        }

        self.pending_size = Some(new_size);
        self.last_event_time = Instant::now();
        self.is_actively_resizing = true;

        log::trace!(
            "Resize event: {}x{} (pending, debounce={}ms)",
            width,
            height,
            self.debounce_duration.as_millis()
        );
    }

    /// Check if resize should be applied.
    ///
    /// Call this every frame. Returns `Some(ResizeEvent)` if the debounce
    /// period has elapsed and the swapchain should be recreated.
    ///
    /// # Returns
    ///
    /// - `Some(ResizeEvent)` - Apply this resize to the swapchain
    /// - `None` - No resize needed this frame
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(event) = resize_manager.update() {
    ///     pipeline.wait_current_slot();
    ///     surface.resize(event.width, event.height);
    /// }
    /// ```
    pub fn update(&mut self) -> Option<ResizeEvent> {
        if let Some((width, height)) = self.pending_size
            && self.last_event_time.elapsed() >= self.debounce_duration
        {
            let previous = self.current_size;
            self.pending_size = None;
            self.current_size = (width, height);
            self.is_actively_resizing = false;

            log::trace!(
                "Resize applied: {}x{} -> {}x{}",
                previous.0,
                previous.1,
                width,
                height
            );

            return Some(ResizeEvent {
                width,
                height,
                previous_width: previous.0,
                previous_height: previous.1,
            });
        }
        None
    }

    /// Force immediate resize without waiting for debounce.
    ///
    /// Use this when you need to resize immediately, such as when
    /// entering/exiting fullscreen.
    ///
    /// # Returns
    ///
    /// - `Some(ResizeEvent)` if there was a pending resize
    /// - `None` if no resize was pending
    pub fn force_resize(&mut self) -> Option<ResizeEvent> {
        if let Some((width, height)) = self.pending_size.take() {
            let previous = self.current_size;
            self.current_size = (width, height);
            self.is_actively_resizing = false;

            log::trace!(
                "Resize forced: {}x{} -> {}x{}",
                previous.0,
                previous.1,
                width,
                height
            );

            return Some(ResizeEvent {
                width,
                height,
                previous_width: previous.0,
                previous_height: previous.1,
            });
        }
        None
    }

    /// Get the size to render at.
    ///
    /// This may differ from the current swapchain size during active resize,
    /// depending on the configured [`ResizeStrategy`].
    ///
    /// # Returns
    ///
    /// The (width, height) that the scene should be rendered at.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (width, height) = resize_manager.render_size();
    /// let render_target = create_render_target(width, height);
    /// ```
    pub fn render_size(&self) -> (u32, u32) {
        match self.strategy {
            ResizeStrategy::Stretch => {
                // Render at current swapchain size, OS stretches
                self.current_size
            }
            ResizeStrategy::IntermediateTarget { width, height } => {
                // Always render at fixed size
                (width, height)
            }
            ResizeStrategy::DynamicResolution {
                scale_during_resize,
            } => {
                if self.is_actively_resizing {
                    // Use pending size as target, scaled down
                    let target = self.pending_size.unwrap_or(self.current_size);
                    let scaled_width = ((target.0 as f32) * scale_during_resize).max(1.0) as u32;
                    let scaled_height = ((target.1 as f32) * scale_during_resize).max(1.0) as u32;
                    (
                        scaled_width.max(self.min_size.0),
                        scaled_height.max(self.min_size.1),
                    )
                } else {
                    self.current_size
                }
            }
        }
    }

    /// Get the current confirmed swapchain size.
    ///
    /// This is the size that was last applied to the swapchain.
    pub fn swapchain_size(&self) -> (u32, u32) {
        self.current_size
    }

    /// Get the pending size, if any.
    ///
    /// Returns the size that will be applied after the debounce period.
    pub fn pending_size(&self) -> Option<(u32, u32)> {
        self.pending_size
    }

    /// Check if resize is actively in progress.
    ///
    /// Returns `true` if there are pending resize events that haven't
    /// been applied yet. Useful for:
    /// - Showing a resize indicator in the UI
    /// - Reducing render quality during resize
    /// - Skipping expensive effects during resize
    pub fn is_resizing(&self) -> bool {
        self.is_actively_resizing
    }

    /// Get the time since the last resize event.
    ///
    /// Useful for debugging or custom resize logic.
    pub fn time_since_last_event(&self) -> Duration {
        self.last_event_time.elapsed()
    }

    /// Get the remaining debounce time.
    ///
    /// Returns how much time is left before the pending resize will be applied.
    /// Returns `Duration::ZERO` if no resize is pending.
    pub fn remaining_debounce(&self) -> Duration {
        if self.pending_size.is_some() {
            let elapsed = self.last_event_time.elapsed();
            self.debounce_duration.saturating_sub(elapsed)
        } else {
            Duration::ZERO
        }
    }

    /// Cancel any pending resize.
    ///
    /// Use this if you need to abort a resize operation.
    pub fn cancel_pending(&mut self) {
        self.pending_size = None;
        self.is_actively_resizing = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_new() {
        let manager = ResizeManager::new((1920, 1080), 50, ResizeStrategy::Stretch);
        assert_eq!(manager.swapchain_size(), (1920, 1080));
        assert_eq!(manager.render_size(), (1920, 1080));
        assert!(!manager.is_resizing());
        assert!(manager.pending_size().is_none());
    }

    #[test]
    fn test_resize_event_sets_pending() {
        let mut manager = ResizeManager::new((1920, 1080), 50, ResizeStrategy::Stretch);

        manager.on_resize_event(1024, 768);

        assert!(manager.is_resizing());
        assert_eq!(manager.pending_size(), Some((1024, 768)));
        // Current size unchanged until debounce
        assert_eq!(manager.swapchain_size(), (1920, 1080));
    }

    #[test]
    fn test_duplicate_event_ignored() {
        let mut manager = ResizeManager::new((1920, 1080), 50, ResizeStrategy::Stretch);

        manager.on_resize_event(1024, 768);
        let first_time = manager.last_event_time;

        // Same size should be ignored
        thread::sleep(Duration::from_millis(5));
        manager.on_resize_event(1024, 768);
        assert_eq!(manager.last_event_time, first_time);

        // Current size should also be ignored
        manager.on_resize_event(1920, 1080);
        assert_eq!(manager.pending_size(), Some((1024, 768)));
    }

    #[test]
    fn test_update_before_debounce() {
        let mut manager = ResizeManager::new((1920, 1080), 100, ResizeStrategy::Stretch);

        manager.on_resize_event(1024, 768);

        // Immediately after event, update should return None
        assert!(manager.update().is_none());
        assert!(manager.is_resizing());
    }

    #[test]
    fn test_update_after_debounce() {
        let mut manager = ResizeManager::new((1920, 1080), 10, ResizeStrategy::Stretch);

        manager.on_resize_event(1024, 768);

        // Wait for debounce
        thread::sleep(Duration::from_millis(15));

        let event = manager.update();
        assert!(event.is_some());

        let event = event.unwrap();
        assert_eq!(event.width, 1024);
        assert_eq!(event.height, 768);
        assert_eq!(event.previous_width, 1920);
        assert_eq!(event.previous_height, 1080);

        // After update, state should be updated
        assert!(!manager.is_resizing());
        assert_eq!(manager.swapchain_size(), (1024, 768));
        assert!(manager.pending_size().is_none());
    }

    #[test]
    fn test_force_resize() {
        let mut manager = ResizeManager::new((1920, 1080), 1000, ResizeStrategy::Stretch);

        manager.on_resize_event(1024, 768);

        // Force resize without waiting
        let event = manager.force_resize();
        assert!(event.is_some());
        assert_eq!(manager.swapchain_size(), (1024, 768));
        assert!(!manager.is_resizing());
    }

    #[test]
    fn test_force_resize_no_pending() {
        let mut manager = ResizeManager::new((1920, 1080), 50, ResizeStrategy::Stretch);

        assert!(manager.force_resize().is_none());
    }

    #[test]
    fn test_min_size_clamping() {
        let mut manager = ResizeManager::new((1920, 1080), 10, ResizeStrategy::Stretch);
        manager.set_min_size(100, 100);

        manager.on_resize_event(50, 50);

        thread::sleep(Duration::from_millis(15));
        let event = manager.update().unwrap();

        assert_eq!(event.width, 100);
        assert_eq!(event.height, 100);
    }

    #[test]
    fn test_zero_size_clamped_to_one() {
        let mut manager = ResizeManager::new((1920, 1080), 10, ResizeStrategy::Stretch);

        manager.on_resize_event(0, 0);

        assert_eq!(manager.pending_size(), Some((1, 1)));
    }

    #[test]
    fn test_strategy_stretch() {
        let mut manager = ResizeManager::new((1920, 1080), 100, ResizeStrategy::Stretch);

        manager.on_resize_event(1024, 768);

        // Render size is current swapchain size (not pending)
        assert_eq!(manager.render_size(), (1920, 1080));
    }

    #[test]
    fn test_strategy_intermediate_target() {
        let mut manager = ResizeManager::new(
            (1920, 1080),
            100,
            ResizeStrategy::IntermediateTarget {
                width: 1280,
                height: 720,
            },
        );

        manager.on_resize_event(1024, 768);

        // Always renders at intermediate size
        assert_eq!(manager.render_size(), (1280, 720));
    }

    #[test]
    fn test_strategy_dynamic_resolution_during_resize() {
        let mut manager = ResizeManager::new(
            (1920, 1080),
            100,
            ResizeStrategy::DynamicResolution {
                scale_during_resize: 0.5,
            },
        );

        manager.on_resize_event(1000, 800);

        // During resize, renders at scaled pending size
        let render_size = manager.render_size();
        assert_eq!(render_size, (500, 400));
    }

    #[test]
    fn test_strategy_dynamic_resolution_after_resize() {
        let mut manager = ResizeManager::new(
            (1920, 1080),
            10,
            ResizeStrategy::DynamicResolution {
                scale_during_resize: 0.5,
            },
        );

        manager.on_resize_event(1000, 800);
        thread::sleep(Duration::from_millis(15));
        manager.update();

        // After resize, renders at full size
        assert_eq!(manager.render_size(), (1000, 800));
    }

    #[test]
    fn test_cancel_pending() {
        let mut manager = ResizeManager::new((1920, 1080), 100, ResizeStrategy::Stretch);

        manager.on_resize_event(1024, 768);
        assert!(manager.is_resizing());

        manager.cancel_pending();
        assert!(!manager.is_resizing());
        assert!(manager.pending_size().is_none());
        assert_eq!(manager.swapchain_size(), (1920, 1080));
    }

    #[test]
    fn test_multiple_events_uses_latest() {
        let mut manager = ResizeManager::new((1920, 1080), 10, ResizeStrategy::Stretch);

        manager.on_resize_event(1024, 768);
        manager.on_resize_event(800, 600);
        manager.on_resize_event(640, 480);

        thread::sleep(Duration::from_millis(15));
        let event = manager.update().unwrap();

        // Should use the latest size
        assert_eq!(event.width, 640);
        assert_eq!(event.height, 480);
    }

    #[test]
    fn test_remaining_debounce() {
        let mut manager = ResizeManager::new((1920, 1080), 100, ResizeStrategy::Stretch);

        // No pending resize
        assert_eq!(manager.remaining_debounce(), Duration::ZERO);

        manager.on_resize_event(1024, 768);

        // Should have some remaining time
        let remaining = manager.remaining_debounce();
        assert!(remaining > Duration::ZERO);
        assert!(remaining <= Duration::from_millis(100));
    }

    #[test]
    fn test_set_debounce() {
        let mut manager = ResizeManager::new((1920, 1080), 50, ResizeStrategy::Stretch);

        manager.set_debounce(200);
        manager.on_resize_event(1024, 768);

        thread::sleep(Duration::from_millis(100));
        assert!(manager.update().is_none()); // Still waiting

        thread::sleep(Duration::from_millis(150));
        assert!(manager.update().is_some()); // Now ready
    }

    #[test]
    fn test_set_strategy() {
        let mut manager = ResizeManager::new((1920, 1080), 50, ResizeStrategy::Stretch);

        manager.set_strategy(ResizeStrategy::IntermediateTarget {
            width: 1280,
            height: 720,
        });

        assert_eq!(
            manager.strategy(),
            ResizeStrategy::IntermediateTarget {
                width: 1280,
                height: 720
            }
        );
    }

    #[test]
    fn test_default_strategy() {
        assert_eq!(ResizeStrategy::default(), ResizeStrategy::Stretch);
    }
}
