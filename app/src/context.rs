//! Application and draw contexts.

use std::sync::Arc;

use redlilium_graphics::{
    FramePipeline, FrameSchedule, GraphicsDevice, GraphicsInstance, RenderGraph, Surface,
    SurfaceTexture,
};

/// Application context providing access to graphics resources.
///
/// This context is available during all application callbacks and provides
/// access to the graphics device, window dimensions, and frame timing.
pub struct AppContext {
    /// The graphics instance.
    pub(crate) instance: Arc<GraphicsInstance>,
    /// The graphics device.
    pub(crate) device: Arc<GraphicsDevice>,
    /// The surface for presenting to the window.
    pub(crate) surface: Arc<Surface>,
    /// The frame pipeline for managing frames in flight.
    pub(crate) pipeline: FramePipeline,
    /// Current window width in physical pixels.
    pub(crate) width: u32,
    /// Current window height in physical pixels.
    pub(crate) height: u32,
    /// Current scale factor (DPI scaling).
    pub(crate) scale_factor: f64,
    /// Current frame number.
    pub(crate) frame_number: u64,
    /// Delta time since last frame in seconds.
    pub(crate) delta_time: f32,
    /// Time since application start in seconds.
    pub(crate) elapsed_time: f32,
}

impl AppContext {
    /// Get the graphics instance.
    pub fn instance(&self) -> &Arc<GraphicsInstance> {
        &self.instance
    }

    /// Get the graphics device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Get the surface.
    pub fn surface(&self) -> &Arc<Surface> {
        &self.surface
    }

    /// Get the frame pipeline.
    pub fn pipeline(&self) -> &FramePipeline {
        &self.pipeline
    }

    /// Get mutable access to the frame pipeline.
    pub fn pipeline_mut(&mut self) -> &mut FramePipeline {
        &mut self.pipeline
    }

    /// Get the current window width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the current window height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the window aspect ratio.
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }

    /// Get the current scale factor (DPI scaling).
    ///
    /// This is the ratio between physical pixels and logical pixels.
    /// A scale factor of 2.0 means the display is a HiDPI/Retina display.
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Get the current frame number.
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Get the delta time since last frame in seconds.
    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }

    /// Get the elapsed time since application start in seconds.
    pub fn elapsed_time(&self) -> f32 {
        self.elapsed_time
    }
}

/// Draw context for rendering a frame.
///
/// This context is provided during `on_draw` callbacks and includes
/// the frame schedule for submitting render graphs and the current
/// swapchain texture.
pub struct DrawContext<'a> {
    /// The application context.
    pub(crate) app: &'a mut AppContext,
    /// The frame schedule for submitting render graphs.
    pub(crate) schedule: FrameSchedule,
    /// The current swapchain texture.
    pub(crate) swapchain_texture: SurfaceTexture,
}

impl<'a> DrawContext<'a> {
    /// Get the graphics instance.
    pub fn instance(&self) -> &Arc<GraphicsInstance> {
        &self.app.instance
    }

    /// Get the graphics device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.app.device
    }

    /// Get the current window width.
    pub fn width(&self) -> u32 {
        self.app.width
    }

    /// Get the current window height.
    pub fn height(&self) -> u32 {
        self.app.height
    }

    /// Get the window aspect ratio.
    pub fn aspect_ratio(&self) -> f32 {
        self.app.aspect_ratio()
    }

    /// Get the current scale factor (DPI scaling).
    pub fn scale_factor(&self) -> f64 {
        self.app.scale_factor
    }

    /// Get the current frame number.
    pub fn frame_number(&self) -> u64 {
        self.app.frame_number
    }

    /// Get the delta time since last frame in seconds.
    pub fn delta_time(&self) -> f32 {
        self.app.delta_time
    }

    /// Get the elapsed time since application start in seconds.
    pub fn elapsed_time(&self) -> f32 {
        self.app.elapsed_time
    }

    /// Get the current swapchain texture.
    pub fn swapchain_texture(&self) -> &SurfaceTexture {
        &self.swapchain_texture
    }

    /// Submit a render graph to the frame schedule.
    ///
    /// Returns a handle that can be used as a dependency for other graphs.
    pub fn submit(
        &mut self,
        name: impl Into<String>,
        graph: &RenderGraph,
        wait_for: &[redlilium_graphics::GraphHandle],
    ) -> redlilium_graphics::GraphHandle {
        self.schedule.submit(name, graph, wait_for)
    }

    /// Finish the frame with the given dependencies.
    ///
    /// This should be called at the end of `on_draw` to signal that
    /// all render graphs have been submitted. Returns the FrameSchedule
    /// which must be returned from `on_draw` for proper frame management.
    pub fn finish(mut self, wait_for: &[redlilium_graphics::GraphHandle]) -> FrameSchedule {
        self.schedule.finish(wait_for);
        self.swapchain_texture.present();
        self.schedule
    }
}
