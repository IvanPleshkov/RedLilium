//! Application handler trait.

use std::path::PathBuf;

use crate::context::{AppContext, DrawContext};
use redlilium_graphics::FrameSchedule;
use winit::event::{KeyEvent, MouseButton};

/// Trait for handling application events and draw requests.
///
/// Implement this trait to create custom application logic.
///
/// # Lifecycle
///
/// 1. `on_init` - Called once when the application starts
/// 2. `on_resize` - Called when the window is resized
/// 3. `on_update` - Called every frame before drawing
/// 4. `on_draw` - Called every frame to render
/// 5. `on_shutdown` - Called when the application is closing
///
/// # Example
///
/// ```ignore
/// use redlilium_app::{AppHandler, AppContext, DrawContext};
///
/// struct MyApp {
///     frame_count: u64,
/// }
///
/// impl AppHandler for MyApp {
///     fn on_init(&mut self, ctx: &mut AppContext) {
///         log::info!("Application initialized");
///     }
///
///     fn on_draw(&mut self, ctx: DrawContext) -> FrameSchedule {
///         self.frame_count += 1;
///         // Render your frame here
///         ctx.finish(&[])
///     }
/// }
/// ```
pub trait AppHandler {
    /// Called once when the application initializes.
    ///
    /// Use this to create GPU resources, load assets, etc.
    fn on_init(&mut self, _ctx: &mut AppContext) {}

    /// Called when the window is resized.
    ///
    /// The new size is available in `ctx.width()` and `ctx.height()`.
    fn on_resize(&mut self, _ctx: &mut AppContext) {}

    /// Called every frame before drawing.
    ///
    /// Use this for game logic, input processing, etc.
    /// Returns `true` to continue running, `false` to exit.
    fn on_update(&mut self, _ctx: &mut AppContext) -> bool {
        true
    }

    /// Called every frame to render.
    ///
    /// This is where you submit render graphs and draw commands.
    /// The returned `FrameSchedule` is used by the App to complete the frame.
    /// You must call `ctx.finish()` at the end of your rendering and return
    /// the resulting schedule.
    fn on_draw(&mut self, ctx: DrawContext) -> FrameSchedule;

    /// Called when a key is pressed or released.
    fn on_key(&mut self, _ctx: &mut AppContext, _event: &KeyEvent) {}

    /// Called when the mouse is moved.
    fn on_mouse_move(&mut self, _ctx: &mut AppContext, _x: f64, _y: f64) {}

    /// Called when a mouse button is pressed or released.
    fn on_mouse_button(&mut self, _ctx: &mut AppContext, _button: MouseButton, _pressed: bool) {}

    /// Called when the mouse wheel is scrolled.
    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, _delta_x: f32, _delta_y: f32) {}

    /// Called when a file is dropped onto the window.
    fn on_file_dropped(&mut self, _ctx: &mut AppContext, _path: PathBuf) {}

    /// Called when a file is being dragged over the window.
    fn on_file_hovered(&mut self, _ctx: &mut AppContext, _path: PathBuf) {}

    /// Called when a file drag leaves the window without dropping.
    fn on_file_hover_cancelled(&mut self, _ctx: &mut AppContext) {}

    /// Called when the user requests to close the window (e.g. clicking the
    /// close button or pressing Alt+F4).
    ///
    /// Return `true` to allow the window to close immediately (the default).
    /// Return `false` to cancel the close — for example to show an "unsaved
    /// changes" dialog first. The handler can later close the application by
    /// returning `false` from [`on_update`](Self::on_update).
    fn on_close_requested(&mut self, _ctx: &mut AppContext) -> bool {
        true
    }

    /// Called when the application is closing.
    ///
    /// Use this to clean up resources.
    fn on_shutdown(&mut self, _ctx: &mut AppContext) {}
}
