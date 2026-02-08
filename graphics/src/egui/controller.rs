//! Egui controller that manages the UI lifecycle.
//!
//! The controller handles input processing, UI updates, and rendering.

use std::sync::Arc;

use egui::{Context, TextureId};
use winit::event::{KeyEvent, MouseButton, MouseScrollDelta};
use winit::keyboard::ModifiersState;

use super::ArcEguiApp;
use super::input::EguiInputState;
use super::renderer::EguiRenderer;
use crate::GraphicsDevice;
use crate::graph::{GraphicsPass, RenderTarget};
use crate::resources::Texture;

/// Controller for egui UI integration.
///
/// Manages the egui context, input handling, and rendering.
///
/// # Example
///
/// ```ignore
/// use std::sync::{Arc, RwLock};
/// use redlilium_graphics::egui::{EguiApp, EguiController};
///
/// struct MyUi { counter: i32 }
///
/// impl EguiApp for MyUi {
///     fn update(&mut self, ctx: &egui::Context) {
///         egui::Window::new("Counter").show(ctx, |ui| {
///             if ui.button("Increment").clicked() {
///                 self.counter += 1;
///             }
///             ui.label(format!("Count: {}", self.counter));
///         });
///     }
/// }
///
/// let ui = Arc::new(RwLock::new(MyUi { counter: 0 }));
/// let controller = EguiController::new(device, ui);
/// ```
pub struct EguiController {
    ctx: Context,
    app: ArcEguiApp,
    input_state: EguiInputState,
    renderer: EguiRenderer,
    setup_done: bool,
    /// Whether egui wants keyboard input this frame.
    pub wants_keyboard_input: bool,
    /// Whether egui wants pointer input this frame.
    pub wants_pointer_input: bool,
}

impl EguiController {
    /// Create a new egui controller.
    ///
    /// # Arguments
    ///
    /// * `device` - The graphics device for creating GPU resources
    /// * `app` - The egui application implementing the UI logic
    /// * `width` - Initial screen width in physical pixels
    /// * `height` - Initial screen height in physical pixels
    /// * `scale_factor` - The DPI scale factor (pixels per point)
    pub fn new(
        device: Arc<GraphicsDevice>,
        app: ArcEguiApp,
        width: u32,
        height: u32,
        scale_factor: f64,
        surface_format: crate::types::TextureFormat,
    ) -> Self {
        let ctx = Context::default();
        let input_state = EguiInputState::new(width, height, scale_factor as f32);
        let renderer = EguiRenderer::new(device, surface_format);

        Self {
            ctx,
            app,
            input_state,
            renderer,
            setup_done: false,
            wants_keyboard_input: false,
            wants_pointer_input: false,
        }
    }

    /// Get the egui context.
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Handle window resize.
    pub fn on_resize(&mut self, width: u32, height: u32) {
        self.input_state.set_screen_size(width, height);
        self.renderer.update_screen_size(width, height);
    }

    /// Handle scale factor (DPI) change.
    pub fn on_scale_factor_changed(&mut self, scale_factor: f64) {
        self.input_state.set_pixels_per_point(scale_factor as f32);
    }

    /// Handle mouse move event.
    ///
    /// Returns `true` if egui wants to capture this input.
    pub fn on_mouse_move(&mut self, x: f64, y: f64) -> bool {
        self.input_state.on_mouse_move(x, y);
        self.wants_pointer_input
    }

    /// Handle mouse button event.
    ///
    /// Returns `true` if egui wants to capture this input.
    pub fn on_mouse_button(&mut self, button: MouseButton, pressed: bool) -> bool {
        self.input_state.on_mouse_button(button, pressed);
        self.wants_pointer_input
    }

    /// Handle mouse scroll event.
    ///
    /// Returns `true` if egui wants to capture this input.
    pub fn on_mouse_scroll(&mut self, delta: MouseScrollDelta) -> bool {
        self.input_state.on_mouse_scroll(delta);
        self.wants_pointer_input
    }

    /// Handle modifier keys change.
    pub fn on_modifiers_changed(&mut self, state: ModifiersState) {
        self.input_state.on_modifiers_changed(state);
    }

    /// Handle key event.
    ///
    /// Returns `true` if egui wants to capture this input.
    pub fn on_key(&mut self, event: &KeyEvent) -> bool {
        self.input_state
            .on_key(event.physical_key, event.state.is_pressed());

        // Handle text input
        if event.state.is_pressed()
            && let Some(ref text) = event.text
        {
            self.input_state.on_text_input(text.as_str());
        }

        self.wants_keyboard_input
    }

    /// Begin a new frame and run the egui app.
    ///
    /// Call this at the start of your frame, before creating the render graph.
    ///
    /// # Arguments
    ///
    /// * `elapsed_time` - Time since application start in seconds
    pub fn begin_frame(&mut self, elapsed_time: f64) {
        // Run setup once
        if !self.setup_done {
            if let Ok(mut app) = self.app.write() {
                app.setup(&self.ctx);
            }
            self.setup_done = true;
        }

        // Get raw input for this frame
        let raw_input = self.input_state.take_raw_input(elapsed_time);

        // Begin egui frame
        self.ctx.begin_pass(raw_input);

        // Run the user's UI code
        if let Ok(mut app) = self.app.write() {
            app.update(&self.ctx);
        }
    }

    /// End the frame and get rendering data.
    ///
    /// Call this after `begin_frame` to finalize the UI and get primitives for rendering.
    ///
    /// Returns the graphics pass for rendering egui, or `None` if there's nothing to render.
    pub fn end_frame(
        &mut self,
        render_target: &RenderTarget,
        screen_width: u32,
        screen_height: u32,
    ) -> Option<GraphicsPass> {
        // End egui frame
        let output = self.ctx.end_pass();

        // Update input state based on output
        self.wants_keyboard_input = self.ctx.wants_keyboard_input();
        self.wants_pointer_input = self.ctx.wants_pointer_input();
        self.input_state.update_from_output(&output.platform_output);

        // Update textures
        self.renderer.update_textures(&output.textures_delta);

        // Tessellate shapes into primitives
        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);

        if primitives.is_empty() {
            return None;
        }

        // Update screen size uniforms - egui outputs vertices in POINTS, not pixels
        // So we need to pass screen size in points to the shader
        let screen_width_points = screen_width as f32 / output.pixels_per_point;
        let screen_height_points = screen_height as f32 / output.pixels_per_point;
        self.renderer
            .update_screen_size_f32(screen_width_points, screen_height_points);

        // Create graphics pass
        Some(self.renderer.create_graphics_pass(
            &primitives,
            render_target,
            screen_width,
            screen_height,
            output.pixels_per_point,
        ))
    }

    /// Register a user-managed texture with egui.
    ///
    /// This allows external textures (such as render targets, offscreen buffers,
    /// or any GPU texture) to be displayed in egui UI elements like `ui.image()`.
    ///
    /// # Arguments
    ///
    /// * `texture` - The GPU texture to register
    ///
    /// # Returns
    ///
    /// A `TextureId` that can be used with egui's image widgets.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Register a render target texture
    /// let texture_id = controller.register_user_texture(my_render_target);
    ///
    /// // In your EguiApp::update():
    /// ui.image(egui::load::SizedTexture::new(texture_id, [256.0, 256.0]));
    /// ```
    pub fn register_user_texture(&mut self, texture: Arc<Texture>) -> TextureId {
        self.renderer.register_user_texture(texture)
    }

    /// Update a previously registered user texture.
    ///
    /// This is useful when the underlying texture has been recreated (e.g., on resize).
    ///
    /// # Arguments
    ///
    /// * `id` - The texture ID returned from `register_user_texture`
    /// * `texture` - The new GPU texture
    pub fn update_user_texture(&mut self, id: TextureId, texture: Arc<Texture>) {
        self.renderer.update_user_texture(id, texture);
    }

    /// Unregister a user-managed texture.
    ///
    /// The texture will no longer be available for rendering in egui.
    ///
    /// # Arguments
    ///
    /// * `id` - The texture ID returned from `register_user_texture`
    pub fn unregister_user_texture(&mut self, id: TextureId) {
        self.renderer.unregister_user_texture(id);
    }
}
