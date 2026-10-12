//! Egui controller that manages the UI lifecycle.
//!
//! The controller handles input processing, UI updates, and rendering.

use std::path::PathBuf;
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

#[cfg(test)]
mod input_tests {
    use super::*;
    use redlilium_core::input::{TouchId, TouchInput, TouchPhase};

    struct Ui {
        clicks: usize,
        button: egui::Rect,
    }
    impl super::super::EguiApp for Ui {
        fn update(&mut self, ctx: &Context) {
            egui::SidePanel::left("controls")
                .exact_width(120.0)
                .show(ctx, |ui| {
                    let response = ui.button("Restart");
                    self.button = response.rect;
                    self.clicks += usize::from(response.clicked());
                });
        }
    }

    fn setup() -> (EguiController, Arc<parking_lot::RwLock<Ui>>) {
        let device = crate::GraphicsInstance::new()
            .unwrap()
            .create_device()
            .unwrap();
        let ui = Arc::new(parking_lot::RwLock::new(Ui {
            clicks: 0,
            button: egui::Rect::NOTHING,
        }));
        let mut controller = EguiController::new(
            device,
            ui.clone(),
            800,
            600,
            2.0,
            crate::TextureFormat::Bgra8UnormSrgb,
        );
        for frame in 0..2 {
            controller.begin_frame(frame as f64 / 60.0);
            controller.discard_frame();
        }
        (controller, ui)
    }

    fn touch(controller: &mut EguiController, id: u64, phase: TouchPhase, pos: egui::Pos2) {
        controller.on_touch(TouchInput {
            id: TouchId(id),
            phase,
            position: [pos.x as f64 * 2.0, pos.y as f64 * 2.0],
        });
    }

    #[test]
    fn hit_test_uses_new_contact_position_and_dpi_without_mouse_hover() {
        let (controller, ui) = setup();
        let point = ui.read().button.center();
        assert!(controller.captures_pointer_at([point.x as f64 * 2.0, point.y as f64 * 2.0]));
        assert!(!controller.captures_pointer_at([700.0, 400.0]));
    }

    #[test]
    fn touch_tap_clicks_but_cancel_or_focus_loss_never_clicks() {
        for cancel in [None, Some(false), Some(true)] {
            let (mut controller, ui) = setup();
            let point = ui.read().button.center();
            touch(&mut controller, 1, TouchPhase::Started, point);
            controller.begin_frame(0.1);
            controller.discard_frame();
            assert!(controller.context().input(|i| i.pointer.primary_down()));
            match cancel {
                None => touch(&mut controller, 1, TouchPhase::Ended, point),
                Some(false) => touch(&mut controller, 1, TouchPhase::Cancelled, point),
                Some(true) => {
                    controller.on_focus_changed(false);
                    controller.on_focus_changed(true);
                }
            }
            controller.begin_frame(0.2);
            controller.discard_frame();
            assert!(!controller.context().input(|i| i.pointer.any_down()));
            assert_eq!(ui.read().clicks, usize::from(cancel.is_none()));
        }
    }

    #[test]
    fn secondary_finger_never_takes_over_primary_pointer() {
        let (mut controller, ui) = setup();
        let point = ui.read().button.center();
        touch(&mut controller, 1, TouchPhase::Started, point);
        touch(
            &mut controller,
            2,
            TouchPhase::Started,
            point + egui::vec2(0.0, 50.0),
        );
        controller.begin_frame(0.1);
        controller.discard_frame();
        touch(&mut controller, 1, TouchPhase::Ended, point);
        controller.begin_frame(0.2);
        controller.discard_frame();
        touch(&mut controller, 2, TouchPhase::Moved, point);
        controller.begin_frame(0.3);
        controller.discard_frame();
        assert!(!controller.context().input(|i| i.pointer.any_down()));
        touch(&mut controller, 2, TouchPhase::Ended, point);
        controller.begin_frame(0.4);
        controller.discard_frame();
        assert_eq!(ui.read().clicks, 1);
    }
}

impl EguiController {
    /// Hit-test the position against UI geometry, without relying on the
    /// previous frame's mouse hover flag. Coordinates are physical pixels.
    pub fn captures_pointer_at(&self, position: [f64; 2]) -> bool {
        let pos = egui::pos2(
            position[0] as f32 / self.input_state.pixels_per_point,
            position[1] as f32 / self.input_state.pixels_per_point,
        );
        self.ctx.layer_id_at(pos).is_some_and(|layer| {
            layer.order != egui::Order::Background || !self.ctx.available_rect().contains(pos)
        })
    }

    pub fn on_touch(&mut self, touch: redlilium_core::input::TouchInput) {
        if self.input_state.on_touch(touch) {
            self.reset_pointer();
        }
    }

    fn reset_pointer(&self) {
        self.ctx
            .input_mut(|input| input.pointer = Default::default());
        self.ctx.stop_dragging();
    }

    /// End gestures without synthesizing a click on the widget underneath.
    pub fn cancel_pointers(&mut self) {
        self.input_state.cancel_pointers();
        self.reset_pointer();
    }

    pub fn cancel_mouse(&mut self) {
        if self.input_state.cancel_mouse() {
            self.reset_pointer();
        }
    }

    pub fn on_focus_changed(&mut self, focused: bool) {
        self.input_state.on_focus_changed(focused);
        if !focused {
            self.reset_pointer();
        }
    }

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

    /// Handle a file being hovered over the window.
    pub fn on_file_hovered(&mut self, path: PathBuf) {
        self.input_state.on_file_hovered(path);
    }

    /// Handle file hover leaving the window.
    pub fn on_file_hover_cancelled(&mut self) {
        self.input_state.on_file_hover_cancelled();
    }

    /// Handle a file dropped onto the window.
    pub fn on_file_dropped(&mut self, path: PathBuf) {
        self.input_state.on_file_dropped(path);
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
            {
                let mut app = self.app.write();
                app.setup(&self.ctx);
            }
            self.setup_done = true;
        }

        // Get raw input for this frame
        let raw_input = self.input_state.take_raw_input(elapsed_time);

        // Begin egui frame
        self.ctx.begin_pass(raw_input);

        // Run the user's UI code
        {
            let mut app = self.app.write();
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

        // Create graphics pass. The screen-size uniform (egui outputs vertices
        // in POINTS) is written into the renderer's uniform ring inside the pass.
        Some(self.renderer.create_graphics_pass(
            &primitives,
            render_target,
            screen_width,
            screen_height,
            output.pixels_per_point,
        ))
    }

    /// Abandon an open frame without producing a draw pass.
    ///
    /// For hosts that split the bracket across callbacks ([`begin_frame`]
    /// before game/UI logic, [`end_frame`](Self::end_frame) in the draw phase):
    /// when the draw phase is skipped (outdated surface, busy frame slot), the
    /// pass is still open and the next [`begin_frame`](Self::begin_frame) would
    /// stack a second pass onto it. Call this first to close the stale pass.
    /// Texture-atlas deltas and platform output are preserved (they are
    /// incremental and must not be lost); only the frame's shapes are dropped.
    ///
    /// [`begin_frame`]: Self::begin_frame
    pub fn discard_frame(&mut self) {
        let output = self.ctx.end_pass();
        self.wants_keyboard_input = self.ctx.wants_keyboard_input();
        self.wants_pointer_input = self.ctx.wants_pointer_input();
        self.input_state.update_from_output(&output.platform_output);
        self.renderer.update_textures(&output.textures_delta);
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

    /// Flush queued egui atlas uploads into the current frame's render graph.
    ///
    /// Call once per frame after [`end_frame`](Self::end_frame), passing the
    /// same graph the egui pass is added to. The graph compiler orders the
    /// upload before the egui draw, so atlas updates take effect the same frame.
    pub fn flush_uploads(&mut self, graph: &mut crate::graph::RenderGraph) {
        self.renderer.flush_uploads(graph);
    }
}
