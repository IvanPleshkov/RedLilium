use std::collections::HashSet;

use redlilium_core::input::KeyCode;

/// Platform-agnostic window and input state, updated by the application layer.
///
/// This is an ECS **resource** (not a component). Insert it into the world with
/// [`World::insert_resource`] and access it in systems via [`Res<WindowInput>`]
/// or [`ResMut<WindowInput>`].
///
/// The application layer is responsible for:
/// 1. Calling [`begin_frame()`](WindowInput::begin_frame) after systems have
///    consumed the current frame's deltas.
/// 2. Forwarding platform events via `on_mouse_move`, `on_mouse_button`,
///    `on_scroll`, `on_key_pressed`, `on_key_released`.
/// 3. Setting `ui_wants_input` when a UI layer (e.g. egui) consumes input.
///
/// This type intentionally has no dependency on `winit` or any windowing crate.
#[derive(Debug, Clone)]
pub struct WindowInput {
    /// Current cursor position in physical pixels.
    pub cursor_position: [f32; 2],
    /// Cursor movement accumulated this frame (physical pixels).
    pub cursor_delta: [f32; 2],
    /// Window width in physical pixels.
    pub window_width: f32,
    /// Window height in physical pixels.
    pub window_height: f32,
    /// Whether the left mouse button is currently held.
    pub mouse_left: bool,
    /// Whether the right mouse button is currently held.
    pub mouse_right: bool,
    /// Whether the middle mouse button is currently held.
    pub mouse_middle: bool,
    /// Scroll delta accumulated this frame (x, y). Positive y = scroll up.
    pub scroll_delta: [f32; 2],

    /// Set of currently pressed keyboard keys.
    pressed_keys: HashSet<KeyCode>,

    /// When `true`, a UI layer (e.g. egui) wants keyboard/mouse input.
    /// Systems that consume input should skip processing when this is set.
    pub ui_wants_input: bool,
}

impl WindowInput {
    /// Reset per-frame deltas. Call **after** systems have consumed the
    /// current frame's input, before forwarding new input events.
    pub fn begin_frame(&mut self) {
        self.cursor_delta = [0.0, 0.0];
        self.scroll_delta = [0.0, 0.0];
    }

    /// Update cursor position and accumulate delta from a mouse-move event.
    pub fn on_mouse_move(&mut self, x: f64, y: f64) {
        let new_x = x as f32;
        let new_y = y as f32;
        self.cursor_delta[0] += new_x - self.cursor_position[0];
        self.cursor_delta[1] += new_y - self.cursor_position[1];
        self.cursor_position = [new_x, new_y];
    }

    /// Update a mouse button state.
    ///
    /// `button_index`: 0 = left, 1 = right, 2 = middle.
    pub fn on_mouse_button(&mut self, button_index: u8, pressed: bool) {
        match button_index {
            0 => self.mouse_left = pressed,
            1 => self.mouse_right = pressed,
            2 => self.mouse_middle = pressed,
            _ => {}
        }
    }

    /// Accumulate scroll delta for this frame.
    pub fn on_scroll(&mut self, dx: f32, dy: f32) {
        self.scroll_delta[0] += dx;
        self.scroll_delta[1] += dy;
    }

    /// Record a key press.
    pub fn on_key_pressed(&mut self, key: KeyCode) {
        self.pressed_keys.insert(key);
    }

    /// Record a key release.
    pub fn on_key_released(&mut self, key: KeyCode) {
        self.pressed_keys.remove(&key);
    }

    /// Check whether a key is currently pressed.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    /// Window aspect ratio (width / height). Returns 1.0 if height is zero.
    pub fn aspect_ratio(&self) -> f32 {
        if self.window_height > 0.0 {
            self.window_width / self.window_height
        } else {
            1.0
        }
    }
}

impl Default for WindowInput {
    fn default() -> Self {
        Self {
            cursor_position: [0.0, 0.0],
            cursor_delta: [0.0, 0.0],
            window_width: 800.0,
            window_height: 600.0,
            mouse_left: false,
            mouse_right: false,
            mouse_middle: false,
            scroll_delta: [0.0, 0.0],
            pressed_keys: HashSet::new(),
            ui_wants_input: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_frame_clears_deltas() {
        let mut input = WindowInput {
            cursor_delta: [10.0, 20.0],
            scroll_delta: [1.0, 2.0],
            ..WindowInput::default()
        };
        input.begin_frame();
        assert_eq!(input.cursor_delta, [0.0, 0.0]);
        assert_eq!(input.scroll_delta, [0.0, 0.0]);
    }

    #[test]
    fn on_mouse_move_accumulates_delta() {
        let mut input = WindowInput::default();
        input.on_mouse_move(100.0, 200.0);
        assert_eq!(input.cursor_position, [100.0, 200.0]);
        assert_eq!(input.cursor_delta, [100.0, 200.0]);

        // Second move accumulates
        input.on_mouse_move(110.0, 205.0);
        assert_eq!(input.cursor_position, [110.0, 205.0]);
        assert_eq!(input.cursor_delta, [110.0, 205.0]);
    }

    #[test]
    fn on_mouse_button_sets_flags() {
        let mut input = WindowInput::default();
        input.on_mouse_button(0, true);
        assert!(input.mouse_left);
        input.on_mouse_button(1, true);
        assert!(input.mouse_right);
        input.on_mouse_button(2, true);
        assert!(input.mouse_middle);
        input.on_mouse_button(0, false);
        assert!(!input.mouse_left);
    }

    #[test]
    fn on_scroll_accumulates() {
        let mut input = WindowInput::default();
        input.on_scroll(1.0, 2.0);
        input.on_scroll(0.5, -1.0);
        assert!((input.scroll_delta[0] - 1.5).abs() < f32::EPSILON);
        assert!((input.scroll_delta[1] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn key_press_and_release() {
        let mut input = WindowInput::default();
        assert!(!input.is_key_pressed(KeyCode::W));

        input.on_key_pressed(KeyCode::W);
        assert!(input.is_key_pressed(KeyCode::W));

        input.on_key_released(KeyCode::W);
        assert!(!input.is_key_pressed(KeyCode::W));
    }

    #[test]
    fn multiple_keys_pressed() {
        let mut input = WindowInput::default();
        input.on_key_pressed(KeyCode::W);
        input.on_key_pressed(KeyCode::ShiftLeft);
        assert!(input.is_key_pressed(KeyCode::W));
        assert!(input.is_key_pressed(KeyCode::ShiftLeft));
        assert!(!input.is_key_pressed(KeyCode::A));
    }

    #[test]
    fn aspect_ratio_edge_cases() {
        let mut input = WindowInput {
            window_width: 1920.0,
            window_height: 1080.0,
            ..WindowInput::default()
        };
        let aspect = input.aspect_ratio();
        assert!((aspect - 16.0 / 9.0).abs() < 0.01);

        input.window_height = 0.0;
        assert_eq!(input.aspect_ratio(), 1.0);
    }
}
