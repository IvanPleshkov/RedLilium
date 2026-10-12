use std::collections::{BTreeMap, HashSet};

pub use redlilium_core::input::{
    GamepadAxis, GamepadButton, GamepadButtonState, GamepadEvent, GamepadId, GamepadMapping,
    GamepadState, InputCancelReason, InputEvent, InputEventReader, InputOwner, TouchContact,
    TouchId, TouchInput, TouchPhase,
};
use redlilium_core::input::{InputEventStream, KeyCode};

/// Platform-agnostic window and input state, updated by the application layer.
///
/// This is an ECS **resource** (not a component). Insert it into the world with
/// [`World::insert_resource`](crate::World::insert_resource) and access it in
/// systems via [`Res`](crate::Res) or [`ResMut`](crate::ResMut).
///
/// The application layer is responsible for:
/// 1. Calling [`begin_frame()`](WindowInput::begin_frame) after systems have
///    consumed the current frame's deltas.
/// 2. Forwarding keyboard, pointer, gamepad and effective focus events via
///    the corresponding `on_*` methods.
/// 3. Latching UI ownership on pointer start and updating the input viewport.
///
/// Read held state directly; use [`subscribe`](Self::subscribe) for independent
/// durable transition readers in Update and FixedUpdate, including paused frames.
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
    /// Whether the host accepts input. Focus/visibility loss neutralizes held
    /// controls. Regaining focus does not restore keyboard or pointer holds;
    /// gamepads resume from a fresh platform sample.
    pub focused: bool,
    /// Physical pixels per logical pixel.
    pub scale_factor: f64,
    mouse_owners: [Option<InputOwner>; 3],
    touches: BTreeMap<TouchId, TouchContact>,
    gamepads: BTreeMap<GamepadId, GamepadState>,
    events: InputEventStream,
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
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        let new_x = x as f32;
        let new_y = y as f32;
        if self.focused {
            self.cursor_delta[0] += new_x - self.cursor_position[0];
            self.cursor_delta[1] += new_y - self.cursor_position[1];
        }
        self.cursor_position = [new_x, new_y];
    }

    /// Update a mouse button state.
    ///
    /// `button_index`: 0 = left, 1 = right, 2 = middle.
    pub fn on_mouse_button(&mut self, button_index: u8, pressed: bool) {
        self.on_mouse_button_owned(
            button_index,
            pressed,
            if self.ui_wants_input {
                InputOwner::Ui
            } else {
                InputOwner::Game
            },
        );
    }

    /// Ownership is latched on press until release/cancel. Use
    /// `game_mouse_pressed` in gameplay; legacy mouse flags describe raw holds.
    pub fn on_mouse_button_owned(&mut self, button_index: u8, pressed: bool, owner: InputOwner) {
        if button_index > 2 || (pressed && !self.focused) {
            return;
        }
        let old = [self.mouse_left, self.mouse_right, self.mouse_middle][button_index as usize];
        if old == pressed {
            return;
        }
        let owner = if pressed {
            owner
        } else {
            self.mouse_owners[button_index as usize].unwrap_or(owner)
        };
        self.mouse_owners[button_index as usize] = pressed.then_some(owner);
        match button_index {
            0 => self.mouse_left = pressed,
            1 => self.mouse_right = pressed,
            2 => self.mouse_middle = pressed,
            _ => {}
        }
        self.events.send(InputEvent::MouseButton {
            button: button_index,
            pressed,
            owner,
        });
    }

    /// Accumulate scroll delta for this frame.
    pub fn on_scroll(&mut self, dx: f32, dy: f32) {
        if !self.focused || !dx.is_finite() || !dy.is_finite() {
            return;
        }
        self.scroll_delta[0] += dx;
        self.scroll_delta[1] += dy;
    }

    /// Record a key press.
    pub fn on_key_pressed(&mut self, key: KeyCode) {
        if self.focused && self.pressed_keys.insert(key) {
            self.events.send(InputEvent::Key { key, pressed: true });
        }
    }

    /// Record a key release.
    pub fn on_key_released(&mut self, key: KeyCode) {
        if self.pressed_keys.remove(&key) {
            self.events.send(InputEvent::Key {
                key,
                pressed: false,
            });
        }
    }

    /// Check whether a key is currently pressed.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    /// Independent durable events for a frame or fixed-step consumer. Frame
    /// cleanup never clears a reader. Subscribe before generating input.
    pub fn subscribe(&mut self) -> InputEventReader {
        self.events.subscribe()
    }

    pub fn game_mouse_pressed(&self, button: u8) -> bool {
        self.mouse_owners.get(button as usize) == Some(&Some(InputOwner::Game))
    }
    pub fn touches(&self) -> &BTreeMap<TouchId, TouchContact> {
        &self.touches
    }
    pub fn gamepads(&self) -> &BTreeMap<GamepadId, GamepadState> {
        &self.gamepads
    }

    /// Inject native, browser, or test input. A contact cannot change owner.
    /// Unknown move/end events are ignored, including late events after cancel.
    pub fn on_touch(&mut self, input: TouchInput, owner: InputOwner) {
        if !self.focused || !input.position.iter().all(|v| v.is_finite()) {
            return;
        }
        if input.phase == TouchPhase::Started {
            if self.touches.contains_key(&input.id) {
                return;
            }
            self.touches.insert(
                input.id,
                TouchContact {
                    id: input.id,
                    origin: input.position,
                    position: input.position,
                    owner,
                },
            );
        }
        let Some(contact) = self.touches.get_mut(&input.id) else {
            return;
        };
        contact.position = input.position;
        self.events.send(InputEvent::Touch {
            contact: *contact,
            phase: input.phase,
        });
        if matches!(input.phase, TouchPhase::Ended | TouchPhase::Cancelled) {
            self.touches.remove(&input.id);
        }
    }

    pub fn on_gamepad(&mut self, mut event: GamepadEvent) {
        match &mut event {
            GamepadEvent::Connected { id, name, mapping } => {
                if self.gamepads.contains_key(id) {
                    return;
                }
                self.gamepads.insert(
                    *id,
                    GamepadState {
                        id: *id,
                        name: name.clone(),
                        mapping: *mapping,
                        axes: BTreeMap::new(),
                        buttons: BTreeMap::new(),
                    },
                );
            }
            GamepadEvent::Disconnected { id } => {
                self.neutralize_gamepad(*id);
                if self.gamepads.remove(id).is_none() {
                    return;
                }
            }
            GamepadEvent::Axis { id, axis, value } => {
                let Some(pad) = self.gamepads.get_mut(id) else {
                    return;
                };
                *value = if self.focused && value.is_finite() {
                    value.clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                if pad.axes.get(axis).copied().unwrap_or(0.0) == *value {
                    return;
                }
                pad.axes.insert(*axis, *value);
            }
            GamepadEvent::Button {
                id,
                button,
                value,
                pressed,
            } => {
                let Some(pad) = self.gamepads.get_mut(id) else {
                    return;
                };
                *value = if self.focused && value.is_finite() {
                    value.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                *pressed &= self.focused;
                let state = GamepadButtonState {
                    value: *value,
                    pressed: *pressed,
                };
                if pad.buttons.get(button).copied().unwrap_or_default() == state {
                    return;
                }
                pad.buttons.insert(*button, state);
            }
        }
        self.events.send(InputEvent::Gamepad(event));
    }

    fn neutralize_gamepad(&mut self, id: GamepadId) {
        if let Some(pad) = self.gamepads.get(&id).cloned() {
            for axis in pad.axes.keys() {
                self.on_gamepad(GamepadEvent::Axis {
                    id,
                    axis: *axis,
                    value: 0.0,
                });
            }
            for button in pad.buttons.keys() {
                self.on_gamepad(GamepadEvent::Button {
                    id,
                    button: *button,
                    value: 0.0,
                    pressed: false,
                });
            }
        }
    }

    /// Cancel all pointer gestures, retaining keyboard/gamepad state. UI can
    /// explicitly cancel gameplay drags before opening a modal.
    pub fn cancel_pointers(&mut self, reason: InputCancelReason) {
        for (_, contact) in std::mem::take(&mut self.touches) {
            self.events.send(InputEvent::Touch {
                contact,
                phase: TouchPhase::Cancelled,
            });
        }
        self.cancel_mouse(reason);
    }

    /// Cursor exit cancels mouse drags; captured touch contacts remain active.
    pub fn cancel_mouse(&mut self, reason: InputCancelReason) {
        for button in 0..3 {
            self.on_mouse_button_owned(button, false, InputOwner::Game);
        }
        self.cursor_delta = [0.0; 2];
        self.scroll_delta = [0.0; 2];
        self.events.send(InputEvent::Cancelled(reason));
    }

    /// Neutralize every held control immediately, even without another frame.
    pub fn cancel_all(&mut self, reason: InputCancelReason) {
        for key in self.pressed_keys.clone() {
            self.on_key_released(key);
        }
        for id in self.gamepads.keys().copied().collect::<Vec<_>>() {
            self.neutralize_gamepad(id);
        }
        self.cancel_pointers(reason);
    }

    pub fn on_focus_changed(&mut self, focused: bool) {
        if self.focused == focused {
            return;
        }
        self.focused = focused;
        if !focused {
            self.cancel_all(InputCancelReason::FocusLost);
        }
        self.events.send(InputEvent::Focused(focused));
    }

    /// Changing the coordinate space ends active drags rather than silently
    /// moving their origins. The next contact starts in the new viewport.
    /// Returns whether the coordinate space changed and gestures were cancelled.
    pub fn set_viewport(&mut self, width: f32, height: f32, scale_factor: f64) -> bool {
        if (self.window_width, self.window_height, self.scale_factor)
            != (width, height, scale_factor)
        {
            self.cancel_pointers(InputCancelReason::ViewportChanged);
            self.window_width = width;
            self.window_height = height;
            self.scale_factor = scale_factor;
            true
        } else {
            false
        }
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
            focused: true,
            scale_factor: 1.0,
            mouse_owners: [None; 3],
            touches: BTreeMap::new(),
            gamepads: BTreeMap::new(),
            events: InputEventStream::default(),
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
