//! Input handling for egui integration.
//!
//! This module provides translation from winit events to egui input events.

use std::path::PathBuf;

use egui::{Key, Modifiers, PointerButton, Pos2, RawInput, Rect, Vec2};
use winit::event::{MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};

/// Input state for egui.
///
/// Collects input events throughout a frame and produces egui's RawInput.
pub struct EguiInputState {
    /// Current mouse position in points.
    pub mouse_pos: Pos2,
    /// Accumulated scroll delta.
    pub scroll_delta: Vec2,
    /// Current modifier keys state.
    pub modifiers: Modifiers,
    /// Screen rectangle in points.
    pub screen_rect: Rect,
    /// Pixels per point (DPI scaling).
    pub pixels_per_point: f32,
    /// Events collected this frame.
    pub events: Vec<egui::Event>,
    /// Files being hovered over the window.
    hovered_files: Vec<egui::HoveredFile>,
    /// Files dropped onto the window this frame.
    dropped_files: Vec<egui::DroppedFile>,
    /// System clipboard for copy/paste (desktop only).
    #[cfg(not(target_arch = "wasm32"))]
    clipboard: Option<arboard::Clipboard>,
}

impl EguiInputState {
    /// Create a new input state with the given screen size.
    pub fn new(width: u32, height: u32, pixels_per_point: f32) -> Self {
        Self {
            mouse_pos: Pos2::ZERO,
            scroll_delta: Vec2::ZERO,
            modifiers: Modifiers::default(),
            screen_rect: Rect::from_min_size(
                Pos2::ZERO,
                Vec2::new(
                    width as f32 / pixels_per_point,
                    height as f32 / pixels_per_point,
                ),
            ),
            pixels_per_point,
            events: Vec::new(),
            hovered_files: Vec::new(),
            dropped_files: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            clipboard: arboard::Clipboard::new().ok(),
        }
    }

    /// Update screen size.
    pub fn set_screen_size(&mut self, width: u32, height: u32) {
        self.screen_rect = Rect::from_min_size(
            Pos2::ZERO,
            Vec2::new(
                width as f32 / self.pixels_per_point,
                height as f32 / self.pixels_per_point,
            ),
        );
    }

    /// Update pixels per point (DPI scaling).
    ///
    /// Note: Call `set_screen_size` after this to update the screen rect
    /// with the new scale factor.
    pub fn set_pixels_per_point(&mut self, pixels_per_point: f32) {
        self.pixels_per_point = pixels_per_point;
    }

    /// Handle mouse move event.
    pub fn on_mouse_move(&mut self, x: f64, y: f64) {
        let pos = Pos2::new(
            x as f32 / self.pixels_per_point,
            y as f32 / self.pixels_per_point,
        );
        self.mouse_pos = pos;
        self.events.push(egui::Event::PointerMoved(pos));
    }

    /// Handle mouse button event.
    pub fn on_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        let egui_button = match button {
            MouseButton::Left => Some(PointerButton::Primary),
            MouseButton::Right => Some(PointerButton::Secondary),
            MouseButton::Middle => Some(PointerButton::Middle),
            MouseButton::Back => Some(PointerButton::Extra1),
            MouseButton::Forward => Some(PointerButton::Extra2),
            MouseButton::Other(_) => None,
        };

        if let Some(button) = egui_button {
            self.events.push(egui::Event::PointerButton {
                pos: self.mouse_pos,
                button,
                pressed,
                modifiers: self.modifiers,
            });
        }
    }

    /// Handle mouse scroll event.
    pub fn on_mouse_scroll(&mut self, delta: MouseScrollDelta) {
        let delta = match delta {
            MouseScrollDelta::LineDelta(x, y) => {
                // Line delta is typically in "lines", convert to points
                Vec2::new(x * 24.0, y * 24.0)
            }
            MouseScrollDelta::PixelDelta(pos) => Vec2::new(
                pos.x as f32 / self.pixels_per_point,
                pos.y as f32 / self.pixels_per_point,
            ),
        };

        self.scroll_delta += delta;
        self.events.push(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta,
            modifiers: self.modifiers,
        });
    }

    /// Handle keyboard modifiers change.
    pub fn on_modifiers_changed(&mut self, state: ModifiersState) {
        self.modifiers = Modifiers {
            alt: state.alt_key(),
            ctrl: state.control_key(),
            shift: state.shift_key(),
            mac_cmd: cfg!(target_os = "macos") && state.super_key(),
            command: if cfg!(target_os = "macos") {
                state.super_key()
            } else {
                state.control_key()
            },
        };
    }

    /// Handle key event.
    pub fn on_key(&mut self, physical_key: PhysicalKey, pressed: bool) {
        if let PhysicalKey::Code(keycode) = physical_key
            && let Some(key) = translate_keycode(keycode)
        {
            self.events.push(egui::Event::Key {
                key,
                physical_key: Some(translate_physical_key(keycode)),
                pressed,
                repeat: false,
                modifiers: self.modifiers,
            });

            // Handle paste (Ctrl+V / Cmd+V)
            #[cfg(not(target_arch = "wasm32"))]
            if pressed
                && key == Key::V
                && self.modifiers.command
                && let Some(clipboard) = &mut self.clipboard
                && let Ok(text) = clipboard.get_text()
                && !text.is_empty()
            {
                self.events.push(egui::Event::Paste(text));
            }

            // Handle copy (Ctrl+C / Cmd+C)
            #[cfg(not(target_arch = "wasm32"))]
            if pressed && key == Key::C && self.modifiers.command {
                self.events.push(egui::Event::Copy);
            }

            // Handle cut (Ctrl+X / Cmd+X)
            #[cfg(not(target_arch = "wasm32"))]
            if pressed && key == Key::X && self.modifiers.command {
                self.events.push(egui::Event::Cut);
            }
        }
    }

    /// Handle text input.
    pub fn on_text_input(&mut self, text: &str) {
        // Filter out control characters
        let filtered: String = text
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
            .collect();
        if !filtered.is_empty() {
            self.events.push(egui::Event::Text(filtered));
        }
    }

    /// Handle a file being hovered over the window.
    pub fn on_file_hovered(&mut self, path: PathBuf) {
        self.hovered_files.push(egui::HoveredFile {
            path: Some(path),
            ..Default::default()
        });
    }

    /// Handle file hover leaving the window.
    pub fn on_file_hover_cancelled(&mut self) {
        self.hovered_files.clear();
    }

    /// Handle a file dropped onto the window.
    pub fn on_file_dropped(&mut self, path: PathBuf) {
        self.dropped_files.push(egui::DroppedFile {
            path: Some(path),
            ..Default::default()
        });
    }

    /// Take the raw input for this frame and prepare for next frame.
    pub fn take_raw_input(&mut self, time: f64) -> RawInput {
        let events = std::mem::take(&mut self.events);
        self.scroll_delta = Vec2::ZERO;

        // Create viewport info with native pixels per point
        let mut viewports = egui::viewport::ViewportIdMap::default();
        viewports.insert(
            egui::ViewportId::ROOT,
            egui::ViewportInfo {
                native_pixels_per_point: Some(self.pixels_per_point),
                ..Default::default()
            },
        );

        RawInput {
            viewport_id: egui::ViewportId::ROOT,
            viewports,
            screen_rect: Some(self.screen_rect),
            max_texture_side: Some(8192),
            time: Some(time),
            predicted_dt: 1.0 / 60.0,
            modifiers: self.modifiers,
            events,
            hovered_files: std::mem::take(&mut self.hovered_files),
            dropped_files: std::mem::take(&mut self.dropped_files),
            focused: true,
            ..Default::default()
        }
    }

    /// Update the state based on egui's output.
    pub fn update_from_output(&mut self, output: &egui::PlatformOutput) {
        // Handle cursor icon changes, copy/paste, etc.
        // In egui 0.33, clipboard operations are in output.commands
        for command in &output.commands {
            if let egui::OutputCommand::CopyText(text) = command {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(clipboard) = &mut self.clipboard
                    && let Err(e) = clipboard.set_text(text)
                {
                    log::warn!("Failed to copy to clipboard: {}", e);
                }
            }
        }
    }
}

/// Translate winit keycode to egui key.
fn translate_keycode(keycode: KeyCode) -> Option<Key> {
    Some(match keycode {
        KeyCode::Escape => Key::Escape,
        KeyCode::Insert => Key::Insert,
        KeyCode::Home => Key::Home,
        KeyCode::Delete => Key::Delete,
        KeyCode::End => Key::End,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::ArrowLeft => Key::ArrowLeft,
        KeyCode::ArrowUp => Key::ArrowUp,
        KeyCode::ArrowRight => Key::ArrowRight,
        KeyCode::ArrowDown => Key::ArrowDown,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Enter | KeyCode::NumpadEnter => Key::Enter,
        KeyCode::Tab => Key::Tab,
        KeyCode::Space => Key::Space,

        KeyCode::KeyA => Key::A,
        KeyCode::KeyB => Key::B,
        KeyCode::KeyC => Key::C,
        KeyCode::KeyD => Key::D,
        KeyCode::KeyE => Key::E,
        KeyCode::KeyF => Key::F,
        KeyCode::KeyG => Key::G,
        KeyCode::KeyH => Key::H,
        KeyCode::KeyI => Key::I,
        KeyCode::KeyJ => Key::J,
        KeyCode::KeyK => Key::K,
        KeyCode::KeyL => Key::L,
        KeyCode::KeyM => Key::M,
        KeyCode::KeyN => Key::N,
        KeyCode::KeyO => Key::O,
        KeyCode::KeyP => Key::P,
        KeyCode::KeyQ => Key::Q,
        KeyCode::KeyR => Key::R,
        KeyCode::KeyS => Key::S,
        KeyCode::KeyT => Key::T,
        KeyCode::KeyU => Key::U,
        KeyCode::KeyV => Key::V,
        KeyCode::KeyW => Key::W,
        KeyCode::KeyX => Key::X,
        KeyCode::KeyY => Key::Y,
        KeyCode::KeyZ => Key::Z,

        KeyCode::Digit0 | KeyCode::Numpad0 => Key::Num0,
        KeyCode::Digit1 | KeyCode::Numpad1 => Key::Num1,
        KeyCode::Digit2 | KeyCode::Numpad2 => Key::Num2,
        KeyCode::Digit3 | KeyCode::Numpad3 => Key::Num3,
        KeyCode::Digit4 | KeyCode::Numpad4 => Key::Num4,
        KeyCode::Digit5 | KeyCode::Numpad5 => Key::Num5,
        KeyCode::Digit6 | KeyCode::Numpad6 => Key::Num6,
        KeyCode::Digit7 | KeyCode::Numpad7 => Key::Num7,
        KeyCode::Digit8 | KeyCode::Numpad8 => Key::Num8,
        KeyCode::Digit9 | KeyCode::Numpad9 => Key::Num9,

        KeyCode::F1 => Key::F1,
        KeyCode::F2 => Key::F2,
        KeyCode::F3 => Key::F3,
        KeyCode::F4 => Key::F4,
        KeyCode::F5 => Key::F5,
        KeyCode::F6 => Key::F6,
        KeyCode::F7 => Key::F7,
        KeyCode::F8 => Key::F8,
        KeyCode::F9 => Key::F9,
        KeyCode::F10 => Key::F10,
        KeyCode::F11 => Key::F11,
        KeyCode::F12 => Key::F12,

        KeyCode::Minus | KeyCode::NumpadSubtract => Key::Minus,
        KeyCode::Equal | KeyCode::NumpadAdd => Key::Equals,

        _ => return None,
    })
}

/// Translate winit keycode to egui physical key.
fn translate_physical_key(keycode: KeyCode) -> egui::Key {
    translate_keycode(keycode).unwrap_or(Key::Escape)
}
