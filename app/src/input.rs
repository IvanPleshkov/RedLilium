//! Input conversion utilities.
//!
//! Maps platform-specific (winit) key codes to engine-agnostic
//! [`redlilium_core::input::KeyCode`] values.

use redlilium_core::input::KeyCode;
use winit::keyboard;

pub(crate) mod gamepads;
#[cfg(target_arch = "wasm32")]
pub(crate) mod web;
#[cfg(target_os = "windows")]
pub(crate) mod windows;

/// Normalizes platform IDs to connection/contact-lifetime identities.
#[derive(Default)]
pub(crate) struct TouchTracker {
    next: u64,
    active:
        std::collections::HashMap<(winit::event::DeviceId, u64), redlilium_core::input::TouchInput>,
}
impl TouchTracker {
    pub fn update(
        &mut self,
        touch: winit::event::Touch,
    ) -> Option<redlilium_core::input::TouchInput> {
        use redlilium_core::input::{TouchId, TouchInput, TouchPhase};
        let key = (touch.device_id, touch.id);
        let phase = match touch.phase {
            winit::event::TouchPhase::Started => TouchPhase::Started,
            winit::event::TouchPhase::Moved => TouchPhase::Moved,
            winit::event::TouchPhase::Ended => TouchPhase::Ended,
            winit::event::TouchPhase::Cancelled => TouchPhase::Cancelled,
        };
        if phase == TouchPhase::Started {
            if self.active.contains_key(&key) {
                return None;
            }
            self.next += 1;
            self.active.insert(
                key,
                TouchInput {
                    id: TouchId(self.next),
                    phase,
                    position: [touch.location.x, touch.location.y],
                },
            );
        }
        let input = self.active.get_mut(&key)?;
        input.phase = phase;
        input.position = [touch.location.x, touch.location.y];
        let input = *input;
        if matches!(phase, TouchPhase::Ended | TouchPhase::Cancelled) {
            self.active.remove(&key);
        }
        Some(input)
    }
    pub fn clear(&mut self) {
        self.active.clear();
    }
    #[cfg(any(target_arch = "wasm32", target_os = "windows", test))]
    pub fn capture_lost(&mut self, raw_id: u64) -> Option<redlilium_core::input::TouchInput> {
        let key = self.active.keys().find(|(_, id)| *id == raw_id).copied()?;
        let mut input = self.active.remove(&key)?;
        input.phase = redlilium_core::input::TouchPhase::Cancelled;
        Some(input)
    }
}

/// Convert a winit [`keyboard::KeyCode`] to an engine [`KeyCode`], if a
/// mapping exists.
pub fn map_winit_key(key: keyboard::KeyCode) -> Option<KeyCode> {
    Some(match key {
        // Letters
        keyboard::KeyCode::KeyA => KeyCode::A,
        keyboard::KeyCode::KeyB => KeyCode::B,
        keyboard::KeyCode::KeyC => KeyCode::C,
        keyboard::KeyCode::KeyD => KeyCode::D,
        keyboard::KeyCode::KeyE => KeyCode::E,
        keyboard::KeyCode::KeyF => KeyCode::F,
        keyboard::KeyCode::KeyG => KeyCode::G,
        keyboard::KeyCode::KeyH => KeyCode::H,
        keyboard::KeyCode::KeyI => KeyCode::I,
        keyboard::KeyCode::KeyJ => KeyCode::J,
        keyboard::KeyCode::KeyK => KeyCode::K,
        keyboard::KeyCode::KeyL => KeyCode::L,
        keyboard::KeyCode::KeyM => KeyCode::M,
        keyboard::KeyCode::KeyN => KeyCode::N,
        keyboard::KeyCode::KeyO => KeyCode::O,
        keyboard::KeyCode::KeyP => KeyCode::P,
        keyboard::KeyCode::KeyQ => KeyCode::Q,
        keyboard::KeyCode::KeyR => KeyCode::R,
        keyboard::KeyCode::KeyS => KeyCode::S,
        keyboard::KeyCode::KeyT => KeyCode::T,
        keyboard::KeyCode::KeyU => KeyCode::U,
        keyboard::KeyCode::KeyV => KeyCode::V,
        keyboard::KeyCode::KeyW => KeyCode::W,
        keyboard::KeyCode::KeyX => KeyCode::X,
        keyboard::KeyCode::KeyY => KeyCode::Y,
        keyboard::KeyCode::KeyZ => KeyCode::Z,

        // Digits
        keyboard::KeyCode::Digit0 => KeyCode::Digit0,
        keyboard::KeyCode::Digit1 => KeyCode::Digit1,
        keyboard::KeyCode::Digit2 => KeyCode::Digit2,
        keyboard::KeyCode::Digit3 => KeyCode::Digit3,
        keyboard::KeyCode::Digit4 => KeyCode::Digit4,
        keyboard::KeyCode::Digit5 => KeyCode::Digit5,
        keyboard::KeyCode::Digit6 => KeyCode::Digit6,
        keyboard::KeyCode::Digit7 => KeyCode::Digit7,
        keyboard::KeyCode::Digit8 => KeyCode::Digit8,
        keyboard::KeyCode::Digit9 => KeyCode::Digit9,

        // Function keys
        keyboard::KeyCode::F1 => KeyCode::F1,
        keyboard::KeyCode::F2 => KeyCode::F2,
        keyboard::KeyCode::F3 => KeyCode::F3,
        keyboard::KeyCode::F4 => KeyCode::F4,
        keyboard::KeyCode::F5 => KeyCode::F5,
        keyboard::KeyCode::F6 => KeyCode::F6,
        keyboard::KeyCode::F7 => KeyCode::F7,
        keyboard::KeyCode::F8 => KeyCode::F8,
        keyboard::KeyCode::F9 => KeyCode::F9,
        keyboard::KeyCode::F10 => KeyCode::F10,
        keyboard::KeyCode::F11 => KeyCode::F11,
        keyboard::KeyCode::F12 => KeyCode::F12,

        // Modifiers
        keyboard::KeyCode::ShiftLeft => KeyCode::ShiftLeft,
        keyboard::KeyCode::ShiftRight => KeyCode::ShiftRight,
        keyboard::KeyCode::ControlLeft => KeyCode::ControlLeft,
        keyboard::KeyCode::ControlRight => KeyCode::ControlRight,
        keyboard::KeyCode::AltLeft => KeyCode::AltLeft,
        keyboard::KeyCode::AltRight => KeyCode::AltRight,
        keyboard::KeyCode::SuperLeft => KeyCode::SuperLeft,
        keyboard::KeyCode::SuperRight => KeyCode::SuperRight,

        // Arrows
        keyboard::KeyCode::ArrowUp => KeyCode::ArrowUp,
        keyboard::KeyCode::ArrowDown => KeyCode::ArrowDown,
        keyboard::KeyCode::ArrowLeft => KeyCode::ArrowLeft,
        keyboard::KeyCode::ArrowRight => KeyCode::ArrowRight,

        // Common
        keyboard::KeyCode::Space => KeyCode::Space,
        keyboard::KeyCode::Enter => KeyCode::Enter,
        keyboard::KeyCode::Escape => KeyCode::Escape,
        keyboard::KeyCode::Tab => KeyCode::Tab,
        keyboard::KeyCode::Backspace => KeyCode::Backspace,
        keyboard::KeyCode::Delete => KeyCode::Delete,
        keyboard::KeyCode::Insert => KeyCode::Insert,
        keyboard::KeyCode::Home => KeyCode::Home,
        keyboard::KeyCode::End => KeyCode::End,
        keyboard::KeyCode::PageUp => KeyCode::PageUp,
        keyboard::KeyCode::PageDown => KeyCode::PageDown,

        // Punctuation / symbols
        keyboard::KeyCode::Minus => KeyCode::Minus,
        keyboard::KeyCode::Equal => KeyCode::Equal,
        keyboard::KeyCode::BracketLeft => KeyCode::BracketLeft,
        keyboard::KeyCode::BracketRight => KeyCode::BracketRight,
        keyboard::KeyCode::Backslash => KeyCode::Backslash,
        keyboard::KeyCode::Semicolon => KeyCode::Semicolon,
        keyboard::KeyCode::Quote => KeyCode::Quote,
        keyboard::KeyCode::Backquote => KeyCode::Backquote,
        keyboard::KeyCode::Comma => KeyCode::Comma,
        keyboard::KeyCode::Period => KeyCode::Period,
        keyboard::KeyCode::Slash => KeyCode::Slash,

        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::{
        dpi::PhysicalPosition,
        event::{DeviceId, Touch, TouchPhase},
    };

    fn event(id: u64, phase: TouchPhase) -> Touch {
        Touch {
            device_id: DeviceId::dummy(),
            phase,
            location: PhysicalPosition::new(-20.0, 900.0),
            force: None,
            id,
        }
    }

    #[test]
    fn winit_contacts_keep_identity_until_end_and_ignore_late_events() {
        let mut tracker = TouchTracker::default();
        let first = tracker.update(event(7, TouchPhase::Started)).unwrap();
        let second = tracker.update(event(8, TouchPhase::Started)).unwrap();
        assert_ne!(first.id, second.id);
        assert_eq!(first.position, [-20.0, 900.0]);
        assert_eq!(
            tracker.update(event(7, TouchPhase::Moved)).unwrap().id,
            first.id
        );
        assert_eq!(
            tracker.update(event(8, TouchPhase::Ended)).unwrap().id,
            second.id
        );
        assert!(tracker.update(event(8, TouchPhase::Moved)).is_none());
        let replacement = tracker.update(event(8, TouchPhase::Started)).unwrap();
        assert_ne!(replacement.id, second.id);
        let cancelled = tracker.capture_lost(8).unwrap();
        assert_eq!(cancelled.id, replacement.id);
        assert_eq!(
            cancelled.phase,
            redlilium_core::input::TouchPhase::Cancelled
        );
        assert!(tracker.update(event(8, TouchPhase::Ended)).is_none());
        assert_eq!(
            tracker.update(event(7, TouchPhase::Moved)).unwrap().id,
            first.id
        );
        tracker.clear();
        assert!(tracker.update(event(7, TouchPhase::Moved)).is_none());
        assert_ne!(
            tracker.update(event(7, TouchPhase::Started)).unwrap().id,
            first.id
        );
    }
}
