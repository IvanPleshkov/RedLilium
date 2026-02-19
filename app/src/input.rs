//! Input conversion utilities.
//!
//! Maps platform-specific (winit) key codes to engine-agnostic
//! [`redlilium_core::input::KeyCode`] values.

use redlilium_core::input::KeyCode;
use winit::keyboard;

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
