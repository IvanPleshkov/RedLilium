//! Platform-agnostic input types.
//!
//! Provides a [`KeyCode`] enum that identifies physical keyboard keys
//! without depending on any windowing crate.

/// Physical keyboard key identifier.
///
/// Matches common physical key positions (US QWERTY layout names).
/// Platform layers (e.g. winit) map their native key codes to this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KeyCode {
    // Letters
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,

    // Digits
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,

    // Function keys
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,

    // Modifiers
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    SuperLeft,
    SuperRight,

    // Arrows
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,

    // Common keys
    Space,
    Enter,
    Escape,
    Tab,
    Backspace,
    Delete,
    Insert,
    Home,
    End,
    PageUp,
    PageDown,

    // Punctuation / symbols
    Minus,
    Equal,
    BracketLeft,
    BracketRight,
    Backslash,
    Semicolon,
    Quote,
    Backquote,
    Comma,
    Period,
    Slash,
}
