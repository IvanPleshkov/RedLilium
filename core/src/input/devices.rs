//! Device-neutral input protocol. Hosts and headless tests use the same events.
use super::KeyCode;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

/// Connection identity, stable until disconnect. Reconnection gets a new ID;
/// browser slots and native device indices can be reused without aliasing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GamepadId(pub u64);

/// Semantic indices follow the Web Gamepad standard layout on every backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamepadMapping {
    Standard,
    Unmapped,
}

/// Stick axes are in [-1, 1], X right-positive and Y down-positive.
/// No gameplay dead zone, radial clamp, sensitivity curve, or inversion is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum GamepadAxis {
    LeftX,
    LeftY,
    RightX,
    RightY,
}

/// Standard physical positions, independent of a manufacturer's printed labels.
/// Triggers are analog buttons in [0, 1]; the D-pad is four buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum GamepadButton {
    South,
    East,
    West,
    North,
    LeftShoulder,
    RightShoulder,
    LeftTrigger,
    RightTrigger,
    Select,
    Start,
    LeftStick,
    RightStick,
    DpadUp,
    DpadDown,
    DpadLeft,
    DpadRight,
    Guide,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GamepadButtonState {
    pub value: f32,
    pub pressed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GamepadState {
    pub id: GamepadId,
    pub name: String,
    pub mapping: GamepadMapping,
    /// Raw normalized indices, also available for unmapped devices.
    pub axes: BTreeMap<u32, f32>,
    pub buttons: BTreeMap<u32, GamepadButtonState>,
}

impl GamepadState {
    /// Unmapped devices have no guaranteed semantic layout; use raw indices.
    pub fn axis(&self, axis: GamepadAxis) -> f32 {
        if self.mapping != GamepadMapping::Standard {
            return 0.0;
        }
        self.axes.get(&(axis as u32)).copied().unwrap_or(0.0)
    }
    pub fn button(&self, button: GamepadButton) -> GamepadButtonState {
        if self.mapping != GamepadMapping::Standard {
            return GamepadButtonState::default();
        }
        self.buttons
            .get(&(button as u32))
            .copied()
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GamepadEvent {
    Connected {
        id: GamepadId,
        name: String,
        mapping: GamepadMapping,
    },
    Disconnected {
        id: GamepadId,
    },
    Axis {
        id: GamepadId,
        axis: u32,
        value: f32,
    },
    Button {
        id: GamepadId,
        button: u32,
        value: f32,
        pressed: bool,
    },
}

/// A contact identity, unique for its lifetime even when a platform reuses IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TouchId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TouchPhase {
    Started,
    Moved,
    Ended,
    Cancelled,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TouchInput {
    pub id: TouchId,
    pub phase: TouchPhase,
    /// Physical pixels relative to the viewport's top-left, X right, Y down.
    /// Divide by WindowInput::scale_factor for logical pixels. May be outside
    /// the viewport while captured. A viewport/DPI change cancels active drags.
    pub position: [f64; 2],
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputOwner {
    Game,
    Ui,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TouchContact {
    pub id: TouchId,
    pub position: [f64; 2],
    pub origin: [f64; 2],
    /// Fixed at Started; crossing UI or lifting another finger cannot transfer it.
    pub owner: InputOwner,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputCancelReason {
    FocusLost,
    Hidden,
    PointerLeft,
    Platform,
    ViewportChanged,
    UiCaptured,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputEvent {
    Key {
        key: KeyCode,
        pressed: bool,
    },
    MouseButton {
        button: u8,
        pressed: bool,
        owner: InputOwner,
    },
    Touch {
        contact: TouchContact,
        phase: TouchPhase,
    },
    Gamepad(GamepadEvent),
    Focused(bool),
    Cancelled(InputCancelReason),
}

/// An independent event subscription. Drain from Update OR FixedUpdate, not
/// both. Events survive any number of frames with zero fixed steps and are
/// delivered once to each subscriber, in order. Subscribe before injecting
/// input; use the current WindowInput state to initialize a late subscriber.
///
/// A reader retains its unread events until drained or dropped (no frame TTL
/// or silent overflow). Drop readers for inactive consumers. During an extended
/// pause, drain in a frame system if retaining all transitions is unnecessary.
#[derive(Debug)]
pub struct InputEventReader(Arc<Mutex<VecDeque<InputEvent>>>);
impl InputEventReader {
    pub fn drain(&mut self) -> Vec<InputEvent> {
        self.0
            .lock()
            .expect("input reader lock poisoned")
            .drain(..)
            .collect()
    }
}

#[derive(Debug, Default)]
pub struct InputEventStream {
    readers: Vec<Weak<Mutex<VecDeque<InputEvent>>>>,
}
// A cloned state is a snapshot, not another producer for existing subscribers.
impl Clone for InputEventStream {
    fn clone(&self) -> Self {
        Self::default()
    }
}
impl InputEventStream {
    pub fn subscribe(&mut self) -> InputEventReader {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        self.readers.push(Arc::downgrade(&queue));
        InputEventReader(queue)
    }
    pub fn send(&mut self, event: InputEvent) {
        self.readers.retain(|reader| {
            if let Some(queue) = reader.upgrade() {
                queue
                    .lock()
                    .expect("input reader lock poisoned")
                    .push_back(event.clone());
                true
            } else {
                false
            }
        });
    }
}
