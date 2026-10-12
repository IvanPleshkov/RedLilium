//! Platform discovery is owned by the host, not by ECS gameplay.
use redlilium_core::input::{GamepadEvent, GamepadId, GamepadMapping};
use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct GamepadBackend {
    gilrs: Option<gilrs::Gilrs>,
    ids: HashMap<gilrs::GamepadId, GamepadId>,
    next_id: u64,
}
#[cfg(not(target_arch = "wasm32"))]
impl GamepadBackend {
    pub fn new() -> Self {
        let gilrs = gilrs::GilrsBuilder::new()
            .with_default_filters(false)
            .with_force_feedback(false)
            .build()
            .map_err(|e| log::warn!("Gamepad initialization failed: {e}"))
            .ok();
        Self {
            gilrs,
            ids: HashMap::new(),
            next_id: 1,
        }
    }
    pub fn poll(&mut self) -> Vec<GamepadEvent> {
        let mut out = Vec::new();
        let Some(gilrs) = &mut self.gilrs else {
            return out;
        };
        // Snapshot after EACH event, not only after draining the backend: a
        // down/up pair received between render frames must remain two events.
        while let Some(event) = gilrs.next_event() {
            if matches!(event.event, gilrs::EventType::Disconnected) {
                if let Some(id) = self.ids.remove(&event.id) {
                    out.push(GamepadEvent::Disconnected { id });
                }
            } else {
                sample_native(
                    gilrs.gamepad(event.id),
                    &mut self.ids,
                    &mut self.next_id,
                    &mut out,
                );
            }
        }
        // Also discovers controllers already attached at boot and refreshes
        // current state after focus returns, without requiring a new axis event.
        for (_, pad) in gilrs.gamepads() {
            sample_native(pad, &mut self.ids, &mut self.next_id, &mut out);
        }
        out
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn sample_native(
    pad: gilrs::Gamepad<'_>,
    ids: &mut HashMap<gilrs::GamepadId, GamepadId>,
    next: &mut u64,
    out: &mut Vec<GamepadEvent>,
) {
    use gilrs::{Axis as A, Button as B};
    if !pad.is_connected() {
        return;
    }
    let mapping = if matches!(pad.mapping_source(), gilrs::MappingSource::None) {
        GamepadMapping::Unmapped
    } else {
        GamepadMapping::Standard
    };
    let id = *ids.entry(pad.id()).or_insert_with(|| {
        let id = GamepadId(*next);
        *next += 1;
        out.push(GamepadEvent::Connected {
            id,
            name: pad.name().into(),
            mapping,
        });
        id
    });
    if mapping == GamepadMapping::Unmapped {
        for (axis, value) in pad.state().axes() {
            out.push(GamepadEvent::Axis {
                id,
                axis: axis.into_u32(),
                value: value.value(),
            });
        }
        for (button, value) in pad.state().buttons() {
            out.push(GamepadEvent::Button {
                id,
                button: button.into_u32(),
                value: value.value(),
                pressed: value.is_pressed(),
            });
        }
        return;
    }
    for (axis, value) in [
        pad.value(A::LeftStickX),
        -pad.value(A::LeftStickY),
        pad.value(A::RightStickX),
        -pad.value(A::RightStickY),
    ]
    .into_iter()
    .enumerate()
    {
        out.push(GamepadEvent::Axis {
            id,
            axis: axis as u32,
            value,
        });
    }
    for (button, native) in [
        B::South,
        B::East,
        B::West,
        B::North,
        B::LeftTrigger,
        B::RightTrigger,
        B::LeftTrigger2,
        B::RightTrigger2,
        B::Select,
        B::Start,
        B::LeftThumb,
        B::RightThumb,
        B::DPadUp,
        B::DPadDown,
        B::DPadLeft,
        B::DPadRight,
        B::Mode,
    ]
    .into_iter()
    .enumerate()
    {
        // Some native devices expose their digital D-pad as a pair of axes.
        let axis_pressed = match native {
            B::DPadUp => pad.value(A::DPadY) > 0.0,
            B::DPadDown => pad.value(A::DPadY) < 0.0,
            B::DPadLeft => pad.value(A::DPadX) < 0.0,
            B::DPadRight => pad.value(A::DPadX) > 0.0,
            _ => false,
        };
        let pressed = pad.is_pressed(native) || axis_pressed;
        let value = if axis_pressed {
            1.0
        } else {
            pad.button_data(native).map(|v| v.value()).unwrap_or(0.0)
        };
        out.push(GamepadEvent::Button {
            id,
            button: button as u32,
            value,
            pressed,
        });
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) struct GamepadBackend {
    ids: HashMap<u32, (GamepadId, String)>,
    next_id: u64,
}
#[cfg(target_arch = "wasm32")]
impl GamepadBackend {
    pub fn new() -> Self {
        Self {
            ids: HashMap::new(),
            next_id: 1,
        }
    }
    pub fn poll(&mut self) -> Vec<GamepadEvent> {
        use wasm_bindgen::JsCast;
        let mut out = Vec::new();
        // Poll on every real frame: controllers become visible after the
        // browser's activation gesture, which need not emit a connect event.
        let pads = web_sys::window().and_then(|w| w.navigator().get_gamepads().ok());
        let mut seen = Vec::new();
        if let Some(pads) = pads {
            for pad in pads
                .iter()
                .filter_map(|p| p.dyn_into::<web_sys::Gamepad>().ok())
                .filter(|p| p.connected())
            {
                let index = pad.index();
                let name = pad.id();
                seen.push(index);
                if self
                    .ids
                    .get(&index)
                    .is_some_and(|(_, previous)| previous != &name)
                {
                    out.push(GamepadEvent::Disconnected {
                        id: self.ids.remove(&index).unwrap().0,
                    });
                }
                let id = self
                    .ids
                    .entry(index)
                    .or_insert_with(|| {
                        let id = GamepadId(self.next_id);
                        self.next_id += 1;
                        out.push(GamepadEvent::Connected {
                            id,
                            name: name.clone(),
                            mapping: if pad.mapping() == web_sys::GamepadMappingType::Standard {
                                GamepadMapping::Standard
                            } else {
                                GamepadMapping::Unmapped
                            },
                        });
                        (id, name)
                    })
                    .0;
                for (axis, value) in pad.axes().iter().enumerate() {
                    out.push(GamepadEvent::Axis {
                        id,
                        axis: axis as u32,
                        value: value.as_f64().unwrap_or(0.0) as f32,
                    });
                }
                for (button, value) in pad.buttons().iter().enumerate() {
                    if let Ok(value) = value.dyn_into::<web_sys::GamepadButton>() {
                        out.push(GamepadEvent::Button {
                            id,
                            button: button as u32,
                            value: value.value() as f32,
                            pressed: value.pressed(),
                        });
                    }
                }
            }
        }
        self.ids.retain(|index, (id, _)| {
            if seen.contains(index) {
                true
            } else {
                out.push(GamepadEvent::Disconnected { id: *id });
                false
            }
        });
        out
    }
    /// A browser may reuse a slot for the same model between animation frames.
    pub fn disconnected(&mut self, index: u32) -> Option<GamepadEvent> {
        self.ids
            .remove(&index)
            .map(|(id, _)| GamepadEvent::Disconnected { id })
    }
}
