//! DOM events winit 0.30 does not forward. Owned listeners are removed on drop.
use std::{cell::RefCell, rc::Rc};
use wasm_bindgen::{JsCast, closure::Closure};

pub(crate) use super::web_pointer::WebInputEvent;
use super::web_pointer::WebPointerQueue;
type Listener = (
    web_sys::EventTarget,
    &'static str,
    Closure<dyn FnMut(web_sys::Event)>,
);
pub struct WebInputBridge {
    pending: Rc<RefCell<WebPointerQueue>>,
    listeners: Vec<Listener>,
}
impl WebInputBridge {
    pub fn new(canvas: &web_sys::HtmlCanvasElement) -> Self {
        // Engine-owned game surface: the browser must not turn a drag into
        // scrolling/zooming, nor generate compatibility mouse events.
        let _ = canvas.style().set_property("touch-action", "none");
        let mut this = Self {
            pending: Rc::default(),
            listeners: Vec::new(),
        };
        let target: web_sys::EventTarget = canvas.clone().into();
        let captured = canvas.clone();
        let queue = this.pending.clone();
        this.listen(target.clone(), "pointerdown", move |event| {
            if let Some(event) = event.dyn_ref::<web_sys::PointerEvent>() {
                queue.borrow_mut().down(event.pointer_id() as u64);
                if event.pointer_type() == "touch" {
                    event.prevent_default();
                    let _ = captured.focus();
                    let _ = captured.set_pointer_capture(event.pointer_id());
                }
            }
        });
        for name in [
            "pointerup",
            "pointercancel",
            "lostpointercapture",
            "pointermove",
        ] {
            let queue = this.pending.clone();
            this.listen(target.clone(), name, move |event| {
                if let Some(event) = event.dyn_ref::<web_sys::PointerEvent>() {
                    let id = event.pointer_id() as u64;
                    let touch = event.pointer_type() == "touch";
                    let mut queue = queue.borrow_mut();
                    match name {
                        "pointerup" => queue.up(id),
                        "pointercancel" => queue.cancel(id, touch),
                        "lostpointercapture" => queue.capture_lost(id, touch),
                        // winit forwards chorded transitions as MouseInput.
                        "pointermove" if event.button() != -1 => queue.observe_transition(),
                        _ => {}
                    }
                }
            });
        }
        if let Some(window) = web_sys::window() {
            let queue = this.pending.clone();
            this.listen(window.into(), "gamepaddisconnected", move |event| {
                if let Some(event) = event.dyn_ref::<web_sys::GamepadEvent>()
                    && let Some(gamepad) = event.gamepad()
                {
                    queue
                        .borrow_mut()
                        .push(WebInputEvent::GamepadDisconnected(gamepad.index()));
                }
            });
        }
        this
    }
    fn listen(
        &mut self,
        target: web_sys::EventTarget,
        name: &'static str,
        callback: impl FnMut(web_sys::Event) + 'static,
    ) {
        let callback = Closure::wrap(Box::new(callback) as Box<dyn FnMut(web_sys::Event)>);
        if target
            .add_event_listener_with_callback_and_bool(
                name,
                callback.as_ref().unchecked_ref(),
                true,
            )
            .is_ok()
        {
            self.listeners.push((target, name, callback));
        }
    }
    pub fn delivered_transition(&self) {
        self.pending.borrow_mut().delivered_transition();
    }
    pub fn drain(&self) -> Vec<WebInputEvent> {
        self.pending.borrow_mut().drain_ready()
    }
}
impl Drop for WebInputBridge {
    fn drop(&mut self) {
        for (target, name, callback) in &self.listeners {
            let _ = target.remove_event_listener_with_callback_and_bool(
                name,
                callback.as_ref().unchecked_ref(),
                true,
            );
        }
    }
}
