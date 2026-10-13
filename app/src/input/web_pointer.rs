//! Merge DOM-only cancellations with winit's possibly buffered button/contact
//! transitions. DOM listeners run in capture phase, before winit's listeners.
//! A cancellation cannot overtake the input transitions that preceded it.
use std::collections::{HashSet, VecDeque};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WebInputEvent {
    PointerCancelled { id: u64, touch: bool },
    GamepadDisconnected(u32),
}

#[derive(Default)]
pub(super) struct WebPointerQueue {
    active: HashSet<u64>,
    observed: u64,
    delivered: u64,
    pending: VecDeque<(u64, WebInputEvent)>,
}
impl WebPointerQueue {
    /// One DOM event which winit translates to MouseInput or a non-Moved Touch.
    /// Includes chorded button transitions, carried by pointermove in the DOM.
    pub fn observe_transition(&mut self) {
        self.observed += 1;
    }
    pub fn down(&mut self, id: u64) {
        self.observe_transition();
        self.active.insert(id);
    }
    pub fn up(&mut self, id: u64) {
        self.observe_transition();
        self.active.remove(&id);
    }
    pub fn cancel(&mut self, id: u64, touch: bool) {
        let active = self.active.remove(&id);
        if touch {
            // winit already forwards Touch(Cancelled), including its ordering.
            self.observe_transition();
        } else if active {
            // winit 0.30 does not forward mouse/pen pointercancel.
            self.push(WebInputEvent::PointerCancelled { id, touch });
        }
    }
    pub fn capture_lost(&mut self, id: u64, touch: bool) {
        // Normal pointerup and pointercancel have already ended this lifetime.
        // Do not infer its cause from `buttons`: both can report zero.
        if self.active.remove(&id) {
            self.push(WebInputEvent::PointerCancelled { id, touch });
        }
    }
    pub fn push(&mut self, event: WebInputEvent) {
        self.pending.push_back((self.observed, event));
    }
    /// Called after the corresponding winit transition reaches the handler,
    /// even if the handler ignores it while unfocused or before initialization.
    pub fn delivered_transition(&mut self) {
        self.delivered += 1;
        debug_assert!(
            self.delivered <= self.observed,
            "unobserved web pointer transition"
        );
    }
    pub fn drain_ready(&mut self) -> Vec<WebInputEvent> {
        let mut events = Vec::new();
        while self
            .pending
            .front()
            .is_some_and(|(after, _)| *after <= self.delivered)
        {
            events.push(self.pending.pop_front().unwrap().1);
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn non_pointer_events_share_order_without_creating_a_barrier() {
        let mut q = WebPointerQueue::default();
        q.push(WebInputEvent::GamepadDisconnected(3));
        assert_eq!(q.drain_ready(), vec![WebInputEvent::GamepadDisconnected(3)]);
        q.down(1);
        q.push(WebInputEvent::GamepadDisconnected(4));
        assert!(q.drain_ready().is_empty());
        q.delivered_transition();
        assert_eq!(q.drain_ready(), vec![WebInputEvent::GamepadDisconnected(4)]);
    }
    #[test]
    fn completed_click_and_touch_never_cancel_even_before_winit_delivery() {
        for touch in [false, true] {
            let mut q = WebPointerQueue::default();
            q.down(1);
            q.up(1);
            q.capture_lost(1, touch);
            assert!(q.drain_ready().is_empty());
            q.delivered_transition();
            q.delivered_transition();
            assert!(q.drain_ready().is_empty());
        }
    }
    #[test]
    fn loss_waits_for_its_press_but_precedes_the_next_gesture() {
        for touch in [false, true] {
            let mut q = WebPointerQueue::default();
            q.down(1);
            q.capture_lost(1, touch);
            q.down(1);
            q.up(1);
            q.capture_lost(1, touch);
            assert!(q.drain_ready().is_empty());
            q.delivered_transition();
            assert_eq!(
                q.drain_ready(),
                vec![WebInputEvent::PointerCancelled { id: 1, touch }]
            );
            q.delivered_transition();
            q.delivered_transition();
            assert!(q.drain_ready().is_empty());
        }
    }
    #[test]
    fn actual_mouse_cancel_deduplicates_loss_and_late_release() {
        let mut q = WebPointerQueue::default();
        q.down(1);
        q.cancel(1, false);
        q.capture_lost(1, false);
        q.up(1);
        assert!(q.drain_ready().is_empty());
        q.delivered_transition();
        assert_eq!(
            q.drain_ready(),
            vec![WebInputEvent::PointerCancelled {
                id: 1,
                touch: false
            }]
        );
        q.delivered_transition();
        assert!(q.drain_ready().is_empty());
    }
    #[test]
    fn touch_cancel_uses_winit_and_capture_loss_only_cancels_active_contact() {
        let mut q = WebPointerQueue::default();
        q.down(1);
        q.down(2);
        q.cancel(1, true);
        q.capture_lost(1, true);
        q.capture_lost(2, true);
        for _ in 0..2 {
            q.delivered_transition();
            assert!(q.drain_ready().is_empty());
        }
        q.delivered_transition();
        assert_eq!(
            q.drain_ready(),
            vec![WebInputEvent::PointerCancelled { id: 2, touch: true }]
        );
    }
    #[test]
    fn chorded_buttons_keep_capture_active_until_final_up() {
        let mut q = WebPointerQueue::default();
        q.down(1);
        q.observe_transition();
        q.observe_transition();
        q.capture_lost(1, false);
        for _ in 0..2 {
            q.delivered_transition();
            assert!(q.drain_ready().is_empty());
        }
        q.delivered_transition();
        assert_eq!(
            q.drain_ready(),
            vec![WebInputEvent::PointerCancelled {
                id: 1,
                touch: false
            }]
        );
    }
}
