//! Scene-view mouse gesture routing: who owns a click, and what a release
//! resolves to.
//!
//! The decisions here used to live as a nested if-tree inside the shell's
//! `on_mouse_button` — which is exactly where the "gizmo press cleared the
//! selection" bug hid (refs #85). The routing is now a pure function over a
//! small state struct, unit-tested headlessly; the shell gathers the flags,
//! calls [`SceneGestures`], and acts on the returned decision.
//!
//! Ownership precedence for a primary-button press over the scene view:
//!
//! 1. **egui** — a popup/panel under the cursor consumes the click.
//! 2. **gizmo** — a hovered handle turns the press into a manipulation
//!    (no selection clear, no pick; the drag flows via `WindowInput`).
//! 3. **scene gesture** — otherwise the press arms select/box-select:
//!    the selection clears immediately, and the release resolves to a
//!    GPU pick (small movement) or a box selection (drag past threshold).

/// Pixels of cursor travel that turn a press-hold into a box selection.
const BOX_DRAG_THRESHOLD_PX: f32 = 5.0;

/// What a primary-button press resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressRouting {
    /// egui owns the pointer (popup over the scene, panel focus, …).
    EguiOwns,
    /// A gizmo handle is hovered: the press is a manipulation. The shell
    /// must not clear the selection or arm any select gesture.
    GizmoOwns,
    /// The press arms a scene select gesture (clear selection now; the
    /// release decides between click-pick and box-select).
    SceneGesture,
    /// The press is outside the scene view — nothing to do.
    Outside,
}

/// What a primary-button release resolves to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReleaseAction {
    /// Select every entity intersecting the rectangle (physical pixels).
    BoxSelect { start: [f32; 2], end: [f32; 2] },
    /// GPU-pick the entity under the cursor (physical pixels).
    ClickPick { x: u32, y: u32 },
    /// The gizmo owned this gesture (its own MergeBarrier seals the drag —
    /// the shell must NOT break undo merging here).
    GizmoOwned,
    /// No armed gesture (press was consumed elsewhere or out of view).
    Nothing,
}

/// The scene view's gesture state machine. Owns the press/drag bookkeeping
/// that used to live as loose fields on the shell.
#[derive(Debug, Default)]
pub struct SceneGestures {
    cursor_pos: [f32; 2],
    drag_start: Option<[f32; 2]>,
    dragging_box: bool,
}

impl SceneGestures {
    /// Current cursor position (window space, physical pixels).
    pub fn cursor_pos(&self) -> [f32; 2] {
        self.cursor_pos
    }

    /// The armed box-selection rectangle, when a drag is past the threshold:
    /// `(start, current)`. The shell draws the marquee from this.
    pub fn box_rect(&self) -> Option<([f32; 2], [f32; 2])> {
        self.dragging_box
            .then_some(())
            .and(self.drag_start)
            .map(|s| (s, self.cursor_pos))
    }

    /// Track cursor movement; arms box selection once a pressed drag moves
    /// past the threshold.
    pub fn on_move(&mut self, x: f32, y: f32) {
        self.cursor_pos = [x, y];
        if let Some(start) = self.drag_start
            && !self.dragging_box
        {
            let dx = x - start[0];
            let dy = y - start[1];
            if (dx * dx + dy * dy).sqrt() > BOX_DRAG_THRESHOLD_PX {
                self.dragging_box = true;
            }
        }
    }

    /// Route a primary-button press. `gizmo_wants_cursor` must be sampled
    /// BEFORE the press mutates anything — hover state is maintained by
    /// cursor moves and is current here.
    pub fn on_press(
        &mut self,
        in_scene_view: bool,
        egui_wants_pointer: bool,
        gizmo_wants_cursor: bool,
    ) -> PressRouting {
        if !in_scene_view {
            return PressRouting::Outside;
        }
        if egui_wants_pointer {
            return PressRouting::EguiOwns;
        }
        if gizmo_wants_cursor {
            // Manipulation: leave the selection alone, arm no gesture.
            return PressRouting::GizmoOwns;
        }
        self.drag_start = Some(self.cursor_pos);
        self.dragging_box = false;
        PressRouting::SceneGesture
    }

    /// Resolve a primary-button release. Always disarms the gesture.
    pub fn on_release(&mut self, gizmo_wants_cursor: bool) -> ReleaseAction {
        let start = self.drag_start.take();
        let was_box = std::mem::take(&mut self.dragging_box);
        if gizmo_wants_cursor {
            // A live gizmo drag ends here; its final delta actions drain
            // AFTER this event, so undo-merge must stay open for them.
            return ReleaseAction::GizmoOwned;
        }
        match (start, was_box) {
            (Some(start), true) => ReleaseAction::BoxSelect {
                start,
                end: self.cursor_pos,
            },
            (Some(_), false) => ReleaseAction::ClickPick {
                x: self.cursor_pos[0].max(0.0) as u32,
                y: self.cursor_pos[1].max(0.0) as u32,
            },
            (None, _) => ReleaseAction::Nothing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pressed(g: &mut SceneGestures, gizmo: bool) -> PressRouting {
        g.on_press(true, false, gizmo)
    }

    #[test]
    fn click_without_movement_is_a_pick() {
        let mut g = SceneGestures::default();
        g.on_move(100.0, 50.0);
        assert_eq!(pressed(&mut g, false), PressRouting::SceneGesture);
        g.on_move(102.0, 51.0); // under the threshold
        assert_eq!(
            g.on_release(false),
            ReleaseAction::ClickPick { x: 102, y: 51 }
        );
    }

    #[test]
    fn drag_past_threshold_is_a_box_select() {
        let mut g = SceneGestures::default();
        g.on_move(100.0, 50.0);
        pressed(&mut g, false);
        g.on_move(160.0, 90.0);
        assert!(g.box_rect().is_some(), "marquee armed");
        assert_eq!(
            g.on_release(false),
            ReleaseAction::BoxSelect {
                start: [100.0, 50.0],
                end: [160.0, 90.0]
            }
        );
    }

    /// The #85 regression, pinned as a pure-logic test: a press on a hovered
    /// gizmo handle must neither clear the selection (SceneGesture implies
    /// that) nor resolve to a pick on release.
    #[test]
    fn gizmo_hover_owns_the_whole_gesture() {
        let mut g = SceneGestures::default();
        g.on_move(100.0, 50.0);
        assert_eq!(pressed(&mut g, true), PressRouting::GizmoOwns);
        g.on_move(200.0, 50.0); // dragging the handle
        assert!(g.box_rect().is_none(), "no marquee during a gizmo drag");
        assert_eq!(g.on_release(true), ReleaseAction::GizmoOwned);
        // The next ordinary click behaves normally again.
        assert_eq!(pressed(&mut g, false), PressRouting::SceneGesture);
    }

    #[test]
    fn egui_popup_swallows_the_press() {
        let mut g = SceneGestures::default();
        assert_eq!(g.on_press(true, true, false), PressRouting::EguiOwns);
        assert_eq!(g.on_release(false), ReleaseAction::Nothing);
    }

    #[test]
    fn press_outside_scene_view_is_ignored() {
        let mut g = SceneGestures::default();
        assert_eq!(g.on_press(false, false, false), PressRouting::Outside);
        assert_eq!(g.on_release(false), ReleaseAction::Nothing);
    }

    /// Release with the gizmo still hot disarms any accidentally armed
    /// gesture instead of leaking it into the next click.
    #[test]
    fn gizmo_release_disarms_stale_gesture() {
        let mut g = SceneGestures::default();
        g.on_move(10.0, 10.0);
        pressed(&mut g, false); // armed
        assert_eq!(g.on_release(true), ReleaseAction::GizmoOwned);
        assert_eq!(g.on_release(false), ReleaseAction::Nothing, "disarmed");
    }
}
