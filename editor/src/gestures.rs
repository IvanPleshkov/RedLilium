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
    /// An active viewport tool owns the press: the release is delivered to
    /// the tool as a click; selection and gizmo are untouched.
    ToolOwns,
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
    /// The active viewport tool owned the press: deliver a click at the
    /// release cursor (window space, physical pixels).
    ToolClick { x: f32, y: f32 },
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
    /// The current primary press was granted to the active viewport tool.
    tool_armed: bool,
    /// Secondary-button press position; a clean (no-drag) release opens the
    /// viewport context menu. A drag is the fly-camera's, not ours.
    secondary_start: Option<[f32; 2]>,
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
        tool_active: bool,
    ) -> PressRouting {
        if !in_scene_view {
            return PressRouting::Outside;
        }
        if egui_wants_pointer {
            return PressRouting::EguiOwns;
        }
        if tool_active {
            // The tool outranks the gizmo: while a mode like "connect roads"
            // is live, every scene click belongs to it.
            self.tool_armed = true;
            return PressRouting::ToolOwns;
        }
        if gizmo_wants_cursor {
            // Manipulation: leave the selection alone, arm no gesture.
            return PressRouting::GizmoOwns;
        }
        self.drag_start = Some(self.cursor_pos);
        self.dragging_box = false;
        PressRouting::SceneGesture
    }

    /// Route a secondary-button press: arms the context-menu candidate when
    /// the press lands in the scene view and egui does not want the pointer.
    pub fn on_secondary_press(&mut self, in_scene_view: bool, egui_wants_pointer: bool) {
        self.secondary_start = (in_scene_view && !egui_wants_pointer).then_some(self.cursor_pos);
    }

    /// Resolve a secondary-button release: `Some(position)` when it was a
    /// clean click (no drag past the threshold) — open the context menu
    /// there. A drag was the fly-camera navigating; no menu.
    pub fn on_secondary_release(&mut self) -> Option<[f32; 2]> {
        let start = self.secondary_start.take()?;
        let dx = self.cursor_pos[0] - start[0];
        let dy = self.cursor_pos[1] - start[1];
        ((dx * dx + dy * dy).sqrt() <= BOX_DRAG_THRESHOLD_PX).then_some(self.cursor_pos)
    }

    /// Resolve a primary-button release. Always disarms the gesture.
    pub fn on_release(&mut self, gizmo_wants_cursor: bool) -> ReleaseAction {
        let start = self.drag_start.take();
        let was_box = std::mem::take(&mut self.dragging_box);
        if std::mem::take(&mut self.tool_armed) {
            return ReleaseAction::ToolClick {
                x: self.cursor_pos[0],
                y: self.cursor_pos[1],
            };
        }
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
        g.on_press(true, false, gizmo, false)
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
        assert_eq!(g.on_press(true, true, false, false), PressRouting::EguiOwns);
        assert_eq!(g.on_release(false), ReleaseAction::Nothing);
    }

    #[test]
    fn press_outside_scene_view_is_ignored() {
        let mut g = SceneGestures::default();
        assert_eq!(
            g.on_press(false, false, false, false),
            PressRouting::Outside
        );
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

#[cfg(test)]
mod tool_and_menu_tests {
    use super::*;

    #[test]
    fn active_tool_owns_click_over_gizmo() {
        let mut g = SceneGestures::default();
        g.on_move(50.0, 60.0);
        // Even with a hovered gizmo handle, the active tool wins.
        assert_eq!(g.on_press(true, false, true, true), PressRouting::ToolOwns);
        assert_eq!(
            g.on_release(true),
            ReleaseAction::ToolClick { x: 50.0, y: 60.0 }
        );
        // Tool off again: normal gizmo precedence is restored.
        assert_eq!(
            g.on_press(true, false, true, false),
            PressRouting::GizmoOwns
        );
    }

    #[test]
    fn clean_secondary_click_opens_menu() {
        let mut g = SceneGestures::default();
        g.on_move(120.0, 80.0);
        g.on_secondary_press(true, false);
        g.on_move(122.0, 81.0); // under the drag threshold
        assert_eq!(g.on_secondary_release(), Some([122.0, 81.0]));
    }

    #[test]
    fn secondary_drag_is_fly_camera_not_menu() {
        let mut g = SceneGestures::default();
        g.on_move(120.0, 80.0);
        g.on_secondary_press(true, false);
        g.on_move(220.0, 160.0); // fly-camera look
        assert_eq!(g.on_secondary_release(), None);
    }

    #[test]
    fn secondary_press_over_egui_never_arms() {
        let mut g = SceneGestures::default();
        g.on_secondary_press(true, true);
        assert_eq!(g.on_secondary_release(), None);
    }
}
