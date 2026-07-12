//! The translate-gizmo interaction state machine: hover → drag → commit.
//!
//! The gizmo never mutates a scene. It consumes camera + cursor state each
//! frame and emits [`GizmoEvent`]s; the consumer applies the deltas to its
//! own model (an ECS `Transform` through an undoable edit action, a DSL
//! literal in source code, …) and re-feeds the target position via
//! [`set_target`](TranslateGizmo::set_target).

use std::collections::VecDeque;

use redlilium_core::math::Vec3;

use crate::math::{
    GizmoCamera, Ray, closest_ray_line_params, ray_capsule_hit, ray_plane_intersect, ray_quad_hit,
    screen_scale,
};

/// A pickable part of the translate gizmo. Axes translate along one world
/// axis; planes translate in the two spanned axes at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handle {
    AxisX,
    AxisY,
    AxisZ,
    PlaneXY,
    PlaneXZ,
    PlaneYZ,
}

impl Handle {
    /// The unit direction of an axis handle, or the plane normal.
    pub fn primary_dir(self) -> Vec3 {
        match self {
            Handle::AxisX => Vec3::new(1.0, 0.0, 0.0),
            Handle::AxisY => Vec3::new(0.0, 1.0, 0.0),
            Handle::AxisZ | Handle::PlaneXY => Vec3::new(0.0, 0.0, 1.0),
            Handle::PlaneXZ => Vec3::new(0.0, 1.0, 0.0),
            Handle::PlaneYZ => Vec3::new(1.0, 0.0, 0.0),
        }
    }

    /// Whether this is a plane handle.
    pub fn is_plane(self) -> bool {
        matches!(self, Handle::PlaneXY | Handle::PlaneXZ | Handle::PlaneYZ)
    }

    const ALL: [Handle; 6] = [
        Handle::AxisX,
        Handle::AxisY,
        Handle::AxisZ,
        Handle::PlaneXY,
        Handle::PlaneXZ,
        Handle::PlaneYZ,
    ];
}

/// Cursor input for one frame, in viewport pixels (origin top-left).
/// `None` position means the cursor is outside the viewport.
#[derive(Debug, Clone, Copy, Default)]
pub struct CursorState {
    pub position: Option<(f32, f32)>,
    /// Primary button held this frame.
    pub pressed: bool,
}

/// What the gizmo did this frame. Consumers poll these after
/// [`frame`](TranslateGizmo::frame).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GizmoEvent {
    /// The primary button went down on a handle.
    DragStart { handle: Handle },
    /// The cursor moved during a drag: the world-space translation since the
    /// previous frame. Zero-length deltas are not emitted.
    DragDelta { handle: Handle, world_delta: Vec3 },
    /// The primary button was released: the total translation of the whole
    /// drag (the consumer's commit point — e.g. the single undo entry).
    DragEnd { handle: Handle, total_delta: Vec3 },
}

/// Tuning knobs. The defaults give a gizmo roughly 15% of the eye-target
/// distance long, with generously forgiving pick shapes.
#[derive(Debug, Clone, Copy)]
pub struct GizmoConfig {
    /// Gizmo length as a fraction of eye–target distance (screen-constant).
    pub size_factor: f32,
    /// Pick-capsule radius around axis arrows, in gizmo-local units.
    pub axis_pick_radius: f32,
    /// Plane-handle quad bounds in gizmo-local units (`min..max` on both axes).
    pub plane_min: f32,
    pub plane_max: f32,
    /// Axis pick segment in gizmo-local units.
    pub axis_pick_min: f32,
    pub axis_pick_max: f32,
}

impl Default for GizmoConfig {
    fn default() -> Self {
        Self {
            size_factor: 0.15,
            axis_pick_radius: 0.1,
            plane_min: 0.3,
            plane_max: 0.6,
            axis_pick_min: 0.2,
            axis_pick_max: 1.1,
        }
    }
}

/// The drag session: everything captured at DragStart that later frames
/// solve against. The anchor keeps the grab point stable — the gizmo does
/// not "jump" to put its origin under the cursor.
struct DragSession {
    handle: Handle,
    /// Target position when the drag started.
    start_target: Vec3,
    /// Axis handles: the axis parameter under the cursor at drag start.
    /// Plane handles: the plane point under the cursor at drag start.
    axis_anchor: f32,
    plane_anchor: Vec3,
    /// Total translation applied so far.
    total: Vec3,
}

/// The interactive translate gizmo (#80). World-axis aligned (global mode).
///
/// Drive it once per frame with [`frame`](Self::frame), then drain
/// [`poll_event`](Self::poll_event). While a drag is live the gizmo tracks
/// its own position (`start + total`), so a consumer with write-back latency
/// (e.g. a source-code round-trip) still gets smooth dragging;
/// [`set_target`](Self::set_target) re-synchronizes it outside drags.
pub struct TranslateGizmo {
    config: GizmoConfig,
    target: Option<Vec3>,
    hovered: Option<Handle>,
    drag: Option<DragSession>,
    prev_pressed: bool,
    events: VecDeque<GizmoEvent>,
}

impl TranslateGizmo {
    pub fn new(config: GizmoConfig) -> Self {
        Self {
            config,
            target: None,
            hovered: None,
            drag: None,
            prev_pressed: false,
            events: VecDeque::new(),
        }
    }

    /// Where the gizmo sits, or `None` to hide it (no target selected).
    /// Ignored while a drag is live — the drag session owns the position
    /// until the button is released.
    pub fn set_target(&mut self, target: Option<Vec3>) {
        if self.drag.is_none() {
            self.target = target;
            if target.is_none() {
                self.hovered = None;
            }
        }
    }

    /// The position the gizmo renders at this frame (live drag included).
    pub fn effective_target(&self) -> Option<Vec3> {
        match (&self.drag, self.target) {
            (Some(drag), _) => Some(drag.start_target + drag.total),
            (None, t) => t,
        }
    }

    /// The handle under the cursor (when not dragging).
    pub fn hovered(&self) -> Option<Handle> {
        self.hovered
    }

    /// The handle being dragged.
    pub fn active(&self) -> Option<Handle> {
        self.drag.as_ref().map(|d| d.handle)
    }

    /// Whether the gizmo currently owns the cursor (a drag is live or a
    /// handle is hovered). Consumers use this to suppress their own
    /// click-through behavior (e.g. scene picking) while interacting.
    pub fn wants_cursor(&self) -> bool {
        self.drag.is_some() || self.hovered.is_some()
    }

    pub fn config(&self) -> &GizmoConfig {
        &self.config
    }

    /// Next event, in emission order.
    pub fn poll_event(&mut self) -> Option<GizmoEvent> {
        self.events.pop_front()
    }

    /// Advance the state machine one frame.
    pub fn frame(&mut self, camera: &GizmoCamera, cursor: CursorState) {
        let pressed_edge = cursor.pressed && !self.prev_pressed;
        let released = !cursor.pressed && self.prev_pressed;
        self.prev_pressed = cursor.pressed;

        let Some(base_target) = self.effective_target() else {
            self.hovered = None;
            return;
        };
        let ray = cursor.position.and_then(|p| camera.ray_from_screen(p));

        if let Some(drag) = &mut self.drag {
            // --- Live drag: solve the anchor against this frame's ray. ---
            if let Some(ray) = &ray {
                let scale = screen_scale(
                    camera,
                    drag.start_target + drag.total,
                    self.config.size_factor,
                );
                let new_total = solve_drag(drag, ray, scale);
                if let Some(new_total) = new_total {
                    let step = new_total - drag.total;
                    if step.norm() > 1e-6 {
                        drag.total = new_total;
                        self.events.push_back(GizmoEvent::DragDelta {
                            handle: drag.handle,
                            world_delta: step,
                        });
                    }
                }
            }
            if released {
                let drag = self.drag.take().expect("drag present");
                // The consumer owns the final position now; keep rendering at
                // the dragged spot until the next set_target.
                self.target = Some(drag.start_target + drag.total);
                self.events.push_back(GizmoEvent::DragEnd {
                    handle: drag.handle,
                    total_delta: drag.total,
                });
                self.hovered = None;
            }
            return;
        }

        // --- Idle: hover pick; a press on a handle starts a drag. ---
        let scale = screen_scale(camera, base_target, self.config.size_factor);
        self.hovered = ray.as_ref().and_then(|r| self.pick(r, base_target, scale));

        if pressed_edge
            && let (Some(handle), Some(ray)) = (self.hovered, &ray)
            && let Some(session) = start_drag(handle, ray, base_target, scale)
        {
            self.events.push_back(GizmoEvent::DragStart { handle });
            self.drag = Some(session);
        }
    }

    /// The closest handle under the ray, if any.
    fn pick(&self, ray: &Ray, target: Vec3, scale: f32) -> Option<Handle> {
        let c = &self.config;
        let mut best: Option<(f32, Handle)> = None;
        for handle in Handle::ALL {
            let t = if handle.is_plane() {
                let (u, v) = plane_axes(handle);
                ray_quad_hit(ray, target, u, v, c.plane_min * scale, c.plane_max * scale)
            } else {
                ray_capsule_hit(
                    ray,
                    target,
                    handle.primary_dir(),
                    c.axis_pick_min * scale,
                    c.axis_pick_max * scale,
                    c.axis_pick_radius * scale,
                )
            };
            if let Some(t) = t
                && best.is_none_or(|(bt, _)| t < bt)
            {
                best = Some((t, handle));
            }
        }
        best.map(|(_, h)| h)
    }
}

/// The two in-plane axes of a plane handle.
pub(crate) fn plane_axes(handle: Handle) -> (Vec3, Vec3) {
    match handle {
        Handle::PlaneXY => (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
        Handle::PlaneXZ => (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        Handle::PlaneYZ => (Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        _ => unreachable!("axis handles have no plane"),
    }
}

/// Capture the drag anchor. Returns `None` when the view direction makes the
/// handle degenerate (axis near-parallel to the ray, plane near-grazing) —
/// the press is ignored rather than producing wild deltas.
fn start_drag(handle: Handle, ray: &Ray, target: Vec3, _scale: f32) -> Option<DragSession> {
    let (axis_anchor, plane_anchor) = if handle.is_plane() {
        let p = ray_plane_intersect(ray, target, handle.primary_dir())?;
        (0.0, p)
    } else {
        let (_, t_axis) = closest_ray_line_params(ray, target, handle.primary_dir())?;
        (t_axis, Vec3::zeros())
    };
    Some(DragSession {
        handle,
        start_target: target,
        axis_anchor,
        plane_anchor,
        total: Vec3::zeros(),
    })
}

/// Solve this frame's total translation from the drag anchor. `None` keeps
/// the previous total (degenerate frame — e.g. the axis swung parallel to
/// the view ray mid-drag).
fn solve_drag(drag: &DragSession, ray: &Ray, _scale: f32) -> Option<Vec3> {
    if drag.handle.is_plane() {
        // The plane stays where the drag started (start_target), so the
        // anchor and the current hit share one plane and subtract cleanly.
        let p = ray_plane_intersect(ray, drag.start_target, drag.handle.primary_dir())?;
        Some(p - drag.plane_anchor)
    } else {
        let axis = drag.handle.primary_dir();
        let (_, t_axis) = closest_ray_line_params(ray, drag.start_target, axis)?;
        Some(axis * (t_axis - drag.axis_anchor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::{look_at_rh, perspective_rh};

    fn camera() -> GizmoCamera {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let view = look_at_rh(&eye, &Vec3::zeros(), &Vec3::new(0.0, 1.0, 0.0));
        let proj = perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
        GizmoCamera {
            view_proj: proj * view,
            eye,
            viewport: (1600.0, 900.0),
        }
    }

    /// Screen position over the middle of the +X arrow.
    fn over_x_arrow(cam: &GizmoCamera, gizmo: &TranslateGizmo) -> (f32, f32) {
        let scale = screen_scale(
            cam,
            gizmo.effective_target().unwrap(),
            gizmo.config().size_factor,
        );
        cam.project(Vec3::new(0.6 * scale, 0.0, 0.0)).unwrap()
    }

    fn drain(gizmo: &mut TranslateGizmo) -> Vec<GizmoEvent> {
        std::iter::from_fn(|| gizmo.poll_event()).collect()
    }

    #[test]
    fn hidden_gizmo_ignores_input() {
        let cam = camera();
        let mut gizmo = TranslateGizmo::new(GizmoConfig::default());
        gizmo.frame(
            &cam,
            CursorState {
                position: Some((800.0, 450.0)),
                pressed: true,
            },
        );
        assert!(gizmo.hovered().is_none());
        assert!(drain(&mut gizmo).is_empty());
    }

    #[test]
    fn hover_x_arrow() {
        let cam = camera();
        let mut gizmo = TranslateGizmo::new(GizmoConfig::default());
        gizmo.set_target(Some(Vec3::zeros()));
        let pos = over_x_arrow(&cam, &gizmo);
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(pos),
                pressed: false,
            },
        );
        assert_eq!(gizmo.hovered(), Some(Handle::AxisX));
        assert!(gizmo.wants_cursor());
    }

    #[test]
    fn drag_x_arrow_produces_x_deltas_and_commit() {
        let cam = camera();
        let mut gizmo = TranslateGizmo::new(GizmoConfig::default());
        gizmo.set_target(Some(Vec3::zeros()));

        // Press on the arrow.
        let start = over_x_arrow(&cam, &gizmo);
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(start),
                pressed: true,
            },
        );
        assert_eq!(
            drain(&mut gizmo),
            vec![GizmoEvent::DragStart {
                handle: Handle::AxisX
            }]
        );

        // Drag: move the cursor to where world (1, 0, 0) projects — the grab
        // point should follow, translating the target by ~+1 on X.
        let scale = screen_scale(&cam, Vec3::zeros(), gizmo.config().size_factor);
        let dest = cam.project(Vec3::new(0.6 * scale + 1.0, 0.0, 0.0)).unwrap();
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(dest),
                pressed: true,
            },
        );
        let events = drain(&mut gizmo);
        assert_eq!(events.len(), 1);
        let GizmoEvent::DragDelta {
            handle,
            world_delta,
        } = events[0]
        else {
            panic!("expected DragDelta, got {events:?}");
        };
        assert_eq!(handle, Handle::AxisX);
        assert!((world_delta.x - 1.0).abs() < 1e-2, "dx = {}", world_delta.x);
        assert!(world_delta.y.abs() < 1e-3 && world_delta.z.abs() < 1e-3);

        // Release: commit carries the total.
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(dest),
                pressed: false,
            },
        );
        let events = drain(&mut gizmo);
        assert_eq!(events.len(), 1);
        let GizmoEvent::DragEnd { total_delta, .. } = events[0] else {
            panic!("expected DragEnd, got {events:?}");
        };
        assert!((total_delta.x - 1.0).abs() < 1e-2);
        // The gizmo kept tracking its own position through the drag.
        assert!((gizmo.effective_target().unwrap().x - 1.0).abs() < 1e-2);
    }

    #[test]
    fn plane_drag_moves_in_two_axes() {
        let cam = camera();
        let mut gizmo = TranslateGizmo::new(GizmoConfig::default());
        gizmo.set_target(Some(Vec3::zeros()));
        let scale = screen_scale(&cam, Vec3::zeros(), gizmo.config().size_factor);

        // Press in the middle of the XY plane handle (facing the camera).
        let grab = Vec3::new(0.45 * scale, 0.45 * scale, 0.0);
        let start = cam.project(grab).unwrap();
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(start),
                pressed: true,
            },
        );
        let events = drain(&mut gizmo);
        assert_eq!(
            events,
            vec![GizmoEvent::DragStart {
                handle: Handle::PlaneXY
            }]
        );

        // Drag diagonally by (+1, +0.5) in the XY plane.
        let dest = cam.project(grab + Vec3::new(1.0, 0.5, 0.0)).unwrap();
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(dest),
                pressed: true,
            },
        );
        let events = drain(&mut gizmo);
        let GizmoEvent::DragDelta { world_delta, .. } = events[0] else {
            panic!("expected DragDelta");
        };
        assert!((world_delta.x - 1.0).abs() < 1e-2);
        assert!((world_delta.y - 0.5).abs() < 1e-2);
        assert!(world_delta.z.abs() < 1e-3);
    }

    #[test]
    fn set_target_ignored_during_drag() {
        let cam = camera();
        let mut gizmo = TranslateGizmo::new(GizmoConfig::default());
        gizmo.set_target(Some(Vec3::zeros()));
        let start = over_x_arrow(&cam, &gizmo);
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(start),
                pressed: true,
            },
        );
        drain(&mut gizmo);

        // A laggy consumer writes back an old position mid-drag: ignored.
        gizmo.set_target(Some(Vec3::new(50.0, 0.0, 0.0)));
        assert!(gizmo.effective_target().unwrap().norm() < 1.0);
    }

    #[test]
    fn press_off_gizmo_does_nothing() {
        let cam = camera();
        let mut gizmo = TranslateGizmo::new(GizmoConfig::default());
        gizmo.set_target(Some(Vec3::zeros()));
        gizmo.frame(
            &cam,
            CursorState {
                position: Some((100.0, 100.0)),
                pressed: true,
            },
        );
        assert!(drain(&mut gizmo).is_empty());
        assert!(!gizmo.wants_cursor());
    }
}
