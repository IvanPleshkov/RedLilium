//! The translate-gizmo interaction state machine: hover → drag → commit.
//!
//! The gizmo never mutates a scene. It consumes camera + cursor state each
//! frame and emits [`GizmoEvent`]s; the consumer applies the deltas to its
//! own model (an ECS `Transform` through an undoable edit action, a DSL
//! literal in source code, …) and re-feeds the target position via
//! [`set_target`](TransformGizmo::set_target).

use std::collections::VecDeque;

use redlilium_core::math::Vec3;

use crate::math::{
    GizmoCamera, Ray, closest_ray_line_params, ray_capsule_hit, ray_plane_intersect, ray_quad_hit,
    ray_ring_hit, ring_direction, screen_scale, signed_angle,
};

/// Which transform aspect the gizmo currently manipulates. World-axis
/// aligned (global mode) in v1 for all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoMode {
    #[default]
    Translate,
    Rotate,
    Scale,
}

/// A pickable part of the gizmo. Which set is live depends on
/// [`GizmoMode`]: axes+planes translate, rings rotate, scale handles scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handle {
    AxisX,
    AxisY,
    AxisZ,
    PlaneXY,
    PlaneXZ,
    PlaneYZ,
    RingX,
    RingY,
    RingZ,
    ScaleX,
    ScaleY,
    ScaleZ,
    ScaleUniform,
}

impl Handle {
    /// The unit direction of an axis/scale handle, the plane normal of a
    /// plane handle, or the rotation axis of a ring.
    pub fn primary_dir(self) -> Vec3 {
        match self {
            Handle::AxisX | Handle::ScaleX | Handle::RingX | Handle::PlaneYZ => {
                Vec3::new(1.0, 0.0, 0.0)
            }
            Handle::AxisY | Handle::ScaleY | Handle::RingY | Handle::PlaneXZ => {
                Vec3::new(0.0, 1.0, 0.0)
            }
            Handle::AxisZ | Handle::ScaleZ | Handle::RingZ | Handle::PlaneXY => {
                Vec3::new(0.0, 0.0, 1.0)
            }
            // Uniform scale has no single direction; callers must not ask.
            Handle::ScaleUniform => Vec3::new(0.0, 0.0, 0.0),
        }
    }

    /// Whether this is a plane handle.
    pub fn is_plane(self) -> bool {
        matches!(self, Handle::PlaneXY | Handle::PlaneXZ | Handle::PlaneYZ)
    }

    /// Whether this is a rotate ring.
    pub fn is_ring(self) -> bool {
        matches!(self, Handle::RingX | Handle::RingY | Handle::RingZ)
    }

    /// Whether this is a scale handle.
    pub fn is_scale(self) -> bool {
        matches!(
            self,
            Handle::ScaleX | Handle::ScaleY | Handle::ScaleZ | Handle::ScaleUniform
        )
    }

    const TRANSLATE: [Handle; 6] = [
        Handle::AxisX,
        Handle::AxisY,
        Handle::AxisZ,
        Handle::PlaneXY,
        Handle::PlaneXZ,
        Handle::PlaneYZ,
    ];
    const ROTATE: [Handle; 3] = [Handle::RingX, Handle::RingY, Handle::RingZ];
    const SCALE: [Handle; 4] = [
        Handle::ScaleX,
        Handle::ScaleY,
        Handle::ScaleZ,
        Handle::ScaleUniform,
    ];

    /// The live handle set for a mode.
    fn for_mode(mode: GizmoMode) -> &'static [Handle] {
        match mode {
            GizmoMode::Translate => &Self::TRANSLATE,
            GizmoMode::Rotate => &Self::ROTATE,
            GizmoMode::Scale => &Self::SCALE,
        }
    }
}

/// A mode-typed transform delta, in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GizmoDelta {
    /// World-space translation.
    Translate(Vec3),
    /// Rotation around a world axis (radians, right-handed).
    Rotate { axis: Vec3, angle: f32 },
    /// Per-component scale factor (uniform handles emit equal components).
    Scale(Vec3),
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
/// [`frame`](TransformGizmo::frame).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GizmoEvent {
    /// The primary button went down on a handle.
    DragStart { handle: Handle },
    /// The cursor moved during a drag: the change since the previous frame
    /// (translation step / rotation-angle step / scale-factor step).
    /// No-op deltas are not emitted.
    DragDelta { handle: Handle, delta: GizmoDelta },
    /// The primary button was released: the total change of the whole drag
    /// (the consumer's commit point — e.g. the single undo entry).
    DragEnd { handle: Handle, total: GizmoDelta },
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
    /// Rotate-ring radius and pick half-width, in gizmo-local units.
    pub ring_radius: f32,
    pub ring_pick_band: f32,
    /// Uniform-scale center cube pick radius, in gizmo-local units.
    pub uniform_pick_radius: f32,
}

impl Default for GizmoConfig {
    fn default() -> Self {
        Self {
            size_factor: 0.11,
            axis_pick_radius: 0.1,
            plane_min: 0.3,
            plane_max: 0.6,
            axis_pick_min: 0.2,
            axis_pick_max: 1.1,
            ring_radius: 0.9,
            ring_pick_band: 0.1,
            uniform_pick_radius: 0.15,
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
    /// Axis/scale handles: the axis parameter under the cursor at drag start.
    /// Uniform scale: the start distance from center on the view plane.
    axis_anchor: f32,
    /// Plane handles: the plane point under the cursor at drag start.
    plane_anchor: Vec3,
    /// Rings: the ring-plane direction under the cursor last frame (angle
    /// increments accumulate frame-to-frame, so >180° drags never wrap).
    prev_ring_dir: Vec3,
    /// Accumulated translation (Translate mode).
    total_translation: Vec3,
    /// Accumulated angle in radians (Rotate mode).
    total_angle: f32,
    /// Accumulated per-axis factor (Scale mode); starts at 1.
    total_scale: Vec3,
}

impl DragSession {
    fn total_delta(&self) -> GizmoDelta {
        match self.handle {
            h if h.is_ring() => GizmoDelta::Rotate {
                axis: h.primary_dir(),
                angle: self.total_angle,
            },
            h if h.is_scale() => GizmoDelta::Scale(self.total_scale),
            _ => GizmoDelta::Translate(self.total_translation),
        }
    }
}

/// The interactive transform gizmo (#80/#85): translate, rotate, and scale
/// modes over world axes (global mode).
///
/// Drive it once per frame with [`frame`](Self::frame), then drain
/// [`poll_event`](Self::poll_event). While a drag is live the gizmo tracks
/// its own position (`start + total`), so a consumer with write-back latency
/// (e.g. a source-code round-trip) still gets smooth dragging;
/// [`set_target`](Self::set_target) re-synchronizes it outside drags.
pub struct TransformGizmo {
    config: GizmoConfig,
    mode: GizmoMode,
    target: Option<Vec3>,
    hovered: Option<Handle>,
    drag: Option<DragSession>,
    prev_pressed: bool,
    events: VecDeque<GizmoEvent>,
}

impl TransformGizmo {
    pub fn new(config: GizmoConfig) -> Self {
        Self {
            config,
            mode: GizmoMode::default(),
            target: None,
            hovered: None,
            drag: None,
            prev_pressed: false,
            events: VecDeque::new(),
        }
    }

    /// The current manipulation mode.
    pub fn mode(&self) -> GizmoMode {
        self.mode
    }

    /// Switch modes. Ignored during a live drag (the drag finishes in the
    /// mode it started in); clears hover so stale handles don't highlight.
    pub fn set_mode(&mut self, mode: GizmoMode) {
        if self.drag.is_none() && self.mode != mode {
            self.mode = mode;
            self.hovered = None;
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
            (Some(drag), _) => Some(drag.start_target + drag.total_translation),
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
            if let Some(ray) = &ray
                && let Some(step) = solve_drag_step(drag, ray)
            {
                self.events.push_back(GizmoEvent::DragDelta {
                    handle: drag.handle,
                    delta: step,
                });
            }
            if released {
                let drag = self.drag.take().expect("drag present");
                // The consumer owns the final value now; keep rendering at
                // the dragged spot until the next set_target (translation
                // moves the gizmo; rotation/scale leave it in place).
                self.target = Some(drag.start_target + drag.total_translation);
                self.events.push_back(GizmoEvent::DragEnd {
                    handle: drag.handle,
                    total: drag.total_delta(),
                });
                self.hovered = None;
            }
            return;
        }

        // --- Idle: hover pick; a press on a handle starts a drag. ---
        let scale = screen_scale(camera, base_target, self.config.size_factor);
        self.hovered = ray.as_ref().and_then(|r| self.pick(r, base_target, scale));

        let _ = scale;
        if pressed_edge
            && let (Some(handle), Some(ray)) = (self.hovered, &ray)
            && let Some(session) = start_drag(handle, ray, base_target, ray.dir)
        {
            self.events.push_back(GizmoEvent::DragStart { handle });
            self.drag = Some(session);
        }
    }

    /// The closest live-mode handle under the ray, if any.
    fn pick(&self, ray: &Ray, target: Vec3, scale: f32) -> Option<Handle> {
        let c = &self.config;
        let mut best: Option<(f32, Handle)> = None;
        for &handle in Handle::for_mode(self.mode) {
            let t = if handle.is_plane() {
                let (u, v) = plane_axes(handle);
                ray_quad_hit(ray, target, u, v, c.plane_min * scale, c.plane_max * scale)
            } else if handle.is_ring() {
                ray_ring_hit(
                    ray,
                    target,
                    handle.primary_dir(),
                    c.ring_radius * scale,
                    c.ring_pick_band * scale,
                )
            } else if handle == Handle::ScaleUniform {
                ray_sphere_like_hit(ray, target, c.uniform_pick_radius * scale)
            } else {
                // Axis arrows and per-axis scale handles share the capsule.
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

/// Closest-approach sphere test (uniform-scale center handle).
fn ray_sphere_like_hit(ray: &Ray, center: Vec3, radius: f32) -> Option<f32> {
    let t = (center - ray.origin).dot(&ray.dir).max(0.0);
    let p = ray.origin + ray.dir * t;
    ((p - center).norm() <= radius).then_some(t)
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
/// handle degenerate (axis near-parallel to the ray, plane/ring near-grazing,
/// scale grabbed at the exact center) — the press is ignored rather than
/// producing wild deltas.
fn start_drag(handle: Handle, ray: &Ray, target: Vec3, view_dir: Vec3) -> Option<DragSession> {
    let mut session = DragSession {
        handle,
        start_target: target,
        axis_anchor: 0.0,
        plane_anchor: Vec3::zeros(),
        prev_ring_dir: Vec3::zeros(),
        total_translation: Vec3::zeros(),
        total_angle: 0.0,
        total_scale: Vec3::new(1.0, 1.0, 1.0),
    };
    if handle.is_plane() {
        session.plane_anchor = ray_plane_intersect(ray, target, handle.primary_dir())?;
    } else if handle.is_ring() {
        session.prev_ring_dir = ring_direction(ray, target, handle.primary_dir())?;
    } else if handle == Handle::ScaleUniform {
        // Uniform scale: radial distance on the view-facing plane.
        let p = ray_plane_intersect(ray, target, view_dir)?;
        let d = (p - target).norm();
        if d < 1e-4 {
            return None; // grabbed the exact center — the ratio would explode
        }
        session.axis_anchor = d;
    } else {
        let (_, t_axis) = closest_ray_line_params(ray, target, handle.primary_dir())?;
        if handle.is_scale() && t_axis.abs() < 1e-4 {
            return None; // scale ratio needs a non-zero start parameter
        }
        session.axis_anchor = t_axis;
    }
    Some(session)
}

/// Advance the drag by this frame's ray: updates the session totals and
/// returns the per-frame step. `None` on a degenerate/no-op frame.
fn solve_drag_step(drag: &mut DragSession, ray: &Ray) -> Option<GizmoDelta> {
    let handle = drag.handle;
    if handle.is_ring() {
        let axis = handle.primary_dir();
        let dir = ring_direction(ray, drag.start_target, axis)?;
        let step = signed_angle(drag.prev_ring_dir, dir, axis);
        drag.prev_ring_dir = dir;
        if step.abs() < 1e-6 {
            return None;
        }
        drag.total_angle += step;
        Some(GizmoDelta::Rotate { axis, angle: step })
    } else if handle == Handle::ScaleUniform {
        // The view plane is fixed at drag start via the anchor distance; use
        // the plane through the start target facing the current ray.
        let p = ray_plane_intersect(ray, drag.start_target, ray.dir)?;
        let d = (p - drag.start_target).norm();
        let new_total = d / drag.axis_anchor;
        if new_total <= 1e-4 {
            return None;
        }
        let prev = drag.total_scale.x;
        let ratio = new_total / prev;
        if (ratio - 1.0).abs() < 1e-6 {
            return None;
        }
        drag.total_scale = Vec3::new(new_total, new_total, new_total);
        Some(GizmoDelta::Scale(Vec3::new(ratio, ratio, ratio)))
    } else if handle.is_scale() {
        let axis = handle.primary_dir();
        let (_, t_axis) = closest_ray_line_params(ray, drag.start_target, axis)?;
        let new_total = t_axis / drag.axis_anchor;
        if new_total <= 1e-4 {
            return None; // dragged through/past the center — hold
        }
        let axis_index = if axis.x != 0.0 {
            0
        } else if axis.y != 0.0 {
            1
        } else {
            2
        };
        let prev = drag.total_scale[axis_index];
        let ratio = new_total / prev;
        if (ratio - 1.0).abs() < 1e-6 {
            return None;
        }
        drag.total_scale[axis_index] = new_total;
        let mut step = Vec3::new(1.0, 1.0, 1.0);
        step[axis_index] = ratio;
        Some(GizmoDelta::Scale(step))
    } else if handle.is_plane() {
        let p = ray_plane_intersect(ray, drag.start_target, handle.primary_dir())?;
        let new_total = p - drag.plane_anchor;
        let step = new_total - drag.total_translation;
        if step.norm() < 1e-6 {
            return None;
        }
        drag.total_translation = new_total;
        Some(GizmoDelta::Translate(step))
    } else {
        let axis = handle.primary_dir();
        let (_, t_axis) = closest_ray_line_params(ray, drag.start_target, axis)?;
        let new_total = axis * (t_axis - drag.axis_anchor);
        let step = new_total - drag.total_translation;
        if step.norm() < 1e-6 {
            return None;
        }
        drag.total_translation = new_total;
        Some(GizmoDelta::Translate(step))
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
    fn over_x_arrow(cam: &GizmoCamera, gizmo: &TransformGizmo) -> (f32, f32) {
        let scale = screen_scale(
            cam,
            gizmo.effective_target().unwrap(),
            gizmo.config().size_factor,
        );
        cam.project(Vec3::new(0.6 * scale, 0.0, 0.0)).unwrap()
    }

    fn drain(gizmo: &mut TransformGizmo) -> Vec<GizmoEvent> {
        std::iter::from_fn(|| gizmo.poll_event()).collect()
    }

    #[test]
    fn hidden_gizmo_ignores_input() {
        let cam = camera();
        let mut gizmo = TransformGizmo::new(GizmoConfig::default());
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
        let mut gizmo = TransformGizmo::new(GizmoConfig::default());
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
        let mut gizmo = TransformGizmo::new(GizmoConfig::default());
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
            delta: GizmoDelta::Translate(world_delta),
        } = events[0]
        else {
            panic!("expected translate DragDelta, got {events:?}");
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
        let GizmoEvent::DragEnd {
            total: GizmoDelta::Translate(total_delta),
            ..
        } = events[0]
        else {
            panic!("expected translate DragEnd, got {events:?}");
        };
        assert!((total_delta.x - 1.0).abs() < 1e-2);
        // The gizmo kept tracking its own position through the drag.
        assert!((gizmo.effective_target().unwrap().x - 1.0).abs() < 1e-2);
    }

    #[test]
    fn plane_drag_moves_in_two_axes() {
        let cam = camera();
        let mut gizmo = TransformGizmo::new(GizmoConfig::default());
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
        let GizmoEvent::DragDelta {
            delta: GizmoDelta::Translate(world_delta),
            ..
        } = events[0]
        else {
            panic!("expected translate DragDelta");
        };
        assert!((world_delta.x - 1.0).abs() < 1e-2);
        assert!((world_delta.y - 0.5).abs() < 1e-2);
        assert!(world_delta.z.abs() < 1e-3);
    }

    #[test]
    fn set_target_ignored_during_drag() {
        let cam = camera();
        let mut gizmo = TransformGizmo::new(GizmoConfig::default());
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
        let mut gizmo = TransformGizmo::new(GizmoConfig::default());
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

    #[test]
    fn mode_switch_changes_handle_set_and_blocks_during_drag() {
        let cam = camera();
        let mut gizmo = TransformGizmo::new(GizmoConfig::default());
        gizmo.set_target(Some(Vec3::zeros()));
        assert_eq!(gizmo.mode(), GizmoMode::Translate);

        // In rotate mode the X arrow position no longer hovers anything.
        gizmo.set_mode(GizmoMode::Rotate);
        let pos = over_x_arrow(&cam, &gizmo);
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(pos),
                pressed: false,
            },
        );
        assert_eq!(
            gizmo.hovered(),
            None,
            "translate handles are gone in rotate mode"
        );

        // Start a rotate drag; a mode switch mid-drag is ignored.
        let scale = screen_scale(&cam, Vec3::zeros(), gizmo.config().size_factor);
        let ring_r = gizmo.config().ring_radius * scale;
        // The Z ring lies in the XY plane, facing this camera dead-on.
        let grab = cam.project(Vec3::new(ring_r, 0.0, 0.0)).unwrap();
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(grab),
                pressed: false,
            },
        );
        assert_eq!(gizmo.hovered(), Some(Handle::RingZ));
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(grab),
                pressed: true,
            },
        );
        assert!(gizmo.active().is_some());
        gizmo.set_mode(GizmoMode::Scale);
        assert_eq!(gizmo.mode(), GizmoMode::Rotate, "mode locked during drag");
    }

    #[test]
    fn rotate_drag_produces_angle_and_commit() {
        let cam = camera();
        let mut gizmo = TransformGizmo::new(GizmoConfig::default());
        gizmo.set_target(Some(Vec3::zeros()));
        gizmo.set_mode(GizmoMode::Rotate);

        let scale = screen_scale(&cam, Vec3::zeros(), gizmo.config().size_factor);
        let r = gizmo.config().ring_radius * scale;

        // Grab the Z ring (in the XY plane) at angle 0, drag to 90°.
        let grab = cam.project(Vec3::new(r, 0.0, 0.0)).unwrap();
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(grab),
                pressed: true,
            },
        );
        assert_eq!(
            drain(&mut gizmo),
            vec![GizmoEvent::DragStart {
                handle: Handle::RingZ
            }]
        );

        // Sweep in two 45° steps so per-frame increments accumulate.
        for angle in [std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2] {
            let p = Vec3::new(r * angle.cos(), r * angle.sin(), 0.0);
            let cursor = cam.project(p).unwrap();
            gizmo.frame(
                &cam,
                CursorState {
                    position: Some(cursor),
                    pressed: true,
                },
            );
        }
        let mut total = 0.0;
        for ev in drain(&mut gizmo) {
            let GizmoEvent::DragDelta {
                delta: GizmoDelta::Rotate { axis, angle },
                ..
            } = ev
            else {
                panic!("expected rotate deltas, got {ev:?}");
            };
            assert!((axis - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-5);
            total += angle;
        }
        assert!(
            (total - std::f32::consts::FRAC_PI_2).abs() < 1e-2,
            "swept 90°, got {total}"
        );

        // Release: the commit carries the whole angle.
        let end = cam.project(Vec3::new(0.0, r, 0.0)).unwrap();
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(end),
                pressed: false,
            },
        );
        let events = drain(&mut gizmo);
        let GizmoEvent::DragEnd {
            total: GizmoDelta::Rotate { angle, .. },
            ..
        } = events[0]
        else {
            panic!("expected rotate DragEnd, got {events:?}");
        };
        assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 1e-2);
        // Rotation does not move the gizmo.
        assert!(gizmo.effective_target().unwrap().norm() < 1e-6);
    }

    #[test]
    fn scale_drag_produces_axis_factor() {
        let cam = camera();
        let mut gizmo = TransformGizmo::new(GizmoConfig::default());
        gizmo.set_target(Some(Vec3::zeros()));
        gizmo.set_mode(GizmoMode::Scale);

        let scale = screen_scale(&cam, Vec3::zeros(), gizmo.config().size_factor);
        // Grab the X scale handle mid-shaft and pull it to double the reach.
        let grab_t = 0.6 * scale;
        let grab = cam.project(Vec3::new(grab_t, 0.0, 0.0)).unwrap();
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(grab),
                pressed: true,
            },
        );
        assert_eq!(
            drain(&mut gizmo),
            vec![GizmoEvent::DragStart {
                handle: Handle::ScaleX
            }]
        );

        let dest = cam.project(Vec3::new(2.0 * grab_t, 0.0, 0.0)).unwrap();
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(dest),
                pressed: true,
            },
        );
        gizmo.frame(
            &cam,
            CursorState {
                position: Some(dest),
                pressed: false,
            },
        );

        let mut total = Vec3::new(1.0, 1.0, 1.0);
        let mut committed: Option<Vec3> = None;
        for ev in drain(&mut gizmo) {
            match ev {
                GizmoEvent::DragDelta {
                    delta: GizmoDelta::Scale(f),
                    ..
                } => {
                    total = Vec3::new(total.x * f.x, total.y * f.y, total.z * f.z);
                }
                GizmoEvent::DragEnd {
                    total: GizmoDelta::Scale(t),
                    ..
                } => {
                    committed = Some(t);
                }
                other => panic!("unexpected event {other:?}"),
            }
        }
        assert!((total.x - 2.0).abs() < 5e-2, "x doubled: {}", total.x);
        assert!((total.y - 1.0).abs() < 1e-5 && (total.z - 1.0).abs() < 1e-5);
        let committed = committed.expect("commit");
        assert!((committed.x - 2.0).abs() < 5e-2);
    }
}
