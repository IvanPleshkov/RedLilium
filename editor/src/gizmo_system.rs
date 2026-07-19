//! Editor orchestration for the translate gizmo (#85).
//!
//! The gizmo crate is scene-agnostic; this module binds it to the editor:
//! anchors come from [`GizmoAnchors`](redlilium_ecs::ui::GizmoAnchors)
//! providers on the selected entity (type-erased — `Transform` and any
//! custom level-generator component look identical here), drags become
//! merging [`EditAction`]s through the [`ActionQueue`] (one undo entry per
//! drag, HARD RULE #1 intact), and rendering rides the frame graph after
//! the debug pass.

use redlilium_core::abstract_editor::{ActionQueue, MergeBarrier};
use redlilium_core::math::{Vec3, mat4_to_cols_array_2d};
use redlilium_ecs::ui::{AnchorEdit, GizmoCaps};
use redlilium_ecs::{
    Camera, CameraTarget, Entity, RenderSchedule, ScenePass, System, SystemContext, SystemError,
    WindowInput, World, ui::Selection,
};
use redlilium_gizmo::{
    CursorState, GizmoCamera, GizmoDelta, GizmoEvent, GizmoMode, GizmoRenderer, TransformGizmo,
    build_anchor_dots, build_vertices,
};
use redlilium_graphics::graph::RenderTarget;

/// Where the scene image sits inside the window, in physical pixels. The
/// shells publish it each frame (headless: the whole frame; windowed: the
/// egui scene panel), so world-resident systems can translate
/// `WindowInput`'s window-space cursor into scene-image space — the same
/// space remote picks and screenshots use.
#[derive(Debug, Clone, Copy, Default)]
pub struct SceneViewRect {
    pub offset: [f32; 2],
    pub size: [f32; 2],
}

/// Which anchor currently carries the gizmo, plus the interaction state the
/// per-frame system needs across frames.
#[derive(Default)]
pub struct GizmoUiState {
    /// The focused anchor: (entity, provider component name, anchor id).
    /// The component name is owned — provider names live in dylib images
    /// and must not be held as `&'static str` across a module reload.
    pub focus: Option<(Entity, String, u32)>,
    /// Previous frame's mouse-left, for click-edge detection.
    prev_pressed: bool,
    /// This frame's anchor dots (world position, is-focused) for rendering.
    pub dots: Vec<(Vec3, bool)>,
}

/// Pixel radius for clicking an anchor dot to move the gizmo onto it.
const DOT_PICK_RADIUS_PX: f32 = 14.0;

/// The scene camera as the gizmo sees it: first `Camera` component (the
/// editor camera in editor mode), eye recovered from the view matrix.
pub(crate) fn gizmo_camera(world: &World, viewport: (f32, f32)) -> Option<GizmoCamera> {
    let cameras = world.read_all::<Camera>().ok()?;
    let (_, cam) = cameras.iter().next()?;
    let eye = cam
        .view_matrix
        .try_inverse()
        .map(|inv| Vec3::new(inv[(0, 3)], inv[(1, 3)], inv[(2, 3)]))?;
    Some(GizmoCamera {
        view_proj: cam.view_projection(),
        eye,
        viewport,
    })
}

/// Per-frame gizmo interaction: anchors → focus → drag → undoable actions.
///
/// Runs in `PostUpdate` after `UpdateCameraMatrices` (fresh view-proj) and
/// before the asset chain (#54 ordering). Editor-only, but unconditional —
/// this system never runs in a game world.
pub struct GizmoInteract;

impl System for GizmoInteract {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<TransformGizmo>()
            || !world.has_resource::<GizmoUiState>()
            || !world.has_resource::<SceneViewRect>()
            || !world.has_resource::<Selection>()
            || !world.has_resource::<WindowInput>()
            || !world.has_resource::<ActionQueue<World>>()
        {
            return Ok(());
        }

        let rect = *world.resource::<SceneViewRect>();
        if rect.size[0] < 1.0 || rect.size[1] < 1.0 {
            return Ok(());
        }
        let Some(camera) = gizmo_camera(world, (rect.size[0], rect.size[1])) else {
            return Ok(());
        };

        // Cursor in scene-image space; None outside the scene rect.
        let (cursor_pos, pressed) = {
            let input = world.resource::<WindowInput>();
            let p = input.cursor_position;
            let local = (p[0] - rect.offset[0], p[1] - rect.offset[1]);
            let inside = local.0 >= 0.0
                && local.1 >= 0.0
                && local.0 < rect.size[0]
                && local.1 < rect.size[1];
            (inside.then_some(local), input.mouse_left)
        };

        let mut state = world.resource_mut::<GizmoUiState>();
        let pressed_edge = pressed && !state.prev_pressed;
        state.prev_pressed = pressed;

        // Mode shortcuts (Unity-style W/E/R), suppressed while the RMB
        // fly-camera is navigating and while a drag is live (set_mode is
        // drag-safe anyway, but skipping avoids hover flicker).
        {
            use redlilium_core::input::KeyCode;
            let input = world.resource::<WindowInput>();
            if !input.mouse_right {
                let mut gizmo = world.resource_mut::<TransformGizmo>();
                if input.is_key_pressed(KeyCode::W) {
                    gizmo.set_mode(GizmoMode::Translate);
                } else if input.is_key_pressed(KeyCode::E) {
                    gizmo.set_mode(GizmoMode::Rotate);
                } else if input.is_key_pressed(KeyCode::R) {
                    gizmo.set_mode(GizmoMode::Scale);
                }
            }
        }

        // The selection's primary entity provides the anchors.
        let primary = world
            .resource::<Selection>()
            .entities()
            .first()
            .copied()
            .filter(|e| world.is_alive(*e));
        let mut gizmo = world.resource_mut::<TransformGizmo>();
        let Some(entity) = primary else {
            state.focus = None;
            state.dots.clear();
            gizmo.set_target(None);
            return Ok(());
        };

        let mode_cap = match gizmo.mode() {
            GizmoMode::Translate => GizmoCaps::TRANSLATE,
            GizmoMode::Rotate => GizmoCaps::ROTATE,
            GizmoMode::Scale => GizmoCaps::SCALE,
        };
        // Only anchors supporting the current mode participate: a
        // translate-only generator corner simply hides the gizmo in
        // rotate/scale modes.
        let anchors: Vec<_> = world
            .gizmo_anchors_of(entity)
            .into_iter()
            .filter(|(_, a)| a.caps.contains(mode_cap))
            .collect();
        if anchors.is_empty() {
            state.focus = None;
            state.dots.clear();
            gizmo.set_target(None);
            return Ok(());
        }

        // Resolve focus: keep it if still valid, else default to the first
        // anchor (Transform's origin for ordinary entities).
        let focus_valid = state.focus.as_ref().is_some_and(|(e, name, id)| {
            *e == entity && anchors.iter().any(|(n, a)| n == name && a.id == *id)
        });
        if !focus_valid {
            let (name, anchor) = &anchors[0];
            state.focus = Some((entity, (*name).to_owned(), anchor.id));
        }
        let (_, focus_name, focus_id) = state.focus.clone().expect("focus set above");
        let focused_pos = anchors
            .iter()
            .find(|(n, a)| *n == focus_name && a.id == focus_id)
            .map(|(_, a)| a.position)
            .expect("focus validated against anchors");

        // Anchor-dot click: a press that is NOT on a gizmo handle but lands
        // near another anchor's projected dot re-focuses the gizmo there.
        if pressed_edge
            && gizmo.active().is_none()
            && !gizmo.wants_cursor()
            && let Some(cursor) = cursor_pos
        {
            let hit = anchors
                .iter()
                .filter(|(n, a)| !(*n == focus_name && a.id == focus_id))
                .filter_map(|(n, a)| {
                    let s = camera.project(a.position)?;
                    let d = ((s.0 - cursor.0).powi(2) + (s.1 - cursor.1).powi(2)).sqrt();
                    (d <= DOT_PICK_RADIUS_PX).then_some((d, n, a))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0));
            if let Some((_, name, anchor)) = hit {
                state.focus = Some((entity, (*name).to_owned(), anchor.id));
                gizmo.set_target(Some(anchor.position));
                state.dots = anchors
                    .iter()
                    .map(|(n, a)| (a.position, *n == *name && a.id == anchor.id))
                    .collect();
                return Ok(());
            }
        }

        gizmo.set_target(Some(focused_pos));
        gizmo.frame(
            &camera,
            CursorState {
                position: cursor_pos,
                pressed,
            },
        );

        // Drain events into undoable actions. Per-frame deltas merge into a
        // single undo entry (set_component_action semantics); the barrier at
        // drag end seals it.
        let queue = world.resource::<ActionQueue<World>>();
        while let Some(event) = gizmo.poll_event() {
            match event {
                GizmoEvent::DragStart { .. } => {}
                GizmoEvent::DragDelta { delta, .. } => {
                    let edit = match delta {
                        GizmoDelta::Translate(v) => AnchorEdit::Translate(v),
                        GizmoDelta::Rotate { axis, angle } => AnchorEdit::Rotate { axis, angle },
                        GizmoDelta::Scale(f) => AnchorEdit::Scale(f),
                    };
                    if let Some(action) =
                        world.gizmo_edit_action(entity, &focus_name, focus_id, edit)
                    {
                        queue.push(action);
                    }
                }
                GizmoEvent::DragEnd { .. } => {
                    queue.push(Box::new(MergeBarrier));
                }
            }
        }

        // Dots for the renderer (only meaningful with more than one anchor).
        state.dots = if anchors.len() > 1 {
            anchors
                .iter()
                .map(|(n, a)| (a.position, *n == focus_name && a.id == focus_id))
                .collect()
        } else {
            Vec::new()
        };
        Ok(())
    }
}

/// Contributes the gizmo overlay pass to the frame graph, ordered after the
/// current `ScenePass` writer (the debug pass) and becoming the new last
/// writer so egui composites after it.
pub struct GizmoRender;

impl System for GizmoRender {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<TransformGizmo>()
            || !world.has_resource::<GizmoRenderer>()
            || !world.has_resource::<RenderSchedule>()
            || !world.has_resource::<SceneViewRect>()
        {
            return Ok(());
        }
        let rect = *world.resource::<SceneViewRect>();
        let Some(camera) = gizmo_camera(world, (rect.size[0], rect.size[1])) else {
            return Ok(());
        };

        let mut vertices = {
            let gizmo = world.resource::<TransformGizmo>();
            build_vertices(&gizmo, &camera)
        };
        if world.has_resource::<GizmoUiState>() {
            let state = world.resource::<GizmoUiState>();
            if !state.dots.is_empty() {
                let size_factor = world.resource::<TransformGizmo>().config().size_factor;
                vertices.extend(build_anchor_dots(&state.dots, &camera, size_factor));
            }
        }
        if vertices.is_empty() {
            return Ok(());
        }

        let Some(color) = world
            .read_all::<CameraTarget>()
            .ok()
            .and_then(|t| t.iter().next().map(|(_, target)| target.color.clone()))
        else {
            return Ok(());
        };

        let gizmo_handle = {
            let mut renderer = world.resource_mut::<GizmoRenderer>();
            renderer.update_view_proj(mat4_to_cols_array_2d(&camera.view_proj));
            let rt = RenderTarget::from_texture(color);
            let Some(pass) = renderer.create_graphics_pass(&vertices, &rt) else {
                return Ok(());
            };
            let prev_writer = world.resource::<ScenePass>().0;
            let mut schedule = world.resource_mut::<RenderSchedule>();
            schedule.graph_mut().map(|graph| {
                let h = graph.add_graphics_pass(pass);
                if let Some(prev) = prev_writer {
                    graph.add_dependency(h, prev);
                }
                h
            })
        };
        if let Some(h) = gizmo_handle {
            world.resource_mut::<ScenePass>().0 = Some(h);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EditorWorldParams, create_editor_world};
    use crate::scene_view::SceneViewState;
    use redlilium_ecs::{EcsRunner, MeshRenderer, Transform};
    use redlilium_gizmo::screen_scale;
    use redlilium_graphics::{GraphicsInstance, TextureFormat};
    use redlilium_vfs::Vfs;

    /// #85 end-to-end: select a scene entity, drag the gizmo's X arrow with
    /// a synthetic cursor through real `run_frame`s, and verify the edit
    /// went through the ActionQueue → history (single undo entry reverts
    /// the whole drag). Exercises the Transform provider, the interact
    /// system, scene-rect cursor mapping, and the merge machinery together.
    #[test]
    fn gizmo_drag_moves_transform_and_undoes_as_one_entry() {
        let instance = GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("graphics device");
        let engine = redlilium_runtime::EngineContext::with_vfs(device.clone(), Vfs::new());
        let mut scene_view = SceneViewState::new(device, TextureFormat::Bgra8UnormSrgb);
        let runner = EcsRunner::single_thread();

        let mut ew = create_editor_world(
            &EditorWorldParams {
                remote: false,
                egui: false,
            },
            &engine,
            &mut scene_view,
            16.0 / 9.0,
        );
        ew.world.insert_resource(SceneViewRect {
            offset: [0.0, 0.0],
            size: [1600.0, 900.0],
        });

        // Pick a demo entity carrying a mesh (and a Transform).
        let target = ew
            .world
            .iter_entities()
            .find(|&e| ew.world.get::<MeshRenderer>(e).is_some())
            .expect("demo scene has meshes");
        let start_translation = ew.world.get::<Transform>(target).unwrap().translation;
        ew.world
            .resource_mut::<redlilium_ecs::ui::Selection>()
            .set(vec![target]);

        let dt = 1.0 / 60.0;
        let tick = |ew: &mut crate::core::EditorWorld| {
            ew.drain_actions();
            ew.schedules.run_frame(&mut ew.world, &runner, dt);
        };

        // Frame 1: camera matrices populate; gizmo takes its target.
        tick(&mut ew);

        // Build the same camera the systems see, to steer the cursor.
        let camera = gizmo_camera(&ew.world, (1600.0, 900.0)).expect("camera present");
        let anchors = ew.world.gizmo_anchors_of(target);
        let anchor_pos = anchors[0].1.position;
        let scale = {
            let f = ew
                .world
                .resource::<redlilium_gizmo::TransformGizmo>()
                .config()
                .size_factor;
            screen_scale(&camera, anchor_pos, f)
        };
        let grab_world = anchor_pos + Vec3::new(0.6 * scale, 0.0, 0.0);
        let grab = camera.project(grab_world).expect("on screen");

        // Hover first: the shell's press-time gesture routing relies on
        // wants_cursor() being true BEFORE the click lands (a press on a
        // hovered handle must not clear the selection or start a pick).
        {
            let mut input = ew.window_input.write();
            input.cursor_position = [grab.0, grab.1];
        }
        tick(&mut ew);
        assert!(
            ew.world
                .resource::<redlilium_gizmo::TransformGizmo>()
                .wants_cursor(),
            "hovering the X arrow must claim the cursor before the press"
        );

        // Press on the X arrow.
        {
            let mut input = ew.window_input.write();
            input.mouse_left = true;
        }
        tick(&mut ew);

        // Drag: aim the grab point at +1.5 on world X, hold for two frames.
        let dest = camera
            .project(grab_world + Vec3::new(1.5, 0.0, 0.0))
            .expect("on screen");
        {
            let mut input = ew.window_input.write();
            input.cursor_position = [dest.0, dest.1];
        }
        tick(&mut ew);
        tick(&mut ew); // actions from the drag frame drain here

        // Release.
        {
            let mut input = ew.window_input.write();
            input.mouse_left = false;
        }
        tick(&mut ew);
        tick(&mut ew); // barrier drains

        let dragged = ew.world.get::<Transform>(target).unwrap().translation;
        assert!(
            (dragged.x - start_translation.x - 1.5).abs() < 0.05,
            "dragged +1.5 on X: {} -> {}",
            start_translation.x,
            dragged.x
        );
        assert!((dragged.y - start_translation.y).abs() < 1e-3);
        assert!((dragged.z - start_translation.z).abs() < 1e-3);

        // The whole drag is ONE undo entry.
        assert!(ew.history.can_undo());
        ew.history.undo(&mut ew.world).unwrap();
        let reverted = ew.world.get::<Transform>(target).unwrap().translation;
        assert!(
            (reverted - start_translation).norm() < 1e-4,
            "single undo reverts the whole drag: {reverted:?}"
        );
        assert!(!ew.history.can_undo(), "drag collapsed into one entry");
    }

    /// #85 v2: rotate mode end-to-end — switch mode via the remote-style
    /// resource call, drag the Z ring through real frames, verify the
    /// Transform rotation changed and a single undo reverts it.
    #[test]
    fn gizmo_rotate_drag_rotates_transform_and_undoes() {
        use redlilium_gizmo::GizmoMode;

        let instance = GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("graphics device");
        let engine = redlilium_runtime::EngineContext::with_vfs(device.clone(), Vfs::new());
        let mut scene_view = SceneViewState::new(device, TextureFormat::Bgra8UnormSrgb);
        let runner = EcsRunner::single_thread();

        let mut ew = create_editor_world(
            &EditorWorldParams {
                remote: false,
                egui: false,
            },
            &engine,
            &mut scene_view,
            16.0 / 9.0,
        );
        ew.world.insert_resource(SceneViewRect {
            offset: [0.0, 0.0],
            size: [1600.0, 900.0],
        });

        let target = ew
            .world
            .iter_entities()
            .find(|&e| ew.world.get::<MeshRenderer>(e).is_some())
            .expect("demo scene has meshes");
        let start_rotation = ew.world.get::<Transform>(target).unwrap().rotation;
        ew.world
            .resource_mut::<redlilium_ecs::ui::Selection>()
            .set(vec![target]);
        ew.world
            .resource_mut::<redlilium_gizmo::TransformGizmo>()
            .set_mode(GizmoMode::Rotate);

        let dt = 1.0 / 60.0;
        let tick = |ew: &mut crate::core::EditorWorld| {
            ew.drain_actions();
            ew.schedules.run_frame(&mut ew.world, &runner, dt);
        };
        tick(&mut ew); // camera matrices

        let camera = gizmo_camera(&ew.world, (1600.0, 900.0)).expect("camera");
        let anchors = ew.world.gizmo_anchors_of(target);
        let center = anchors[0].1.position;
        let scale = {
            let f = ew
                .world
                .resource::<redlilium_gizmo::TransformGizmo>()
                .config()
                .size_factor;
            screen_scale(&camera, center, f)
        };
        let r = 0.9 * scale; // ring_radius default

        // The editor camera looks at the scene from an angle, so pick the
        // ring whose plane faces the camera best: project two points on each
        // ring and pick... keep it simple — grab the Y ring (XZ plane) at
        // its +X point; the default camera looks down enough for it to pick.
        let grab_world = center + Vec3::new(r, 0.0, 0.0);
        let grab = camera.project(grab_world).expect("on screen");
        {
            let mut input = ew.window_input.write();
            input.cursor_position = [grab.0, grab.1];
        }
        tick(&mut ew);
        let hovered = ew
            .world
            .resource::<redlilium_gizmo::TransformGizmo>()
            .hovered();
        assert!(hovered.is_some(), "a ring hovers at the grab point");

        {
            let mut input = ew.window_input.write();
            input.mouse_left = true;
        }
        tick(&mut ew);

        // Sweep the cursor to a rotated position on the same ring plane.
        let axis = hovered.unwrap().primary_dir();
        // Rotate the grab offset 40° around the hovered ring's axis.
        let offset = grab_world - center;
        let angle = 40f32.to_radians();
        let (s_a, c_a) = (angle.sin(), angle.cos());
        // Rodrigues rotation of `offset` around `axis`.
        let rotated =
            offset * c_a + axis.cross(&offset) * s_a + axis * axis.dot(&offset) * (1.0 - c_a);
        let dest = camera.project(center + rotated).expect("on screen");
        {
            let mut input = ew.window_input.write();
            input.cursor_position = [dest.0, dest.1];
        }
        tick(&mut ew);
        tick(&mut ew); // drain delta actions

        {
            let mut input = ew.window_input.write();
            input.mouse_left = false;
        }
        tick(&mut ew);
        tick(&mut ew);

        let rotated_q = ew.world.get::<Transform>(target).unwrap().rotation;
        let dot = (rotated_q.i * start_rotation.i
            + rotated_q.j * start_rotation.j
            + rotated_q.k * start_rotation.k
            + rotated_q.w * start_rotation.w)
            .abs();
        assert!(
            dot < 0.999,
            "rotation changed by a measurable angle (dot = {dot})"
        );

        // One undo reverts the whole rotation drag.
        assert!(ew.history.can_undo());
        ew.history.undo(&mut ew.world).unwrap();
        let reverted = ew.world.get::<Transform>(target).unwrap().rotation;
        let dot = rotated_q.i * reverted.i
            + rotated_q.j * reverted.j
            + rotated_q.k * reverted.k
            + rotated_q.w * reverted.w;
        let back = reverted.i * start_rotation.i
            + reverted.j * start_rotation.j
            + reverted.k * start_rotation.k
            + reverted.w * start_rotation.w;
        assert!(back.abs() > 0.9999, "undo restores the original rotation");
        let _ = dot;
        assert!(!ew.history.can_undo(), "whole drag was one entry");
    }
}
