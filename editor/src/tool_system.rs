//! Generic driver for pluggable viewport tools ([`ViewportTools`] registry).
//!
//! Runs in `PostUpdate` after `UpdateCameraMatrices` (the cursor ray needs
//! fresh view-proj), mirroring `GizmoInteract`'s resource access pattern.
//! The shell routes clicks to the registry (gestures grant the active tool
//! ownership); this system prepares the per-frame [`ViewportToolInput`]
//! (cursor, world ray, click/escape edges) and ticks the active tool.

use redlilium_core::abstract_editor::ActionQueue;
use redlilium_core::input::KeyCode;
use redlilium_ecs::ui::{ViewportRay, ViewportToolCtx, ViewportToolInput, ViewportTools};
use redlilium_ecs::{System, SystemContext, SystemError, WindowInput, World};

use crate::gizmo_system::SceneViewRect;

pub struct RunViewportTool;

impl System for RunViewportTool {
    type Result = ();

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<ViewportTools>()
            || !world.has_resource::<SceneViewRect>()
            || !world.has_resource::<WindowInput>()
            || !world.has_resource::<ActionQueue<World>>()
        {
            return Ok(());
        }

        let mut tools = world.resource_mut::<ViewportTools>();
        tools.resolve_pending();

        // Escape edge detection lives here (not in the shell) so headless
        // and windowed shells behave identically.
        let escape_down = world
            .resource::<WindowInput>()
            .is_key_pressed(KeyCode::Escape);
        let escape = escape_down && !tools.prev_escape;
        tools.prev_escape = escape_down;

        if !tools.is_active() {
            tools.pending_click = false;
            return Ok(());
        }

        let rect = *world.resource::<SceneViewRect>();
        if rect.size[0] < 1.0 || rect.size[1] < 1.0 {
            return Ok(());
        }

        let (cursor, ray) = {
            let input = world.resource::<WindowInput>();
            let p = input.cursor_position;
            let local = [p[0] - rect.offset[0], p[1] - rect.offset[1]];
            let inside = local[0] >= 0.0
                && local[1] >= 0.0
                && local[0] < rect.size[0]
                && local[1] < rect.size[1];
            let ray = inside
                .then(|| crate::gizmo_system::gizmo_camera(world, (rect.size[0], rect.size[1])))
                .flatten()
                .and_then(|cam| cam.ray_from_screen((local[0], local[1])))
                .map(|r| ViewportRay {
                    origin: r.origin,
                    dir: r.dir,
                });
            (inside.then_some(local), ray)
        };

        let input = ViewportToolInput {
            cursor,
            scene_size: rect.size,
            ray,
            clicked: std::mem::take(&mut tools.pending_click),
            escape,
        };
        let actions = world.resource::<ActionQueue<World>>();
        let mut tool_ctx = ViewportToolCtx {
            world,
            actions: &actions,
            input: &input,
        };
        tools.run_active(&mut tool_ctx);
        Ok(())
    }
}
