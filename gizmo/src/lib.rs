//! Interactive translate gizmo for RedLilium Engine (#80).
//!
//! Consumer-agnostic by design: the gizmo **never mutates a scene**. It
//! takes camera + cursor state each frame, picks its handles analytically
//! (ray-vs-capsule / ray-vs-quad — exact, zero-latency, no GPU readback),
//! and emits translation deltas that the consumer applies to its own model:
//!
//! - the editor wraps them in undoable edit actions (merged per drag),
//! - the tyroxine VSCode preview writes them back into DSL source literals.
//!
//! Works on wasm: no threads, no IO, no GPU dependency in the interaction
//! core ([`TranslateGizmo`]); rendering is a separate opt-in piece
//! ([`GizmoRenderer`]) patterned on the debug drawer.
//!
//! # Usage
//!
//! ```ignore
//! let mut gizmo = TranslateGizmo::new(GizmoConfig::default());
//! let mut renderer = GizmoRenderer::new(device, surface_format);
//!
//! // Each frame:
//! gizmo.set_target(selection_position);          // Option<Vec3>
//! gizmo.frame(&camera, cursor);                  // camera: GizmoCamera
//! while let Some(event) = gizmo.poll_event() {
//!     match event {
//!         GizmoEvent::DragStart { .. } => begin_edit(),
//!         GizmoEvent::DragDelta { world_delta, .. } => preview(world_delta),
//!         GizmoEvent::DragEnd { total_delta, .. } => commit(total_delta),
//!     }
//! }
//! if gizmo.wants_cursor() { /* suppress scene picking */ }
//!
//! // At render time:
//! renderer.update_view_proj(view_proj);
//! let verts = build_vertices(&gizmo, &camera);
//! renderer.append_to_pass(&mut overlay_pass, &verts);
//! ```

mod math;
mod mesh;
mod renderer;
mod state;

pub use math::{GizmoCamera, Ray, screen_scale};
pub use mesh::{GizmoUniforms, GizmoVertex, build_anchor_dots, build_vertices};
pub use renderer::GizmoRenderer;
pub use state::{CursorState, GizmoConfig, GizmoEvent, Handle, TranslateGizmo};
