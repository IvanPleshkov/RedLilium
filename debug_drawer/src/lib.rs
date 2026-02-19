//! Debug drawing utilities for RedLilium Engine.
//!
//! Provides immediate-mode debug line drawing that integrates with the
//! RedLilium graphics pipeline. Thread-safe with one-frame latency.
//!
//! # Architecture
//!
//! - [`DebugDrawer`] — Thread-safe accumulator (store as a shared resource)
//! - [`DebugDrawerContext`] — Short-lived drawing context (created per-system)
//! - [`DebugDrawerRenderer`] — GPU resource manager (creates [`GraphicsPass`](redlilium_graphics::graph::GraphicsPass))
//!
//! # Usage
//!
//! ```ignore
//! // Setup (once)
//! let drawer = Arc::new(DebugDrawer::new());
//! let mut renderer = DebugDrawerRenderer::new(device, surface_format);
//!
//! // Each frame:
//! drawer.advance_tick();
//!
//! // In any ECS system (can run in parallel):
//! let mut ctx = drawer.context();
//! ctx.draw_line([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
//! ctx.draw_aabb([-1.0; 3], [1.0; 3], [0.0, 1.0, 0.0, 1.0]);
//! drop(ctx); // or let it go out of scope
//!
//! // At render time:
//! renderer.update_view_proj(camera_view_proj_matrix);
//! let render_data = drawer.take_render_data();
//! if let Some(pass) = renderer.create_graphics_pass(&render_data, &render_target) {
//!     graph.add_graphics_pass(pass);
//! }
//! ```

mod draw_api;
mod drawer;
mod renderer;
mod shader;
mod vertex;

pub use drawer::{DebugDrawer, DebugDrawerContext};
pub use renderer::DebugDrawerRenderer;
pub use vertex::{DebugUniforms, DebugVertex};
