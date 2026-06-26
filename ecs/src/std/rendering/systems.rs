//! Render systems — they run in the [`Render`](crate::Render) schedule and
//! contribute passes to the frame graph held in the [`RenderSchedule`] resource.

use crate::system::SystemError;
use crate::{System, SystemContext};

use super::{MaterialManager, MeshManager, RenderSchedule, TextureManager};

/// Flushes the GPU-upload managers' pending transfers into the frame graph
/// (deferred, graph-ordered — never a synchronous write). Passes that read the
/// uploaded meshes/textures/materials should depend on this system so their
/// transfer pass is scheduled first.
pub struct FlushUploads;

impl System for FlushUploads {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.world();
        let mut schedule = world.resource_mut::<RenderSchedule>();
        let Some(graph) = schedule.graph_mut() else {
            return Ok(()); // no frame graph bound (not in the render bracket)
        };
        if world.has_resource::<TextureManager>() {
            world.resource_mut::<TextureManager>().flush_uploads(graph);
        }
        if world.has_resource::<MeshManager>() {
            world.resource_mut::<MeshManager>().flush_uploads(graph);
        }
        if world.has_resource::<MaterialManager>() {
            world.resource_mut::<MaterialManager>().flush_uploads(graph);
        }
        Ok(())
    }
}
