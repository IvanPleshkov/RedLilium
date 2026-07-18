//! Rendering component types.

mod camera_output;
mod camera_target;
mod mesh_renderer;
mod pipeline_targets;
mod render_path;

pub use camera_output::{CameraOutput, CameraTargetSpec, OutputFormat, SizePolicy};
pub use camera_target::CameraTarget;
pub use mesh_renderer::{MeshRenderer, Primitive};
pub use pipeline_targets::PipelineTargets;
pub use render_path::{DEFERRED_PIPELINE, FORWARD_PIPELINE, RenderPath};
