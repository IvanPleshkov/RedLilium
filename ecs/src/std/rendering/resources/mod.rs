//! Rendering resource types.

mod frame_ring;
#[cfg(feature = "rendering")]
mod material_asset_manager;
#[cfg(feature = "rendering")]
mod material_instance_manager;
mod material_manager;
mod mesh_manager;
#[cfg(feature = "rendering")]
mod pipeline_cache;
mod render_schedule;
#[cfg(feature = "rendering")]
mod shader_manager;
mod texture_manager;
#[cfg(feature = "rendering")]
mod vertex_layout_manager;

pub use frame_ring::FrameRing;
#[cfg(feature = "rendering")]
pub use material_asset_manager::{MaterialAssetManager, ResolvedMaterial};
#[cfg(feature = "rendering")]
pub use material_instance_manager::{InstanceHandle, MaterialInstanceManager, ResolvedInstance};
pub use material_manager::{CpuBundleInfo, MaterialManager, MaterialManagerError};
pub use mesh_manager::{MeshHandle, MeshManager};
#[cfg(feature = "rendering")]
pub use pipeline_cache::PipelineCache;
pub use render_schedule::RenderSchedule;
#[cfg(feature = "rendering")]
pub use shader_manager::ShaderManager;
pub use texture_manager::{TextureManager, TextureManagerError};
#[cfg(feature = "rendering")]
pub use vertex_layout_manager::VertexLayoutManager;

// Re-export pack_uniform_bytes at module level
pub use material_manager::pack_uniform_bytes;
