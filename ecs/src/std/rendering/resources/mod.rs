//! Rendering resource types.

mod frame_ring;
mod material_manager;
mod mesh_manager;
mod render_schedule;
mod texture_manager;
#[cfg(feature = "rendering")]
mod vertex_layout_manager;

pub use frame_ring::FrameRing;
pub use material_manager::{CpuBundleInfo, MaterialManager, MaterialManagerError};
pub use mesh_manager::{MeshHandle, MeshManager};
pub use render_schedule::RenderSchedule;
pub use texture_manager::{TextureManager, TextureManagerError};
#[cfg(feature = "rendering")]
pub use vertex_layout_manager::VertexLayoutManager;

// Re-export pack_uniform_bytes at module level
pub use material_manager::pack_uniform_bytes;
