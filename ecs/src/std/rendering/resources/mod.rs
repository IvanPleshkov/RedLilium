//! Rendering resource types.

mod material_manager;
mod mesh_manager;
mod render_schedule;
mod texture_manager;

pub use material_manager::{CpuBundleInfo, MaterialManager, MaterialManagerError};
pub use mesh_manager::MeshManager;
pub use render_schedule::RenderSchedule;
pub use texture_manager::{TextureManager, TextureManagerError};

// Re-export pack_uniform_bytes at module level
pub use material_manager::pack_uniform_bytes;
