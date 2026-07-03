//! Rendering resource types.

#[cfg(feature = "rendering")]
mod changed_assets;
mod frame_ring;
#[cfg(feature = "rendering")]
mod material_asset_manager;
#[cfg(feature = "rendering")]
mod material_instance_manager;
mod mesh_manager;
#[cfg(feature = "rendering")]
mod pipeline_cache;
mod render_schedule;
#[cfg(feature = "rendering")]
mod shader_manager;
mod texture_manager;
#[cfg(feature = "rendering")]
mod vertex_layout_manager;

#[cfg(feature = "rendering")]
pub use changed_assets::ChangedAssets;
pub use frame_ring::FrameRing;
#[cfg(feature = "rendering")]
pub use material_asset_manager::{MaterialAssetManager, ResolvedMaterial};
#[cfg(feature = "rendering")]
pub use material_instance_manager::{MaterialInstanceManager, ResolvedInstance};
pub use mesh_manager::MeshManager;
#[cfg(feature = "rendering")]
pub use pipeline_cache::PipelineCache;
pub use render_schedule::RenderSchedule;
#[cfg(feature = "rendering")]
pub use shader_manager::ShaderManager;
pub use texture_manager::TextureManager;
#[cfg(feature = "rendering")]
pub use vertex_layout_manager::VertexLayoutManager;
