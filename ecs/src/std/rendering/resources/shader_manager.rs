//! Sharing manager for shader assets — the standard single-phase
//! [`AssetManager`](redlilium_assets::AssetManager) over the shader loader:
//! single requester per guid, resident `Arc<Shader>` sharing, failure latching,
//! and hot-reload invalidation. Pulled by the material template manager.

use redlilium_assets::AssetManager;

use crate::std::rendering::loaders::ShaderLoader;

/// Owns and shares resident shaders (an ECS resource).
pub type ShaderManager = AssetManager<ShaderLoader>;
