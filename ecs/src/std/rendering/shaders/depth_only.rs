//! Depth-only shader access (#129) — the shared vertex-only shader depth
//! passes draw every mesh with (see `std-assets/shaders/depth_only.slang`).
//!
//! Reuses [`CameraUniforms`](super::CameraUniforms) (external set) and
//! [`ModelUniforms`](super::ModelUniforms) (dynamic set); it declares no
//! static material set.
//!
//! The shader is a std asset (`kind: "shader"`), but depth passes are engine
//! infrastructure, not authored content — so it is also embedded here (the
//! `entity_index` precedent) and handed to the pipeline cache directly,
//! without an asset-manager round trip. Both routes resolve to the same
//! baked artifact: the bake keys on the normalized source bytes, which are
//! identical.

use std::sync::Arc;

use redlilium_assets::Guid;

use crate::std::rendering::loaders::Shader;

/// Mount-relative asset path of the depth-only shader (its
/// [`Guid::stable`] identity).
pub const DEPTH_ONLY_SHADER_PATH: &str = "shaders/depth_only.slang";

/// The embedded Slang source — byte-identical to the std asset.
const SHADER_SLANG: &str = include_str!("../../../../../std-assets/shaders/depth_only.slang");

/// The depth-only shader's asset guid (the pipeline-cache key).
pub fn depth_only_guid() -> Guid {
    Guid::stable(DEPTH_ONLY_SHADER_PATH)
}

/// The resident depth-only shader, ready for
/// [`PipelineCache::get_or_build_depth_only`](crate::std::rendering::PipelineCache::get_or_build_depth_only)
/// via [`DrawArgs::depth_only`](crate::std::rendering::DrawArgs::depth_only).
pub fn depth_only_shader() -> Arc<Shader> {
    Arc::new(Shader {
        source: SHADER_SLANG.as_bytes().to_vec(),
    })
}
