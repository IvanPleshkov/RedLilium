//! Pipeline-derived auxiliary render targets (ADR-035, #128).

use std::collections::HashMap;
use std::sync::Arc;

use redlilium_graphics::Texture;

/// Runtime-only home for the intermediate GPU targets a camera's render
/// pipeline derives beyond the color+depth pair in
/// [`CameraTarget`](super::CameraTarget) — a shadow map, an HDR intermediate,
/// a bloom chain.
///
/// Owned by the pipeline's
/// [`ensure_targets`](crate::std::rendering::CameraRenderPipeline::ensure_targets)
/// hook under the same discipline as `CameraTarget` itself: re-derived
/// whenever a target disagrees with its spec, never serialized. Keys are
/// pipeline-chosen names (e.g. `"shadow_map"`).
#[derive(Debug, Clone, Default, crate::Component)]
#[skip_serialization]
pub struct PipelineTargets {
    targets: HashMap<String, Arc<Texture>>,
}

impl PipelineTargets {
    /// The target registered under `name`, if any.
    pub fn get(&self, name: &str) -> Option<&Arc<Texture>> {
        self.targets.get(name)
    }

    /// Register (or replace) a target under `name`.
    pub fn set(&mut self, name: impl Into<String>, texture: Arc<Texture>) {
        self.targets.insert(name.into(), texture);
    }

    /// Remove the target registered under `name`.
    pub fn remove(&mut self, name: &str) -> Option<Arc<Texture>> {
        self.targets.remove(name)
    }

    /// Number of registered targets.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Whether no targets are registered.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}
