//! Asset-based material (template) resolver.
//!
//! `MaterialAssetManager` is the consumer facade for the **material template**
//! (`docs/MATERIAL_ASSETS.md` Decision 5): given a material guid it resolves the
//! record's [`MaterialData`] (via an embedded
//! [`AssetManager`](redlilium_assets::AssetManager) — the data phase), looks the
//! shading model up in the [`ShadingRegistry`], pulls the model's shader via the
//! [`ShaderManager`], and publishes a [`ResolvedMaterial`] into its
//! [`ResidentCache`]. It owns no pipeline — the pipeline is specialized at draw
//! time by the [`PipelineCache`](super::PipelineCache).
//!
//! Hot reload: a resident hit pull-validates the shader `Arc` (a reloaded
//! shader re-resolves the template, serving last-good meanwhile), and
//! [`invalidate`](MaterialAssetManager::invalidate) drops both the resolution
//! and the data so an edited record reloads from fresh settings.

use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetManager, AssetProcessor, Guid, ResidentCache};

use super::ShaderManager;
use crate::std::rendering::loaders::{MaterialLoader, Shader};
use crate::std::rendering::shading::{PropValue, ShadingRegistry};

/// A fully resolved material template: the shader to specialize a pipeline from,
/// and the property values in the shading model's schema order (model defaults
/// overlaid with the material's authored values).
#[derive(Debug)]
pub struct ResolvedMaterial {
    /// The shading model id this material uses.
    pub shading_model: String,
    /// The shader source asset guid (the pipeline cache key).
    pub shader_guid: Guid,
    /// The resident shader source.
    pub shader: Arc<Shader>,
    /// Property values in schema order — the canonical packing order.
    pub properties: Vec<(String, PropValue)>,
}

/// Owns and shares resolved material templates (an ECS resource).
#[derive(Default)]
pub struct MaterialAssetManager {
    /// The data phase: guid → resident [`MaterialData`].
    data: AssetManager<MaterialLoader>,
    /// The resolution: guid → the single shared [`ResolvedMaterial`].
    cache: ResidentCache<Guid, ResolvedMaterial>,
}

impl MaterialAssetManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// The resolved template for `guid`, driving its resolution one step if not
    /// yet resident. Returns `None` while the material data or its shader is
    /// still loading (or after a failure). Call again next frame to advance.
    pub fn get_or_request(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        shader_mgr: &mut ShaderManager,
        registry: &ShadingRegistry,
        guid: Guid,
    ) -> Option<Arc<ResolvedMaterial>> {
        if let Some(resolved) = self.cache.get(&guid).cloned() {
            // Pull-validation (hot reload): is the shader we resolved with still
            // current? `get_or_request` re-requests an invalidated shader; while
            // it reloads (`None`) we keep serving the last-good resolution, and
            // once a *different* `Arc` lands we drop ours and re-resolve below.
            match shader_mgr.get_or_request(processor, db, resolved.shader_guid) {
                Some(shader) if !Arc::ptr_eq(&shader, &resolved.shader) => {
                    self.cache.invalidate(&guid);
                }
                _ => return Some(resolved),
            }
        }
        if self.cache.is_failed(&guid) {
            return None;
        }

        // Phase 1: the material data (request once, then poll).
        let data = self.data.get_or_request(processor, db, guid)?;

        // Phase 2: resolve the shading model + its shader.
        let Some(model) = registry.get(&data.shading_model) else {
            log::warn!(
                "material {guid:?}: unknown shading model '{}'",
                data.shading_model
            );
            self.cache.fail(guid);
            return None;
        };
        let shader = shader_mgr.get_or_request(processor, db, model.shader)?; // None → still loading

        // Phase 3: build + publish the resolved template.
        let resolved = Arc::new(ResolvedMaterial {
            shading_model: data.shading_model.clone(),
            shader_guid: model.shader,
            shader,
            properties: model.resolve(&data.properties),
        });
        self.cache.publish(guid, resolved.clone());
        Some(resolved)
    }

    /// The resolved template for `guid` if already resident — no request side effect.
    pub fn get(&self, guid: Guid) -> Option<&Arc<ResolvedMaterial>> {
        self.cache.get(&guid)
    }

    /// Drop all state for `guid` — the resolution *and* the data, so the next
    /// `get_or_request` reloads from fresh record settings (hot reload).
    /// Dependants keep serving the old resolution until the new one lands, then
    /// rebuild by pointer identity.
    pub fn invalidate(&mut self, guid: Guid) {
        self.cache.invalidate(&guid);
        self.data.invalidate(guid);
    }
}
