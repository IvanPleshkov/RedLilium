//! Asset-based material (template) resolver.
//!
//! `MaterialAssetManager` is the consumer facade for the **material template**
//! (`docs/MATERIAL_ASSETS.md` Decision 5): given a [`MaterialSource`] guid it
//! resolves the record's [`MaterialData`] (shading-model id + property values),
//! looks the model up in the [`ShadingRegistry`], pulls the model's shader via
//! the [`ShaderManager`], and produces a [`ResolvedMaterial`] (shader + property
//! values in schema order). It owns no pipeline — the pipeline is specialized at
//! draw time by the [`PipelineCache`](super::PipelineCache), keyed by the mesh's
//! vertex layout.
//!
//! It is the single requester per material guid and is pulled by the material
//! *instance* manager (like the [`VertexLayoutManager`](super::VertexLayoutManager)
//! is pulled by the mesh manager). It chains two async resolutions — the material
//! data, then its shader — returning `None` until both are ready.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetHandle, AssetProcessor, Guid};

use super::ShaderManager;
use crate::std::rendering::loaders::{MaterialData, MaterialLoader, MaterialSource, Shader};
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
    /// guid → resolved template (the single shared resolution per material).
    resident: HashMap<Guid, Arc<ResolvedMaterial>>,
    /// In-flight material-data requests (this manager is the sole requester).
    data_pending: HashMap<Guid, AssetHandle<MaterialData>>,
    /// Material data that has loaded but whose shader is still resolving.
    data_ready: HashMap<Guid, Arc<MaterialData>>,
    /// Materials whose resolution failed — not retried (avoids per-frame spam).
    failed: HashSet<Guid>,
}

impl MaterialAssetManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// The resolved template for `guid`, driving its resolution one step if not
    /// yet resident. Returns `None` while the material data or its shader is still
    /// loading (or after a failure). Call again next frame to advance.
    pub fn get_or_request(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        shader_mgr: &mut ShaderManager,
        registry: &ShadingRegistry,
        guid: Guid,
    ) -> Option<Arc<ResolvedMaterial>> {
        if let Some(resolved) = self.resident.get(&guid) {
            return Some(resolved.clone());
        }
        if self.failed.contains(&guid) {
            return None;
        }

        // Phase 1: obtain the material data (request once, then poll).
        if !self.data_ready.contains_key(&guid) {
            match self.data_pending.get(&guid).map(|h| h.get()) {
                None => {
                    let handle = processor.request::<MaterialLoader>(db, MaterialSource { guid }, ());
                    self.data_pending.insert(guid, handle);
                    return None;
                }
                Some(None) => return None,
                Some(Some(Ok(data))) => {
                    self.data_pending.remove(&guid);
                    self.data_ready.insert(guid, data);
                }
                Some(Some(Err(e))) => {
                    log::warn!("material {guid:?} data failed to load: {e}");
                    self.data_pending.remove(&guid);
                    self.failed.insert(guid);
                    return None;
                }
            }
        }

        // Phase 2: resolve the shading model + its shader.
        let data = self.data_ready.get(&guid).expect("data_ready populated above");
        let Some(model) = registry.get(&data.shading_model) else {
            log::warn!(
                "material {guid:?}: unknown shading model '{}'",
                data.shading_model
            );
            self.data_ready.remove(&guid);
            self.failed.insert(guid);
            return None;
        };
        let shader = shader_mgr.get_or_request(processor, db, model.shader)?; // None → still loading

        // Phase 3: build + cache the resolved template.
        let resolved = Arc::new(ResolvedMaterial {
            shading_model: data.shading_model.clone(),
            shader_guid: model.shader,
            shader,
            properties: model.resolve(&data.properties),
        });
        self.data_ready.remove(&guid);
        self.resident.insert(guid, resolved.clone());
        Some(resolved)
    }

    /// The resolved template for `guid` if already resident — no request side effect.
    pub fn get(&self, guid: Guid) -> Option<Arc<ResolvedMaterial>> {
        self.resident.get(&guid).cloned()
    }
}
