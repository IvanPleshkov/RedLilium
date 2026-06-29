//! Sharing manager for shader assets — the single requester per guid, caching
//! the resident `Arc<Shader>` so materials binding the same shader share it.
//! Pulled by the material consumer (like [`VertexLayoutManager`] is pulled by the
//! mesh manager).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetHandle, AssetProcessor, Guid};

use crate::std::rendering::loaders::{Shader, ShaderLoader, ShaderSource};

/// Owns and shares resident shaders (an ECS resource).
#[derive(Default)]
pub struct ShaderManager {
    resident: HashMap<Guid, Arc<Shader>>,
    pending: HashMap<Guid, AssetHandle<Shader>>,
    failed: HashSet<Guid>,
}

impl ShaderManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// The resident shader for `guid`, requesting it once if not yet seen.
    /// `None` while loading (or after a failure).
    pub fn get_or_request(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        guid: Guid,
    ) -> Option<Arc<Shader>> {
        if let Some(shader) = self.resident.get(&guid) {
            return Some(shader.clone());
        }
        if self.failed.contains(&guid) {
            return None;
        }

        match self.pending.get(&guid).map(|h| h.get()) {
            None => {}
            Some(None) => return None,
            Some(Some(Ok(shader))) => {
                self.pending.remove(&guid);
                self.resident.insert(guid, shader.clone());
                return Some(shader);
            }
            Some(Some(Err(e))) => {
                log::warn!("shader {guid:?} failed to load: {e}");
                self.pending.remove(&guid);
                self.failed.insert(guid);
                return None;
            }
        }

        let handle = processor.request::<ShaderLoader>(db, ShaderSource { guid }, ());
        self.pending.insert(guid, handle);
        None
    }

    /// The resident shader for `guid` if already loaded — no request side effect.
    pub fn get(&self, guid: Guid) -> Option<Arc<Shader>> {
        self.resident.get(&guid).cloned()
    }
}
