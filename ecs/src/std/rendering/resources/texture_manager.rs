//! Asset-based GPU texture management.
//!
//! `TextureManager` is the consumer-facing facade for textures and the single
//! source of truth for resident [`ResolvedTexture`]s: material instances
//! resolve their texture properties against it, and a component can hold an
//! [`AssetRef<TextureSource>`](redlilium_assets::AssetRef) that the `MeshLoad`
//! sync resolves the same way. Loading is driven by [`drive`](Self::drive);
//! pixel uploads flow through the loader's GPU stage + `AssetGpuFlush` (never a
//! synchronous write).
//!
//! A sampler is **texture metadata, not an asset**: its parameters live in the
//! record's [`TextureSettings`] and the manager interns the GPU objects by
//! content — textures sharing parameters share one `Arc<Sampler>`, and a
//! settings edit simply resolves to a different interned sampler on reload
//! (samplers themselves are immutable, so the intern cache never invalidates).
//!
//! Texture sources aren't plain guids (`File | Solid`), so this manager keeps
//! its own (dependency-free) drive loop on top of a
//! [`ResidentCache`](redlilium_assets::ResidentCache) instead of being a plain
//! [`AssetManager`](redlilium_assets::AssetManager).

use std::collections::HashMap;
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetHandle, AssetProcessor, ResidentCache};
use redlilium_graphics::{GraphicsDevice, Sampler, Texture};

use crate::std::rendering::loaders::{TextureLoader, TextureSettings, TextureSource};

/// A fully resolved texture: the resident GPU texture plus the sampler its
/// record settings describe (shared/interned — equal parameters, one `Arc`).
#[derive(Debug)]
pub struct ResolvedTexture {
    pub texture: Arc<Texture>,
    pub sampler: Arc<Sampler>,
}

// A component-side `AssetRef<TextureSource>` resolves to the manager's product.
impl redlilium_assets::AssetRefSource for TextureSource {
    type Asset = ResolvedTexture;
    const KIND: &'static str = "texture";
}

/// Owns and shares resident GPU textures (an ECS resource).
pub struct TextureManager {
    device: Arc<GraphicsDevice>,
    cache: ResidentCache<TextureSource, ResolvedTexture>,
    /// In-flight loads.
    pending: HashMap<TextureSource, AssetHandle<Texture>>,
    /// Sources demanded but not yet requested (no processor at `request` time).
    demanded: Vec<TextureSource>,
    /// Settings → the canonical shared sampler. Samplers are immutable value
    /// objects: a settings edit is a *different* key, never an invalidation.
    samplers: HashMap<TextureSettings, Arc<Sampler>>,
}

impl TextureManager {
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            cache: ResidentCache::new(),
            pending: HashMap::new(),
            demanded: Vec::new(),
            samplers: HashMap::new(),
        }
    }

    /// Ensure `source` is loading (idempotent; no-op if resident, in flight, or
    /// failed). Consumers (the instance manager, the sync system) call this for
    /// unresolved sources.
    pub fn request(&mut self, source: &TextureSource) {
        if self.cache.get(source).is_some()
            || self.cache.is_failed(source)
            || self.pending.contains_key(source)
            || self.demanded.contains(source)
        {
            return;
        }
        self.demanded.push(source.clone());
    }

    /// The resolved texture for `source`, if loaded.
    pub fn get(&self, source: &TextureSource) -> Option<&Arc<ResolvedTexture>> {
        self.cache.get(source)
    }

    /// Whether `source` failed to load (latched until invalidated).
    pub fn is_failed(&self, source: &TextureSource) -> bool {
        self.cache.is_failed(source)
    }

    /// Bumped whenever the resident set changes (load / reload).
    pub fn generation(&self) -> u64 {
        self.cache.generation()
    }

    /// Drop all state for the file texture `guid` so it reloads (hot reload —
    /// file content *or* settings edits; both re-resolve on reload). Consumers
    /// keep serving the old `Arc` until the new texture lands, then rebuild by
    /// pointer identity.
    pub fn invalidate_file(&mut self, guid: redlilium_assets::Guid) {
        let source = TextureSource::File(guid);
        self.cache.invalidate(&source);
        self.pending.remove(&source);
        self.demanded.retain(|s| s != &source);
    }

    /// Advance all in-flight loads: request demanded sources, poll pending
    /// ones, pair finished textures with their (interned) sampler and publish.
    /// Call from a load system.
    pub fn drive(&mut self, processor: &mut AssetProcessor, db: &AssetDb) {
        for source in self.demanded.drain(..) {
            let handle = processor.request::<TextureLoader>(db, source.clone(), ());
            self.pending.insert(source, handle);
        }

        let mut done: Vec<(TextureSource, Option<Arc<Texture>>)> = Vec::new();
        for (source, handle) in self.pending.iter() {
            match handle.get() {
                None => {}
                Some(Ok(texture)) => done.push((source.clone(), Some(texture))),
                Some(Err(e)) => {
                    log::warn!("texture {source:?} failed to load: {e}");
                    done.push((source.clone(), None));
                }
            }
        }
        for (source, texture) in done {
            self.pending.remove(&source);
            let Some(texture) = texture else {
                self.cache.fail(source);
                continue;
            };
            let settings = record_settings(db, &source);
            match self.intern_sampler(&settings) {
                Ok(sampler) => {
                    self.cache
                        .publish(source, Arc::new(ResolvedTexture { texture, sampler }));
                }
                Err(e) => {
                    log::warn!("texture {source:?}: sampler failed: {e}");
                    self.cache.fail(source);
                }
            }
        }
    }

    /// The shared sampler for `settings` — one GPU object per distinct
    /// parameter set.
    fn intern_sampler(
        &mut self,
        settings: &TextureSettings,
    ) -> Result<Arc<Sampler>, redlilium_graphics::GraphicsError> {
        if let Some(sampler) = self.samplers.get(settings) {
            return Ok(sampler.clone());
        }
        let sampler = self
            .device
            .create_sampler_from_cpu(&settings.to_sampler())?;
        self.samplers.insert(settings.clone(), sampler.clone());
        Ok(sampler)
    }
}

/// The texture settings of `source`'s DB record (defaults for `Solid` sources
/// and records without settings).
fn record_settings(db: &AssetDb, source: &TextureSource) -> TextureSettings {
    let TextureSource::File(guid) = source else {
        return TextureSettings::default();
    };
    db.record(guid)
        .and_then(|r| r.settings.as_deref())
        .and_then(|s| ron::from_str::<TextureSettings>(s).ok())
        .unwrap_or_default()
}
