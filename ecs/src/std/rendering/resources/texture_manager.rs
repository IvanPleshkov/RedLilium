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

use crate::PlayModeAware;
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
        // Virtual textures are published by their producing system (ADR-029),
        // never loaded — an unpublished one simply stays unresolved, exactly
        // like a still-loading file, until the producer publishes it.
        if matches!(source, TextureSource::Virtual(_)) {
            return;
        }
        self.demanded.push(source.clone());
    }

    /// Publish a runtime-produced texture as the virtual asset `guid`
    /// (ADR-029) — e.g. a camera's offscreen color output. Consumers resolve
    /// it through `TextureSource::Virtual(guid)` like any other texture.
    ///
    /// Re-publishing (a resized target) replaces the resident entry and bumps
    /// the generation, so `AssetRef` holders re-resolve to the new texture.
    pub fn publish_virtual(
        &mut self,
        guid: redlilium_assets::Guid,
        texture: Arc<Texture>,
    ) -> Result<(), redlilium_graphics::GraphicsError> {
        // Camera outputs are linear render targets: sample linearly, clamp at
        // the edges, no sRGB decode (the data is not an encoded image).
        use redlilium_core::sampler::{AddressMode, FilterMode};
        let settings = TextureSettings {
            srgb: false,
            filter: FilterMode::Linear,
            address: AddressMode::ClampToEdge,
            anisotropy: 1,
            // Virtual textures are published GPU targets, not blit-generated.
            generate_mips: false,
        };
        let sampler = self.intern_sampler(&settings)?;
        let source = TextureSource::Virtual(guid);
        self.cache.invalidate(&source);
        self.cache
            .publish(source, Arc::new(ResolvedTexture { texture, sampler }));
        Ok(())
    }

    /// The resolved texture for `source`, if loaded.
    pub fn get(&self, source: &TextureSource) -> Option<&Arc<ResolvedTexture>> {
        self.cache.get(source)
    }

    /// The graphics device textures are created on (shared engine-wide).
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
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

    /// Force MeshLoad to re-scan all asset refs by bumping generation.
    /// Called on snapshot restore to ensure unresolved refs get re-requested.
    /// See `MeshManager::request_rescan` — rescan without reloading.
    pub(crate) fn request_rescan(&mut self) {
        self.cache.bump_generation();
    }

    pub(crate) fn force_rescan(&mut self) {
        // Invalidate all RESIDENT texture sources (the actual loaded textures)
        // so the generation bumps. This forces MeshLoad to re-scan all asset refs
        // even in steady state (when nothing is demanded/pending). On re-scan,
        // unresolved refs get re-requested and re-resolved.
        // We collect keys first because invalidate mutates the cache.
        let sources: Vec<_> = self.cache.iter().map(|(k, _)| k.clone()).collect();
        for source in sources {
            self.cache.invalidate(&source);
        }
    }
}

impl PlayModeAware for TextureManager {
    fn on_stop(&mut self) {
        self.force_rescan();
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
