//! Asset-based GPU texture management.
//!
//! `TextureManager` is the consumer-facing facade for textures and the single
//! source of truth for resident `Arc<Texture>`es: material instances resolve
//! their texture properties against it, and a component can hold an
//! [`AssetRef<TextureSource>`](redlilium_assets::AssetRef) that the `MeshLoad`
//! sync resolves the same way. Loading is driven by [`drive`](Self::drive);
//! pixel uploads flow through the loader's GPU stage + `AssetGpuFlush` (never a
//! synchronous write).
//!
//! Texture sources aren't plain guids (`File | Solid`), so this manager keeps
//! its own (dependency-free) drive loop on top of a
//! [`ResidentCache`](redlilium_assets::ResidentCache) instead of being a plain
//! [`AssetManager`](redlilium_assets::AssetManager).

use std::collections::HashMap;
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetHandle, AssetProcessor, ResidentCache};
use redlilium_graphics::Texture;

use crate::std::rendering::loaders::{TextureLoader, TextureSource};

// A component-side `AssetRef<TextureSource>` resolves to the shared GPU texture.
impl redlilium_assets::AssetRefSource for TextureSource {
    type Asset = Texture;
}

/// Owns and shares resident GPU textures (an ECS resource).
#[derive(Default)]
pub struct TextureManager {
    cache: ResidentCache<TextureSource, Texture>,
    /// In-flight loads.
    pending: HashMap<TextureSource, AssetHandle<Texture>>,
    /// Sources demanded but not yet requested (no processor at `request` time).
    demanded: Vec<TextureSource>,
}

impl TextureManager {
    pub fn new() -> Self {
        Self::default()
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

    /// The resident texture for `source`, if loaded.
    pub fn get(&self, source: &TextureSource) -> Option<&Arc<Texture>> {
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

    /// Drop all state for the file texture `guid` so it reloads (hot reload).
    /// Consumers keep serving the old `Arc` until the new texture lands, then
    /// rebuild by pointer identity.
    pub fn invalidate_file(&mut self, guid: redlilium_assets::Guid) {
        let source = TextureSource::File(guid);
        self.cache.invalidate(&source);
        self.pending.remove(&source);
        self.demanded.retain(|s| s != &source);
    }

    /// Advance all in-flight loads: request demanded sources, poll pending
    /// ones, publish finished textures as resident. Call from a load system.
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
            match texture {
                Some(texture) => self.cache.publish(source, texture),
                None => self.cache.fail(source),
            }
        }
    }
}
