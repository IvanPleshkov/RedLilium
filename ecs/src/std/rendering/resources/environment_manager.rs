//! Asset-based IBL-environment resolver (#145).
//!
//! [`EnvironmentManager`] is the consumer facade a
//! [`CameraEnvironment`](crate::std::rendering::CameraEnvironment) binds (via an
//! `AssetRef<EnvironmentSource>`). Resolution ([`drive`](Self::drive)) loads the
//! environment data (the three cubemap guids, via an embedded
//! [`AssetManager`](redlilium_assets::AssetManager)) and pulls those textures
//! from the [`TextureManager`] — far lighter than the material path (no property
//! packing, no binding groups: the deferred pipeline builds its own bindings
//! from the resolved textures).
//!
//! Hot reload: `drive` pull-validates every resident environment's cubemap
//! `Arc`s (a re-resolved texture rebuilds the environment, serving last-good
//! meanwhile), and [`invalidate`](Self::invalidate) drops the resolution + data
//! so an edited record reloads from fresh settings.

use std::collections::HashSet;
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetManager, AssetProcessor, Guid, ResidentCache};

use super::{ResolvedTexture, TextureManager};
use crate::std::rendering::loaders::{EnvironmentLoader, EnvironmentSource, TextureSource};

/// One resolved cubemap: its source (for hot-reload pull-validation) and the
/// resident texture. Environments carry exactly three (irradiance, prefilter,
/// sky).
type ResolvedCube = (TextureSource, Arc<ResolvedTexture>);

/// A fully resolved IBL environment: the three resident cubemaps plus the
/// reflection-LOD range derived from the prefilter chain. The resolved textures
/// are retained so hot reload can pull-validate (an `Arc` mismatch rebuilds).
#[derive(Debug)]
pub struct ResolvedEnvironment {
    /// Diffuse irradiance cubemap.
    pub irradiance: Arc<ResolvedTexture>,
    /// Specular prefiltered-environment cubemap.
    pub prefilter: Arc<ResolvedTexture>,
    /// Sky background cubemap.
    pub sky: Arc<ResolvedTexture>,
    /// Highest prefilter mip index (`mip_count - 1`) — the value the resolve
    /// shader multiplies roughness by. Derived from the resolved prefilter
    /// texture, so it always matches the baked chain (no hardcoded constant).
    pub max_reflection_lod: f32,
    /// The (source, resolution) the cubemaps were built from, in
    /// irradiance/prefilter/sky order — hot-reload pull-validation compares
    /// these `Arc`s against the texture manager's current ones.
    sources: [ResolvedCube; 3],
}

// A component-side `AssetRef<EnvironmentSource>` resolves to the manager's
// product — the fully resolved environment, not the loader's raw data.
impl redlilium_assets::AssetRefSource for EnvironmentSource {
    type Asset = ResolvedEnvironment;
    const KIND: &'static str = "environment";
}

/// Owns and shares resolved IBL environments (an ECS resource).
#[derive(Default)]
pub struct EnvironmentManager {
    /// The data phase: guid → resident [`EnvironmentData`].
    data: AssetManager<EnvironmentLoader>,
    /// The resolution: guid → the single shared [`ResolvedEnvironment`].
    cache: ResidentCache<Guid, ResolvedEnvironment>,
    /// Environments being (re)resolved — demanded but not yet published.
    demanded: HashSet<Guid>,
}

impl EnvironmentManager {
    /// Create an empty environment manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure the environment is resolving (idempotent; no-op if resident, in
    /// flight, or failed). The sync system calls this for unresolved refs.
    pub fn request(&mut self, source: &EnvironmentSource) {
        let guid = source.guid;
        if self.cache.get(&guid).is_some() || self.cache.is_failed(&guid) {
            return;
        }
        self.demanded.insert(guid);
    }

    /// The resolved environment for `guid`, if resolved.
    pub fn get(&self, guid: Guid) -> Option<&Arc<ResolvedEnvironment>> {
        self.cache.get(&guid)
    }

    /// Bumped whenever the resident set changes (load / reload).
    pub fn generation(&self) -> u64 {
        self.cache.generation()
    }

    /// Drop all state for `guid` — the resolution *and* the data — so it
    /// re-resolves from fresh record settings (hot reload).
    pub fn invalidate(&mut self, guid: Guid) {
        self.cache.invalidate(&guid);
        self.data.invalidate(guid);
        self.demanded.remove(&guid);
    }

    /// Advance all demanded environments: load the data, resolve the three
    /// cubemaps via the texture manager, publish when all are resident. Also
    /// pull-validates resident environments against their textures (hot
    /// reload). Call from the environment-load system, which co-locks the
    /// managers.
    pub fn drive(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        texture_mgr: &mut TextureManager,
    ) {
        // Pull-validation (hot reload): re-resolve resident environments whose
        // any bound cubemap has been re-resolved (`Arc` mismatch) or
        // invalidated. The old environment keeps serving until the rebuild
        // republishes over it.
        for (guid, resolved) in self.cache.iter() {
            if self.demanded.contains(guid) {
                continue;
            }
            for (source, texture) in &resolved.sources {
                match texture_mgr.get(source) {
                    Some(current) if !Arc::ptr_eq(current, texture) => {
                        self.demanded.insert(*guid);
                        break;
                    }
                    Some(_) => {}
                    None => texture_mgr.request(source),
                }
            }
        }

        let mut ready: Vec<(Guid, [ResolvedCube; 3])> = Vec::new();
        let mut failed_now: Vec<Guid> = Vec::new();

        'demanded: for guid in self.demanded.iter().copied() {
            // Phase 1: the environment data (request once, then poll).
            let Some(data) = self.data.get_or_request(processor, db, guid) else {
                if self.data.is_failed(guid) {
                    failed_now.push(guid);
                }
                continue;
            };

            // Phase 2: resolve the three cubemaps. All must be resident before
            // publishing.
            let sources = [
                TextureSource::File(data.irradiance),
                TextureSource::File(data.prefilter),
                TextureSource::File(data.sky),
            ];
            let mut resolved: Vec<ResolvedCube> = Vec::with_capacity(3);
            for source in sources {
                if texture_mgr.is_failed(&source) {
                    log::warn!("environment {guid:?}: cubemap {source:?} failed to load");
                    failed_now.push(guid);
                    continue 'demanded;
                }
                match texture_mgr.get(&source) {
                    Some(texture) => resolved.push((source, texture.clone())),
                    None => {
                        texture_mgr.request(&source);
                        continue 'demanded; // still loading
                    }
                }
            }
            let arr: [ResolvedCube; 3] = resolved.try_into().expect("exactly three cubemaps");
            ready.push((guid, arr));
        }

        for (guid, sources) in ready {
            let max_reflection_lod = (sources[1].1.texture.mip_level_count().max(1) - 1) as f32;
            let resolved = Arc::new(ResolvedEnvironment {
                irradiance: sources[0].1.clone(),
                prefilter: sources[1].1.clone(),
                sky: sources[2].1.clone(),
                max_reflection_lod,
                sources,
            });
            self.demanded.remove(&guid);
            self.cache.publish(guid, resolved);
        }

        for guid in failed_now {
            self.demanded.remove(&guid);
            self.cache.fail(guid);
        }
    }
}
