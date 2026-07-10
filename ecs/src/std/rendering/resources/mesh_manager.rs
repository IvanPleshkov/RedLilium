//! Asset-based GPU mesh management.
//!
//! `MeshManager` is the consumer-facing facade for meshes and the single source
//! of truth for resident `Arc<Mesh>`es: a component holds an
//! [`AssetRef<MeshSource>`](redlilium_assets::AssetRef) and the `MeshLoad` system
//! resolves it against this manager ([`request`](Self::request) +
//! [`get`](Self::get)). Loading is driven by [`drive`](Self::drive) (which
//! co-locks this manager + the layout manager + processor + DB), so consumers
//! never touch the vertex layout, the asset processor, or the DB.
//!
//! The mesh pipeline takes a **dependency** (the shared vertex layout `Arc`,
//! injected per request) and its sources aren't plain guids (`File | Generated`),
//! so this manager keeps its drive logic explicit on top of a
//! [`ResidentCache`](redlilium_assets::ResidentCache) — unlike the plain
//! guid-keyed managers, which are just
//! [`AssetManager`](redlilium_assets::AssetManager)s.

use std::collections::HashMap;
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetHandle, AssetProcessor, ResidentCache};
use redlilium_graphics::Mesh;

use super::VertexLayoutManager;
use crate::PlayModeAware;
use crate::std::rendering::loaders::{MeshLoader, MeshSource};

// A component-side `AssetRef<MeshSource>` resolves to the shared GPU mesh.
impl redlilium_assets::AssetRefSource for MeshSource {
    type Asset = Mesh;
    const KIND: &'static str = "mesh";
}

/// Owns and shares resident meshes (an ECS resource).
#[derive(Default)]
pub struct MeshManager {
    cache: ResidentCache<MeshSource, Mesh>,
    /// In-flight loads; the inner request is `None` until the layout dependency
    /// has resolved.
    pending: HashMap<MeshSource, Option<AssetHandle<Mesh>>>,
}

impl MeshManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure `source` is loading (idempotent; no-op if resident, in flight, or
    /// failed). The sync system calls this for unresolved refs.
    pub fn request(&mut self, source: &MeshSource) {
        if self.cache.get(source).is_some()
            || self.pending.contains_key(source)
            || self.cache.is_failed(source)
        {
            return;
        }
        self.pending.insert(source.clone(), None);
    }

    /// The resident mesh for `source`, if loaded.
    pub fn get(&self, source: &MeshSource) -> Option<&Arc<Mesh>> {
        self.cache.get(source)
    }

    /// Bumped whenever the resident set changes (load / reload).
    pub fn generation(&self) -> u64 {
        self.cache.generation()
    }

    /// Drop all state for the file mesh `guid` so it reloads (hot reload).
    /// Component refs keep serving the old `Arc` until the demand-driven sync
    /// re-requests and the new mesh lands.
    pub fn invalidate_file(&mut self, guid: redlilium_assets::Guid) {
        let source = MeshSource::File(guid);
        self.cache.invalidate(&source);
        self.pending.remove(&source);
    }

    /// Advance all in-flight loads: resolve each one's shared vertex layout (a
    /// file mesh references it in its DB record; a generated mesh gets it from
    /// the generator), request the mesh asset with that layout injected, then
    /// publish finished meshes as resident. Also pull-validates resident file
    /// meshes against their shared layout (hot reload — an unchanged layout
    /// re-interns to the same `Arc` and is skipped). Call from `MeshLoad`.
    pub fn drive(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        layout_mgr: &mut VertexLayoutManager,
    ) {
        // Pull-validation (hot reload): a file mesh whose shared layout has been
        // re-resolved to a *different* `Arc` rebuilds through the normal pending
        // flow; the resident mesh keeps serving until the new one lands.
        let mut revalidate: Vec<MeshSource> = Vec::new();
        for (source, mesh) in self.cache.iter() {
            if self.pending.contains_key(source) {
                continue;
            }
            let MeshSource::File(guid) = source else {
                continue; // generated meshes derive their layout from the generator
            };
            let Some(layout_guid) = db.record(guid).and_then(|r| r.reference("layout")) else {
                continue;
            };
            if let Some(layout) = layout_mgr.get_or_request(processor, db, layout_guid)
                && !Arc::ptr_eq(&layout, mesh.layout())
            {
                revalidate.push(source.clone());
            }
        }
        for source in revalidate {
            self.pending.insert(source, None);
        }

        let mut done: Vec<(MeshSource, Option<Arc<Mesh>>)> = Vec::new();

        for (source, request) in self.pending.iter_mut() {
            // Resolve the shared layout and kick off the mesh asset request once.
            if request.is_none() {
                let layout = match source {
                    MeshSource::File(guid) => {
                        let Some(layout_guid) = db.record(guid).and_then(|r| r.reference("layout"))
                        else {
                            continue; // record/layout reference not (yet) known
                        };
                        match layout_mgr.get_or_request(processor, db, layout_guid) {
                            Some(layout) => layout,
                            None => continue, // layout still loading
                        }
                    }
                    MeshSource::Generated(generator) => {
                        layout_mgr.intern((*generator.layout()).clone())
                    }
                };
                *request = Some(processor.request::<MeshLoader>(db, source.clone(), layout));
            }

            // Poll the mesh asset; publish when it lands.
            if let Some(handle) = request {
                match handle.get() {
                    None => {}
                    Some(Ok(mesh)) => done.push((source.clone(), Some(mesh))),
                    Some(Err(e)) => {
                        log::warn!("mesh {source:?} failed to load: {e}");
                        done.push((source.clone(), None));
                    }
                }
            }
        }

        for (source, mesh) in done {
            self.pending.remove(&source);
            match mesh {
                Some(mesh) => self.cache.publish(source, mesh),
                None => self.cache.fail(source),
            }
        }
    }

    /// Force MeshLoad to re-scan all asset refs by bumping generation.
    /// Called on snapshot restore to ensure unresolved refs get re-requested.
    pub(crate) fn force_rescan(&mut self) {
        // Invalidate all RESIDENT mesh sources (the actual loaded meshes) so the
        // generation bumps. This forces MeshLoad to re-scan all asset refs even
        // in steady state (when nothing is in-flight). On re-scan, unresolved refs
        // get re-requested and re-resolved.
        // We collect keys first because invalidate mutates the cache.
        let keys: Vec<_> = self.cache.iter().map(|(k, _)| k.clone()).collect();
        for source in keys {
            self.cache.invalidate(&source);
        }
    }
}

impl PlayModeAware for MeshManager {
    fn on_stop(&mut self) {
        self.force_rescan();
    }
}
