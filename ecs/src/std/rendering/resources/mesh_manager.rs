//! Asset-based GPU mesh management.
//!
//! `MeshManager` is the consumer-facing facade for meshes: you `request` a mesh
//! by [`MeshSource`] and get a [`MeshHandle`] that resolves to an `Arc<Mesh>`
//! once loaded — you never touch the vertex layout, the asset processor, or the
//! DB. The actual loading is driven by the `MeshLoad` system (which co-locks this
//! manager + the layout manager + processor + DB), so the consumer side here has
//! no dependencies of its own.
//!
//! This manager is the single requester per `MeshSource` (the asset processor
//! does not dedup), so all consumers of the same source share one `Arc<Mesh>`.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use redlilium_assets::{AssetDb, AssetHandle, AssetProcessor};
use redlilium_graphics::Mesh;

use super::VertexLayoutManager;
use crate::std::rendering::loaders::{MeshLoader, MeshSource};

/// A demand-to-load handle for a mesh. Cloneable and cheap; poll [`get`](Self::get)
/// for the resident `Arc<Mesh>` once it has loaded (`None` until then).
#[derive(Clone, Default, Debug)]
pub struct MeshHandle {
    slot: Arc<RwLock<Option<Arc<Mesh>>>>,
}

impl MeshHandle {
    /// The resident mesh, if it has finished loading.
    pub fn get(&self) -> Option<Arc<Mesh>> {
        self.slot.read().clone()
    }

    fn fulfill(&self, mesh: Arc<Mesh>) {
        *self.slot.write() = Some(mesh);
    }
}

/// In-flight mesh request: the consumer handle to fulfil, and the inner mesh-asset
/// request (set once the layout dependency has resolved).
struct PendingMesh {
    handle: MeshHandle,
    asset: Option<AssetHandle<Mesh>>,
}

/// Owns and shares resident meshes (an ECS resource).
#[derive(Default)]
pub struct MeshManager {
    resident: HashMap<MeshSource, Arc<Mesh>>,
    pending: HashMap<MeshSource, PendingMesh>,
}

impl MeshManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request a mesh by source, returning a handle that resolves once loaded.
    /// Deduplicates by source: repeat requests of an in-flight source share the
    /// same handle, and a resident source resolves immediately.
    pub fn request(&mut self, source: MeshSource) -> MeshHandle {
        if let Some(mesh) = self.resident.get(&source) {
            let handle = MeshHandle::default();
            handle.fulfill(mesh.clone());
            return handle;
        }
        if let Some(pending) = self.pending.get(&source) {
            return pending.handle.clone();
        }
        let handle = MeshHandle::default();
        self.pending.insert(
            source,
            PendingMesh {
                handle: handle.clone(),
                asset: None,
            },
        );
        handle
    }

    /// The resident mesh for `source` if already loaded — no request side effect.
    pub fn get(&self, source: &MeshSource) -> Option<Arc<Mesh>> {
        self.resident.get(source).cloned()
    }

    /// Advance all in-flight mesh requests: resolve each one's shared vertex
    /// layout (a file mesh references it in its DB record; a generated mesh gets
    /// it from the generator), request the mesh asset with that layout injected,
    /// then fulfil handles as meshes finish. Call from the `MeshLoad` system,
    /// which provides the co-locked processor / DB / layout manager.
    pub fn drive(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        layout_mgr: &mut VertexLayoutManager,
    ) {
        let mut done: Vec<(MeshSource, Option<Arc<Mesh>>)> = Vec::new();

        for (source, pending) in self.pending.iter_mut() {
            // Resolve the shared layout and kick off the mesh asset request once.
            if pending.asset.is_none() {
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
                pending.asset = Some(processor.request::<MeshLoader>(db, source.clone(), layout));
            }

            // Poll the mesh asset; fulfil the handle when it lands.
            if let Some(handle) = &pending.asset {
                match handle.get() {
                    None => {}
                    Some(Ok(mesh)) => {
                        pending.handle.fulfill(mesh.clone());
                        done.push((source.clone(), Some(mesh)));
                    }
                    Some(Err(e)) => {
                        log::warn!("mesh {source:?} failed to load: {e}");
                        done.push((source.clone(), None));
                    }
                }
            }
        }

        for (source, mesh) in done {
            self.pending.remove(&source);
            if let Some(mesh) = mesh {
                self.resident.insert(source, mesh);
            }
        }
    }
}
