//! Sharing manager for [`VertexLayout`] assets.
//!
//! `VertexLayout` must be shared via `Arc` between meshes and materials — the
//! renderer batches by `Arc` pointer-equality, so two consumers binding the same
//! layout must hold the *same* `Arc`. The [`AssetProcessor`] does not deduplicate
//! requests (sharing is a consumer concern by design), so this manager is the
//! **single requester per source guid**: it caches the resident `Arc`, and
//! additionally interns by content so distinct sources (or generated layouts) of
//! equal content collapse to one `Arc`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetHandle, AssetProcessor, Guid};
use redlilium_core::mesh::VertexLayout;

use crate::std::rendering::loaders::{VertexLayoutLoader, VertexLayoutSource};

/// Owns and shares `Arc<VertexLayout>` instances (an ECS resource).
#[derive(Default)]
pub struct VertexLayoutManager {
    /// guid → the single shared resident layout for that source.
    resident: HashMap<Guid, Arc<VertexLayout>>,
    /// In-flight requests awaiting delivery (this manager is the sole requester).
    pending: HashMap<Guid, AssetHandle<VertexLayout>>,
    /// Content → shared `Arc`: identical layouts (across guids, or generated)
    /// collapse to one `Arc` so pointer-equality (and thus batching) holds.
    interned: HashMap<VertexLayout, Arc<VertexLayout>>,
    /// Sources whose load failed — not re-requested (avoids per-frame retry spam).
    failed: HashSet<Guid>,
}

impl VertexLayoutManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a layout by content → one shared `Arc` per distinct content.
    ///
    /// Use this for generated / non-file layouts so they still share a pointer
    /// with file-loaded layouts of equal content.
    pub fn intern(&mut self, layout: VertexLayout) -> Arc<VertexLayout> {
        if let Some(arc) = self.interned.get(&layout) {
            return arc.clone();
        }
        let arc = Arc::new(layout.clone());
        self.interned.insert(layout, arc.clone());
        arc
    }

    /// Intern an already-`Arc`'d layout by content (reuse an equal existing
    /// `Arc` if present, else adopt this one as the canonical share).
    fn intern_arc(&mut self, arc: Arc<VertexLayout>) -> Arc<VertexLayout> {
        if let Some(existing) = self.interned.get(arc.as_ref()) {
            return existing.clone();
        }
        self.interned.insert((*arc).clone(), arc.clone());
        arc
    }

    /// The resident layout for `guid`, requesting it once if not yet seen.
    /// Returns `None` while loading (or after a failure). Call again next frame to
    /// advance the request.
    pub fn get_or_request(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        guid: Guid,
    ) -> Option<Arc<VertexLayout>> {
        if let Some(arc) = self.resident.get(&guid) {
            return Some(arc.clone());
        }
        if self.failed.contains(&guid) {
            return None;
        }

        // Poll the pending handle (if any) without holding a borrow of `pending`.
        match self.pending.get(&guid).map(|h| h.get()) {
            // Not requested yet → request below.
            None => {}
            // Requested, still loading.
            Some(None) => return None,
            // Delivered: intern by content and cache as resident.
            Some(Some(Ok(arc))) => {
                self.pending.remove(&guid);
                let shared = self.intern_arc(arc);
                self.resident.insert(guid, shared.clone());
                return Some(shared);
            }
            // Failed: drop and remember so we don't re-request every frame.
            Some(Some(Err(e))) => {
                log::warn!("vertex layout {guid:?} failed to load: {e}");
                self.pending.remove(&guid);
                self.failed.insert(guid);
                return None;
            }
        }

        let handle = processor.request::<VertexLayoutLoader>(db, VertexLayoutSource { guid });
        self.pending.insert(guid, handle);
        None
    }

    /// The resident layout for `guid` if already loaded — no request side effect.
    pub fn get(&self, guid: Guid) -> Option<Arc<VertexLayout>> {
        self.resident.get(&guid).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Interning collapses equal-content layouts to one `Arc` (pointer-equal,
    /// which is what the renderer batches on) and keeps distinct ones apart.
    #[test]
    fn intern_dedups_by_content() {
        let mut m = VertexLayoutManager::new();
        let a = m.intern((*VertexLayout::pbr()).clone());
        let b = m.intern((*VertexLayout::pbr()).clone());
        assert!(Arc::ptr_eq(&a, &b), "equal content must share one Arc");

        let c = m.intern((*VertexLayout::position_only()).clone());
        assert!(
            !Arc::ptr_eq(&a, &c),
            "distinct content must not share an Arc"
        );
    }
}
