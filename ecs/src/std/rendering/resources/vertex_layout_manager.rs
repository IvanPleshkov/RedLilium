//! Sharing manager for [`VertexLayout`] assets.
//!
//! `VertexLayout` must be shared via `Arc` between meshes and materials — the
//! renderer batches by `Arc` pointer-equality, so two consumers binding the same
//! layout must hold the *same* `Arc`. On top of the standard
//! [`AssetManager`](redlilium_assets::AssetManager) (single requester, failure
//! latch, hot reload), this manager **interns by content**: distinct sources (or
//! generated layouts) of equal content collapse to one `Arc`. On hot reload an
//! unchanged layout re-interns to the *same* `Arc`, so dependants (which
//! pull-validate by pointer identity) skip rebuilding.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use redlilium_assets::{AssetDb, AssetManager, AssetProcessor, Guid};
use redlilium_core::mesh::VertexLayout;

use crate::std::rendering::loaders::VertexLayoutLoader;

/// Owns and shares `Arc<VertexLayout>` instances (an ECS resource).
#[derive(Default)]
pub struct VertexLayoutManager {
    inner: AssetManager<VertexLayoutLoader>,
    /// Content → the canonical shared `Arc`: identical layouts (across guids, or
    /// generated) collapse to one `Arc` so pointer-equality batching holds.
    interned: HashMap<VertexLayout, Weak<VertexLayout>>,
    /// Current canonical version by guid. Weak identity cannot suffer address
    /// reuse; invalidation clears it along with the inner cache.
    memo: HashMap<Guid, Weak<VertexLayout>>,
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
        if let Some(arc) = self.interned.get(&layout).and_then(Weak::upgrade) {
            return arc;
        }
        let arc = Arc::new(layout.clone());
        self.interned.insert(layout, Arc::downgrade(&arc));
        arc
    }

    /// Intern an already-`Arc`'d layout by content (reuse an equal existing
    /// `Arc` if present, else adopt this one as the canonical share).
    fn intern_arc(&mut self, arc: Arc<VertexLayout>) -> Arc<VertexLayout> {
        if let Some(existing) = self.interned.get(arc.as_ref()).and_then(Weak::upgrade) {
            return existing;
        }
        self.interned.insert((*arc).clone(), Arc::downgrade(&arc));
        arc
    }

    /// The shared (interned) layout for `guid`, requesting it once if not yet
    /// seen. `None` while loading (or after a failure) — call again next frame.
    pub fn get_or_request(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        guid: Guid,
    ) -> Option<Arc<VertexLayout>> {
        if let Some(shared) = self.memo.get(&guid).and_then(Weak::upgrade) {
            return Some(shared);
        }
        let raw = self.inner.get_or_request(processor, db, guid)?;
        let shared = self.intern_arc(raw);
        // Retain the canonical allocation, not a second equal raw layout.
        self.inner.publish(guid, shared.clone());
        self.memo.insert(guid, Arc::downgrade(&shared));
        Some(shared)
    }

    /// Release the file cache's ownership without losing a live canonical Arc.
    pub fn release(&mut self, guid: Guid) {
        self.inner.release(guid);
    }

    /// Collect cache-only layouts and prune expired content/memo weak entries.
    pub fn collect_unused(&mut self) -> usize {
        let removed = self.inner.collect_unused();
        self.memo.retain(|_, value| value.strong_count() > 0);
        self.interned.retain(|_, value| value.strong_count() > 0);
        removed
    }

    /// Drop the loaded state for `guid` so it reloads (hot reload). Unchanged
    /// content re-interns to the same `Arc` — dependants see pointer equality
    /// and skip rebuilding.
    pub fn invalidate(&mut self, guid: Guid) {
        self.inner.invalidate(guid);
        self.memo.remove(&guid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_weak_and_collection_prunes_dead_allocations() {
        let mut manager = VertexLayoutManager::new();
        let layout = (*VertexLayout::position_only()).clone();
        let shared = manager.intern(layout.clone());
        assert_eq!(Arc::strong_count(&shared), 1);
        manager.collect_unused();
        assert!(Arc::ptr_eq(&manager.intern(layout.clone()), &shared));
        drop(shared);
        manager.collect_unused();
        assert!(manager.interned.is_empty());
    }

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
