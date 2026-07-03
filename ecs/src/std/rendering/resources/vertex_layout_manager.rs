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
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetManager, AssetProcessor, Guid};
use redlilium_core::mesh::VertexLayout;

use crate::std::rendering::loaders::VertexLayoutLoader;

/// Owns and shares `Arc<VertexLayout>` instances (an ECS resource).
#[derive(Default)]
pub struct VertexLayoutManager {
    inner: AssetManager<VertexLayoutLoader>,
    /// Content → the canonical shared `Arc`: identical layouts (across guids, or
    /// generated) collapse to one `Arc` so pointer-equality batching holds.
    interned: HashMap<VertexLayout, Arc<VertexLayout>>,
    /// Per-guid memo `(inner Arc ptr → interned Arc)` so a resident hit skips
    /// the content hash; invalidated implicitly when the inner `Arc` changes.
    memo: HashMap<Guid, (usize, Arc<VertexLayout>)>,
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

    /// The shared (interned) layout for `guid`, requesting it once if not yet
    /// seen. `None` while loading (or after a failure) — call again next frame.
    pub fn get_or_request(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        guid: Guid,
    ) -> Option<Arc<VertexLayout>> {
        let raw = self.inner.get_or_request(processor, db, guid)?;
        let raw_ptr = Arc::as_ptr(&raw) as usize;
        if let Some((ptr, shared)) = self.memo.get(&guid)
            && *ptr == raw_ptr
        {
            return Some(shared.clone());
        }
        let shared = self.intern_arc(raw);
        self.memo.insert(guid, (raw_ptr, shared.clone()));
        Some(shared)
    }

    /// Drop the loaded state for `guid` so it reloads (hot reload). Unchanged
    /// content re-interns to the same `Arc` — dependants see pointer equality
    /// and skip rebuilding.
    pub fn invalidate(&mut self, guid: Guid) {
        self.inner.invalidate(guid);
        // The memo entry self-invalidates: the reloaded inner `Arc` is new, so
        // the stored pointer no longer matches.
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
