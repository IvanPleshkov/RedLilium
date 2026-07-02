//! [`ChangedAssets`] — the hot-reload inbox (an ECS resource).
//!
//! Anything that observes an asset change pushes the guid here: the editor's
//! asset inspector (settings edits) and the filesystem watcher (external file
//! edits). The `HotReload` system drains it each frame and invalidates the
//! *owning* manager per guid; everything downstream (dependent managers, the
//! pipeline cache, component `AssetRef`s) catches up through pull-validation by
//! `Arc` pointer identity — see `docs/ASSETS.md` §9.

use redlilium_assets::Guid;

/// Guids whose source data changed and should be reloaded (an ECS resource).
#[derive(Default)]
pub struct ChangedAssets {
    guids: Vec<Guid>,
}

impl ChangedAssets {
    pub fn new() -> Self {
        Self::default()
    }

    /// Report that an asset's source data changed.
    pub fn push(&mut self, guid: Guid) {
        if !self.guids.contains(&guid) {
            self.guids.push(guid);
        }
    }

    /// Take all pending change reports.
    pub fn drain(&mut self) -> Vec<Guid> {
        std::mem::take(&mut self.guids)
    }

    /// Whether there are pending change reports.
    pub fn is_empty(&self) -> bool {
        self.guids.is_empty()
    }
}
