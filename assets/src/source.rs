//! Asset identity: the stable [`Guid`] and the [`AssetSource`] trait every asset
//! kind declares.

use serde::{Deserialize, Serialize};

/// Stable, content-/path-independent asset identity, anchored in the central DB
/// (`guid -> (mount, path)`). Survives renames and content edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Guid(pub uuid::Uuid);

impl Guid {
    /// Mint a fresh random identity (assigned once, on first import).
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }

    /// A deterministic identity derived from a stable `name` (e.g. an asset's
    /// mount-relative path). The same name always yields the same guid, so a
    /// generated/committed pack regenerates without churn and references to it
    /// survive regeneration.
    pub fn stable(name: &str) -> Self {
        // Fixed namespace for RedLilium engine assets (UUIDv5).
        const NS: uuid::Uuid = uuid::Uuid::from_u128(0x52ed_1111_0000_0000_0000_0000_0000_0001);
        Self(uuid::Uuid::new_v5(&NS, name.as_bytes()))
    }
}

impl Default for Guid {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Guid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A per-kind description of *where an asset comes from*. It is the cache key
/// (so it must be `Hash + Eq`) and the serialized form stored in components /
/// prefabs.
///
/// File-backed sources expose their [`Guid`] so the processor can resolve a path
/// via the central DB and (re)read it; generated/embedded sources return `None`
/// and are synthesized without any I/O.
/// (`Send + Sync` so a source can be moved into an async `parse` task.)
pub trait AssetSource: Clone + Eq + std::hash::Hash + Send + Sync + 'static {
    /// The file identity backing this source, if any.
    fn file_guid(&self) -> Option<Guid>;
}
