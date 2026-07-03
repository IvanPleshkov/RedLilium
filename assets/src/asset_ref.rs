//! [`AssetRef`] — an asset reference held in component data.
//!
//! The field carries both the asset's *identity* (its serialized [`AssetSource`])
//! and the *cached resolution* (the shared `Arc` of the resident resource), as
//! plain data — no locks, no indirection. Reads are a field access; all mutation
//! flows through a sync system that resolves refs against the owning manager, so
//! it stays visible to the ECS (borrows + change detection). Hot reload is the
//! same write path: the manager republishes a new `Arc` and the sync system
//! re-resolves stale refs (`Arc` pointer identity is the version).

use std::sync::Arc;

use crate::AssetSource;

/// Binds a source type to the resident resource it resolves to, so a single
/// generic [`AssetRef`] field works for any asset kind. The `Asset` is the
/// consumer-facing product (what the *manager* publishes), which may be richer
/// than what the loader produces.
pub trait AssetRefSource: AssetSource {
    type Asset;
    /// The DB record kind a file-backed source of this type references —
    /// what an inspector drop target accepts (e.g. `"mesh"`, `"texture"`).
    const KIND: &'static str;
}

/// An asset reference in component data: the source (serialized identity) plus
/// the cached resolved resource (runtime-only, filled by a sync system).
pub struct AssetRef<S: AssetRefSource> {
    source: S,
    resource: Option<Arc<S::Asset>>,
}

// Manual impl: a derive would require `S::Asset: Clone`, but cloning shares the
// `Arc` — the resource itself is never cloned.
impl<S: AssetRefSource + Clone> Clone for AssetRef<S> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            resource: self.resource.clone(),
        }
    }
}

impl<S: AssetRefSource> AssetRef<S> {
    /// An unresolved reference to `source` (resolves once the asset loads).
    pub fn new(source: S) -> Self {
        Self {
            source,
            resource: None,
        }
    }

    /// The asset's identity (what gets serialized).
    pub fn source(&self) -> &S {
        &self.source
    }

    /// The resolved resource, if it has loaded.
    pub fn get(&self) -> Option<&Arc<S::Asset>> {
        self.resource.as_ref()
    }

    /// Replace the cached resolution (sync system / hot reload).
    pub fn resolve(&mut self, resource: Arc<S::Asset>) {
        self.resource = Some(resource);
    }

    /// True if `resource` is the currently cached resolution (pointer identity).
    pub fn is_current(&self, resource: &Arc<S::Asset>) -> bool {
        self.resource
            .as_ref()
            .is_some_and(|r| Arc::ptr_eq(r, resource))
    }
}

impl<S: AssetRefSource + std::fmt::Debug> std::fmt::Debug for AssetRef<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssetRef")
            .field("source", &self.source)
            .field("resolved", &self.resource.is_some())
            .finish()
    }
}
