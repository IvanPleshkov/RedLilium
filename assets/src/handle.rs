//! The consumer-facing [`AssetHandle`] and the per-request slot behind it.
//!
//! Contract: a consumer calls `request(source)` and gets an `AssetHandle<T>` —
//! an `Arc`-backed handle that (a) carries the eventual result and (b) **is the
//! demand for loading** while held. The consumer polls [`get`](AssetHandle::get)
//! each tick (or `await`s); once it has the `Arc<T>`, it drops the handle and
//! keeps the `Arc<T>` (that becomes the resource's lifetime — one refcount).
//!
//! Dropping the handle **before** completion cancels the in-flight load: the
//! processor sees the request is no longer demanded (the handle's strong count
//! dropped) and trips the [`CancellationToken`], so the read/decode tasks bail
//! at their next checkpoint.
//!
//! There is **no dedup** — each `request` is an independent load. Sharing/reuse
//! is the consumer's responsibility.

// The slot's setters/token are driven by the request processor (next chunk).
#![allow(dead_code)]

use std::sync::Arc;

use parking_lot::RwLock;

use crate::error::AssetError;

/// The result-delivery channel behind an [`AssetHandle`]: a one-shot cell the
/// processor fills. Behind an `RwLock` so the handle is `Send + Sync` (it can
/// ride in an ECS component) — the processor only locks to write the result and
/// consumers lock to read it. Demand is the handle's `Arc` strong count (the
/// processor holds only a `Weak`).
pub(crate) struct RequestSlot<T> {
    /// `None` while loading; `Some(Ok)` ready; `Some(Err)` failed.
    result: RwLock<Option<Result<Arc<T>, AssetError>>>,
}

impl<T> RequestSlot<T> {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            result: RwLock::new(None),
        })
    }

    pub(crate) fn fulfill(&self, result: Result<Arc<T>, AssetError>) {
        *self.result.write() = Some(result);
    }
}

/// A handle to a requested asset — demand-while-loading + access to the result.
///
/// Cheap to clone (`Arc`). Hold it until loaded; then take the `Arc<T>` and drop
/// the handle. Dropping before completion cancels the load.
pub struct AssetHandle<T> {
    slot: Arc<RequestSlot<T>>,
}

impl<T> AssetHandle<T> {
    pub(crate) fn new(slot: Arc<RequestSlot<T>>) -> Self {
        Self { slot }
    }

    /// The current result: `None` while loading, `Some(Ok)` when ready,
    /// `Some(Err)` on failure. Cheap (clones the `Arc<T>` / error).
    pub fn get(&self) -> Option<Result<Arc<T>, AssetError>> {
        self.slot.result.read().clone()
    }

    /// The loaded resource if ready, else `None`.
    pub fn ready(&self) -> Option<Arc<T>> {
        match self.get() {
            Some(Ok(arc)) => Some(arc),
            _ => None,
        }
    }
}

impl<T> Clone for AssetHandle<T> {
    fn clone(&self) -> Self {
        Self {
            slot: Arc::clone(&self.slot),
        }
    }
}
