//! The [`AssetLoader`] trait — identity + a method that assembles the pipeline.
//!
//! The stage sequence is **runtime data, not types**: `pipeline(source, env)`
//! decides which stages to run for *this* source — e.g. a file mesh produces
//! `[read, decode, upload]` while a generated mesh produces `[generate, upload]`
//! (no IO). Stages are [`AssetStage`] objects; the processor runs the list,
//! placing each on its executor.

use crate::source::AssetSource;
use crate::stage::{AssetStage, LoadEnv};

/// One asset type = one loader: its identity + how to build the pipeline for a
/// given source.
pub trait AssetLoader: 'static {
    /// Stable name (DB routing + diagnostics).
    const NAME: &'static str;

    /// Identity / cache key, serialized in components & prefabs.
    type Source: AssetSource;
    /// The final resident produced by the pipeline (e.g. `Mesh`, `Prefab`).
    /// `request` returns an `AssetHandle<Asset>`.
    type Asset: 'static;

    /// Assemble the stage sequence for `source`. Decided at runtime — omit the
    /// IO stage for generated sources, the decode stage for GPU-ready formats,
    /// the GPU stage for prefabs, etc. The last stage's output must be the
    /// (boxed) `Asset`.
    fn pipeline(&self, source: &Self::Source, env: &LoadEnv) -> Vec<Box<dyn AssetStage>>;
}
