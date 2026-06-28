//! The [`AssetLoader`] trait — identity + a method that assembles the pipeline.
//!
//! The stage sequence is **runtime data, not types**: `pipeline(source, env)`
//! decides which stages to run for *this* source — e.g. a file mesh produces
//! `[read, decode, upload]` while a generated mesh produces `[generate, upload]`
//! (no IO). Stages are [`AssetStage`] objects; the processor runs the list,
//! placing each on its executor.

use crate::source::AssetSource;
use crate::stage::{AssetStage, LoadEnv};

/// One asset type = one loader: its identity, the file formats it imports, and
/// how to build the pipeline for a given source.
///
/// Loaders are **stateless and type-level** (no instance, no erasure): the
/// processor builder registers them by type (`with_loader::<L>()`), and a load
/// is `request::<L>(db, source)`. A loader lives in whatever crate owns its
/// dependencies (mesh/texture here; a prefab loader in `ecs/std`; etc.).
pub trait AssetLoader: 'static {
    /// Stable kind name (DB `kind` field + extension routing + diagnostics).
    const NAME: &'static str;

    /// File extensions this loader imports (lowercase, no dot) — populates the
    /// builder's `ext -> kind` route table for scanning. Empty for loaders with
    /// no file form (purely generated).
    const EXTENSIONS: &'static [&'static str] = &[];

    /// Identity / cache key, serialized in components & prefabs.
    type Source: AssetSource;
    /// The final resident produced by the pipeline (e.g. `Mesh`, `Prefab`).
    /// `request` returns an `AssetHandle<Asset>`. `Send + Sync` so the handle can
    /// ride in an ECS component and the resolved `Arc<Asset>` can be shared.
    type Asset: Send + Sync + 'static;
    /// Runtime dependencies the caller resolves and injects into the pipeline —
    /// e.g. a shared `Arc<VertexLayout>` for a mesh, so the loaded mesh binds the
    /// *same* layout `Arc` as everything else (pointer-equality drives batching).
    /// `()` when the loader needs none. The deps are consumed when the pipeline
    /// is assembled (the stages capture what they need), so they need no bounds.
    type Deps;

    /// Assemble the stage sequence for `source`, given any resolved `deps`.
    /// Decided at runtime — omit the IO stage for generated sources, the decode
    /// stage for GPU-ready formats, the GPU stage for prefabs, etc. The last
    /// stage's output must be the (boxed) `Asset`.
    fn pipeline(
        source: &Self::Source,
        deps: &Self::Deps,
        env: &LoadEnv,
    ) -> Vec<Box<dyn AssetStage>>;
}
