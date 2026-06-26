//! # RedLilium asset system (prototype)
//!
//! A trait-based asset pipeline. Each asset type is one [`AssetLoader`]
//! declaration (`Source`/`Settings`/`Raw`/`Cpu` + async `read`/`decode` +
//! `decoded_size`); types that need a GPU resource add the optional [`GpuStage`]
//! (`Cpu -> Gpu` through the frame graph) — folded into the same loader, no
//! separate render-asset layer.
//!
//! **Contract:** a consumer `request(source)`s and gets an [`AssetHandle<T>`] —
//! an `Arc`-backed handle that is the *demand to load* while held; poll
//! [`AssetHandle::get`] until ready, then keep the `Arc<T>` and drop the handle.
//! Dropping before completion cancels the in-flight load. **No dedup/cache** —
//! sharing/reuse is the consumer's job.
//!
//! Pipeline stages run on different executors (read → IO runtime, decode →
//! compute pool, upload → render thread/frame graph). The processor (an ECS
//! system; *next chunk*) drives them under two mutable budgets: **RAM** (gates
//! decode intake) and **GPU bytes/frame** (gates uploads).
//!
//! Identity: a stable [`Guid`] resolved to `(mount, path)` via the central
//! bijection-checked [`AssetDb`]; components store the [`AssetSource`].
//!
//! *This module currently provides the contract shapes (loaders + handle); the
//! request processor / ECS bridge is the next step.*

mod db;
mod error;
mod handle;
mod loader;
mod mesh;
mod persist;
mod processor;
mod source;
mod stage;

pub use db::{AssetDb, AssetPath, AssetRecord, DbError};
pub use error::AssetError;
pub use handle::AssetHandle;
pub use loader::AssetLoader;
pub use mesh::{MeshGenerator, MeshLoader, MeshSource};
pub use persist::{DB_FILE_NAME, load_mount_db, save_mount_db};
pub use processor::{AssetProcessor, AsyncTask};
pub use source::{AssetSource, Guid};
pub use stage::{AnyAsset, AssetStage, Executor, GpuValue, LoadEnv, StageFuture};
