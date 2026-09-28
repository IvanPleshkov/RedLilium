//! # RedLilium asset system
//!
//! An [`AssetLoader`] builds a nonempty sequence of IO/CPU [`AssetStage`]s,
//! optionally ending in one GPU stage. [`AssetProcessor`] hands asynchronous
//! work to the ECS executors and emits GPU transfers through render graphs.
//!
//! [`AssetHandle`] carries demand and the eventual result. Dropping every clone
//! abandons the request: an issued stage finishes or is dropped by its executor,
//! its result is discarded, and no following stage starts. The processor retains
//! that task's admission slot until its completion is collected. It does not
//! interrupt an executing stage or shut down the executors.
//!
//! Budgets count admitted requests (including intermediate data waiting for GPU)
//! and GPU stages per flush. They are not byte limits. Resident caches are owned
//! by managers; each processor request is independent and is not deduplicated.
//!
//! [`AssetManager`] / [`ResidentCache`] share resident `Arc`s. Components retain
//! [`AssetRef`]s, while [`AssetDb`] maps stable [`Guid`]s to mounted paths.
//! Managers can `release` ownership while preserving live identity through
//! `Weak`. Explicit `collect_unused` removes cache-only resources and expired
//! weak entries; `invalidate` forgets even a live version for hot reload.

mod asset_ref;
mod db;
mod error;
mod handle;
mod loader;
mod manager;
mod persist;
mod processor;
mod scan;
mod source;
mod stage;
mod task;

pub use asset_ref::{AssetRef, AssetRefSource};
pub use db::{AssetDb, AssetPath, AssetRecord, DbError, extract_guids};
pub use error::AssetError;
pub use handle::AssetHandle;
pub use loader::AssetLoader;
pub use manager::{AssetManager, ResidentCache};
pub use persist::{DB_FILE_NAME, load_mount_db, save_mount_db};
pub use processor::{AssetProcessor, AssetProcessorBuilder, AsyncTask};
pub use scan::{ScanReport, scan_mount};
pub use source::{AssetSource, Guid};
pub use stage::{AnyAsset, AssetStage, Executor, GpuValue, LoadEnv, StageFuture};
