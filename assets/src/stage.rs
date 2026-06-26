//! A pipeline stage — a self-contained unit that transforms erased data on a
//! given executor. A loader assembles a `Vec<Box<dyn AssetStage>>` per source
//! ([`AssetLoader::pipeline`](crate::AssetLoader::pipeline)); the processor runs
//! them in order, placing each on its executor and threading the (type-erased)
//! value from one to the next.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use redlilium_graphics::{GraphicsDevice, TransferOperation};

use crate::error::AssetError;

/// Type-erased value flowing between stages (e.g. file bytes, `CpuMesh`).
/// `Send + Sync` so the request (and the processor holding it) stays `Send +
/// Sync` — it crosses into the IO/compute task that produces it.
pub type AnyAsset = Box<dyn Any + Send + Sync>;

/// Type-erased resident produced by a GPU stage (holds e.g. `Arc<Mesh>`). Lives
/// on the render thread, so it is **not** `Send`.
pub type GpuValue = Box<dyn Any>;

/// The future an async (IO/CPU) stage returns.
pub type StageFuture = Pin<Box<dyn Future<Output = Result<AnyAsset, AssetError>> + Send>>;

/// Which executor a stage runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Executor {
    /// IO runtime (file reads).
    Io,
    /// Compute pool (decode/generate; cooperatively yields if heavy).
    Cpu,
    /// Render thread, in `on_draw` with the frame graph (GPU residency).
    Gpu,
}

/// One pipeline stage. Most stages are async ([`run_async`](Self::run_async),
/// IO/CPU); a GPU stage instead implements [`run_gpu`](Self::run_gpu) (sync,
/// render thread, returns transfer ops for the frame graph). The processor picks
/// the method by [`executor`](Self::executor).
///
/// `Send + Sync`: a built pipeline lives inside the processor (an ECS resource),
/// and async stages run on the compute/IO pools. Stages capture only `Send +
/// Sync` data (vfs, device, settings).
pub trait AssetStage: Send + Sync {
    /// The executor this stage runs on.
    fn executor(&self) -> Executor;

    /// IO/CPU stages: async transform `input -> output`. The returned future is
    /// `Send` (spawned off-thread), so it captures owned clones, not `&self`.
    fn run_async(&self, input: AnyAsset) -> StageFuture {
        let _ = input;
        unreachable!("`run_async` called on a {:?} stage", self.executor());
    }

    /// GPU stage: sync transform on the render thread. Returns the resident
    /// resource (erased) plus the transfer ops to add to the frame graph (no
    /// synchronous GPU writes — Phase 0 rule). The device is captured at build
    /// time (from [`LoadEnv`](crate::LoadEnv)).
    fn run_gpu(&self, input: AnyAsset) -> Result<(GpuValue, Vec<TransferOperation>), AssetError> {
        let _ = input;
        unreachable!("`run_gpu` called on a {:?} stage", self.executor());
    }
}

/// Resources a loader draws on to build its stages: the resolved file path (for
/// file-backed sources), the VFS, and the graphics device.
#[derive(Clone)]
pub struct LoadEnv {
    /// Resolved primary path for file-backed sources (`None` for generated).
    pub path: Option<crate::db::AssetPath>,
    /// VFS router (captured by IO stages).
    pub vfs: redlilium_vfs::Vfs,
    /// Graphics device (captured by GPU stages).
    pub device: Arc<GraphicsDevice>,
}
