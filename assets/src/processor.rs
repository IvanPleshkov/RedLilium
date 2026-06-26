//! The request processor — runs each request's stage list, runtime-agnostic.
//!
//! It owns the in-flight requests but does **not** spawn: `drain_tasks` hands
//! the async (IO/CPU) stage futures to the ECS bridge (which spawns them on
//! `ctx.io()` / `ctx.compute()`), results flow back over a channel and are
//! applied by `collect`; GPU stages run in `flush_gpu` with the frame graph.
//! No cache/dedup — each request is independent; sharing is the consumer's job.
//!
//! Budgets are trivial and settable: `ram_budget` gates how many requests are
//! admitted into flight; `gpu_per_frame` caps GPU stages run per frame. (Both
//! are count-based for now — byte-precise gating needs a per-stage size hint.)

use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use redlilium_graphics::{RenderGraph, TransferConfig, TransferPass};

use crate::error::AssetError;
use crate::handle::{AssetHandle, RequestSlot};
use crate::loader::AssetLoader;
use crate::stage::{AnyAsset, Executor, LoadEnv};

/// Identifies one in-flight request (for routing async results back).
type RequestId = u64;

/// An async stage future to spawn, tagged with its executor.
pub type AsyncTask = (Executor, Pin<Box<dyn Future<Output = ()> + Send>>);

/// Delivers the final (erased) value into the typed handle slot, or an error.
type Deliver = Box<dyn FnOnce(Result<Box<dyn Any>, AssetError>)>;

struct Request {
    stages: Vec<Box<dyn crate::stage::AssetStage>>,
    next: usize,
    /// Input to `stages[next]`; `None` while an async stage is in flight.
    value: Option<AnyAsset>,
    deliver: Option<Deliver>,
    /// Demand check: false once the consumer dropped the handle.
    alive: Box<dyn Fn() -> bool>,
    in_flight: bool,
}

/// Runs requests through their stage pipelines.
pub struct AssetProcessor {
    env: LoadEnv,
    next_id: RequestId,
    requests: HashMap<RequestId, Request>,
    result_tx: Sender<(RequestId, Result<AnyAsset, AssetError>)>,
    result_rx: Receiver<(RequestId, Result<AnyAsset, AssetError>)>,
    /// Max requests in flight (trivial RAM proxy).
    ram_budget: usize,
    /// Max GPU stages run per frame.
    gpu_per_frame: usize,
}

impl AssetProcessor {
    /// Create a processor over the given load environment (vfs + device + path
    /// resolver). Budgets default high; set them from project settings.
    pub fn new(env: LoadEnv) -> Self {
        let (result_tx, result_rx) = channel();
        Self {
            env,
            next_id: 0,
            requests: HashMap::new(),
            result_tx,
            result_rx,
            ram_budget: usize::MAX,
            gpu_per_frame: usize::MAX,
        }
    }

    /// Set the (mutable, per-tick) budgets.
    pub fn set_budgets(&mut self, ram_budget: usize, gpu_per_frame: usize) {
        self.ram_budget = ram_budget;
        self.gpu_per_frame = gpu_per_frame;
    }

    /// Request an asset: builds its pipeline and returns a handle. The handle is
    /// the demand — drop it before completion to cancel the load.
    pub fn request<L: AssetLoader>(
        &mut self,
        loader: &L,
        source: L::Source,
    ) -> AssetHandle<L::Asset> {
        let stages = loader.pipeline(&source, &self.env);

        let slot = RequestSlot::<L::Asset>::new();
        let handle = AssetHandle::new(Arc::clone(&slot));
        let demand = Arc::downgrade(&slot);
        let alive = Box::new(move || demand.strong_count() > 0);
        let deliver_slot = Arc::downgrade(&slot);
        let deliver: Deliver = Box::new(move |result| {
            if let Some(slot) = deliver_slot.upgrade() {
                slot.fulfill(result.and_then(downcast_final::<L::Asset>));
            }
        });

        let id = self.next_id;
        self.next_id += 1;
        self.requests.insert(
            id,
            Request {
                stages,
                next: 0,
                value: Some(Box::new(())), // first stage's input is unit
                deliver: Some(deliver),
                alive,
                in_flight: false,
            },
        );
        handle
    }

    /// Drop requests whose handle is gone (coarse cancel: any in-flight task
    /// completes but its result is discarded).
    fn drop_abandoned(&mut self) {
        self.requests.retain(|_, r| (r.alive)());
    }

    /// Produce async (IO/CPU) stage tasks to spawn, up to the RAM budget. The
    /// ECS bridge spawns each on its executor; results return via the channel.
    pub fn drain_tasks(&mut self) -> Vec<AsyncTask> {
        self.drop_abandoned();
        let mut tasks = Vec::new();
        let mut in_flight = self.requests.values().filter(|r| r.in_flight).count();

        for (&id, r) in self.requests.iter_mut() {
            if in_flight >= self.ram_budget {
                break;
            }
            if r.in_flight || r.next >= r.stages.len() {
                continue;
            }
            if r.stages[r.next].executor() == Executor::Gpu {
                continue; // GPU stages run in flush_gpu
            }
            let Some(value) = r.value.take() else {
                continue;
            };
            let fut = r.stages[r.next].run_async(value);
            let tx = self.result_tx.clone();
            r.in_flight = true;
            in_flight += 1;
            let executor = r.stages[r.next].executor();
            tasks.push((
                executor,
                Box::pin(async move {
                    let _ = tx.send((id, fut.await));
                }) as Pin<Box<dyn Future<Output = ()> + Send>>,
            ));
        }
        tasks
    }

    /// Apply finished async stage results, advancing each request (and
    /// delivering when its last stage was async).
    pub fn collect(&mut self) {
        while let Ok((id, result)) = self.result_rx.try_recv() {
            let Some(r) = self.requests.get_mut(&id) else {
                continue; // request was abandoned
            };
            r.in_flight = false;
            match result {
                Ok(value) => {
                    r.next += 1;
                    if r.next >= r.stages.len() {
                        let deliver = r.deliver.take().unwrap();
                        deliver(Ok(value));
                        self.requests.remove(&id);
                    } else {
                        r.value = Some(value);
                    }
                }
                Err(e) => {
                    let deliver = r.deliver.take().unwrap();
                    deliver(Err(e));
                    self.requests.remove(&id);
                }
            }
        }
    }

    /// Run pending GPU stages (up to `gpu_per_frame`), adding their upload ops to
    /// `graph`. Call in `on_draw`.
    pub fn flush_gpu(&mut self, graph: &mut RenderGraph) {
        let ready: Vec<RequestId> = self
            .requests
            .iter()
            .filter(|(_, r)| {
                !r.in_flight
                    && r.next < r.stages.len()
                    && r.stages[r.next].executor() == Executor::Gpu
            })
            .map(|(&id, _)| id)
            .take(self.gpu_per_frame)
            .collect();

        let mut ops = Vec::new();
        for id in ready {
            let r = self.requests.get_mut(&id).unwrap();
            let Some(value) = r.value.take() else {
                continue;
            };
            match r.stages[r.next].run_gpu(value) {
                Ok((gpu_value, mut stage_ops)) => {
                    ops.append(&mut stage_ops);
                    r.next += 1;
                    // GPU is the residency tail → this is the final value.
                    let deliver = r.deliver.take().unwrap();
                    deliver(Ok(gpu_value));
                    self.requests.remove(&id);
                }
                Err(e) => {
                    let deliver = r.deliver.take().unwrap();
                    deliver(Err(e));
                    self.requests.remove(&id);
                }
            }
        }

        if !ops.is_empty() {
            let mut pass = TransferPass::new("asset_uploads".into());
            pass.set_transfer_config(TransferConfig::new().with_operations(ops));
            graph.add_transfer_pass(pass);
        }
    }
}

/// Downcast a pipeline's final erased value to `Arc<T>`: a GPU stage already
/// produced `Arc<T>`; a CPU-final pipeline produced an owned `T` to wrap.
fn downcast_final<T: 'static>(value: Box<dyn Any>) -> Result<Arc<T>, AssetError> {
    match value.downcast::<Arc<T>>() {
        Ok(arc) => Ok(*arc),
        Err(value) => match value.downcast::<T>() {
            Ok(owned) => Ok(Arc::new(*owned)),
            Err(_) => Err(AssetError::Decode(
                "pipeline produced an unexpected final type".into(),
            )),
        },
    }
}
