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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use parking_lot::Mutex;
use redlilium_graphics::{GraphicsDevice, RenderGraph, TransferConfig, TransferPass};
use redlilium_vfs::Vfs;

use crate::db::{AssetDb, AssetPath};
use crate::error::AssetError;
use crate::handle::{AssetHandle, RequestSlot};
use crate::loader::AssetLoader;
use crate::source::AssetSource;
use crate::stage::{AnyAsset, Executor, LoadEnv};

/// Identifies one in-flight request (for routing async results back).
type RequestId = u64;

/// An async stage future to spawn, tagged with its executor.
pub type AsyncTask = (Executor, Pin<Box<dyn Future<Output = ()> + Send>>);

/// Delivers the final (erased) value into the typed handle slot, or an error.
/// `Send + Sync` (captures only a `Weak<RequestSlot>`) so the request stays so.
type Deliver = Box<dyn FnOnce(Result<Box<dyn Any>, AssetError>) + Send + Sync>;

/// Distinguishes concurrent processors (e.g. blue-green swap) so their per-frame
/// transfer passes don't collide on a name.
static PROCESSOR_SEQ: AtomicU64 = AtomicU64::new(0);

struct Request {
    stages: Vec<Box<dyn crate::stage::AssetStage>>,
    next: usize,
    /// Input to `stages[next]`; `None` while an async stage is in flight.
    value: Option<AnyAsset>,
    deliver: Option<Deliver>,
    /// Demand check: false once the consumer dropped the handle.
    alive: Box<dyn Fn() -> bool + Send + Sync>,
    in_flight: bool,
}

/// Runs requests through their stage pipelines.
pub struct AssetProcessor {
    /// Unique per instance (transfer-pass naming across concurrent processors).
    id: u64,
    /// Stable env parts; the per-request `path` is resolved from the DB.
    vfs: Vfs,
    device: Arc<GraphicsDevice>,
    /// `extension -> kind` routes (from registered loaders), for scan/import.
    routes: HashMap<String, String>,
    next_id: RequestId,
    requests: HashMap<RequestId, Request>,
    result_tx: Sender<(RequestId, Result<AnyAsset, AssetError>)>,
    /// `mpsc::Receiver` is `!Sync`; the `Mutex` makes the processor `Sync` so it
    /// can be an ECS resource. Only `collect` locks it (briefly, to drain).
    result_rx: Mutex<Receiver<(RequestId, Result<AnyAsset, AssetError>)>>,
    /// Max requests in flight (trivial RAM proxy).
    ram_budget: usize,
    /// Max GPU stages run per frame.
    gpu_per_frame: usize,
}

impl AssetProcessor {
    /// Builder for a processor with a static set of loaders/routes. The loaders
    /// are fixed at build (no dynamic registration); the editor/game assembles
    /// them on top of the built-ins here.
    pub fn builder(vfs: Vfs, device: Arc<GraphicsDevice>) -> AssetProcessorBuilder {
        AssetProcessorBuilder {
            vfs,
            device,
            routes: HashMap::new(),
        }
    }

    /// Create a processor with no extension routes (loading still works via
    /// `request::<L>`; only scan/import needs routes). Budgets default high.
    pub fn new(vfs: Vfs, device: Arc<GraphicsDevice>) -> Self {
        Self::with_routes(vfs, device, HashMap::new())
    }

    fn with_routes(vfs: Vfs, device: Arc<GraphicsDevice>, routes: HashMap<String, String>) -> Self {
        let (result_tx, result_rx) = channel();
        Self {
            id: PROCESSOR_SEQ.fetch_add(1, Ordering::Relaxed),
            vfs,
            device,
            routes,
            next_id: 0,
            requests: HashMap::new(),
            result_tx,
            result_rx: Mutex::new(result_rx),
            ram_budget: usize::MAX,
            gpu_per_frame: usize::MAX,
        }
    }

    /// The kind a file extension routes to (case-insensitive), if any.
    pub fn kind_for_extension(&self, ext: &str) -> Option<&str> {
        self.routes
            .get(&ext.to_ascii_lowercase())
            .map(String::as_str)
    }

    /// Scan `mount` (or the subtree `scope`) using this processor's routes,
    /// registering assets in `db`. Returns the change delta. See [`scan_mount`].
    ///
    /// [`scan_mount`]: crate::scan::scan_mount
    pub async fn scan(
        &self,
        db: &mut AssetDb,
        mount: &str,
        scope: Option<&str>,
    ) -> Result<crate::scan::ScanReport, AssetError> {
        crate::scan::scan_mount(&self.vfs, db, mount, scope, |ext| {
            self.kind_for_extension(ext).map(str::to_string)
        })
        .await
    }

    /// Set the (mutable, per-tick) budgets.
    pub fn set_budgets(&mut self, ram_budget: usize, gpu_per_frame: usize) {
        self.ram_budget = ram_budget;
        self.gpu_per_frame = gpu_per_frame;
    }

    /// Request an asset: resolves the source, builds its pipeline, returns a
    /// handle. The handle is the demand — drop it before completion to cancel.
    /// A file source whose guid isn't in `db` fails immediately (the returned
    /// handle is already `Err(NotResolved)`).
    pub fn request<L: AssetLoader>(
        &mut self,
        db: &AssetDb,
        source: L::Source,
        deps: L::Deps,
    ) -> AssetHandle<L::Asset> {
        let slot = RequestSlot::<L::Asset>::new();
        let handle = AssetHandle::new(Arc::clone(&slot));

        let path = match resolve_path(db, &source) {
            Ok(path) => path,
            Err(e) => {
                slot.fulfill(Err(e)); // unresolved → nothing queued
                return handle;
            }
        };
        // Per-kind settings from the record (for sources that store their data
        // there, e.g. a vertex layout's parameters).
        let settings = source
            .file_guid()
            .and_then(|guid| db.record(&guid).and_then(|r| r.settings.clone()));
        let env = LoadEnv {
            path,
            settings,
            vfs: self.vfs.clone(),
            device: Arc::clone(&self.device),
        };
        let stages = L::pipeline(&source, &deps, &env);

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
        // Drain under the lock, then process (the body needs `&mut self`).
        let drained: Vec<(RequestId, Result<AnyAsset, AssetError>)> =
            self.result_rx.lock().try_iter().collect();
        for (id, result) in drained {
            let Some(r) = self.requests.get_mut(&id) else {
                continue; // request was abandoned
            };
            r.in_flight = false;
            match result {
                Ok(value) => {
                    r.next += 1;
                    if r.next >= r.stages.len() {
                        let deliver = r.deliver.take().unwrap();
                        let final_value: Box<dyn Any> = value; // drop +Send+Sync for delivery
                        deliver(Ok(final_value));
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
            let mut pass = TransferPass::new(format!("asset_uploads_{}", self.id));
            pass.set_transfer_config(TransferConfig::new().with_operations(ops));
            graph.add_transfer_pass(pass);
        }
    }

    /// Number of requests still in flight (queued, running, or awaiting GPU).
    pub fn pending(&self) -> usize {
        self.requests.len()
    }

    /// Whether the pool has fully drained — every request delivered or abandoned.
    /// A blue-green swap ticks the old processor until this is true, then drops
    /// it (stop calling `request`; keep calling `drain_tasks`/`collect`/
    /// `flush_gpu`).
    pub fn is_idle(&self) -> bool {
        self.requests.is_empty()
    }

    /// The VFS the processor loads from — shared with consumers that need
    /// direct file access on the same mounts (e.g. the editor's remote
    /// channel writing a freshly created asset).
    pub fn vfs(&self) -> &Vfs {
        &self.vfs
    }
}

/// Builds an [`AssetProcessor`] with a fixed set of loaders + routes.
pub struct AssetProcessorBuilder {
    vfs: Vfs,
    device: Arc<GraphicsDevice>,
    routes: HashMap<String, String>,
}

impl AssetProcessorBuilder {
    /// Register a loader: adds its `EXTENSIONS -> NAME` routes (later
    /// registrations override earlier ones — overlay your own on the built-ins).
    pub fn with_loader<L: AssetLoader>(mut self) -> Self {
        for ext in L::EXTENSIONS {
            self.routes
                .insert(ext.to_ascii_lowercase(), L::NAME.to_string());
        }
        self
    }

    /// Add or override a single `extension -> kind` route.
    pub fn with_route(mut self, ext: impl Into<String>, kind: impl Into<String>) -> Self {
        self.routes
            .insert(ext.into().to_ascii_lowercase(), kind.into());
        self
    }

    /// Finish, producing the processor.
    pub fn build(self) -> AssetProcessor {
        AssetProcessor::with_routes(self.vfs, self.device, self.routes)
    }
}

/// Resolve a source's primary path: `None` for non-file sources (generated);
/// the DB path for a file source; `NotResolved` if its guid is unknown.
fn resolve_path(db: &AssetDb, source: &impl AssetSource) -> Result<Option<AssetPath>, AssetError> {
    match source.file_guid() {
        None => Ok(None),
        Some(guid) => db
            .record(&guid)
            .map(|r| Some(r.path.clone()))
            .ok_or(AssetError::NotResolved),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{AssetSource, Guid};

    /// Minimal source fixture: a file-backed or generated identity.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum TestSource {
        File(Guid),
        Generated,
    }

    impl AssetSource for TestSource {
        fn file_guid(&self) -> Option<Guid> {
            match self {
                Self::File(guid) => Some(*guid),
                Self::Generated => None,
            }
        }
    }

    #[test]
    fn generated_source_needs_no_path() {
        let db = AssetDb::new();
        assert!(matches!(
            resolve_path(&db, &TestSource::Generated),
            Ok(None)
        ));
    }

    #[test]
    fn unknown_file_guid_is_not_resolved() {
        let db = AssetDb::new();
        let src = TestSource::File(Guid::new());
        assert!(matches!(
            resolve_path(&db, &src),
            Err(AssetError::NotResolved)
        ));
    }

    #[test]
    fn known_file_guid_resolves_to_its_path() {
        let mut db = AssetDb::new();
        let guid = db.register_path(AssetPath::new("assets", "a.rmesh"), "mesh", 0);
        let src = TestSource::File(guid);
        let path = resolve_path(&db, &src).unwrap().unwrap();
        assert_eq!(path, AssetPath::new("assets", "a.rmesh"));
    }

    #[test]
    fn processor_and_handle_are_send_sync() {
        // The whole point of the retrofit: usable as an ECS resource, and the
        // handle can ride in an ECS component.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AssetProcessor>();
        assert_send_sync::<crate::AssetHandle<redlilium_graphics::Mesh>>();
    }
}
