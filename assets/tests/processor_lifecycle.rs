use redlilium_assets::*;
use redlilium_graphics::{BackendType, GraphicsInstance, InstanceParameters, TransferOperation};
use redlilium_vfs::Vfs;
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::task::{Context, Poll, Waker};

fn processor() -> AssetProcessor {
    let instance = GraphicsInstance::with_parameters(
        InstanceParameters::new().with_backend(BackendType::Dummy),
    )
    .unwrap();
    AssetProcessor::new(Vfs::new(), instance.create_device().unwrap())
}
fn block<T>(future: impl Future<Output = T>) -> T {
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("expected ready test stage"),
    }
}
fn pump(processor: &mut AssetProcessor) {
    for (_, future) in processor.drain_tasks() {
        block(future);
    }
    processor.collect();
}
fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Source;
impl AssetSource for Source {
    fn file_guid(&self) -> Option<Guid> {
        None
    }
}

#[derive(Clone)]
enum Stage {
    Cpu(Arc<AtomicUsize>),
    Gpu(Arc<AtomicUsize>),
    Hold,
    BuildPanic,
    ConstructPanic,
    PollPanic,
    GpuPanic,
    ExecutorPanic,
    DropPanic { poll_panics: bool },
}
struct PanickingDrop {
    poll_panics: bool,
}
impl Future for PanickingDrop {
    type Output = Result<AnyAsset, AssetError>;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        assert!(!self.poll_panics, "poll panic before destructor");
        Poll::Pending
    }
}
impl Drop for PanickingDrop {
    fn drop(&mut self) {
        panic!("future destructor panic");
    }
}
impl AssetStage for Stage {
    fn executor(&self) -> Executor {
        match self {
            Self::ExecutorPanic => panic!("executor selection panic"),
            Self::Gpu(_) | Self::GpuPanic => Executor::Gpu,
            _ => Executor::Cpu,
        }
    }
    fn run_async(&self, _: AnyAsset) -> StageFuture {
        if matches!(self, Self::ConstructPanic) {
            panic!("future construction panic");
        }
        if let Self::DropPanic { poll_panics } = self {
            return Box::pin(PanickingDrop {
                poll_panics: *poll_panics,
            });
        }
        let stage = self.clone();
        Box::pin(async move {
            match stage {
                Self::PollPanic => panic!("poll panic"),
                Self::Hold => std::future::pending().await,
                Self::Cpu(count) => {
                    count.fetch_add(1, Ordering::SeqCst);
                    Ok(Box::new(vec![0u8; 1024]) as AnyAsset)
                }
                _ => unreachable!(),
            }
        })
    }
    fn run_gpu(&self, _: AnyAsset) -> Result<(GpuValue, Vec<TransferOperation>), AssetError> {
        if matches!(self, Self::GpuPanic) {
            panic!("GPU panic");
        }
        if let Self::Gpu(count) = self {
            count.fetch_add(1, Ordering::SeqCst);
        }
        Ok((Box::new(vec![1u8; 1024]), vec![]))
    }
}
struct Loader;
impl AssetLoader for Loader {
    const NAME: &'static str = "test";
    type Source = Source;
    type Asset = Vec<u8>;
    type Deps = Vec<Stage>;
    fn pipeline(_: &Source, deps: &Vec<Stage>, _: &LoadEnv) -> Vec<Box<dyn AssetStage>> {
        assert!(
            !deps.iter().any(|s| matches!(s, Stage::BuildPanic)),
            "pipeline build panic"
        );
        deps.iter()
            .map(|s| Box::new(s.clone()) as Box<dyn AssetStage>)
            .collect()
    }
}
fn request(p: &mut AssetProcessor, stages: Vec<Stage>) -> AssetHandle<Vec<u8>> {
    p.request::<Loader>(&AssetDb::new(), Source, stages)
}
fn pipeline_error(handle: &AssetHandle<Vec<u8>>) {
    assert!(
        matches!(handle.get(), Some(Err(AssetError::Pipeline(_)))),
        "{:?}",
        handle.get()
    );
}

#[test]
fn panics_finish_requests_and_release_admission() {
    for stage in [
        Stage::BuildPanic,
        Stage::ConstructPanic,
        Stage::PollPanic,
        Stage::GpuPanic,
        Stage::ExecutorPanic,
        Stage::DropPanic { poll_panics: true },
    ] {
        let mut p = processor();
        p.set_budgets(1, 1);
        let handle = request(&mut p, vec![stage]);
        pump(&mut p);
        p.flush_gpu();
        pipeline_error(&handle);
        assert!(p.is_idle());
        let next = request(&mut p, vec![Stage::Cpu(counter())]);
        pump(&mut p);
        assert!(next.ready().is_some());
    }
}

#[test]
fn dropping_issued_tasks_reports_failure_even_before_first_poll() {
    for poll_first in [false, true] {
        for panicking_drop in [false, true] {
            let mut p = processor();
            let stage = if panicking_drop {
                Stage::DropPanic { poll_panics: false }
            } else {
                Stage::Hold
            };
            let handle = request(&mut p, vec![stage]);
            let (_, mut task) = p.drain_tasks().pop().unwrap();
            if poll_first {
                assert!(
                    task.as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
            }
            drop(task);
            p.collect();
            pipeline_error(&handle);
            assert!(p.is_idle());
        }
    }
}

#[test]
fn abandonment_keeps_running_task_accounted_until_disposal() {
    let mut p = processor();
    p.set_budgets(1, 1);
    let first = request(&mut p, vec![Stage::Hold]);
    let tasks = p.drain_tasks();
    drop(first);
    let second = request(&mut p, vec![Stage::Cpu(counter())]);
    assert!(p.drain_tasks().is_empty());
    assert!(!p.is_idle());
    assert_eq!(p.pending(), 2);
    drop(tasks);
    pump(&mut p);
    assert!(second.ready().is_some());
    assert!(p.is_idle());
}

#[test]
fn abandoned_requests_do_not_start_cpu_or_gpu_work() {
    for gpu_only in [false, true] {
        let mut p = processor();
        let count = counter();
        let stages = if gpu_only {
            vec![Stage::Gpu(count.clone())]
        } else {
            vec![Stage::Cpu(count.clone())]
        };
        let handle = request(&mut p, stages);
        drop(handle);
        pump(&mut p);
        p.flush_gpu();
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert!(p.is_idle());
    }
    let mut p = processor();
    let gpu = counter();
    let handle = request(&mut p, vec![Stage::Cpu(counter()), Stage::Gpu(gpu.clone())]);
    pump(&mut p);
    drop(handle);
    p.flush_gpu();
    assert_eq!(gpu.load(Ordering::SeqCst), 0);
    assert!(p.is_idle());
}

#[test]
fn admission_covers_intermediate_data_and_allows_admitted_chains_to_finish() {
    let mut p = processor();
    p.set_budgets(1, 0);
    let decoded = counter();
    let gpu = counter();
    let handles: Vec<_> = (0..5)
        .map(|_| {
            request(
                &mut p,
                vec![
                    Stage::Cpu(counter()),
                    Stage::Cpu(decoded.clone()),
                    Stage::Gpu(gpu.clone()),
                ],
            )
        })
        .collect();
    for _ in 0..10 {
        pump(&mut p);
        p.flush_gpu();
    }
    assert_eq!(
        decoded.load(Ordering::SeqCst),
        1,
        "only one decoded payload may wait for GPU"
    );
    p.set_budgets(0, 1); // reducing admission must not strand existing work
    p.flush_gpu();
    assert_eq!(gpu.load(Ordering::SeqCst), 1);
    assert!(p.drain_tasks().is_empty());
    p.set_budgets(1, 1);
    for _ in 0..10 {
        pump(&mut p);
        p.flush_gpu();
    }
    assert!(handles.iter().all(|h| h.ready().is_some()));
    assert!(p.is_idle());
}

#[test]
fn malformed_pipelines_fail_before_any_stage_executes() {
    let count = counter();
    for stages in [
        vec![],
        vec![Stage::Gpu(count.clone()), Stage::Cpu(count.clone())],
        vec![Stage::Gpu(count.clone()), Stage::Gpu(count.clone())],
    ] {
        let mut p = processor();
        let handle = request(&mut p, stages);
        pipeline_error(&handle);
        assert!(p.is_idle());
        pump(&mut p);
        p.flush_gpu();
    }
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

#[test]
fn successful_publication_clears_a_previous_failure_latch() {
    let mut cache = ResidentCache::new();
    cache.fail(1);
    cache.publish(1, Arc::new(42));
    assert!(!cache.is_failed(&1));
    assert_eq!(**cache.get(&1).unwrap(), 42);
}
