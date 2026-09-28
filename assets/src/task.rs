//! Completion reporting survives executor cancellation and stage panics.
use std::any::Any;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::mpsc::Sender;
use std::task::{Context, Poll};

use crate::{AnyAsset, AssetError, StageFuture};

type Completion = (u64, Result<AnyAsset, AssetError>);

pub(crate) fn guarded<T>(operation: &str, f: impl FnOnce() -> T) -> Result<T, AssetError> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let message = panic_message(&*payload);
        // Consume the payload while its defining module is still loaded. Even
        // a panicking payload destructor must not lose request completion.
        if let Err(secondary) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
            std::mem::forget(secondary);
        }
        AssetError::Pipeline(format!("{operation} panicked: {message}"))
    })
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "non-string panic payload".to_owned()
    }
}

// Keep the future as a field: an unwind from poll reaches `guarded` before
// destroying the future, avoiding a double panic if its destructor also panics.
pub(crate) struct StageTask {
    future: Option<StageFuture>,
    report: Option<(u64, Sender<Completion>)>,
}

impl StageTask {
    pub(crate) fn new(id: u64, future: StageFuture, tx: Sender<Completion>) -> Self {
        Self {
            future: Some(future),
            report: Some((id, tx)),
        }
    }

    fn dispose(&mut self) -> Result<(), AssetError> {
        let Some(future) = self.future.take() else {
            return Ok(());
        };
        guarded("stage future destruction", || drop(future))
    }

    fn report(&mut self, result: Result<AnyAsset, AssetError>) {
        if let Some((id, tx)) = self.report.take() {
            let _ = tx.send((id, result));
        }
    }
}

impl Future for StageTask {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let result = match guarded("async stage", || {
            this.future
                .as_mut()
                .expect("stage polled after completion")
                .as_mut()
                .poll(cx)
        }) {
            Ok(Poll::Pending) => return Poll::Pending,
            Ok(Poll::Ready(result)) => result,
            Err(error) => Err(error),
        };
        // Report only after destruction: collect/is_idle must account for the
        // user future's full lifetime, not just its last poll.
        let disposal = this.dispose();
        this.report(disposal.and(result));
        Poll::Ready(())
    }
}

impl Drop for StageTask {
    fn drop(&mut self) {
        let disposal = self.dispose();
        if self.report.is_some() {
            self.report(Err(disposal.err().unwrap_or_else(|| {
                AssetError::Pipeline("async stage task was dropped before completion".into())
            })));
        }
    }
}
