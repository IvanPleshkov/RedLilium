use redlilium_core::compute::{CancellationToken, Checkpoint, ComputeContext, YieldNow, yield_now};

use crate::io_runtime::IoRuntime;

/// ECS-specific compute context implementing [`ComputeContext`].
///
/// Created internally by [`ComputePool`](crate::ComputePool) when spawning
/// tasks. Provides cooperative yielding, cancellation-aware checkpoints,
/// and IO access to compute tasks.
#[derive(Clone)]
pub struct EcsComputeContext {
    io: IoRuntime,
    token: CancellationToken,
}

impl EcsComputeContext {
    /// Creates a new ECS compute context with the given IO runtime
    /// and cancellation token.
    pub(crate) fn new(io: IoRuntime, token: CancellationToken) -> Self {
        Self { io, token }
    }
}

impl ComputeContext for EcsComputeContext {
    type Io = IoRuntime;

    fn yield_now(&self) -> YieldNow {
        yield_now()
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint::with_token(self.token.clone())
    }

    fn io(&self) -> &Self::Io {
        &self.io
    }
}
