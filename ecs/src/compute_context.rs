use redlilium_core::compute::{ComputeContext, YieldNow, yield_now};

use crate::io_runtime::IoRuntime;

/// ECS-specific compute context implementing [`ComputeContext`].
///
/// Created internally by [`ComputePool`](crate::ComputePool) when spawning
/// tasks. Provides cooperative yielding and IO access to compute tasks.
#[derive(Clone)]
pub struct EcsComputeContext {
    io: IoRuntime,
}

impl EcsComputeContext {
    /// Creates a new ECS compute context with the given IO runtime.
    pub(crate) fn new(io: IoRuntime) -> Self {
        Self { io }
    }
}

impl ComputeContext for EcsComputeContext {
    type Io = IoRuntime;

    fn yield_now(&self) -> YieldNow {
        yield_now()
    }

    fn io(&self) -> &Self::Io {
        &self.io
    }
}
