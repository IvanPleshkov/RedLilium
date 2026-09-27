//! Completion of a graph-ordered CPU readback.
use crate::GraphicsError;
use std::sync::Arc;

/// Observable state of a one-shot readback request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadbackStatus {
    Pending,
    Ready,
    Consumed,
}
#[derive(Debug)]
enum State {
    Pending,
    Ready(Result<Vec<u8>, GraphicsError>),
    Consumed,
}
/// A one-shot result destination for `TransferOperation::readback_buffer`.
/// Clones observe the same result. Create a fresh handle for each operation.
#[derive(Debug, Clone)]
pub struct Readback(Arc<Shared>);
#[derive(Debug)]
struct Shared {
    state: parking_lot::Mutex<State>,
    claimed: std::sync::atomic::AtomicBool,
}
impl Default for Readback {
    fn default() -> Self {
        Self::new()
    }
}
impl Readback {
    pub fn new() -> Self {
        Self(Arc::new(Shared {
            state: parking_lot::Mutex::new(State::Pending),
            claimed: std::sync::atomic::AtomicBool::new(false),
        }))
    }
    pub fn status(&self) -> ReadbackStatus {
        match &*self.0.state.lock() {
            State::Pending => ReadbackStatus::Pending,
            State::Ready(_) => ReadbackStatus::Ready,
            State::Consumed => ReadbackStatus::Consumed,
        }
    }
    /// Take a completed success or failure once. An empty successful read is `Some(Ok(vec![]))`.
    pub fn take_result(&self) -> Option<Result<Vec<u8>, GraphicsError>> {
        let mut state = self.0.state.lock();
        if !matches!(*state, State::Ready(_)) {
            return None;
        }
        match std::mem::replace(&mut *state, State::Consumed) {
            State::Ready(result) => Some(result),
            _ => unreachable!(),
        }
    }
    pub(crate) fn complete(&self, result: Result<Vec<u8>, GraphicsError>) {
        let mut state = self.0.state.lock();
        if matches!(*state, State::Pending) {
            *state = State::Ready(result);
        }
    }
}
pub(crate) type ReadbackCallback = Box<dyn FnOnce(Result<Vec<u8>, GraphicsError>) + Send>;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failure_is_observable_once() {
        let request = Readback::new();
        let receiver = request.clone();
        assert_eq!(receiver.status(), ReadbackStatus::Pending);
        request.complete(Err(GraphicsError::DeviceLost));
        assert_eq!(receiver.take_result(), Some(Err(GraphicsError::DeviceLost)));
        assert_eq!(receiver.status(), ReadbackStatus::Consumed);
    }
}

/// Roll reservations back if command recording or submission rejects the graph.
pub(crate) struct ReadbackReservations {
    requests: Vec<Readback>,
    committed: bool,
}
impl ReadbackReservations {
    pub(crate) fn reserve(graph: &crate::RenderGraph) -> Result<Self, GraphicsError> {
        let mut guard = Self {
            requests: Vec::new(),
            committed: false,
        };
        for pass in graph.passes() {
            if let Some(config) = pass.as_transfer().and_then(|p| p.transfer_config()) {
                for op in &config.operations {
                    if let crate::TransferOperation::ReadbackBuffer { dst, .. } = op {
                        if dst
                            .0
                            .claimed
                            .swap(true, std::sync::atomic::Ordering::AcqRel)
                        {
                            return Err(GraphicsError::InvalidParameter(
                                "a readback result can belong to only one submitted operation"
                                    .into(),
                            ));
                        }
                        guard.requests.push(dst.clone());
                    }
                }
            }
        }
        Ok(guard)
    }
    pub(crate) fn commit(mut self) {
        self.committed = true;
    }
}
impl Drop for ReadbackReservations {
    fn drop(&mut self) {
        if !self.committed {
            for request in &self.requests {
                request
                    .0
                    .claimed
                    .store(false, std::sync::atomic::Ordering::Release);
            }
        }
    }
}

pub(crate) type ReadbackBatch = Vec<(std::ops::Range<usize>, Readback)>;

#[cfg(test)]
mod reservations {
    use super::*;
    use crate::*;
    #[test]
    fn duplicate_destinations_rejected_and_failed_reservations_rollback() {
        let device = GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap();
        let buffer = device
            .create_buffer(&BufferDescriptor::new(
                16,
                BufferUsage::MAP_READ | BufferUsage::COPY_DST,
            ))
            .unwrap();
        let result = Readback::new();
        let op = TransferOperation::readback_buffer(buffer, 0..4, result);
        let graph = |count| {
            let mut graph = RenderGraph::new();
            let mut pass = TransferPass::new("readback".into());
            pass.set_transfer_config(
                TransferConfig::new().with_operations(vec![op.clone(); count]),
            );
            graph.add_transfer_pass(pass);
            graph
        };
        assert!(ReadbackReservations::reserve(&graph(2)).is_err());
        let valid = graph(1);
        let reservation = ReadbackReservations::reserve(&valid).unwrap();
        drop(reservation);
        ReadbackReservations::reserve(&valid).unwrap().commit();
        assert!(ReadbackReservations::reserve(&valid).is_err());
    }
}
