//! Automatic cross-graph dependency derivation (#47 phase 3).
//!
//! When several graphs are submitted in one frame, dependency edges between
//! them are derived **automatically from overlapping resource usage** — there
//! is no manual dependency API, and none is planned: all graphics work goes
//! through render graphs, so their declared resource usage is the single
//! source of truth for ordering (decision recorded in #47).
//!
//! Granularity is **graph↔graph**: a semaphore wait happens at the submit
//! boundary anyway, so finer pass↔pass edges across graphs buy nothing.
//!
//! # Current role (single queue)
//!
//! With every submit on the one graphics queue, submission order already
//! provides every derived edge, so the edges change no runtime behavior —
//! they are recorded on the [`FrameSchedule`](super::FrameSchedule) for
//! inspection and tests. Phase 4 (#47) translates the edges whose endpoints
//! land on *different queues* into timeline-semaphore waits at submit time.

use std::collections::HashSet;

use crate::compiler::CompiledGraph;

/// Identity of a resource for cross-graph dependency analysis.
///
/// Keyed by the pointer of the `Arc`'d resource. Unique among live resources,
/// and every submitted graph keeps `Arc` references to its resources until the
/// frame slot is recycled, so keys are stable for the lifetime of the frame —
/// the only scope this analysis runs over.
///
/// The swapchain surface is deliberately absent: it has no resource object,
/// and at most one graph per frame may write it (enforced in `submit`), so
/// no cross-graph surface edge can arise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ResourceKey {
    Texture(usize),
    Buffer(usize),
}

/// Aggregated read/write resource sets of one submitted graph.
///
/// Built from the compiled graph's per-pass usage declarations — the same
/// declarations the barrier generator consumes, so an access visible to the
/// backend is exactly as visible here.
#[derive(Debug, Default)]
pub(crate) struct GraphUsage {
    /// Resources any pass of the graph reads.
    reads: HashSet<ResourceKey>,
    /// Resources any pass of the graph writes.
    writes: HashSet<ResourceKey>,
}

impl GraphUsage {
    /// Aggregate the per-pass resource usages of a compiled graph.
    pub(crate) fn from_compiled(compiled: &CompiledGraph) -> Self {
        let mut usage = Self::default();
        for pass_usage in compiled.pass_usages() {
            for decl in &pass_usage.texture_usages {
                let key = ResourceKey::Texture(std::sync::Arc::as_ptr(&decl.texture) as usize);
                usage.insert(key, decl.access.is_write());
            }
            for decl in &pass_usage.buffer_usages {
                let key = ResourceKey::Buffer(std::sync::Arc::as_ptr(&decl.buffer) as usize);
                usage.insert(key, decl.access.is_write());
            }
        }
        usage
    }

    fn insert(&mut self, key: ResourceKey, is_write: bool) {
        if is_write {
            self.writes.insert(key);
        } else {
            self.reads.insert(key);
        }
    }

    /// Whether a graph with usage `next`, submitted after `self`, depends on
    /// `self` — i.e. shares a resource in a hazardous way:
    ///
    /// - RAW: `next` reads something `self` wrote
    /// - WAW: `next` writes something `self` wrote
    /// - WAR: `next` writes something `self` read
    ///
    /// Read-after-read is not a hazard and derives no edge.
    pub(crate) fn conflicts_with(&self, next: &GraphUsage) -> bool {
        next.reads.iter().any(|key| self.writes.contains(key)) // RAW
            || next
                .writes
                .iter()
                .any(|key| self.writes.contains(key) || self.reads.contains(key)) // WAW, WAR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: usize) -> ResourceKey {
        ResourceKey::Buffer(id)
    }

    fn usage(reads: &[usize], writes: &[usize]) -> GraphUsage {
        GraphUsage {
            reads: reads.iter().map(|&id| key(id)).collect(),
            writes: writes.iter().map(|&id| key(id)).collect(),
        }
    }

    #[test]
    fn read_after_write_conflicts() {
        let a = usage(&[], &[1]);
        let b = usage(&[1], &[]);
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn write_after_write_conflicts() {
        let a = usage(&[], &[1]);
        let b = usage(&[], &[1]);
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn write_after_read_conflicts() {
        let a = usage(&[1], &[]);
        let b = usage(&[], &[1]);
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn read_after_read_is_no_conflict() {
        let a = usage(&[1], &[]);
        let b = usage(&[1], &[]);
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn disjoint_resources_are_no_conflict() {
        let a = usage(&[1], &[2]);
        let b = usage(&[3], &[4]);
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn texture_and_buffer_keys_never_alias() {
        // Same pointer value, different resource kinds: must not collide.
        let mut a = GraphUsage::default();
        a.insert(ResourceKey::Texture(7), true);
        let mut b = GraphUsage::default();
        b.insert(ResourceKey::Buffer(7), false);
        assert!(!a.conflicts_with(&b));
    }
}
