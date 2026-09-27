//! Parallel per-entity iteration support.
//!
//! Splits entity processing across reusable workers owned by the world,
//! falling back to sequential iteration on WASM where threads are
//! unavailable.
//!
//! The core function [`par_for_each_entities`] is used by
//! [`QueryGuard::par_for_each`](crate::QueryGuard::par_for_each),
//! [`ForEachAccess::run_par_for_each`](crate::ForEachAccess::run_par_for_each),
//! and [`LockRequest::par_for_each`](crate::LockRequest::par_for_each).

use crate::query::QueryItem;

/// Configuration for parallel iteration.
///
/// Controls maximum participants (including the caller) and minimum batch size.
/// Use [`Default::default()`] for sensible defaults.
#[derive(Debug, Clone)]
pub struct ParConfig {
    /// Minimum number of entities per batch. Prevents scheduling overhead
    /// from dominating for small workloads. Zero is treated as one. Default: 64.
    pub min_batch_size: usize,
    /// Maximum participants, including the calling thread, capped by the
    /// executor capacity. `None` uses that capacity; zero is treated as one.
    /// Busy workers, small queries, or startup failure can reduce parallelism.
    /// Callbacks must not wait for other callbacks to run concurrently.
    /// Default: `None`.
    pub num_threads: Option<usize>,
}

impl Default for ParConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 64,
            num_threads: None,
        }
    }
}

/// Entity count below which parallel iteration is not worth the overhead.
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_THRESHOLD: usize = 128;

/// Parallel iteration over a slice of entity indices.
///
/// Workers and the caller claim disjoint batches from an atomic counter.
/// Each participant calls `items.query_get(entity)` for its batch and passes
/// matching results to `f`. All borrowed work completes before returning.
///
/// Falls back to sequential for small entity counts (< [`PARALLEL_THRESHOLD`])
/// or when `min_batch_size` would result in a single batch.
///
/// # Safety contract (upheld by callers)
///
/// - Each entity index in `entities` appears at most once (sparse set
///   invariant: dense entity arrays have no duplicates).
/// - [`QueryItem::query_get`] returns disjoint memory for different
///   entity indices (guaranteed by sparse set layout: different entity
///   indices map to different dense slots).
/// - `I: Sync` ensures `&items` can be shared across threads.
/// - The item lifetime `'x` is chosen by the caller and must not outlive
///   the locks backing `items` (callers tie it to a borrow of the
///   owning guard, or protect it with an HRTB closure bound).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn par_for_each_entities<'x, I, F>(
    executor: &crate::ParallelExecutor,
    items: &I,
    entities: &[u32],
    config: &ParConfig,
    f: &F,
) where
    I: QueryItem + Sync,
    F: Fn(u32, I::Item<'x>) + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering};

    let count = entities.len();
    let min_batch = config.min_batch_size.max(1);
    let participants = config
        .num_threads
        .unwrap_or(executor.parallelism())
        .max(1)
        .min(executor.parallelism())
        .min((count / min_batch).max(1));

    if count < PARALLEL_THRESHOLD || participants == 1 {
        for &entity in entities {
            // SAFETY: entities are unique (sparse set invariant), each
            // index visited exactly once.
            if let Some(item) = unsafe { items.query_get(entity) } {
                f(entity, item);
            }
        }
        return;
    }

    let batch_size = count
        .div_ceil(participants.saturating_mul(4))
        .max(min_batch);
    let batches = count.div_ceil(batch_size);
    let next = AtomicUsize::new(0);
    executor.run(participants, || {
        loop {
            let batch = next.fetch_add(1, Ordering::Relaxed);
            if batch >= batches {
                break;
            }
            let start = batch * batch_size;
            let end = start.saturating_add(batch_size).min(count);
            for &entity in &entities[start..end] {
                // SAFETY: the atomic counter assigns disjoint batches; entity
                // indices are unique and the executor drains all borrowed work.
                if let Some(item) = unsafe { items.query_get(entity) } {
                    f(entity, item);
                }
            }
        }
    });
}

/// WASM fallback: sequential iteration (no threads available).
#[cfg(target_arch = "wasm32")]
pub(crate) fn par_for_each_entities<'x, I, F>(
    _executor: &crate::ParallelExecutor,
    items: &I,
    entities: &[u32],
    _config: &ParConfig,
    f: &F,
) where
    I: QueryItem + Sync,
    F: Fn(u32, I::Item<'x>) + Sync,
{
    for &entity in entities {
        if let Some(item) = unsafe { items.query_get(entity) } {
            f(entity, item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_config_default() {
        let config = ParConfig::default();
        assert_eq!(config.min_batch_size, 64);
        assert!(config.num_threads.is_none());
    }
}
