//! Parallel per-entity iteration support.
//!
//! Splits entity processing across threads using [`std::thread::scope`],
//! falling back to sequential iteration on WASM where threads are
//! unavailable.
//!
//! The core function [`par_for_each_entities`] is used by
//! [`QueryGuard::par_for_each`](crate::QueryGuard::par_for_each),
//! [`ForEachAccess::run_par_for_each`](crate::ForEachAccess::run_par_for_each),
//! and [`LockRequest::par_for_each`](crate::LockRequest::par_for_each).

use crate::query_guard::QueryItem;

/// Configuration for parallel iteration.
///
/// Controls the number of worker threads and minimum batch size.
/// Use [`Default::default()`] for sensible defaults.
#[derive(Debug, Clone)]
pub struct ParConfig {
    /// Minimum number of entities per batch. Prevents thread overhead
    /// from dominating for small workloads. Default: 64.
    pub min_batch_size: usize,
    /// Number of worker threads. `None` uses
    /// [`std::thread::available_parallelism`]. Default: `None`.
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

impl ParConfig {
    fn effective_threads(&self) -> usize {
        self.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }
}

/// Entity count below which parallel iteration is not worth the overhead.
const PARALLEL_THRESHOLD: usize = 128;

/// Parallel iteration over a slice of entity indices.
///
/// Splits `entities` into chunks and processes them on separate threads
/// via [`std::thread::scope`]. Each thread calls `items.query_get(entity)`
/// for its chunk and passes matching results to `f`.
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
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn par_for_each_entities<I, F>(items: &I, entities: &[u32], config: &ParConfig, f: &F)
where
    I: QueryItem + Sync,
    F: Fn(u32, I::Item) + Sync,
{
    let count = entities.len();

    if count < PARALLEL_THRESHOLD || count < config.min_batch_size {
        for &entity in entities {
            // SAFETY: entities are unique (sparse set invariant), each
            // index visited exactly once.
            if let Some(item) = unsafe { items.query_get(entity) } {
                f(entity, item);
            }
        }
        return;
    }

    let num_threads = config.effective_threads().max(1);
    let batch_size = (count / (num_threads * 4))
        .max(config.min_batch_size)
        .max(1);

    std::thread::scope(|scope| {
        for chunk in entities.chunks(batch_size) {
            scope.spawn(move || {
                for &entity in chunk {
                    // SAFETY: each entity index is unique within the dense
                    // array, and chunks are disjoint, so no two threads
                    // access the same entity index.
                    if let Some(item) = unsafe { items.query_get(entity) } {
                        f(entity, item);
                    }
                }
            });
        }
    });
}

/// WASM fallback: sequential iteration (no threads available).
#[cfg(target_arch = "wasm32")]
pub(crate) fn par_for_each_entities<I, F>(items: &I, entities: &[u32], _config: &ParConfig, f: &F)
where
    I: QueryItem + Sync,
    F: Fn(u32, I::Item) + Sync,
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

    #[test]
    fn effective_threads_uses_available_parallelism() {
        let config = ParConfig::default();
        let threads = config.effective_threads();
        assert!(threads >= 1);
    }

    #[test]
    fn effective_threads_respects_override() {
        let config = ParConfig {
            num_threads: Some(3),
            ..Default::default()
        };
        assert_eq!(config.effective_threads(), 3);
    }
}
