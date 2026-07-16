//! Play-scoped async task lifecycle management.

use redlilium_core::compute::CancellationToken;
use std::collections::BTreeMap;

/// Manages play-scoped task lifecycle with generation-based cancellation.
///
/// Game plugins should spawn async tasks via [`PlayTasks::spawn`] instead of directly
/// using [`ComputePool`](crate::ComputePool). When a play session ends, all tasks spawned
/// during the current play generation are cancelled, ensuring no game code remains executing
/// before warm-reload dylib unmap.
///
/// This is a critical soundness requirement: without generation-scoped cancellation,
/// futures running dylib code could access dangling pointers during reload.
///
/// # Generation Lifecycle
///
/// - **On Play**: generation counter increments; new tasks join the active generation
/// - **On Stop**: all tokens for the active generation are cancelled; the map entry is cleared
#[derive(Clone)]
pub struct PlayTasks {
    current_generation: u64,
    /// Cancellation tokens per play generation. When Stop occurs, we cancel all
    /// tokens for the active generation and clear that map entry.
    generation_tokens: std::sync::Arc<crate::sync::Mutex<BTreeMap<u64, Vec<CancellationToken>>>>,
}

impl PlayTasks {
    /// Create a new PlayTasks resource in generation 0.
    pub fn new() -> Self {
        Self {
            current_generation: 0,
            generation_tokens: std::sync::Arc::new(crate::sync::Mutex::new(BTreeMap::new())),
        }
    }

    /// Returns the current play generation.
    pub fn current_generation(&self) -> u64 {
        self.current_generation
    }

    /// Spawn an async compute task scoped to the current play generation.
    ///
    /// On Stop, this task's cancellation token will be automatically cancelled,
    /// terminating the task cooperatively (at its next checkpoint or yield).
    pub fn spawn<T, F, Fut>(
        &mut self,
        pool: &crate::ComputePool,
        priority: redlilium_core::compute::Priority,
        f: F,
    ) -> crate::TaskHandle<T>
    where
        T: Send + 'static,
        Fut: std::future::Future<Output = T> + Send + 'static,
        F: FnOnce(crate::EcsComputeContext) -> Fut + Send + 'static,
    {
        let handle = pool.spawn(priority, f);
        let token = handle.cancellation_token();

        let mut tokens = self.generation_tokens.lock();
        tokens
            .entry(self.current_generation)
            .or_default()
            .push(token);

        handle
    }

    /// Begin a new play generation. Call when a play session starts; new tasks
    /// spawned afterward join this generation.
    pub fn begin_generation(&mut self) {
        self.current_generation += 1;
    }

    /// Cancel all tasks belonging to the current play generation. Call when a
    /// play session ends (before the play world is dropped), ensuring no game
    /// code remains executing across a warm-reload dylib unmap.
    pub fn cancel_current_generation(&mut self) {
        let mut tokens = self.generation_tokens.lock();
        if let Some(to_cancel) = tokens.remove(&self.current_generation) {
            for token in to_cancel {
                token.cancel();
            }
        }
    }
}

impl Default for PlayTasks {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_tasks_initial_generation() {
        let tasks = PlayTasks::new();
        assert_eq!(tasks.current_generation(), 0);
    }

    #[test]
    fn play_tasks_increments_generation_on_play_start() {
        let mut tasks = PlayTasks::new();
        assert_eq!(tasks.current_generation(), 0);

        tasks.begin_generation();
        assert_eq!(tasks.current_generation(), 1);

        tasks.begin_generation();
        assert_eq!(tasks.current_generation(), 2);
    }

    #[test]
    fn play_tasks_cancels_tokens_on_stop() {
        use crate::{ComputePool, Priority};
        use redlilium_core::compute::yield_now;

        let pool = ComputePool::new(crate::IoRuntime::new());
        let mut tasks = PlayTasks::new();

        // Increment generation to 1 (simulating Play)
        tasks.begin_generation();

        // Spawn a task that yields
        let handle = tasks.spawn(&pool, Priority::Low, |_ctx| async {
            yield_now().await;
            42u32
        });

        // Task should be pending
        assert!(!handle.is_done());

        // Trigger Stop (cancel tokens for generation 1)
        tasks.cancel_current_generation();

        // Tick the pool — task should complete (be removed/cancelled)
        pool.tick();

        // Pool should be empty (task was cancelled/removed)
        assert_eq!(pool.pending_count(), 0);
    }

    #[test]
    fn play_tasks_only_cancels_current_generation() {
        use crate::{ComputePool, Priority};
        use redlilium_core::compute::yield_now;

        let pool = ComputePool::new(crate::IoRuntime::new());
        let mut tasks = PlayTasks::new();

        // Generation 1: spawn and complete a task
        tasks.begin_generation();
        let h1 = tasks.spawn(&pool, Priority::Low, |_ctx| async { 1u32 });
        pool.tick();
        assert!(h1.is_done());

        // Generation 2: spawn a task that yields
        tasks.begin_generation();
        let h2 = tasks.spawn(&pool, Priority::Low, |_ctx| async {
            yield_now().await;
            2u32
        });
        assert!(!h2.is_done());

        // Stop only cancels generation 2's tasks, not generation 1's (which is already done)
        tasks.cancel_current_generation();
        pool.tick();

        // h2 should be cancelled and pool empty
        assert_eq!(pool.pending_count(), 0);
    }

    #[test]
    fn play_tasks_spawn_across_multiple_generations() {
        use crate::{ComputePool, Priority};
        use redlilium_core::compute::yield_now;

        let pool = ComputePool::new(crate::IoRuntime::new());
        let mut tasks = PlayTasks::new();

        // Generation 1: spawn task (gen 1 == tasks.current_generation() + 1 after begin_generation)
        tasks.begin_generation();
        assert_eq!(tasks.current_generation(), 1);
        let h1_gen1 = tasks.spawn(&pool, Priority::Low, |_ctx| async { 1u32 });
        assert!(!h1_gen1.is_done());

        // Stop gen 1: cancel all gen 1 tasks
        tasks.cancel_current_generation();
        pool.tick();
        assert_eq!(pool.pending_count(), 0, "Gen 1 task should be cancelled");

        // Generation 2: spawn new task
        tasks.begin_generation();
        assert_eq!(tasks.current_generation(), 2);
        let h2_gen2 = tasks.spawn(&pool, Priority::Low, |_ctx| async { 2u32 });
        assert!(!h2_gen2.is_done());

        // Stop gen 2: cancel all gen 2 tasks
        tasks.cancel_current_generation();
        pool.tick();
        assert_eq!(pool.pending_count(), 0, "Gen 2 task should be cancelled");

        // Verify we can start gen 3 with no residual state
        tasks.begin_generation();
        assert_eq!(tasks.current_generation(), 3);
        let h3_gen3 = tasks.spawn(&pool, Priority::Low, |_ctx| async {
            yield_now().await;
            3u32
        });
        assert!(!h3_gen3.is_done());

        tasks.cancel_current_generation();
        pool.tick();
        assert_eq!(pool.pending_count(), 0, "Gen 3 task should be cancelled");
    }

    #[test]
    fn play_tasks_mixed_completed_and_pending_cancel() {
        use crate::{ComputePool, Priority};
        use redlilium_core::compute::yield_now;

        let pool = ComputePool::new(crate::IoRuntime::new());
        let mut tasks = PlayTasks::new();

        tasks.begin_generation();

        // Spawn task that completes immediately
        let h1 = tasks.spawn(&pool, Priority::Low, |_ctx| async { 1u32 });
        pool.tick();
        assert!(h1.is_done());

        // Spawn task that yields (pending)
        let h2 = tasks.spawn(&pool, Priority::Low, |_ctx| async {
            yield_now().await;
            2u32
        });
        assert!(!h2.is_done());

        // Stop cancels only pending tasks for current generation
        tasks.cancel_current_generation();
        pool.tick();

        // Both should be handled (completed one was already done, pending one was cancelled)
        assert_eq!(pool.pending_count(), 0);
    }
}
