use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::vertex::DebugVertex;

/// Accumulated vertex data for a single frame.
pub(crate) struct FrameData {
    pub vertices: Vec<DebugVertex>,
}

impl FrameData {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.vertices.clear();
    }
}

/// Thread-safe debug drawing accumulator.
///
/// Uses double-buffered [`FrameData`]:
/// - `frames[current_tick % 2]` is being written to by [`DebugDrawerContext`]s
/// - `frames[(current_tick + 1) % 2]` holds previous tick data, ready for rendering
///
/// Call [`advance_tick`](Self::advance_tick) once per frame before creating any contexts.
/// The renderer reads the previous tick's data via [`take_render_data`](Self::take_render_data).
pub struct DebugDrawer {
    current_tick: AtomicU64,
    frames: Mutex<[FrameData; 2]>,
}

impl DebugDrawer {
    /// Create a new debug drawer starting at tick 0.
    pub fn new() -> Self {
        Self {
            current_tick: AtomicU64::new(0),
            frames: Mutex::new([FrameData::new(), FrameData::new()]),
        }
    }

    /// Get the current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick.load(Ordering::Acquire)
    }

    /// Advance to the next tick.
    ///
    /// Increments the tick counter and clears the new write buffer.
    /// Call this once per frame at the start, before creating any contexts.
    pub fn advance_tick(&self) {
        let new_tick = self.current_tick.load(Ordering::Acquire) + 1;
        let write_index = (new_tick % 2) as usize;
        {
            let mut frames = self.frames.lock();
            frames[write_index].clear();
        }
        self.current_tick.store(new_tick, Ordering::Release);
    }

    /// Create a drawing context for the current tick.
    ///
    /// The context collects vertices locally and flushes them to the
    /// drawer on [`Drop`]. This minimizes lock contention — the mutex
    /// is only held briefly during the flush.
    pub fn context(&self) -> DebugDrawerContext<'_> {
        let tick = self.current_tick.load(Ordering::Acquire);
        DebugDrawerContext {
            drawer: self,
            tick,
            vertices: Vec::new(),
        }
    }

    /// Take the previous tick's render data.
    ///
    /// Returns the accumulated vertices from tick N-1 (while tick N is being
    /// collected). The internal storage is left empty but retains its allocation.
    pub fn take_render_data(&self) -> Vec<DebugVertex> {
        let tick = self.current_tick.load(Ordering::Acquire);
        let render_index = ((tick + 1) % 2) as usize;
        let mut frames = self.frames.lock();
        std::mem::take(&mut frames[render_index].vertices)
    }

    /// Append vertices from a finished context.
    fn flush(&self, tick: u64, vertices: Vec<DebugVertex>) {
        if vertices.is_empty() {
            return;
        }
        let current = self.current_tick.load(Ordering::Acquire);
        if tick != current {
            log::warn!(
                "DebugDrawerContext flushed for tick {} but current is {}; discarding",
                tick,
                current
            );
            return;
        }
        let write_index = (tick % 2) as usize;
        let mut frames = self.frames.lock();
        frames[write_index].vertices.extend_from_slice(&vertices);
    }
}

impl Default for DebugDrawer {
    fn default() -> Self {
        Self::new()
    }
}

/// A short-lived drawing context.
///
/// Collects debug draw vertices locally. On [`Drop`], flushes them
/// to the parent [`DebugDrawer`] under a brief lock.
///
/// Obtain via [`DebugDrawer::context()`].
pub struct DebugDrawerContext<'a> {
    drawer: &'a DebugDrawer,
    tick: u64,
    pub(crate) vertices: Vec<DebugVertex>,
}

impl DebugDrawerContext<'_> {
    /// Push two vertices forming a line segment.
    #[inline]
    pub fn push_line(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 4]) {
        self.vertices.push(DebugVertex {
            position: start,
            color,
        });
        self.vertices.push(DebugVertex {
            position: end,
            color,
        });
    }
}

impl Drop for DebugDrawerContext<'_> {
    fn drop(&mut self) {
        let vertices = std::mem::take(&mut self.vertices);
        self.drawer.flush(self.tick, vertices);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_drawer() {
        let drawer = DebugDrawer::new();
        assert_eq!(drawer.current_tick(), 0);
    }

    #[test]
    fn test_advance_tick() {
        let drawer = DebugDrawer::new();
        drawer.advance_tick();
        assert_eq!(drawer.current_tick(), 1);
        drawer.advance_tick();
        assert_eq!(drawer.current_tick(), 2);
    }

    #[test]
    fn test_context_flush() {
        let drawer = DebugDrawer::new();
        drawer.advance_tick(); // tick = 1

        {
            let mut ctx = drawer.context();
            ctx.draw_line([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]);
        } // ctx dropped, data flushed to tick 1's buffer

        // Advance to tick 2 so tick 1's data becomes the render data
        drawer.advance_tick();

        let data = drawer.take_render_data();
        assert_eq!(data.len(), 2); // one line = 2 vertices
    }

    #[test]
    fn test_empty_render_data() {
        let drawer = DebugDrawer::new();
        let data = drawer.take_render_data();
        assert!(data.is_empty());
    }

    #[test]
    fn test_stale_context_discarded() {
        let drawer = DebugDrawer::new();
        let mut ctx = drawer.context(); // tick = 0
        ctx.push_line([0.0; 3], [1.0; 3], [1.0; 4]);

        drawer.advance_tick(); // tick = 1
        drawer.advance_tick(); // tick = 2
        drop(ctx); // tries to flush to tick 0, but current is 2 -> discarded

        let data = drawer.take_render_data();
        assert!(data.is_empty());
    }

    #[test]
    fn test_multiple_contexts() {
        let drawer = DebugDrawer::new();
        drawer.advance_tick(); // tick = 1

        {
            let mut ctx1 = drawer.context();
            ctx1.draw_line([0.0; 3], [1.0; 3], [1.0, 0.0, 0.0, 1.0]);

            let mut ctx2 = drawer.context();
            ctx2.draw_line([2.0; 3], [3.0; 3], [0.0, 1.0, 0.0, 1.0]);
        }

        drawer.advance_tick(); // tick = 2
        let data = drawer.take_render_data();
        assert_eq!(data.len(), 4); // 2 lines = 4 vertices
    }
}
