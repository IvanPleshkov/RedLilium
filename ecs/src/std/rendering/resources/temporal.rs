//! Per-frame temporal state for the velocity/jitter contract (#147).

use std::collections::HashMap;

use crate::Entity;

/// Render-side history the temporal contract needs (#147): last frame's
/// unjittered view-projection per camera and model matrix per entity, plus
/// the global frame index driving the jitter sequence.
///
/// Rotated exactly once per frame by the render dispatcher
/// ([`CameraRender`](super::super::CameraRender)) via [`begin_frame`](Self::begin_frame);
/// [`VisibleScene::gather`](super::super::VisibleScene::gather) and the
/// per-view loop then fill the current-frame maps. Ensured by
/// [`EnsureCameraTargets`](super::super::EnsureCameraTargets), so hosts need
/// no extra wiring.
#[derive(Default)]
pub struct TemporalState {
    /// Global frame index — drives the Halton jitter sequence.
    frame: u64,
    /// Unjittered view-projection per camera, previous frame.
    prev_view_proj: HashMap<Entity, [[f32; 4]; 4]>,
    /// Unjittered view-projection per camera, being filled this frame.
    curr_view_proj: HashMap<Entity, [[f32; 4]; 4]>,
    /// Model matrix per visible entity, previous frame.
    prev_models: HashMap<Entity, [[f32; 4]; 4]>,
    /// Model matrix per visible entity, being filled this frame.
    curr_models: HashMap<Entity, [[f32; 4]; 4]>,
}

impl TemporalState {
    /// Advance to the next frame: current maps become the previous ones.
    /// Entities/cameras absent last frame simply miss — consumers fall back
    /// to "previous = current" (zero velocity), so a fresh spawn never
    /// smears.
    pub fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.prev_view_proj = std::mem::take(&mut self.curr_view_proj);
        self.prev_models = std::mem::take(&mut self.curr_models);
    }

    /// The global frame index (jitter-sequence position).
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Last frame's unjittered view-projection of `camera`, if it rendered.
    pub fn prev_view_proj(&self, camera: Entity) -> Option<[[f32; 4]; 4]> {
        self.prev_view_proj.get(&camera).copied()
    }

    /// Record `camera`'s unjittered view-projection for the next frame.
    pub fn record_view_proj(&mut self, camera: Entity, view_proj: [[f32; 4]; 4]) {
        self.curr_view_proj.insert(camera, view_proj);
    }

    /// Last frame's model matrix of `entity`, if it was visible.
    pub fn prev_model(&self, entity: Entity) -> Option<[[f32; 4]; 4]> {
        self.prev_models.get(&entity).copied()
    }

    /// Record `entity`'s model matrix for the next frame.
    pub fn record_model(&mut self, entity: Entity, model: [[f32; 4]; 4]) {
        self.curr_models.insert(entity, model);
    }
}

/// The Halton low-discrepancy sequence (1-based index), the standard TAA
/// jitter source: successive prefixes cover the unit interval evenly.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut fraction = 1.0f32;
    let mut result = 0.0f32;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// Sub-pixel jitter for `frame` in a Halton(2,3) cycle of `cycle` samples,
/// in pixel units within `[-0.5, 0.5)`. Multiply by `2 / target_size` for
/// the NDC offset.
pub fn jitter_pixels(frame: u64, cycle: u32) -> [f32; 2] {
    let cycle = cycle.max(1);
    let index = (frame % cycle as u64) as u32 + 1;
    [halton(index, 2) - 0.5, halton(index, 3) - 0.5]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The classic Halton prefixes (base 2: 1/2, 1/4, 3/4; base 3: 1/3,
    /// 2/3, 1/9).
    #[test]
    fn halton_known_prefixes() {
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
        assert!((halton(3, 3) - 1.0 / 9.0).abs() < 1e-6);
    }

    /// Jitter stays sub-pixel, cycles with the requested period, and the
    /// cycle visits distinct positions.
    #[test]
    fn jitter_bounded_and_cyclic() {
        let mut seen = std::collections::HashSet::new();
        for frame in 0..8u64 {
            let [x, y] = jitter_pixels(frame, 8);
            assert!((-0.5..0.5).contains(&x) && (-0.5..0.5).contains(&y));
            seen.insert((x.to_bits(), y.to_bits()));
            let [nx, ny] = jitter_pixels(frame + 8, 8);
            assert_eq!((x, y), (nx, ny), "period 8");
        }
        assert_eq!(seen.len(), 8, "distinct sub-pixel positions");
    }

    /// A fresh frame rotates current into previous; missing history reads
    /// as None.
    #[test]
    fn state_rotation() {
        let mut state = TemporalState::default();
        let camera = Entity::new(1, 0);
        let entity = Entity::new(2, 0);
        let m = [[1.0f32; 4]; 4];
        state.record_view_proj(camera, m);
        state.record_model(entity, m);
        assert!(state.prev_view_proj(camera).is_none());
        state.begin_frame();
        assert_eq!(state.prev_view_proj(camera), Some(m));
        assert_eq!(state.prev_model(entity), Some(m));
        state.begin_frame();
        assert!(state.prev_view_proj(camera).is_none());
        assert!(state.prev_model(entity).is_none());
    }
}
