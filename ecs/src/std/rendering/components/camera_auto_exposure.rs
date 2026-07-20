//! Per-camera automatic exposure / eye adaptation (#153).

/// Histogram-based auto-exposure on a camera entity: two compute passes build
/// a luminance histogram of the scene-referred image and meter an adapting
/// exposure multiplier the display pass applies. Opt-in like
/// [`CameraBloom`](super::CameraBloom) /
/// [`CameraAmbientOcclusion`](super::CameraAmbientOcclusion) — a camera
/// **without** this component records no compute passes and the display binds
/// a neutral 1.0 exposure buffer, so its image is byte-identical.
///
/// When present this component **drives** the exposure that would otherwise
/// come from [`CameraExposure`](super::CameraExposure): the metered multiplier
/// (clamped to `[2^min_ev, 2^max_ev]`) times `2^compensation`. A camera should
/// carry one or the other; if both are present the auto path wins and only the
/// compensation bias survives.
#[derive(Debug, Clone, Copy, PartialEq, crate::Component)]
pub struct CameraAutoExposure {
    /// Darkest the adaptation may push, in stops relative to middle grey — a
    /// floor on the metered multiplier (`2^min_ev`). Stops a dim scene from
    /// being amplified without bound.
    pub min_ev: f32,
    /// Brightest the adaptation may push, in stops (`2^max_ev`). Caps how far
    /// a bright scene is darkened.
    pub max_ev: f32,
    /// Adaptation speed when the scene **brightens** (per second): how fast the
    /// eye stops down walking into sunlight. Larger = snappier.
    pub speed_up: f32,
    /// Adaptation speed when the scene **darkens** (per second): the slower
    /// dark-adaptation of walking into shade. Usually below [`speed_up`](Self::speed_up).
    pub speed_down: f32,
    /// Exposure-compensation bias in stops, applied on top of the metered
    /// result (`2^compensation`). Positive brightens the adapted image.
    pub compensation: f32,
}

impl CameraAutoExposure {
    /// Auto-exposure with explicit adaptation bounds and speeds.
    pub fn new(
        min_ev: f32,
        max_ev: f32,
        speed_up: f32,
        speed_down: f32,
        compensation: f32,
    ) -> Self {
        Self {
            min_ev,
            max_ev,
            speed_up,
            speed_down,
            compensation,
        }
    }
}

impl Default for CameraAutoExposure {
    fn default() -> Self {
        // A ±6-stop adaptation window (multiplier in [1/64, 64]) covers most
        // scenes; asymmetric speeds (fast to brighten, slower to darken)
        // mimic the eye and hide the metering lag on cuts to bright frames.
        Self {
            min_ev: -6.0,
            max_ev: 6.0,
            speed_up: 3.0,
            speed_down: 1.0,
            compensation: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_window_is_symmetric_and_bounded() {
        let a = CameraAutoExposure::default();
        assert!(a.min_ev < 0.0 && a.max_ev > 0.0);
        assert_eq!(a.min_ev, -a.max_ev);
        // Brighten faster than darken (eye-like).
        assert!(a.speed_up > a.speed_down);
    }
}
