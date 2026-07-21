//! Per-camera screen-space ambient occlusion (#150).

/// Screen-space ambient occlusion on a camera entity: a GTAO-lite (horizon-
/// based) pass that darkens the deferred path's image-based ambient in
/// creases and contacts. Opt-in, exactly like
/// [`TemporalJitter`](super::TemporalJitter) — a camera **without** this
/// component records no SSAO pass and its ambient is unchanged (existing
/// golden images stay byte-stable).
///
/// The occlusion multiplies the IBL diffuse/specular terms only; direct
/// lights stay unoccluded (real shadows are #130), consistent with the
/// resolve pass's existing material-AO handling.
#[derive(Debug, Clone, Copy, PartialEq, crate::Component)]
pub struct CameraAmbientOcclusion {
    /// World-space sampling radius: how far a surface point looks for
    /// occluders. Scene-scale dependent — a value near a typical feature size
    /// (crease width) reads best.
    pub radius: f32,
    /// Strength of the effect (`1.0` = the raw horizon estimate; higher
    /// darkens more). Applied before [`power`](Self::power).
    pub intensity: f32,
    /// Contrast exponent on the AO factor (`ao = pow(ao, power)`): >1 deepens
    /// the darkest occlusion while leaving open surfaces near white.
    pub power: f32,
}

impl CameraAmbientOcclusion {
    /// SSAO with explicit parameters.
    pub fn new(radius: f32, intensity: f32, power: f32) -> Self {
        Self {
            radius,
            intensity,
            power,
        }
    }
}

impl Default for CameraAmbientOcclusion {
    fn default() -> Self {
        // A half-unit radius suits the meter-ish scene scale; intensity 1.0 is
        // the physical estimate; power 1.5 adds a little contact contrast.
        Self {
            radius: 0.5,
            intensity: 1.0,
            power: 1.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_sane() {
        let ao = CameraAmbientOcclusion::default();
        assert!(ao.radius > 0.0 && ao.intensity > 0.0 && ao.power >= 1.0);
    }
}
