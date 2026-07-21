//! Per-camera HDR bloom (#151).

/// HDR bloom on a camera entity: the Jimenez dual-filter (CoD:AW) progressive
/// down/up-sample glow, thresholdless. Opt-in like
/// [`TemporalJitter`](super::TemporalJitter) /
/// [`CameraAmbientOcclusion`](super::CameraAmbientOcclusion) — a camera
/// **without** this component records no bloom passes and its image is
/// byte-identical.
///
/// Thresholdless: every bright region contributes proportionally (a Karis
/// average on the first downsample keeps a lone bright pixel from flickering).
/// The single knob is [`intensity`](Self::intensity), the mix weight of the
/// accumulated glow into the scene-referred color before exposure (the
/// bounded Jimenez/CoD `lerp(scene, bloom, intensity)` composite).
#[derive(Debug, Clone, Copy, PartialEq, crate::Component)]
pub struct CameraBloom {
    /// Mix weight of the bloom into the scene: `lerp(scene, bloom, intensity)`,
    /// applied before exposure/tonemap. Small values (a few percent) read as a
    /// natural glow; near 1.0 the image becomes the blurred bloom.
    pub intensity: f32,
}

impl CameraBloom {
    /// Bloom with an explicit intensity.
    pub fn new(intensity: f32) -> Self {
        Self { intensity }
    }
}

impl Default for CameraBloom {
    fn default() -> Self {
        // A tasteful default glow. With the normalized (energy-preserving)
        // accumulation the bloom is about scene magnitude, so `intensity` is a
        // true blend fraction — a mid-teens percent reads as a gentle glow on
        // highlights, not a haze over the whole frame.
        Self { intensity: 0.15 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_subtle() {
        let b = CameraBloom::default();
        assert!(b.intensity > 0.0 && b.intensity < 0.5);
    }
}
