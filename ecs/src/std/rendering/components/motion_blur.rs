//! Motion-blur opt-in for a camera (#149).

/// Opts a camera into tile-based motion blur (#149, McGuire/Guertin) on the
/// deferred path. The filter runs after TAA, reconstructing each pixel's smear
/// from the completed `GBUFFER_VELOCITY` (geometry + camera, background
/// included via the velocity-completion pass).
///
/// Cameras without the component render exactly as before — motion blur is a
/// fullscreen post pass that only slots in when this component is present.
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, crate::Component)]
#[repr(C)]
pub struct MotionBlur {
    /// Shutter fraction: the portion of the frame interval the shutter stays
    /// open (0.5 ≈ a 180° shutter). Scales the per-frame velocity into blur
    /// length, so it is frame-rate-independent for a fixed shutter *angle*.
    pub shutter: f32,
    /// Reconstruction taps along the dominant motion. More is smoother and
    /// costlier; odd values keep a centred sample. 15 is a good default.
    pub samples: u32,
}

impl Default for MotionBlur {
    fn default() -> Self {
        Self {
            shutter: 0.5,
            samples: 15,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let mb = MotionBlur::default();
        assert_eq!(mb.shutter, 0.5);
        assert_eq!(mb.samples, 15);
    }
}
