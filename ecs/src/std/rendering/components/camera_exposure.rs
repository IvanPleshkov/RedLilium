//! Manual exposure for the standard render path (#142).

/// Manual exposure on a camera entity: a linear multiplier applied to the
/// scene-referred color in the deferred path's display-output pass, before
/// tonemap/encode. A camera without the component renders at `1.0`.
///
/// Deliberately a bare multiplier, not a physical EV — physical units (and
/// auto-exposure) arrive together with photometric lights (#142 follow-up).
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, crate::Component)]
#[repr(C)]
pub struct CameraExposure {
    /// Linear scale on scene-referred radiance (`1.0` = neutral).
    pub exposure: f32,
}

impl CameraExposure {
    /// Neutral exposure.
    pub fn new(exposure: f32) -> Self {
        Self { exposure }
    }
}

impl Default for CameraExposure {
    fn default() -> Self {
        Self { exposure: 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_neutral() {
        assert_eq!(CameraExposure::default().exposure, 1.0);
    }
}
