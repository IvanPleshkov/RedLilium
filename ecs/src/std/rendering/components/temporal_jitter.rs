//! Temporal-jitter opt-in for a camera (#147).

/// Opts a camera into sub-pixel projection jitter — the sampling half of the
/// temporal contract (TAA #148, upscalers later). The dispatcher offsets the
/// camera's projection by a Halton(2,3) sub-pixel amount each frame (±0.5px,
/// fixed by the geometry of temporal supersampling — not a tunable).
///
/// Cameras without the component render bit-identically to before: picking,
/// gizmos, and golden tests on the editor camera are untouched. The G-buffer
/// velocity target is written regardless — velocity is history, jitter is
/// sampling; they opt in independently.
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, crate::Component)]
#[repr(C)]
pub struct TemporalJitter {
    /// Jitter sequence length. 8 matches a TAA history blend of ~0.9;
    /// upscalers stretch it (FSR2: `8 * (display/render)^2`).
    pub cycle: u32,
}

impl Default for TemporalJitter {
    fn default() -> Self {
        Self { cycle: 8 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cycle() {
        assert_eq!(TemporalJitter::default().cycle, 8);
    }
}
