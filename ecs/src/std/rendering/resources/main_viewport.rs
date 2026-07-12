//! Main viewport size resource.

/// Size of the main viewport in pixels, updated by the host every frame
/// (the runtime sets it to the window's inner size; the editor would set it
/// to the game panel). Consumed by
/// [`EnsureCameraTargets`](crate::std::rendering::EnsureCameraTargets) to
/// size `Screen` / `Viewport`-policy camera targets (ADR-029).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MainViewport {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl MainViewport {
    /// Create a viewport size.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}
