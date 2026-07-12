//! Camera output specification (ADR-029, #74).
//!
//! A camera's graphics stack is split in two, following the pattern every
//! engine converged on (Unity `RenderTexture`, Unreal `TextureRenderTarget2D`,
//! Bevy `RenderTarget::Image`):
//!
//! - **[`CameraOutput`] — the serializable *intent*** (this component): what
//!   surface the camera renders to, how it is sized, and under which asset
//!   identity its output is published. Plain data; lives in scene assets.
//! - **[`CameraTarget`](super::CameraTarget) — the runtime *derived cache***:
//!   the actual GPU textures. Never serialized; created and resized by the
//!   [`EnsureCameraTargets`](crate::std::rendering::EnsureCameraTargets)
//!   system whenever it disagrees with the spec.
//!
//! Cameras **without** `CameraOutput` are left alone — their `CameraTarget`
//! (if any) is host-managed (the editor's scene view does this today).

use redlilium_assets::Guid;

/// How an offscreen target is sized.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SizePolicy {
    /// Match the main viewport (requires the
    /// [`MainViewport`](crate::std::rendering::MainViewport) resource).
    Viewport,
    /// A fraction of the main viewport (e.g. `0.5` = half resolution).
    ViewportScale(f32),
    /// Fixed size in pixels.
    Fixed(u32, u32),
}

/// What surface the camera renders to.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CameraTargetSpec {
    /// The main viewport: a viewport-sized offscreen target the host
    /// composites to the swapchain (the runtime blit / editor panel).
    Screen,
    /// An offscreen texture. When `output` is set, the color texture is
    /// published as a **virtual texture asset** under that GUID
    /// (`TextureSource::Virtual`) so materials can sample it like any other
    /// texture — mirrors, minimaps, portals.
    Offscreen {
        /// Target sizing.
        size: SizePolicy,
        /// Asset identity to publish the color output under, if anyone
        /// samples it.
        output: Option<Guid>,
    },
}

/// Serializable camera output spec — see the module docs. Attach next to a
/// [`Camera`](crate::Camera) to opt the entity into system-managed targets.
///
/// Formats are engine-standard for now (`Rgba8Unorm` color +
/// `Depth32Float` depth); a per-camera format choice is a future extension
/// with its own trade-offs (serialization of formats, HDR chains).
#[derive(Debug, Clone, crate::Component)]
pub struct CameraOutput {
    /// What surface to render to.
    pub target: CameraTargetSpec,
    /// Clear color (RGBA) applied at the start of the camera's pass.
    pub clear_color: [f32; 4],
}

impl Default for CameraOutput {
    fn default() -> Self {
        Self::screen()
    }
}

impl CameraOutput {
    /// Render to the main viewport (composited to the swapchain by the host).
    pub fn screen() -> Self {
        Self {
            target: CameraTargetSpec::Screen,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Render offscreen with the given sizing, publishing the color output
    /// as a virtual texture asset under `output`.
    pub fn offscreen(size: SizePolicy, output: Option<Guid>) -> Self {
        Self {
            target: CameraTargetSpec::Offscreen { size, output },
            clear_color: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Set the clear color.
    pub fn with_clear_color(mut self, clear_color: [f32; 4]) -> Self {
        self.clear_color = clear_color;
        self
    }
}
