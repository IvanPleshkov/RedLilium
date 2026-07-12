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
//! (if any) is host-managed; every in-tree camera (runtime primary, editor
//! scene view) now authors a spec.

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

/// Color format of a camera's output — a small *curated* set (the authored
/// choices that exist), deliberately not a mirror of
/// [`TextureFormat`](redlilium_graphics::TextureFormat) (which carries no
/// serde, and most of which makes no sense as a camera output). Same approach
/// as shader variant axes (#6): serialize the intent, map to graphics types
/// at the edge. Depth is always `Depth32Float`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OutputFormat {
    /// Linear 8-bit (`Rgba8Unorm`).
    #[default]
    Standard,
    /// sRGB-encoded 8-bit (`Bgra8UnormSrgb`) — for outputs displayed 1:1 on
    /// an sRGB surface (the editor's scene panel, the runtime blit).
    Srgb,
    /// Linear half-float (`Rgba16Float`) — HDR chains.
    Hdr,
}

impl OutputFormat {
    /// The graphics color format this output renders into.
    pub fn color_format(self) -> redlilium_graphics::TextureFormat {
        use redlilium_graphics::TextureFormat;
        match self {
            Self::Standard => TextureFormat::Rgba8Unorm,
            Self::Srgb => TextureFormat::Bgra8UnormSrgb,
            Self::Hdr => TextureFormat::Rgba16Float,
        }
    }

    /// The output that matches a surface's color space — hosts derive the
    /// primary camera's format from the swapchain so the scene is stored in
    /// the space the compositor expects.
    pub fn matching_surface(surface: redlilium_graphics::TextureFormat) -> Self {
        if surface.is_hdr() {
            Self::Hdr
        } else if surface.is_srgb() {
            Self::Srgb
        } else {
            Self::Standard
        }
    }
}

/// Serializable camera output spec — see the module docs. Attach next to a
/// [`Camera`](crate::Camera) to opt the entity into system-managed targets.
#[derive(Debug, Clone, PartialEq, crate::Component)]
pub struct CameraOutput {
    /// What surface to render to.
    pub target: CameraTargetSpec,
    /// Clear color (RGBA) applied at the start of the camera's pass.
    pub clear_color: [f32; 4],
    /// Color format of the target (depth is always `Depth32Float`).
    pub format: OutputFormat,
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
            format: OutputFormat::Standard,
        }
    }

    /// Render offscreen with the given sizing, publishing the color output
    /// as a virtual texture asset under `output`.
    pub fn offscreen(size: SizePolicy, output: Option<Guid>) -> Self {
        Self {
            target: CameraTargetSpec::Offscreen { size, output },
            clear_color: [0.0, 0.0, 0.0, 1.0],
            format: OutputFormat::Standard,
        }
    }

    /// Set the clear color.
    pub fn with_clear_color(mut self, clear_color: [f32; 4]) -> Self {
        self.clear_color = clear_color;
        self
    }

    /// Set the output color format.
    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }
}
