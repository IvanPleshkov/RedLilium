//! Per-camera render path selection (ADR-035, #128).
//!
//! A camera's graphics stack has two serializable knobs:
//!
//! - **[`CameraOutput`](super::CameraOutput) — *where* the image goes**: the
//!   surface, its sizing, and the asset identity it is published under.
//! - **[`RenderPath`] — *how* the image is produced** (this component): the
//!   stable name of the [`CameraRenderPipeline`](crate::std::rendering::CameraRenderPipeline)
//!   that records the camera's passes.
//!
//! The split is deliberate (ADR-029 amendment): `CameraOutput` is the contract
//! with *consumers* of the camera's image, while the render path is the
//! camera's internal production recipe whose settings grow independently
//! (shadow parameters, quality tiers, HDR chains).
//!
//! Scene files carry only the pipeline *name*; the implementation is resolved
//! at render time through the
//! [`PipelineRegistry`](crate::std::rendering::PipelineRegistry) — the same
//! "serialize the intent, derive the resource" discipline as the rest of the
//! camera stack. A name with no registered pipeline degrades to the forward
//! path with a warning, so scenes stay loadable when a plugin is absent.

/// Stable registry name of the engine's built-in forward pipeline — the
/// default path for every camera without an explicit [`RenderPath`].
pub const FORWARD_PIPELINE: &str = "forward";

/// Serializable choice of the render pipeline that produces this camera's
/// frame — see the module docs. Attach next to a [`Camera`](crate::Camera);
/// a camera without one renders with the [`FORWARD_PIPELINE`].
#[derive(Debug, Clone, PartialEq, Eq, crate::Component)]
pub struct RenderPath {
    /// Registry name of the pipeline (see
    /// [`PipelineRegistry`](crate::std::rendering::PipelineRegistry)).
    pub pipeline: String,
}

impl Default for RenderPath {
    fn default() -> Self {
        Self::forward()
    }
}

impl RenderPath {
    /// The engine's built-in forward path (the default).
    pub fn forward() -> Self {
        Self {
            pipeline: FORWARD_PIPELINE.to_owned(),
        }
    }

    /// A pipeline by registry name.
    pub fn named(pipeline: impl Into<String>) -> Self {
        Self {
            pipeline: pipeline.into(),
        }
    }
}
