//! Per-camera IBL environment selection (#145).

use redlilium_assets::{AssetRef, Guid};

use crate::std::rendering::loaders::EnvironmentSource;

/// Stable guid of the built-in std environment asset (the baked sunrise IBL
/// set). The default [`CameraEnvironment`] references it, so "Add Component"
/// gives a lit sky out of the box.
pub const STD_ENVIRONMENT: &str = "environments/std.env";

/// The IBL environment a camera renders with (#145): an `AssetRef` to an
/// `environment` asset (irradiance + prefilter + sky cubemaps). Read off the
/// camera entity by the deferred pipeline, exactly like
/// [`CameraExposure`](super::CameraExposure) / [`TemporalJitter`](super::TemporalJitter)
/// — so different cameras can render different environments.
///
/// A camera **without** this component (or whose environment has not resolved
/// yet) renders with no skybox and zero image-based ambient — lit by analytic
/// lights only.
#[derive(Debug, Clone, crate::Component)]
pub struct CameraEnvironment {
    /// The environment asset reference (source serialized; resolution runtime).
    pub environment: AssetRef<EnvironmentSource>,
}

impl CameraEnvironment {
    /// Reference the environment asset `guid`.
    pub fn new(guid: Guid) -> Self {
        Self {
            environment: AssetRef::new(EnvironmentSource { guid }),
        }
    }
}

impl Default for CameraEnvironment {
    fn default() -> Self {
        Self::new(Guid::stable(STD_ENVIRONMENT))
    }
}
