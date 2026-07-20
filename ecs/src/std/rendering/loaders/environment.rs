//! The IBL-environment asset (#145): references the three cubemaps that make up
//! a lighting environment — the diffuse irradiance map, the specular prefilter
//! chain, and the sky background. Like a material instance, its data lives in
//! the DB record `settings` (RON) and the file is empty; the
//! [`EnvironmentManager`](super::super::EnvironmentManager) resolves the guids
//! to resident textures. A [`CameraEnvironment`](super::super::CameraEnvironment)
//! component binds one of these per camera.
//!
//! The BRDF integration LUT is *not* part of an environment — it is the same
//! table for every sky, so the deferred pipeline keeps it as a device-wide
//! resource rather than duplicating it into each environment asset.

use redlilium_assets::{
    AnyAsset, AssetError, AssetLoader, AssetSource, AssetStage, Executor, Guid, LoadEnv,
    StageFuture,
};

/// Identity of an environment asset: a record resolved from `guid` via the DB.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentSource {
    pub guid: Guid,
}

impl AssetSource for EnvironmentSource {
    fn file_guid(&self) -> Option<Guid> {
        Some(self.guid)
    }
}

impl From<Guid> for EnvironmentSource {
    fn from(guid: Guid) -> Self {
        Self { guid }
    }
}

/// An environment's authored data (DB record `settings`, RON): the three
/// cubemap texture assets. A growable struct — future scalar settings
/// (intensity, sky yaw, tint) add fields here without touching consumers, and
/// `#[serde(default)]` keeps older records parseable as it grows.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentData {
    /// Diffuse irradiance cubemap.
    pub irradiance: Guid,
    /// Specular prefiltered-environment cubemap (its mip count sets the
    /// reflection-LOD range — see [`EnvironmentManager`](super::super::EnvironmentManager)).
    pub prefilter: Guid,
    /// Sky background cubemap.
    pub sky: Guid,
}

/// Loads an [`EnvironmentData`] from its DB record's `settings` (empty file),
/// exactly like [`MaterialInstanceLoader`](super::MaterialInstanceLoader). The
/// guids are resolved to resident textures by the
/// [`EnvironmentManager`](super::super::EnvironmentManager).
pub struct EnvironmentLoader;

impl AssetLoader for EnvironmentLoader {
    const NAME: &'static str = "environment";
    const EXTENSIONS: &'static [&'static str] = &["env"];
    type Source = EnvironmentSource;
    type Asset = EnvironmentData;
    type Deps = ();

    fn pipeline(
        _source: &EnvironmentSource,
        _deps: &(),
        env: &LoadEnv,
    ) -> Vec<Box<dyn AssetStage>> {
        vec![Box::new(EnvironmentFromSettingsStage {
            settings: env.settings.clone(),
        })]
    }
}

/// CPU stage: deserialize the environment data from the record's settings (RON).
struct EnvironmentFromSettingsStage {
    settings: Option<String>,
}

impl AssetStage for EnvironmentFromSettingsStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let settings = self.settings.clone();
        Box::pin(async move {
            let text = settings.ok_or_else(|| {
                AssetError::Decode("environment: no parameters in the DB record".into())
            })?;
            let data: EnvironmentData = ron::from_str(&text)
                .map_err(|e| AssetError::Decode(format!("environment: ron: {e}")))?;
            Ok(Box::new(data) as AnyAsset)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_data_ron_roundtrip() {
        let data = EnvironmentData {
            irradiance: Guid::stable("textures/ibl/irradiance_cube.ktx2"),
            prefilter: Guid::stable("textures/ibl/prefilter_cube.ktx2"),
            sky: Guid::stable("textures/ibl/sky_cube.ktx2"),
        };
        let ron = ron::to_string(&data).expect("environment -> ron");
        let back: EnvironmentData = ron::from_str(&ron).expect("ron -> environment");
        assert_eq!(data, back);
    }
}
