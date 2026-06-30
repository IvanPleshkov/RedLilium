//! The material-instance asset: references a parent material and overrides some
//! of its property values (`docs/MATERIAL_ASSETS.md` Decision 6 — the
//! Material/MaterialInstance split). Data lives in the DB record `settings`
//! (RON); the file is empty. It resolves to a graphics `MaterialInstance`, and a
//! `Primitive` binds one of these.

use redlilium_assets::{
    AnyAsset, AssetError, AssetLoader, AssetSource, AssetStage, Executor, Guid, LoadEnv, StageFuture,
};

use crate::std::rendering::shading::PropValue;

/// Identity of a material-instance asset: a record resolved from `guid` via the DB.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MaterialInstanceSource {
    pub guid: Guid,
}

impl AssetSource for MaterialInstanceSource {
    fn file_guid(&self) -> Option<Guid> {
        Some(self.guid)
    }
}

/// A material instance's authored data (DB record `settings`, RON): the parent
/// material asset and the property values it overrides (others inherit).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MaterialInstanceData {
    /// The parent `Material` asset guid.
    pub parent: Guid,
    /// Overridden property values by name; unset slots inherit from the parent.
    pub overrides: Vec<(String, PropValue)>,
}

/// Loads a [`MaterialInstanceData`] from its DB record's `settings` (empty file).
/// The resident data is resolved against its parent material + built into a static
/// property binding by the
/// [`MaterialInstanceManager`](super::super::MaterialInstanceManager).
pub struct MaterialInstanceLoader;

impl AssetLoader for MaterialInstanceLoader {
    const NAME: &'static str = "material_instance";
    const EXTENSIONS: &'static [&'static str] = &["matinst"];
    type Source = MaterialInstanceSource;
    type Asset = MaterialInstanceData;
    type Deps = ();

    fn pipeline(
        _source: &MaterialInstanceSource,
        _deps: &(),
        env: &LoadEnv,
    ) -> Vec<Box<dyn AssetStage>> {
        vec![Box::new(InstanceFromSettingsStage {
            settings: env.settings.clone(),
        })]
    }
}

/// CPU stage: deserialize the instance data from the record's settings (RON).
struct InstanceFromSettingsStage {
    settings: Option<String>,
}

impl AssetStage for InstanceFromSettingsStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let settings = self.settings.clone();
        Box::pin(async move {
            let text = settings.ok_or_else(|| {
                AssetError::Decode("material_instance: no parameters in the DB record".into())
            })?;
            let data: MaterialInstanceData = ron::from_str(&text)
                .map_err(|e| AssetError::Decode(format!("material_instance: ron: {e}")))?;
            Ok(Box::new(data) as AnyAsset)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_data_ron_roundtrip() {
        let data = MaterialInstanceData {
            parent: Guid::stable("materials/opaque_white.material"),
            overrides: vec![(
                "base_color".to_owned(),
                PropValue::Vec4([0.2, 0.7, 0.9, 1.0]),
            )],
        };
        let ron = ron::to_string(&data).expect("instance -> ron");
        let back: MaterialInstanceData = ron::from_str(&ron).expect("ron -> instance");
        assert_eq!(data, back);
    }
}
