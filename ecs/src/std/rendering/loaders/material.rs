//! The material (template / surface) asset. A material picks a shading model and
//! provides property *values* for its schema — a settings-agnostic surface, not a
//! shader recipe (`docs/MATERIAL_ASSETS.md` Decision 2). Its data lives in the DB
//! record's `settings` (RON); the file is empty. It resolves to a graphics
//! `Material` pipeline (loader added in a later step).

use redlilium_assets::{
    AnyAsset, AssetError, AssetLoader, AssetSource, AssetStage, Executor, Guid, LoadEnv,
    StageFuture,
};

use crate::std::rendering::shading::PropValue;

/// Identity of a material asset: a record resolved from `guid` via the DB.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MaterialSource {
    pub guid: Guid,
}

impl AssetSource for MaterialSource {
    fn file_guid(&self) -> Option<Guid> {
        Some(self.guid)
    }
}

/// A material's authored data (stored in the DB record `settings` as RON): the
/// shading model it uses and its property values. Settings-agnostic surface.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MaterialData {
    /// The shading model id (looked up in the `ShadingRegistry`).
    pub shading_model: String,
    /// Property values by name; any omitted slot falls back to the model default.
    pub properties: Vec<(String, PropValue)>,
}

/// Loads a [`MaterialData`] from its DB record's `settings` (the file is empty —
/// a material's data is all in the record, like a vertex layout). The resident
/// `MaterialData` is resolved to a shading model + shader by the
/// [`MaterialAssetManager`](super::super::MaterialAssetManager).
pub struct MaterialLoader;

impl AssetLoader for MaterialLoader {
    const NAME: &'static str = "material";
    const EXTENSIONS: &'static [&'static str] = &["material"];
    type Source = MaterialSource;
    type Asset = MaterialData;
    type Deps = ();

    fn pipeline(_source: &MaterialSource, _deps: &(), env: &LoadEnv) -> Vec<Box<dyn AssetStage>> {
        vec![Box::new(MaterialFromSettingsStage {
            settings: env.settings.clone(),
        })]
    }
}

/// CPU stage: deserialize the material from the record's settings (RON).
struct MaterialFromSettingsStage {
    settings: Option<String>,
}

impl AssetStage for MaterialFromSettingsStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let settings = self.settings.clone();
        Box::pin(async move {
            let text = settings.ok_or_else(|| {
                AssetError::Decode("material: no parameters in the DB record".into())
            })?;
            let data: MaterialData = ron::from_str(&text)
                .map_err(|e| AssetError::Decode(format!("material: ron: {e}")))?;
            Ok(Box::new(data) as AnyAsset)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_data_ron_roundtrip() {
        let data = MaterialData {
            shading_model: "opaque".to_owned(),
            properties: vec![(
                "base_color".to_owned(),
                PropValue::Vec4([1.0, 0.5, 0.2, 1.0]),
            )],
        };
        let ron = ron::to_string(&data).expect("material -> ron");
        let back: MaterialData = ron::from_str(&ron).expect("ron -> material");
        assert_eq!(data, back);
    }
}
