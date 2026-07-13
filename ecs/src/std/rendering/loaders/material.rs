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

impl From<Guid> for MaterialSource {
    fn from(guid: Guid) -> Self {
        Self { guid }
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
    /// Feature-axis selections for the shader's `//#pragma variant` axes
    /// (#6, Decision 5's material half). Unset axes take their pragma
    /// defaults; unknown names or system axes fail the material resolve
    /// loudly. Old records without the field parse as "all defaults".
    #[serde(default)]
    pub features: Vec<(String, super::super::shading::FeatureValue)>,
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

    /// A material with feature selections round-trips through RON, and an old
    /// record without the field parses as "all defaults".
    #[test]
    fn material_features_ron_roundtrip_and_default() {
        use crate::std::rendering::shading::FeatureValue;
        let data = MaterialData {
            shading_model: "opaque_textured".to_owned(),
            properties: Vec::new(),
            features: vec![("ALPHA_CUTOUT".to_owned(), FeatureValue::Bool(true))],
        };
        let ron = ron::to_string(&data).expect("material -> ron");
        let back: MaterialData = ron::from_str(&ron).expect("ron -> material");
        assert_eq!(data, back);

        // Pre-features record: field absent → empty (all pragma defaults).
        let old: MaterialData =
            ron::from_str("(shading_model:\"opaque\",properties:[])").expect("old record parses");
        assert!(old.features.is_empty());
    }

    /// The live axis: the real std shader declares ALPHA_CUTOUT, and the
    /// material half of the variant key builds against it — on, off, and
    /// typo'd (loud error naming the axis).
    #[test]
    fn opaque_textured_declares_alpha_cutout() {
        use redlilium_graphics::{ShaderVariantSpace, VariantValue};
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../std-assets/shaders/opaque_textured.slang"
        ))
        .expect("std shader readable");
        let space = ShaderVariantSpace::parse(&source).expect("pragmas parse");

        let on = space
            .build_features(&[("ALPHA_CUTOUT".into(), VariantValue::Bool(true))])
            .unwrap();
        assert_eq!(on.to_string(), "[ALPHA_CUTOUT]");

        // Default (a record without features): the empty key — the same baked
        // permutation as before the axis existed.
        let off = space.build_features(&[]).unwrap();
        assert!(off.is_empty());

        assert!(
            space
                .build_features(&[("ALPHA_CUTOUP".into(), VariantValue::Bool(true))])
                .is_err()
        );
    }

    #[test]
    fn material_data_ron_roundtrip() {
        let data = MaterialData {
            shading_model: "opaque".to_owned(),
            features: Vec::new(),
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
