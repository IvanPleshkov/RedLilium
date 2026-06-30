//! The material-instance asset: references a parent material and overrides some
//! of its property values (`docs/MATERIAL_ASSETS.md` Decision 6 — the
//! Material/MaterialInstance split). Data lives in the DB record `settings`
//! (RON); the file is empty. It resolves to a graphics `MaterialInstance`, and a
//! `Primitive` binds one of these.

use redlilium_assets::{AssetSource, Guid};

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
