//! Backend-independent contact material combination settings.

/// Combines the two contacting colliders' friction or restitution coefficients.
/// When their rules differ, priority is:
/// GeometricMean > ClampedSum > Max > Multiply > Min > Average.
/// Friction and restitution select their rules independently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CoefficientCombineRule {
    /// Arithmetic mean: (a + b) / 2. Also used when the collider override is None.
    #[default]
    Average,
    /// Smaller coefficient.
    Min,
    /// Product of coefficients.
    Multiply,
    /// Larger coefficient.
    Max,
    /// Sum clamped to [0, 1], including for friction.
    ClampedSum,
    /// Square root of the product.
    GeometricMean,
}

macro_rules! native_rule {
    ($rapier:ident) => {
        impl From<CoefficientCombineRule> for super::$rapier::prelude::CoefficientCombineRule {
            fn from(value: CoefficientCombineRule) -> Self {
                match value {
                    CoefficientCombineRule::Average => Self::Average,
                    CoefficientCombineRule::Min => Self::Min,
                    CoefficientCombineRule::Multiply => Self::Multiply,
                    CoefficientCombineRule::Max => Self::Max,
                    CoefficientCombineRule::ClampedSum => Self::ClampedSum,
                    CoefficientCombineRule::GeometricMean => Self::GeometricMean,
                }
            }
        }
    };
}
#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
native_rule!(rapier2d);
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
native_rule!(rapier3d);

impl crate::ComponentField for Option<CoefficientCombineRule> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut value = *self;
        egui::ComboBox::from_label(name)
            .selected_text(
                value.map_or_else(|| "Default (Average)".to_owned(), |r| format!("{r:?}")),
            )
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut value, None, "Default (Average)");
                for rule in [
                    CoefficientCombineRule::Average,
                    CoefficientCombineRule::Min,
                    CoefficientCombineRule::Multiply,
                    CoefficientCombineRule::Max,
                    CoefficientCombineRule::ClampedSum,
                    CoefficientCombineRule::GeometricMean,
                ] {
                    ui.selectable_value(&mut value, Some(rule), format!("{rule:?}"));
                }
            });
        (value != *self).then_some(value)
    }
    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<(), crate::serialize::SerializeError> {
        ctx.write_serde(name, self)
    }
    fn deserialize_field(
        name: &str,
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.read_serde(name)
    }
}
