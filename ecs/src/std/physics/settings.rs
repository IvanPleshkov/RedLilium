//! Shared optional physics settings and their inspector/serialization support.

/// Enables sensor behavior. Reserved fields can be added without changing the builder API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SensorSettings {}

/// Enables collision tracking for pairs involving this collider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CollisionEventSettings {}

/// Enables Rapier's extended ("bullet") CCD for a dynamic body.
///
/// In Rapier 0.36, fast dynamic bodies already sweep against fixed colliders.
/// This setting additionally sweeps against kinematic and non-bullet dynamic
/// bodies. Two dynamic bodies with this setting do not sweep against one another.
/// The setting is retained on other body types but only acts on dynamic bodies.
/// Global `IntegrationParameters::max_ccd_substeps = 0` disables all CCD.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CcdSettings {}

// Keep the inspector's enable/disable control while storing the full optional
// configuration. Future settings fields can extend these widgets in place.
macro_rules! optional_settings_field {
    ($settings:ty $(, $tooltip:literal)?) => {
        impl crate::ComponentField for Option<$settings> {
            fn inspect_field(
                &self,
                name: &str,
                ui: &mut egui::Ui,
                _: &crate::FieldInspectCtx<'_>,
            ) -> Option<Self> {
                let mut enabled = self.is_some();
                let response = ui.checkbox(&mut enabled, name);
                $(let response = response.on_hover_text($tooltip);)?
                response.changed().then(|| {
                    if enabled {
                        Some(self.unwrap_or_default())
                    } else {
                        None
                    }
                })
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
    };
}
optional_settings_field!(SensorSettings);
optional_settings_field!(CollisionEventSettings);
optional_settings_field!(
    CcdSettings,
    "Extended CCD for dynamic bodies against non-bullet moving bodies. Automatic CCD against fixed colliders remains active when unchecked."
);

/// Requests normal contact-force events. `None` on a collider disables reporting.
/// The finite, nonnegative threshold applies to the mean force over a full physics
/// step (sum of normal impulse magnitudes / step duration), strictly greater than
/// `min_force`. Sensors never produce force events. Friction is not included.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContactForceSettings {
    pub min_force: f32,
}

impl crate::ComponentField for Option<ContactForceSettings> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.push_id(name, |ui| {
            let mut enabled = self.is_some();
            let mut changed = ui.checkbox(&mut enabled, name).changed();
            let mut settings = self.unwrap_or_default();
            if enabled {
                ui.horizontal(|ui| {
                    ui.label("Minimum force");
                    changed |= ui
                        .add(egui::DragValue::new(&mut settings.min_force).range(0.0..=f32::MAX))
                        .changed();
                });
            }
            changed.then(|| enabled.then_some(settings))
        })
        .inner
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
