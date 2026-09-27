//! Shared collision-event configuration and lifecycle vocabulary.

/// Enables sensor behavior. Reserved fields can be added without changing the builder API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SensorSettings {}

/// Enables collision tracking for pairs involving this collider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CollisionEventSettings {}

/// A transition of an observed contact/intersection pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionPhase {
    /// Tracking began, possibly because events were enabled inside an existing contact.
    Started,
    Stopped(CollisionStopReason),
}

/// Why tracking of a pair ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionStopReason {
    Separated,
    /// The current collision groups no longer allow the pair.
    FilteredOut,
    Removed,
    TrackingDisabled,
    /// At least one participant changed its sensor role.
    Reconfigured,
}

// Keep the inspector's enable/disable control while storing the full optional
// configuration. Future settings fields can extend these widgets in place.
macro_rules! optional_settings_field {
    ($settings:ty) => {
        impl crate::ComponentField for Option<$settings> {
            fn inspect_field(
                &self,
                name: &str,
                ui: &mut egui::Ui,
                _: &crate::FieldInspectCtx<'_>,
            ) -> Option<Self> {
                let mut enabled = self.is_some();
                ui.checkbox(&mut enabled, name).changed().then(|| {
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
