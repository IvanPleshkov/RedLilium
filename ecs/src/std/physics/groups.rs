//! Dimension-independent, bilateral collision filtering.

/// Membership and acceptance masks for the 32 user-defined collision groups.
///
/// A pair is allowed when both `(a.memberships & b.filter) != 0` and
/// `(b.memberships & a.filter) != 0`. Zero in either mask rejects every pair.
/// The default belongs to and accepts all groups. Body-type collision rules
/// still apply, and these masks affect both solid contacts and sensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CollisionGroups {
    pub memberships: u32,
    pub filter: u32,
}

impl CollisionGroups {
    pub const fn new(memberships: u32, filter: u32) -> Self {
        Self {
            memberships,
            filter,
        }
    }
}

impl Default for CollisionGroups {
    fn default() -> Self {
        Self::new(u32::MAX, u32::MAX)
    }
}

// Also usable with Rapier's free-collider builders and QueryFilter::groups.
macro_rules! rapier_conversion {
    ($rapier:ident) => {
        impl From<CollisionGroups> for super::$rapier::prelude::InteractionGroups {
            fn from(value: CollisionGroups) -> Self {
                use super::$rapier::prelude::{Group, InteractionTestMode};
                Self::new(
                    Group::from_bits_retain(value.memberships),
                    Group::from_bits_retain(value.filter),
                    InteractionTestMode::And,
                )
            }
        }
    };
}
#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
rapier_conversion!(rapier2d);
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
rapier_conversion!(rapier3d);

impl crate::ComponentField for Option<CollisionGroups> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.push_id(name, |ui| {
            let mut enabled = self.is_some();
            let mut changed = ui.checkbox(&mut enabled, name).changed();
            let mut groups = self.unwrap_or_default();
            if enabled {
                for (label, mask) in [
                    ("Memberships", &mut groups.memberships),
                    ("Filter", &mut groups.filter),
                ] {
                    ui.horizontal(|ui| {
                        ui.label(label);
                        changed |= ui
                            .add(egui::DragValue::new(mask).hexadecimal(8, false, true))
                            .changed();
                    });
                }
            }
            changed.then(|| enabled.then_some(groups))
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
