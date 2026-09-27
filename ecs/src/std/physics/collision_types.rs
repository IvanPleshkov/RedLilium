//! Dimension-independent selection of rigid-body type pairs.

/// Collision detection is allowed if either collider enables the body's type pair.
/// Group masks must still allow the interaction on both sides.
/// Both position- and velocity-driven kinematics use the kinematic flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CollisionTypes {
    pub dynamic_dynamic: bool,
    pub dynamic_kinematic: bool,
    pub dynamic_fixed: bool,
    pub kinematic_kinematic: bool,
    pub kinematic_fixed: bool,
    pub fixed_fixed: bool,
}

impl CollisionTypes {
    pub(super) fn restricts_dynamic_pairs(self) -> bool {
        !(self.dynamic_dynamic && self.dynamic_kinematic && self.dynamic_fixed)
    }

    /// Enables detection for every type pair, including fixed/fixed sensors.
    pub const fn all() -> Self {
        Self {
            dynamic_dynamic: true,
            dynamic_kinematic: true,
            dynamic_fixed: true,
            kinematic_kinematic: true,
            kinematic_fixed: true,
            fixed_fixed: true,
        }
    }

    /// Requests no pairs from this collider. The other collider can still enable them.
    /// Use collision groups to unconditionally reject an interaction on one side.
    pub const fn none() -> Self {
        Self {
            dynamic_dynamic: false,
            dynamic_kinematic: false,
            dynamic_fixed: false,
            kinematic_kinematic: false,
            kinematic_fixed: false,
            fixed_fixed: false,
        }
    }
}

impl Default for CollisionTypes {
    /// Rapier's defaults: enable all pairs involving a dynamic body.
    fn default() -> Self {
        Self {
            dynamic_dynamic: true,
            dynamic_kinematic: true,
            dynamic_fixed: true,
            ..Self::none()
        }
    }
}

// Rapier 0.36's CCD sweep filters groups and hooks but omits active collision
// types. Only colliders restricting dynamic pairs enable this hook; default
// settings pay no extra per-pair callback cost.
pub(super) struct CollisionTypeHooks;

macro_rules! rapier_conversion {
    ($rapier:ident) => {
        impl super::$rapier::prelude::PhysicsHooks for CollisionTypeHooks {
            fn filter_contact_pair(
                &self,
                ctx: &super::$rapier::prelude::PairFilterContext<'_>,
            ) -> Option<super::$rapier::prelude::SolverFlags> {
                use super::$rapier::prelude::{RigidBodyType, SolverFlags};
                let kind = |h| {
                    ctx.bodies
                        .get(h)
                        .map_or(RigidBodyType::Fixed, |b| b.body_type())
                };
                let a = ctx.rigid_body1.map_or(RigidBodyType::Fixed, kind);
                let b = ctx.rigid_body2.map_or(RigidBodyType::Fixed, kind);
                (ctx.colliders[ctx.collider1]
                    .active_collision_types()
                    .test(a, b)
                    || ctx.colliders[ctx.collider2]
                        .active_collision_types()
                        .test(a, b))
                .then(SolverFlags::default)
            }
        }

        impl From<CollisionTypes> for super::$rapier::prelude::ActiveCollisionTypes {
            fn from(value: CollisionTypes) -> Self {
                let mut types = Self::empty();
                for (enabled, flag) in [
                    (value.dynamic_dynamic, Self::DYNAMIC_DYNAMIC),
                    (value.dynamic_kinematic, Self::DYNAMIC_KINEMATIC),
                    (value.dynamic_fixed, Self::DYNAMIC_FIXED),
                    (value.kinematic_kinematic, Self::KINEMATIC_KINEMATIC),
                    (value.kinematic_fixed, Self::KINEMATIC_FIXED),
                    (value.fixed_fixed, Self::FIXED_FIXED),
                ] {
                    types.set(flag, enabled);
                }
                types
            }
        }
    };
}
#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
rapier_conversion!(rapier2d);
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
rapier_conversion!(rapier3d);

impl crate::ComponentField for Option<CollisionTypes> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.push_id(name, |ui| {
            let mut custom = self.is_some();
            let mut changed = ui.checkbox(&mut custom, name)
                .on_hover_text("Unchecked uses the default dynamic-body pairs. Either collider may enable a pair; group masks still apply.")
                .changed();
            let mut types = self.unwrap_or_default();
            if custom {
                for (label, value) in [
                    ("Dynamic / Dynamic", &mut types.dynamic_dynamic),
                    ("Dynamic / Kinematic", &mut types.dynamic_kinematic),
                    ("Dynamic / Fixed", &mut types.dynamic_fixed),
                    ("Kinematic / Kinematic", &mut types.kinematic_kinematic),
                    ("Kinematic / Fixed", &mut types.kinematic_fixed),
                    ("Fixed / Fixed", &mut types.fixed_fixed),
                ] {
                    changed |= ui.checkbox(value, label).changed();
                }
            }
            changed.then(|| custom.then_some(types))
        }).inner
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
