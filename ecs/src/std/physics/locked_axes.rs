//! Axis locks are descriptor-owned. They affect dynamic motion in world axes.

macro_rules! axis_settings {
    ($name:ident, translations: [$($t:ident => $tl:literal),+], rotations: [$($r:ident => $rl:literal),+]) => {
        /// World-axis constraints for dynamic bodies. True means locked.
        /// Kept on fixed/kinematic bodies but does not constrain authored motion.
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        pub struct $name {
            $(pub $t: bool,)+
            $(pub $r: bool,)+
        }
        impl $name {
            /// Locks every degree of freedom.
            pub const fn all() -> Self {
                Self { $($t: true,)+ $($r: true,)+ }
            }
            /// Locks all translation; rotation remains free.
            pub const fn translations() -> Self {
                Self { $($t: true,)+ $($r: false,)+ }
            }
            /// Locks all rotation; translation remains free.
            pub const fn rotations() -> Self {
                Self { $($t: false,)+ $($r: true,)+ }
            }
        }
        impl crate::ComponentField for Option<$name> {
            fn inspect_field(
                &self,
                name: &str,
                ui: &mut egui::Ui,
                _: &crate::FieldInspectCtx<'_>,
            ) -> Option<Self> {
                ui.push_id(name, |ui| {
                    let mut enabled = self.is_some();
                    let mut changed = ui.checkbox(&mut enabled, name)
                        .on_hover_text("World-axis locks for dynamic bodies. Fixed/kinematic motion and teleports remain explicit.")
                        .changed();
                    let mut settings = self.unwrap_or_default();
                    if enabled {
                        $(changed |= ui.checkbox(&mut settings.$t, $tl).changed();)+
                        $(changed |= ui.checkbox(&mut settings.$r, $rl).changed();)+
                    }
                    changed.then(|| enabled.then_some(settings))
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
    };
}
axis_settings!(LockedAxes2D,
    translations: [translation_x => "Translation X", translation_y => "Translation Y"],
    rotations: [rotation => "Rotation"]
);
axis_settings!(LockedAxes3D,
    translations: [translation_x => "Translation X", translation_y => "Translation Y", translation_z => "Translation Z"],
    rotations: [rotation_x => "Rotation X", rotation_y => "Rotation Y", rotation_z => "Rotation Z"]
);

impl LockedAxes3D {
    // Rapier 0.36 applies a full-inertia gyroscopic correction that ignores
    // rotation locks. Disable it for constrained bodies instead of letting it
    // inject forbidden angular velocity during the solver's internal substeps.
    #[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
    pub(super) fn has_rotation_locks(self) -> bool {
        self.rotation_x || self.rotation_y || self.rotation_z
    }
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
pub(super) mod dim2 {
    use super::super::rapier2d::prelude::*;
    impl From<super::LockedAxes2D> for LockedAxes {
        fn from(value: super::LockedAxes2D) -> Self {
            let mut flags = Self::empty();
            flags.set(Self::TRANSLATION_LOCKED_X, value.translation_x);
            flags.set(Self::TRANSLATION_LOCKED_Y, value.translation_y);
            flags.set(Self::ROTATION_LOCKED_Z, value.rotation);
            flags
        }
    }
    pub(in crate::std::physics) fn linear(body: &RigidBody, mut velocity: Vector) -> Vector {
        if body.is_dynamic() {
            let flags = body.locked_axes();
            if flags.contains(LockedAxes::TRANSLATION_LOCKED_X) {
                velocity.x = 0.0;
            }
            if flags.contains(LockedAxes::TRANSLATION_LOCKED_Y) {
                velocity.y = 0.0;
            }
        }
        velocity
    }
    pub(in crate::std::physics) fn angular(body: &RigidBody, mut velocity: Real) -> Real {
        if body.is_dynamic() {
            let flags = body.locked_axes();
            if flags.contains(LockedAxes::ROTATION_LOCKED_Z) {
                velocity = 0.0;
            }
        }
        velocity
    }
    // Native set_locked_axes only changes effective mass/inertia. Discard the
    // pre-existing forbidden velocities as part of applying the descriptor.
    pub(in crate::std::physics) fn clamp_velocity(body: &mut RigidBody) {
        if body.is_dynamic() && !body.locked_axes().is_empty() {
            let linvel = linear(body, body.linvel());
            let angvel = angular(body, body.angvel());
            body.set_linvel(linvel, false);
            body.set_angvel(angvel, false);
        }
    }
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
pub(super) mod dim3 {
    use super::super::rapier3d::prelude::*;
    impl From<super::LockedAxes3D> for LockedAxes {
        fn from(value: super::LockedAxes3D) -> Self {
            let mut flags = Self::empty();
            flags.set(Self::TRANSLATION_LOCKED_X, value.translation_x);
            flags.set(Self::TRANSLATION_LOCKED_Y, value.translation_y);
            flags.set(Self::TRANSLATION_LOCKED_Z, value.translation_z);
            flags.set(Self::ROTATION_LOCKED_X, value.rotation_x);
            flags.set(Self::ROTATION_LOCKED_Y, value.rotation_y);
            flags.set(Self::ROTATION_LOCKED_Z, value.rotation_z);
            flags
        }
    }
    pub(in crate::std::physics) fn linear(body: &RigidBody, mut velocity: Vector) -> Vector {
        if body.is_dynamic() {
            let flags = body.locked_axes();
            if flags.contains(LockedAxes::TRANSLATION_LOCKED_X) {
                velocity.x = 0.0;
            }
            if flags.contains(LockedAxes::TRANSLATION_LOCKED_Y) {
                velocity.y = 0.0;
            }
            if flags.contains(LockedAxes::TRANSLATION_LOCKED_Z) {
                velocity.z = 0.0;
            }
        }
        velocity
    }
    pub(in crate::std::physics) fn angular(body: &RigidBody, mut velocity: Vector) -> Vector {
        if body.is_dynamic() {
            let flags = body.locked_axes();
            if flags.contains(LockedAxes::ROTATION_LOCKED_X) {
                velocity.x = 0.0;
            }
            if flags.contains(LockedAxes::ROTATION_LOCKED_Y) {
                velocity.y = 0.0;
            }
            if flags.contains(LockedAxes::ROTATION_LOCKED_Z) {
                velocity.z = 0.0;
            }
        }
        velocity
    }
    // Native set_locked_axes only changes effective mass/inertia. Discard the
    // pre-existing forbidden velocities as part of applying the descriptor.
    pub(in crate::std::physics) fn clamp_velocity(body: &mut RigidBody) {
        if body.is_dynamic() && !body.locked_axes().is_empty() {
            let linvel = linear(body, body.linvel());
            let angvel = angular(body, body.angvel());
            body.set_linvel(linvel, false);
            body.set_angvel(angvel, false);
        }
    }
}
