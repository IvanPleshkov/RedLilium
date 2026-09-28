//! Shared single-axis joint actuation settings.

/// Inclusive coordinate limits: radians for hinges, length units for sliders.
/// Hinge limits must fit in [-π, π] with width less than a full turn.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointLimits {
    pub min: f32,
    pub max: f32,
}

/// Acceleration-based gains give a less mass-dependent response; force-based
/// gains describe physical spring stiffness and damping. Both obey max_effort.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum JointMotorModel {
    #[default]
    AccelerationBased,
    ForceBased,
}

/// Targets are frame 2 relative to frame 1 along the positive joint axis.
/// Angular position is a relative angle, not an accumulated revolution count.
/// Position drives choose the shortest rotation. For a long move inside wide
/// limits, use intermediate targets; the motor does not plan around limit stops.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum JointDrive {
    Velocity {
        velocity: f32,
        damping: f32,
    },
    Position {
        position: f32,
        velocity: f32,
        stiffness: f32,
        damping: f32,
    },
}

/// Optional actuation of the joint's free axis. None on the descriptor disables it.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointMotor {
    pub drive: JointDrive,
    pub model: JointMotorModel,
    /// Force for sliders, torque for hinges; None is uncapped, zero applies no effort.
    pub max_effort: Option<f32>,
}
impl JointMotor {
    pub fn velocity(velocity: f32, damping: f32) -> Self {
        Self {
            drive: JointDrive::Velocity { velocity, damping },
            model: Default::default(),
            max_effort: None,
        }
    }
    pub fn position(position: f32, stiffness: f32, damping: f32) -> Self {
        Self {
            drive: JointDrive::Position {
                position,
                velocity: 0.0,
                stiffness,
                damping,
            },
            model: Default::default(),
            max_effort: None,
        }
    }
    pub fn with_model(mut self, model: JointMotorModel) -> Self {
        self.model = model;
        self
    }
    pub fn with_max_effort(mut self, max_effort: Option<f32>) -> Self {
        self.max_effort = max_effort;
        self
    }
    pub(super) fn set_position(&mut self, value: f32) -> Result<(), JointMotorError> {
        if !value.is_finite() {
            return Err(JointMotorError::Invalid("motor position must be finite"));
        }
        match &mut self.drive {
            JointDrive::Position { position, .. } => {
                *position = value;
                Ok(())
            }
            _ => Err(JointMotorError::RequiresPositionDrive),
        }
    }
    pub(super) fn set_velocity(&mut self, value: f32) -> Result<(), JointMotorError> {
        if !value.is_finite() {
            return Err(JointMotorError::Invalid("motor velocity must be finite"));
        }
        match &mut self.drive {
            JointDrive::Position { velocity, .. } | JointDrive::Velocity { velocity, .. } => {
                *velocity = value
            }
        }
        Ok(())
    }
}

/// A rejected target edit leaves the descriptor unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointMotorError {
    Disabled,
    RequiresPositionDrive,
    Invalid(&'static str),
}
impl std::fmt::Display for JointMotorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Disabled => "joint motor is disabled",
            Self::RequiresPositionDrive => "position target requires a position drive",
            Self::Invalid(message) => message,
        })
    }
}
impl std::error::Error for JointMotorError {}

pub(super) fn validate_settings(
    supported: bool,
    angular: bool,
    limits: Option<JointLimits>,
    motor: Option<JointMotor>,
) -> Result<(), JointMotorError> {
    use JointMotorError::Invalid;
    let pi = std::f32::consts::PI;
    if !supported && (limits.is_some() || motor.is_some()) {
        return Err(Invalid(
            "limits and motors require a revolute or prismatic joint",
        ));
    }
    if let Some(limits) = limits {
        if !limits.min.is_finite() || !limits.max.is_finite() || limits.min > limits.max {
            return Err(Invalid("joint limits must be finite and ordered"));
        }
        if angular
            && (limits.min < -pi
                || limits.max > pi
                || (limits.max as f64 - limits.min as f64) >= 2.0 * pi as f64)
        {
            return Err(Invalid(
                "angular limits must fit in [-pi, pi] with width less than a full turn",
            ));
        }
    }
    if let Some(motor) = motor {
        if motor.max_effort.is_some_and(|v| !v.is_finite() || v < 0.0) {
            return Err(Invalid("motor max_effort must be finite and nonnegative"));
        }
        let (velocity, damping) = match motor.drive {
            JointDrive::Velocity { velocity, damping } => {
                if damping <= 0.0 {
                    return Err(Invalid("velocity motor damping must be positive"));
                }
                (velocity, damping)
            }
            JointDrive::Position {
                position,
                velocity,
                stiffness,
                damping,
            } => {
                if !position.is_finite() || !stiffness.is_finite() || stiffness <= 0.0 {
                    return Err(Invalid(
                        "motor position must be finite and stiffness finite and positive",
                    ));
                }
                if angular && !(-pi..=pi).contains(&position) {
                    return Err(Invalid("angular motor position must fit in [-pi, pi]"));
                }
                if limits.is_some_and(|l| position < l.min || position > l.max) {
                    return Err(Invalid("motor position must be inside joint limits"));
                }
                (velocity, damping)
            }
        };
        if !velocity.is_finite() || !damping.is_finite() || damping < 0.0 {
            return Err(Invalid(
                "motor velocity must be finite and damping finite and nonnegative",
            ));
        }
    }
    Ok(())
}

// Keep dimensional implementations identical. Update only axis parameters;
// unrelated constraints and their warm-start impulses remain intact.
macro_rules! native_parameters {
    ($module:ident, $rapier:ident, $world:ident, $physics:ident, $components:ident, $descriptor:ident) => {
        pub(super) mod $module {
            use super::super::$rapier::prelude::{
                GenericJoint, ImpulseJointHandle, JointAxis, MotorModel, Real,
            };
            use super::{JointDrive, JointLimits, JointMotor, JointMotorModel};
            pub(in crate::std::physics) fn apply(
                joint: &mut GenericJoint,
                axis: JointAxis,
                limits: Option<JointLimits>,
                motor: Option<JointMotor>,
            ) {
                let i = axis as usize;
                if let Some(l) = limits {
                    let old = joint.limits[i];
                    if !joint.limit_axes.contains(axis.into())
                        || old.min != l.min as Real
                        || old.max != l.max as Real
                    {
                        joint.limits[i].impulse = 0.0;
                    }
                    joint.set_limits(axis, [l.min as Real, l.max as Real]);
                } else {
                    joint.limit_axes.remove(axis.into());
                    joint.limits[i] = Default::default();
                }
                if let Some(m) = motor {
                    let model = match m.model {
                        JointMotorModel::AccelerationBased => MotorModel::AccelerationBased,
                        JointMotorModel::ForceBased => MotorModel::ForceBased,
                    };
                    let (p, v, k, d) = match m.drive {
                        JointDrive::Velocity { velocity, damping } => (0.0, velocity, 0.0, damping),
                        JointDrive::Position {
                            position,
                            velocity,
                            stiffness,
                            damping,
                        } => (position, velocity, stiffness, damping),
                    };
                    let old = joint.motors[i];
                    let effort = m.max_effort.map_or(Real::MAX, |v| v as Real);
                    if !joint.motor_axes.contains(axis.into())
                        || old.model != model
                        || old.stiffness != k as Real
                        || old.damping != d as Real
                        || old.max_force != effort
                    {
                        joint.motors[i].impulse = 0.0;
                    }
                    joint.set_motor(axis, p as Real, v as Real, k as Real, d as Real);
                    joint.set_motor_model(axis, model);
                    joint.set_motor_max_force(axis, effort);
                } else {
                    joint.motor_axes.remove(axis.into());
                    joint.motors[i] = Default::default();
                }
            }
            impl super::super::$world::$physics {
                /// Returns true for unchanged or parameter-only edits; false requires rebuild.
                pub(in crate::std::physics) fn update_joint_parameters(
                    &mut self,
                    handle: ImpulseJointHandle,
                    descriptor: &super::super::$components::$descriptor,
                ) -> bool {
                    let Some(previous) = self.applied_joints.get(&handle) else {
                        return false;
                    };
                    if previous == descriptor {
                        return true;
                    }
                    if !previous.same_structure(descriptor) {
                        return false;
                    }
                    let Some(joint) = self.impulse_joints.get_mut(handle, true) else {
                        return false;
                    };
                    descriptor.apply_parameters(&mut joint.data);
                    // Wake immediately as well as notifying Rapier's next step.
                    for handle in [joint.body1(), joint.body2()] {
                        if let Some(body) = self.bodies.get_mut(handle) {
                            body.wake_up(true);
                        }
                    }
                    self.applied_joints.insert(handle, descriptor.clone());
                    true
                }
            }
        }
    };
}
#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
native_parameters!(
    dim2,
    rapier2d,
    world2d,
    PhysicsWorld2D,
    components2d,
    ImpulseJoint2D
);
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
native_parameters!(
    dim3,
    rapier3d,
    world3d,
    PhysicsWorld3D,
    components3d,
    ImpulseJoint3D
);

macro_rules! serde_field {
    () => {
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
    };
}
impl crate::ComponentField for Option<JointLimits> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.push_id(name, |ui| {
            let mut enabled = self.is_some();
            let mut changed = ui.checkbox(&mut enabled, name).changed();
            let mut value = self.unwrap_or(JointLimits { min: 0.0, max: 1.0 });
            if enabled {
                if let Some(v) = value.min.inspect_field("Min", ui, ctx) {
                    value.min = v;
                    changed = true;
                }
                if let Some(v) = value.max.inspect_field("Max", ui, ctx) {
                    value.max = v;
                    changed = true;
                }
            }
            changed.then(|| enabled.then_some(value))
        })
        .inner
    }
    serde_field!();
}
impl crate::ComponentField for Option<JointMotor> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.push_id(name, |ui| {
            let mut enabled = self.is_some();
            let mut changed = ui.checkbox(&mut enabled, name).changed();
            let mut value = self.unwrap_or_else(|| JointMotor::position(0.0, 30.0, 8.0));
            if enabled {
                let mut position_mode = matches!(value.drive, JointDrive::Position { .. });
                if ui.checkbox(&mut position_mode, "Position drive").changed() {
                    value.drive = if position_mode {
                        JointMotor::position(0.0, 30.0, 8.0).drive
                    } else {
                        JointMotor::velocity(0.0, 1.0).drive
                    };
                    changed = true;
                }
                let mut edit = |name: &str, v: &mut f32| {
                    if let Some(new) = v.inspect_field(name, ui, ctx) {
                        *v = new;
                        changed = true;
                    }
                };
                match &mut value.drive {
                    JointDrive::Velocity { velocity, damping } => {
                        edit("Velocity", velocity);
                        edit("Damping", damping);
                    }
                    JointDrive::Position {
                        position,
                        velocity,
                        stiffness,
                        damping,
                    } => {
                        edit("Position", position);
                        edit("Velocity", velocity);
                        edit("Stiffness", stiffness);
                        edit("Damping", damping);
                    }
                }
                egui::ComboBox::from_id_salt("motor_model")
                    .selected_text(format!("{:?}", value.model))
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(
                                &mut value.model,
                                JointMotorModel::AccelerationBased,
                                "Acceleration based",
                            )
                            .changed();
                        changed |= ui
                            .selectable_value(
                                &mut value.model,
                                JointMotorModel::ForceBased,
                                "Force based",
                            )
                            .changed();
                    });
                let mut limited = value.max_effort.is_some();
                if ui.checkbox(&mut limited, "Limit effort").changed() {
                    value.max_effort = limited.then_some(1.0);
                    changed = true;
                }
                if let Some(effort) = value.max_effort {
                    if let Some(v) = effort.inspect_field("Max force / torque", ui, ctx) {
                        value.max_effort = Some(v);
                        changed = true;
                    }
                }
            }
            changed.then(|| enabled.then_some(value))
        })
        .inner
    }
    serde_field!();
}

macro_rules! frame_fields {
    ($components:ident, $frame:ident, $kind:ident, [$($variant:ident),+], $rotation:expr) => {
        impl crate::ComponentField for super::$components::$frame {
            fn inspect_field(&self, name: &str, ui: &mut egui::Ui, ctx: &crate::FieldInspectCtx<'_>) -> Option<Self> {
                ui.push_id(name, |ui| {
                    ui.label(name);
                    let mut value = *self;
                    let mut changed = false;
                    if let Some(v) = value.translation.inspect_field("Translation", ui, ctx) { value.translation = v; changed = true; }
                    changed |= ($rotation)(&mut value, ui, ctx);
                    changed.then_some(value)
                }).inner
            }
            serde_field!();
        }
        impl crate::ComponentField for super::$components::$kind {
            fn inspect_field(&self, name: &str, ui: &mut egui::Ui, _: &crate::FieldInspectCtx<'_>) -> Option<Self> {
                let mut value = *self;
                egui::ComboBox::from_id_salt(name).selected_text(format!("{value:?}")).show_ui(ui, |ui| {
                    $(ui.selectable_value(&mut value, Self::$variant, stringify!($variant));)+
                });
                (value != *self).then_some(value)
            }
            serde_field!();
        }
    };
}
#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
frame_fields!(
    components2d,
    JointFrame2D,
    JointType2D,
    [Fixed, Revolute, Prismatic],
    |v: &mut super::components2d::JointFrame2D,
     ui: &mut egui::Ui,
     ctx: &crate::FieldInspectCtx<'_>| {
        if let Some(r) = v.rotation.inspect_field("Rotation (radians)", ui, ctx) {
            v.rotation = r;
            true
        } else {
            false
        }
    }
);
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
frame_fields!(
    components3d,
    JointFrame3D,
    JointType3D,
    [Fixed, Spherical, Revolute, Prismatic],
    |v: &mut super::components3d::JointFrame3D,
     ui: &mut egui::Ui,
     _: &crate::FieldInspectCtx<'_>| {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Rotation XYZW");
            for v in v.rotation.coords.iter_mut() {
                changed |= ui.add(egui::DragValue::new(v).speed(0.01)).changed();
            }
        });
        changed
    }
);
