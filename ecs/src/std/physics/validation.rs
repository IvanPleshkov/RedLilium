use crate::{Entity, SystemError, Transform};

pub(super) fn invalid(entity: Entity, reason: &str) -> SystemError {
    SystemError::InvalidConfiguration {
        message: format!("physics entity {entity:?}: {reason}"),
    }
}

pub(super) fn transform(
    entity: Entity,
    t: &Transform,
    has_parent: bool,
) -> Result<(), SystemError> {
    if has_parent {
        return Err(invalid(
            entity,
            "rigid bodies must be roots (Parent is not supported)",
        ));
    }
    if t.scale != redlilium_core::math::Vec3::repeat(1.0) {
        return Err(invalid(
            entity,
            "rigid bodies require unit scale; put scaled visuals on a child",
        ));
    }
    if !t
        .translation
        .iter()
        .chain(t.rotation.coords.iter())
        .all(|x| x.is_finite())
        || !t.rotation.norm_squared().is_finite()
        || t.rotation.norm_squared() < 1e-12
    {
        return Err(invalid(
            entity,
            "pose must be finite with a nonzero quaternion",
        ));
    }
    Ok(())
}

fn nonnegative(entity: Entity, field: &str, value: f32) -> Result<(), SystemError> {
    if !value.is_finite() || value < 0.0 {
        return Err(invalid(
            entity,
            &format!("{field} must be finite and nonnegative"),
        ));
    }
    Ok(())
}
fn positive(entity: Entity, field: &str, value: f32) -> Result<(), SystemError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(
            entity,
            &format!("{field} must be finite and positive"),
        ));
    }
    Ok(())
}
fn body(entity: Entity, linear: f32, angular: f32, gravity: f32) -> Result<(), SystemError> {
    nonnegative(entity, "linear_damping", linear)?;
    nonnegative(entity, "angular_damping", angular)?;
    if !gravity.is_finite() {
        return Err(invalid(entity, "gravity_scale must be finite"));
    }
    Ok(())
}
fn material(
    entity: Entity,
    friction: f32,
    restitution: f32,
    density: f32,
) -> Result<(), SystemError> {
    nonnegative(entity, "friction", friction)?;
    nonnegative(entity, "density", density)?;
    if !restitution.is_finite() || !(0.0..=1.0).contains(&restitution) {
        return Err(invalid(
            entity,
            "restitution must be finite and between 0 and 1",
        ));
    }
    Ok(())
}
fn joint(
    entity: Entity,
    body1: Entity,
    body2: Entity,
    anchor1: &[f32],
    anchor2: &[f32],
) -> Result<(), SystemError> {
    if body1 == body2 {
        return Err(invalid(
            entity,
            "joint endpoints must be different entities",
        ));
    }
    if !anchor1.iter().chain(anchor2).all(|x| x.is_finite()) {
        return Err(invalid(entity, "joint anchors must be finite"));
    }
    Ok(())
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod dim2 {
    use super::super::components2d::*;
    use super::*;

    impl RigidBody2D {
        pub(in crate::std::physics) fn validate(&self, entity: Entity) -> Result<(), SystemError> {
            if let Some(settings) = self.mass_properties {
                settings.validate(entity)?;
            }
            body(
                entity,
                self.linear_damping,
                self.angular_damping,
                self.gravity_scale,
            )
        }
    }
    impl Collider2D {
        pub(in crate::std::physics) fn validate(&self, entity: Entity) -> Result<(), SystemError> {
            material(entity, self.friction, self.restitution, self.density)?;
            if let Some(settings) = self.contact_force_events {
                nonnegative(entity, "contact_force_events.min_force", settings.min_force)?;
            }
            self.local_pose.validate(entity)?;
            self.shape.validate(entity)
        }
    }
    impl ColliderShape2D {
        pub(in crate::std::physics) fn validate(&self, entity: Entity) -> Result<(), SystemError> {
            match self {
                ColliderShape2D::Ball { radius } => positive(entity, "radius", *radius)?,
                ColliderShape2D::Cuboid { half_extents } => {
                    for extent in half_extents.iter() {
                        positive(entity, "half_extents", *extent)?;
                    }
                }
                ColliderShape2D::CapsuleY {
                    half_height,
                    radius,
                } => {
                    nonnegative(entity, "half_height", *half_height)?;
                    positive(entity, "radius", *radius)?;
                }
                ColliderShape2D::Compound { parts } => {
                    if parts.is_empty() {
                        return Err(invalid(entity, "compound must contain at least one part"));
                    }
                    for part in parts {
                        part.local_pose.validate(entity)?;
                        ColliderShape2D::from(part.shape.clone()).validate(entity)?;
                    }
                }
            }
            Ok(())
        }
    }
    impl ImpulseJoint2D {
        pub(in crate::std::physics) fn validate(&self, entity: Entity) -> Result<(), SystemError> {
            joint(
                entity,
                self.body1,
                self.body2,
                self.local_frame1.translation.as_slice(),
                self.local_frame2.translation.as_slice(),
            )?;
            self.local_frame1.validate(entity)?;
            self.local_frame2.validate(entity)?;
            super::super::joints::validate_settings(
                self.axis().is_some(),
                self.joint_type == JointType2D::Revolute,
                self.limits,
                self.motor,
            )
            .map_err(|e| invalid(entity, &e.to_string()))
        }
    }
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod dim3 {
    use super::super::components3d::*;
    use super::*;

    impl RigidBody3D {
        pub(in crate::std::physics) fn validate(&self, entity: Entity) -> Result<(), SystemError> {
            if let Some(settings) = self.mass_properties {
                settings.validate(entity)?;
            }
            body(
                entity,
                self.linear_damping,
                self.angular_damping,
                self.gravity_scale,
            )
        }
    }
    impl Collider3D {
        pub(in crate::std::physics) fn validate(&self, entity: Entity) -> Result<(), SystemError> {
            material(entity, self.friction, self.restitution, self.density)?;
            if let Some(settings) = self.contact_force_events {
                nonnegative(entity, "contact_force_events.min_force", settings.min_force)?;
            }
            self.local_pose.validate(entity)?;
            self.shape.validate(entity)
        }
    }
    impl ColliderShape3D {
        pub(in crate::std::physics) fn validate(&self, entity: Entity) -> Result<(), SystemError> {
            match self {
                ColliderShape3D::Ball { radius } => positive(entity, "radius", *radius)?,
                ColliderShape3D::Cuboid { half_extents } => {
                    for extent in half_extents.iter() {
                        positive(entity, "half_extents", *extent)?;
                    }
                }
                ColliderShape3D::CapsuleY {
                    half_height,
                    radius,
                } => {
                    nonnegative(entity, "half_height", *half_height)?;
                    positive(entity, "radius", *radius)?;
                }
                ColliderShape3D::Cylinder {
                    half_height,
                    radius,
                } => {
                    positive(entity, "half_height", *half_height)?;
                    positive(entity, "radius", *radius)?;
                }
                ColliderShape3D::Compound { parts } => {
                    if parts.is_empty() {
                        return Err(invalid(entity, "compound must contain at least one part"));
                    }
                    for part in parts {
                        part.local_pose.validate(entity)?;
                        ColliderShape3D::from(part.shape.clone()).validate(entity)?;
                    }
                }
            }
            Ok(())
        }
    }
    impl ImpulseJoint3D {
        pub(in crate::std::physics) fn validate(&self, entity: Entity) -> Result<(), SystemError> {
            joint(
                entity,
                self.body1,
                self.body2,
                self.local_frame1.translation.as_slice(),
                self.local_frame2.translation.as_slice(),
            )?;
            self.local_frame1.validate(entity)?;
            self.local_frame2.validate(entity)?;
            super::super::joints::validate_settings(
                self.axis().is_some(),
                self.joint_type == JointType3D::Revolute,
                self.limits,
                self.motor,
            )
            .map_err(|e| invalid(entity, &e.to_string()))
        }
    }
}
