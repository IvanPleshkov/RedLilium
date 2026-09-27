//! Physics-owned poses and explicit motion inputs. Transform is presentation for moving bodies.
use super::rapier2d::prelude::*;
use super::world2d::PhysicsWorld2D;
use super::{TeleportVelocity, validation::invalid};
use redlilium_core::math::Vec2;

/// A world-space pose, without visual scale. Rotation is the Z angle in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsPose2D {
    pub translation: Vec2,
    pub rotation: f32,
}
impl Default for PhysicsPose2D {
    fn default() -> Self {
        Self {
            translation: Vec2::zeros(),
            rotation: 0.0,
        }
    }
}
impl PhysicsPose2D {
    pub(super) fn validate(&self, entity: crate::Entity) -> Result<(), crate::SystemError> {
        if !self.translation.iter().all(|x| x.is_finite()) || !(self.rotation.is_finite()) {
            return Err(invalid(
                entity,
                "motion pose must be finite with a valid rotation",
            ));
        }
        Ok(())
    }
    pub(super) fn to_rapier(self) -> Pose {
        Pose::from_parts(
            Vector::new(self.translation.x as Real, self.translation.y as Real),
            Rotation::new(self.rotation as Real),
        )
    }
    pub(super) fn from_rapier(p: &Pose) -> Self {
        Self {
            translation: Vec2::new(p.translation.x as f32, p.translation.y as f32),
            rotation: p.rotation.angle() as f32,
        }
    }
    pub(super) fn from_transform(t: &crate::Transform) -> Self {
        Self {
            translation: Vec2::new(t.translation.x, t.translation.y),
            rotation: super::conversions::quat_to_na(t.rotation).euler_angles().2 as f32,
        }
    }
}

/// Persistent target for a position-based kinematic body, applied before each step.
/// Insert only when the game wants to drive the body; absent input holds its current pose.
#[derive(Debug, Clone, Copy, crate::Component)]
#[skip_serialization]
pub struct KinematicTarget2D {
    pub translation: Vec2,
    pub rotation: f32,
}
impl From<PhysicsPose2D> for KinematicTarget2D {
    fn from(p: PhysicsPose2D) -> Self {
        Self {
            translation: p.translation,
            rotation: p.rotation,
        }
    }
}
impl From<KinematicTarget2D> for PhysicsPose2D {
    fn from(p: KinematicTarget2D) -> Self {
        Self {
            translation: p.translation,
            rotation: p.rotation,
        }
    }
}
/// Persistent world-space velocity for a velocity-based kinematic body.
/// Angular velocity is a scalar in radians per second. Absent input means zero velocity.
#[derive(Debug, Clone, Copy, Default, crate::Component)]
#[skip_serialization]
pub struct KinematicVelocity2D {
    pub linear: Vec2,
    pub angular: f32,
}

impl PhysicsWorld2D {
    /// Read the authoritative simulation pose, without render interpolation.
    pub fn pose(&self, entity: crate::Entity) -> Option<PhysicsPose2D> {
        let handle = self.entity_to_body.get(&entity)?;
        Some(PhysicsPose2D::from_rapier(
            self.bodies.get(*handle)?.position(),
        ))
    }

    /// Queue a teleport for the next StepPhysics2D. Last request for a body wins.
    /// Requires an already synchronized body. The captured handle prevents the
    /// request from following a removed/recreated body. Interpolation resets;
    /// a present kinematic target follows the teleport, and Reset also clears
    /// a present kinematic velocity input. Fixed bodies also update Transform.
    pub fn teleport(
        &mut self,
        entity: crate::Entity,
        pose: PhysicsPose2D,
        velocity: TeleportVelocity,
    ) -> Result<(), crate::SystemError> {
        pose.validate(entity)?;
        let handle = self
            .entity_to_body
            .get(&entity)
            .copied()
            .filter(|h| self.bodies.contains(*h))
            .ok_or_else(|| invalid(entity, "teleport requires a synchronized body"))?;
        self.teleports.insert(entity, (handle, pose, velocity));
        Ok(())
    }
}

impl PhysicsWorld2D {
    pub(super) fn apply_body_settings(
        &mut self,
        handle: RigidBodyHandle,
        desc: &super::components2d::RigidBody2D,
        collider: &super::components2d::Collider2D,
    ) {
        use super::components2d::RigidBodyType2D as Kind;
        let Some((old_body, old_collider, collider_handle)) = self.applied_bodies.get_mut(&handle)
        else {
            return;
        };
        let Some(body) = self.bodies.get_mut(handle) else {
            return;
        };
        if old_body != desc {
            if old_body.body_type != desc.body_type {
                body.set_body_type(
                    match desc.body_type {
                        Kind::Dynamic => RigidBodyType::Dynamic,
                        Kind::Fixed => RigidBodyType::Fixed,
                        Kind::KinematicPosition => RigidBodyType::KinematicPositionBased,
                        Kind::KinematicVelocity => RigidBodyType::KinematicVelocityBased,
                    },
                    true,
                );
                self.pose_resets.insert(handle);
            }
            body.set_linear_damping(desc.linear_damping as Real);
            body.set_angular_damping(desc.angular_damping as Real);
            body.set_gravity_scale(desc.gravity_scale as Real, true);
            body.wake_up(true);
            *old_body = desc.clone();
        }
        if old_collider != collider
            && let Some(live) = self.colliders.get_mut(*collider_handle)
        {
            if old_collider.shape != collider.shape {
                live.set_shape(collider.to_collider().shared_shape().clone());
            }
            live.set_density(collider.density as Real);
            live.set_friction(collider.friction as Real);
            live.set_restitution(collider.restitution as Real);
            live.set_sensor(collider.is_sensor);
            body.wake_up(true);
            *old_collider = collider.clone();
        }
    }
}
