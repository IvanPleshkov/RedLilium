//! Physics-owned poses and explicit motion inputs. Transform is presentation for moving bodies.
use super::rapier3d::prelude::*;
use super::world3d::PhysicsWorld3D;
use super::{TeleportVelocity, validation::invalid};
use redlilium_core::math::{Quat, Vec3};

/// A world-space pose, without visual scale. Rotation is a quaternion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsPose3D {
    pub translation: Vec3,
    pub rotation: Quat,
}
impl Default for PhysicsPose3D {
    fn default() -> Self {
        Self {
            translation: Vec3::zeros(),
            rotation: Quat::identity(),
        }
    }
}
impl PhysicsPose3D {
    pub(super) fn validate(&self, entity: crate::Entity) -> Result<(), crate::SystemError> {
        if !self.translation.iter().all(|x| x.is_finite())
            || !(self.rotation.coords.iter().all(|x| x.is_finite())
                && self.rotation.norm_squared().is_finite()
                && self.rotation.norm_squared() >= 1e-12)
        {
            return Err(invalid(
                entity,
                "motion pose must be finite with a valid rotation",
            ));
        }
        Ok(())
    }
    pub(super) fn to_rapier(self) -> Pose {
        let q = self.rotation.normalize();
        Pose::from_parts(
            Vector::new(
                self.translation.x as Real,
                self.translation.y as Real,
                self.translation.z as Real,
            ),
            Rotation::from_xyzw(q.i as Real, q.j as Real, q.k as Real, q.w as Real).normalize(),
        )
    }
    pub(super) fn from_rapier(p: &Pose) -> Self {
        Self {
            translation: Vec3::new(
                p.translation.x as f32,
                p.translation.y as f32,
                p.translation.z as f32,
            ),
            rotation: redlilium_core::math::quat_from_xyzw(
                p.rotation.x as f32,
                p.rotation.y as f32,
                p.rotation.z as f32,
                p.rotation.w as f32,
            ),
        }
    }
    pub(super) fn from_transform(t: &crate::Transform) -> Self {
        Self {
            translation: t.translation,
            rotation: t.rotation,
        }
    }
}

/// Persistent target for a position-based kinematic body, applied before each step.
/// Insert only when the game wants to drive the body; absent input holds its current pose.
#[derive(Debug, Clone, Copy, crate::Component)]
#[skip_serialization]
pub struct KinematicTarget3D {
    pub translation: Vec3,
    pub rotation: Quat,
}
impl From<PhysicsPose3D> for KinematicTarget3D {
    fn from(p: PhysicsPose3D) -> Self {
        Self {
            translation: p.translation,
            rotation: p.rotation,
        }
    }
}
impl From<KinematicTarget3D> for PhysicsPose3D {
    fn from(p: KinematicTarget3D) -> Self {
        Self {
            translation: p.translation,
            rotation: p.rotation,
        }
    }
}
/// Persistent world-space velocity for a velocity-based kinematic body.
/// Angular velocity is a vector in radians per second. Absent input means zero velocity.
#[derive(Debug, Clone, Copy, Default, crate::Component)]
#[skip_serialization]
pub struct KinematicVelocity3D {
    pub linear: Vec3,
    pub angular: Vec3,
}

impl PhysicsWorld3D {
    /// Read the authoritative simulation pose, without render interpolation.
    pub fn pose(&self, entity: crate::Entity) -> Option<PhysicsPose3D> {
        let handle = self.entity_to_body.get(&entity)?;
        Some(PhysicsPose3D::from_rapier(
            self.bodies.get(*handle)?.position(),
        ))
    }

    /// Queue a teleport for the next StepPhysics3D. Last request for a body wins.
    /// Requires an already synchronized body. The captured handle prevents the
    /// request from following a removed/recreated body. Interpolation resets;
    /// a present kinematic target follows the teleport, and Reset also clears
    /// a present kinematic velocity input. Fixed bodies also update Transform.
    pub fn teleport(
        &mut self,
        entity: crate::Entity,
        pose: PhysicsPose3D,
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

impl PhysicsWorld3D {
    pub(super) fn apply_body_settings(
        &mut self,
        handle: RigidBodyHandle,
        desc: &super::components3d::RigidBody3D,
        collider: &super::components3d::Collider3D,
    ) {
        use super::components3d::RigidBodyType as Kind;
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
            live.set_sensor(collider.sensor.is_some());
            live.set_collision_groups(collider.collision_groups.unwrap_or_default().into());
            live.set_active_events(if collider.collision_events.is_some() {
                ActiveEvents::COLLISION_EVENTS
            } else {
                ActiveEvents::empty()
            });
            if old_collider.sensor != collider.sensor
                || old_collider.collision_events != collider.collision_events
                || old_collider.collision_groups != collider.collision_groups
            {
                self.collision_events
                    .collider_changed(*collider_handle, live);
            }
            body.wake_up(true);
            *old_collider = collider.clone();
        }
    }
}
