//! 2D physics descriptor components.
//!
//! Components that describe rigid body, collider, and joint properties for 2D physics.
//! Use the [`SyncPhysicsBodies2D`](super::physics2d::SyncPhysicsBodies2D) and
//! [`SyncPhysicsJoints2D`](super::physics2d::SyncPhysicsJoints2D) systems
//! to automatically materialize these descriptors into rapier physics objects.

use super::{
    CcdSettings, CollisionEventSettings, CollisionGroups, CollisionTypes, ContactForceSettings,
    LockedAxes2D, MassSettings2D, SensorSettings,
};
use redlilium_core::math::Vec2;

/// 2D collider shape.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ColliderShape2D {
    /// Circle defined by radius.
    Ball { radius: f32 },
    /// Rectangle defined by half extents along each axis.
    Cuboid { half_extents: Vec2 },
    /// Capsule (Y-axis) defined by half height and radius.
    CapsuleY { half_height: f32, radius: f32 },
    /// One collider with a flat list of primitive parts and shared settings.
    Compound { parts: Vec<ColliderPart2D> },
}

/// Explicit physical owner; absent means the body on the same entity.
/// Independent of Parent/Transform. A missing body leaves the collider inactive.
#[derive(Debug, Clone, Copy, PartialEq, crate::Component)]
pub struct ColliderBody2D {
    pub body: crate::Entity,
}

/// Local pose relative to the body (or to the collider for a compound part).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ColliderPose2D {
    pub translation: redlilium_core::math::Vec2,
    pub rotation: f32,
}
impl Default for ColliderPose2D {
    fn default() -> Self {
        Self {
            translation: redlilium_core::math::Vec2::zeros(),
            rotation: 0.0,
        }
    }
}
impl ColliderPose2D {
    pub(super) fn to_rapier(self) -> Pose {
        super::control2d::PhysicsPose2D {
            translation: self.translation,
            rotation: self.rotation,
        }
        .to_rapier()
    }
    pub(super) fn validate(&self, entity: crate::Entity) -> Result<(), crate::SystemError> {
        super::control2d::PhysicsPose2D {
            translation: self.translation,
            rotation: self.rotation,
        }
        .validate(entity)
    }
}

/// Primitive geometry of a compound part. Compounds cannot nest.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ColliderPrimitive2D {
    /// Circle defined by radius.
    Ball { radius: f32 },
    /// Rectangle defined by half extents along each axis.
    Cuboid { half_extents: Vec2 },
    /// Capsule (Y-axis) defined by half height and radius.
    CapsuleY { half_height: f32, radius: f32 },
}

/// Geometry and pose only; material, groups and events belong to the collider.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ColliderPart2D {
    pub shape: ColliderPrimitive2D,
    pub local_pose: ColliderPose2D,
}
impl ColliderPart2D {
    pub fn ball(radius: f32) -> Self {
        Self {
            shape: ColliderPrimitive2D::Ball { radius },
            local_pose: Default::default(),
        }
    }
    pub fn cuboid(half_extents: redlilium_core::math::Vec2) -> Self {
        Self {
            shape: ColliderPrimitive2D::Cuboid { half_extents },
            local_pose: Default::default(),
        }
    }
    pub fn capsule_y(half_height: f32, radius: f32) -> Self {
        Self {
            shape: ColliderPrimitive2D::CapsuleY {
                half_height,
                radius,
            },
            local_pose: Default::default(),
        }
    }
    pub fn with_local_pose(mut self, pose: ColliderPose2D) -> Self {
        self.local_pose = pose;
        self
    }
}
impl From<ColliderPrimitive2D> for ColliderShape2D {
    fn from(value: ColliderPrimitive2D) -> Self {
        match value {
            ColliderPrimitive2D::Ball { radius } => Self::Ball { radius },
            ColliderPrimitive2D::Cuboid { half_extents } => Self::Cuboid { half_extents },
            ColliderPrimitive2D::CapsuleY {
                half_height,
                radius,
            } => Self::CapsuleY {
                half_height,
                radius,
            },
        }
    }
}

/// 2D rigid body type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RigidBodyType2D {
    /// Affected by forces and gravity.
    #[default]
    Dynamic,
    /// Immovable (infinite mass).
    Fixed,
    /// Moved via position, pushes dynamic bodies.
    KinematicPosition,
    /// Moved via velocity, pushes dynamic bodies.
    KinematicVelocity,
}

/// Describes a 2D rigid body's type and physical properties.
///
/// Attach this component to an entity along with [`Transform`](crate::Transform), then run
/// [`SyncPhysicsBodies2D`](super::physics2d::SyncPhysicsBodies2D).
/// Sync validates settings before creating or updating Rapier objects.
#[derive(Debug, Clone, PartialEq, crate::Component)]
pub struct RigidBody2D {
    /// Body type.
    pub body_type: RigidBodyType2D,
    /// Linear velocity damping.
    pub linear_damping: f32,
    /// Angular velocity damping.
    pub angular_damping: f32,
    /// Gravity multiplier (1.0 = normal, 0.0 = no gravity).
    pub gravity_scale: f32,
    /// Extended CCD for dynamic bodies. None keeps automatic CCD against fixed colliders.
    pub ccd: Option<CcdSettings>,
    /// Optional world-axis locks for dynamic motion. None leaves all axes free.
    pub locked_axes: Option<LockedAxes2D>,
    /// Final mass properties. None derives all properties from colliders.
    pub mass_properties: Option<MassSettings2D>,
}

impl RigidBody2D {
    pub fn dynamic() -> Self {
        Self {
            body_type: RigidBodyType2D::Dynamic,
            ..Self::default()
        }
    }

    pub fn fixed() -> Self {
        Self {
            body_type: RigidBodyType2D::Fixed,
            ..Self::default()
        }
    }

    pub fn kinematic_position() -> Self {
        Self {
            body_type: RigidBodyType2D::KinematicPosition,
            ..Self::default()
        }
    }

    pub fn kinematic_velocity() -> Self {
        Self {
            body_type: RigidBodyType2D::KinematicVelocity,
            ..Self::default()
        }
    }

    pub fn with_linear_damping(mut self, v: f32) -> Self {
        self.linear_damping = v;
        self
    }

    pub fn with_angular_damping(mut self, v: f32) -> Self {
        self.angular_damping = v;
        self
    }

    /// Some enables extended CCD; None restores Rapier's automatic fixed-target CCD.
    pub fn with_ccd(mut self, v: Option<CcdSettings>) -> Self {
        self.ccd = v;
        self
    }

    /// Sets dynamic world-axis constraints; None restores all degrees of freedom.
    pub fn with_locked_axes(mut self, v: Option<LockedAxes2D>) -> Self {
        self.locked_axes = v;
        self
    }

    pub fn with_mass_properties(mut self, value: Option<MassSettings2D>) -> Self {
        self.mass_properties = value;
        self
    }

    pub fn with_gravity_scale(mut self, v: f32) -> Self {
        self.gravity_scale = v;
        self
    }
}

impl Default for RigidBody2D {
    fn default() -> Self {
        Self {
            body_type: RigidBodyType2D::Dynamic,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            ccd: None,
            locked_axes: None,
            mass_properties: None,
        }
    }
}

/// Describes a 2D collider's shape and material properties.
#[derive(Debug, Clone, PartialEq, crate::Component)]
pub struct Collider2D {
    /// Collider shape.
    pub shape: ColliderShape2D,
    /// Pose relative to the owning body, independent of visual Transform.
    pub local_pose: ColliderPose2D,
    /// Friction coefficient.
    pub friction: f32,
    /// Restitution (bounciness, 0.0–1.0).
    pub restitution: f32,
    /// Contact friction combination override. None uses Average.
    pub friction_combine_rule: Option<super::CoefficientCombineRule>,
    /// Contact restitution combination override. None uses Average.
    pub restitution_combine_rule: Option<super::CoefficientCombineRule>,
    /// Mass density.
    pub density: f32,
    /// Whether this is a sensor/trigger (no contact forces).
    pub sensor: Option<SensorSettings>,
    /// Opt-in collision tracking. Register Events<CollisionEvent2D> before stepping.
    pub collision_events: Option<CollisionEventSettings>,
    /// Normal contact forces, independently enabled from collision transitions.
    pub contact_force_events: Option<ContactForceSettings>,
    /// Optional group filtering. None allows all groups; both sides must allow a pair.
    pub collision_groups: Option<CollisionGroups>,
    /// None uses the default dynamic-body pairs. Either collider can enable a pair.
    pub collision_types: Option<CollisionTypes>,
}

impl Collider2D {
    pub fn compound(parts: Vec<ColliderPart2D>) -> Self {
        Self {
            shape: ColliderShape2D::Compound { parts },
            ..Default::default()
        }
    }
    pub fn with_local_pose(mut self, pose: ColliderPose2D) -> Self {
        self.local_pose = pose;
        self
    }

    pub fn ball(radius: f32) -> Self {
        Self {
            shape: ColliderShape2D::Ball { radius },
            ..Self::default()
        }
    }

    pub fn cuboid(hx: f32, hy: f32) -> Self {
        Self {
            shape: ColliderShape2D::Cuboid {
                half_extents: Vec2::new(hx, hy),
            },
            ..Self::default()
        }
    }

    pub fn capsule_y(half_height: f32, radius: f32) -> Self {
        Self {
            shape: ColliderShape2D::CapsuleY {
                half_height,
                radius,
            },
            ..Self::default()
        }
    }

    /// Overrides contact friction combination; None restores Average.
    pub fn with_friction_combine_rule(
        mut self,
        rule: Option<super::CoefficientCombineRule>,
    ) -> Self {
        self.friction_combine_rule = rule;
        self
    }
    /// Overrides contact restitution combination; None restores Average.
    pub fn with_restitution_combine_rule(
        mut self,
        rule: Option<super::CoefficientCombineRule>,
    ) -> Self {
        self.restitution_combine_rule = rule;
        self
    }

    pub fn with_friction(mut self, v: f32) -> Self {
        self.friction = v;
        self
    }

    pub fn with_restitution(mut self, v: f32) -> Self {
        self.restitution = v;
        self
    }

    pub fn with_density(mut self, v: f32) -> Self {
        self.density = v;
        self
    }

    /// Sets group filtering; None restores membership in and interaction with all groups.
    pub fn with_collision_groups(mut self, v: Option<CollisionGroups>) -> Self {
        self.collision_groups = v;
        self
    }

    /// Selects body-type pairs; None restores the default pairs involving dynamic bodies.
    pub fn with_collision_types(mut self, v: Option<CollisionTypes>) -> Self {
        self.collision_types = v;
        self
    }

    /// Some enables sensor behavior; None restores a solid collider.
    pub fn with_sensor(mut self, v: Option<SensorSettings>) -> Self {
        self.sensor = v;
        self
    }

    pub(super) fn active_events(&self) -> super::rapier2d::prelude::ActiveEvents {
        use super::rapier2d::prelude::ActiveEvents;
        let mut flags = ActiveEvents::empty();
        flags.set(
            ActiveEvents::COLLISION_EVENTS,
            self.collision_events.is_some(),
        );
        flags.set(
            ActiveEvents::CONTACT_FORCE_EVENTS,
            self.contact_force_events.is_some(),
        );
        flags
    }

    /// Enables normal contact-force reporting, or disables it with `None`.
    pub fn with_contact_force_events(mut self, v: Option<ContactForceSettings>) -> Self {
        self.contact_force_events = v;
        self
    }

    /// Opts into pair transitions. Register Events<CollisionEvent2D> before stepping.
    pub fn with_collision_events(mut self, v: Option<CollisionEventSettings>) -> Self {
        self.collision_events = v;
        self
    }
}

impl Default for Collider2D {
    fn default() -> Self {
        Self {
            shape: ColliderShape2D::Ball { radius: 0.5 },
            local_pose: Default::default(),
            friction: 0.5,
            restitution: 0.0,
            friction_combine_rule: None,
            restitution_combine_rule: None,
            density: 1.0,
            sensor: None,
            collision_events: None,
            contact_force_events: None,
            collision_groups: None,
            collision_types: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Joint descriptor
// ---------------------------------------------------------------------------

/// Joint constraint kind. Limits and motors support revolute and prismatic only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum JointType2D {
    Revolute,
    Fixed,
    Prismatic,
}

/// Body-local joint frame, relative to the body origin (not its centre of mass).
/// Local X is the slider axis and, in 3D, the hinge axis; 2D hinges rotate around Z.
/// Matching frame orientations define zero angle.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointFrame2D {
    pub translation: Vec2,
    pub rotation: f32,
}
impl Default for JointFrame2D {
    fn default() -> Self {
        Self {
            translation: Vec2::zeros(),
            rotation: 0.0,
        }
    }
}
impl JointFrame2D {
    fn to_rapier(self) -> Pose {
        super::control2d::PhysicsPose2D {
            translation: self.translation,
            rotation: self.rotation,
        }
        .to_rapier()
    }
    pub(super) fn validate(&self, entity: crate::Entity) -> Result<(), crate::SystemError> {
        super::control2d::PhysicsPose2D {
            translation: self.translation,
            rotation: self.rotation,
        }
        .validate(entity)
    }
    /// Builds an X-axis frame. Invalid axes produce an invalid frame rejected by sync.
    pub fn from_axis(translation: Vec2, axis: Vec2) -> Self {
        let n = (axis / axis.amax()).normalize();
        let rotation = if n.iter().all(|v| v.is_finite()) {
            n.y.atan2(n.x)
        } else {
            f32::NAN
        };
        Self {
            translation,
            rotation,
        }
    }
}

/// Editable joint configuration and motor targets. Sync applies parameter edits in place.
/// Entity references are remapped during scene/prefab instantiation.
#[derive(Debug, Clone, PartialEq, crate::Component)]
pub struct ImpulseJoint2D {
    pub body1: crate::Entity,
    pub body2: crate::Entity,
    pub joint_type: JointType2D,
    pub local_frame1: JointFrame2D,
    pub local_frame2: JointFrame2D,
    pub limits: Option<super::JointLimits>,
    pub motor: Option<super::JointMotor>,
}
impl ImpulseJoint2D {
    pub fn new(body1: crate::Entity, body2: crate::Entity, joint_type: JointType2D) -> Self {
        Self {
            body1,
            body2,
            joint_type,
            local_frame1: Default::default(),
            local_frame2: Default::default(),
            limits: None,
            motor: None,
        }
    }
    pub fn revolute(
        body1: crate::Entity,
        body2: crate::Entity,
        anchor1: Vec2,
        anchor2: Vec2,
    ) -> Self {
        Self::new(body1, body2, JointType2D::Revolute).with_local_frames(
            JointFrame2D {
                translation: anchor1,
                ..Default::default()
            },
            JointFrame2D {
                translation: anchor2,
                ..Default::default()
            },
        )
    }
    pub fn fixed(body1: crate::Entity, body2: crate::Entity, anchor1: Vec2, anchor2: Vec2) -> Self {
        Self::new(body1, body2, JointType2D::Fixed).with_local_frames(
            JointFrame2D {
                translation: anchor1,
                ..Default::default()
            },
            JointFrame2D {
                translation: anchor2,
                ..Default::default()
            },
        )
    }
    pub fn prismatic(
        body1: crate::Entity,
        body2: crate::Entity,
        axis: Vec2,
        anchor1: Vec2,
        anchor2: Vec2,
    ) -> Self {
        Self::new(body1, body2, JointType2D::Prismatic).with_local_frames(
            JointFrame2D::from_axis(anchor1, axis),
            JointFrame2D::from_axis(anchor2, axis),
        )
    }
    pub fn with_local_frames(mut self, frame1: JointFrame2D, frame2: JointFrame2D) -> Self {
        self.local_frame1 = frame1;
        self.local_frame2 = frame2;
        self
    }
    pub fn with_limits(mut self, limits: Option<super::JointLimits>) -> Self {
        self.limits = limits;
        self
    }
    pub fn with_motor(mut self, motor: Option<super::JointMotor>) -> Self {
        self.motor = motor;
        self
    }
    /// Updates the ECS target, leaving the old value intact on error. Sync applies it.
    pub fn set_motor_position_target(
        &mut self,
        position: f32,
    ) -> Result<(), super::JointMotorError> {
        let mut motor = self.motor.ok_or(super::JointMotorError::Disabled)?;
        motor.set_position(position)?;
        super::joints::validate_settings(
            self.axis().is_some(),
            self.joint_type == JointType2D::Revolute,
            self.limits,
            Some(motor),
        )?;
        self.motor = Some(motor);
        Ok(())
    }
    /// Updates target velocity in either drive mode. Sync applies it.
    pub fn set_motor_velocity_target(
        &mut self,
        velocity: f32,
    ) -> Result<(), super::JointMotorError> {
        let mut motor = self.motor.ok_or(super::JointMotorError::Disabled)?;
        motor.set_velocity(velocity)?;
        super::joints::validate_settings(
            self.axis().is_some(),
            self.joint_type == JointType2D::Revolute,
            self.limits,
            Some(motor),
        )?;
        self.motor = Some(motor);
        Ok(())
    }
    pub(super) fn same_structure(&self, other: &Self) -> bool {
        self.body1 == other.body1
            && self.body2 == other.body2
            && self.joint_type == other.joint_type
            && self.local_frame1 == other.local_frame1
            && self.local_frame2 == other.local_frame2
    }
    pub(super) fn axis(&self) -> Option<JointAxis> {
        match self.joint_type {
            JointType2D::Revolute => Some(JointAxis::AngX),
            JointType2D::Prismatic => Some(JointAxis::LinX),
            _ => None,
        }
    }
    pub(crate) fn to_rapier_joint(&self) -> GenericJoint {
        let mask = match self.joint_type {
            JointType2D::Revolute => JointAxesMask::LOCKED_REVOLUTE_AXES,
            JointType2D::Fixed => JointAxesMask::LOCKED_FIXED_AXES,
            JointType2D::Prismatic => JointAxesMask::LOCKED_PRISMATIC_AXES,
        };
        let mut result = GenericJoint::new(mask);
        result.local_frame1 = self.local_frame1.to_rapier();
        result.local_frame2 = self.local_frame2.to_rapier();
        self.apply_parameters(&mut result);
        result
    }
    pub(super) fn apply_parameters(&self, joint: &mut GenericJoint) {
        if let Some(axis) = self.axis() {
            super::joints::dim2::apply(joint, axis, self.limits, self.motor);
        }
    }
}

// ---------------------------------------------------------------------------
// Build function — materializes descriptors into rapier objects
// ---------------------------------------------------------------------------

use super::rapier2d::prelude::*;
use super::world2d::PhysicsWorld2D;
#[cfg(test)]
use super::world2d::RigidBody2DHandle;

impl RigidBody2D {
    /// Convert this descriptor + transform into a rapier 2D `RigidBody`.
    pub(crate) fn to_rigid_body(&self, transform: &crate::Transform) -> RigidBody {
        use redlilium_core::math::Real;

        let t = &transform.translation;
        let translation = Vector::new(t.x as Real, t.y as Real);

        let builder = match self.body_type {
            RigidBodyType2D::Fixed => RigidBodyBuilder::fixed(),
            RigidBodyType2D::KinematicPosition => RigidBodyBuilder::kinematic_position_based(),
            RigidBodyType2D::KinematicVelocity => RigidBodyBuilder::kinematic_velocity_based(),
            RigidBodyType2D::Dynamic => RigidBodyBuilder::dynamic(),
        };

        // Project the normalized quaternion onto the XY plane's Z rotation.
        let rotation = super::conversions::quat_to_na(transform.rotation)
            .euler_angles()
            .2;
        builder
            .translation(translation)
            .rotation(rotation)
            .linear_damping(self.linear_damping as Real)
            .angular_damping(self.angular_damping as Real)
            .gravity_scale(self.gravity_scale as Real)
            .ccd_enabled(self.ccd.is_some())
            .locked_axes(self.locked_axes.unwrap_or_default().into())
            .build()
    }
}

impl Collider2D {
    /// Convert this descriptor into a rapier 2D `Collider`.
    pub(crate) fn to_collider(&self) -> Collider {
        use redlilium_core::math::Real;

        let shared = self.shape.to_shared_shape();

        ColliderBuilder::new(shared)
            .position(self.local_pose.to_rapier())
            .friction(self.friction as Real)
            .restitution(self.restitution as Real)
            .friction_combine_rule(self.friction_combine_rule.unwrap_or_default().into())
            .restitution_combine_rule(self.restitution_combine_rule.unwrap_or_default().into())
            .density(self.density as Real)
            .sensor(self.sensor.is_some())
            .collision_groups(self.collision_groups.unwrap_or_default().into())
            .active_collision_types(self.collision_types.unwrap_or_default().into())
            .active_hooks(
                if self
                    .collision_types
                    .unwrap_or_default()
                    .restricts_dynamic_pairs()
                {
                    ActiveHooks::FILTER_CONTACT_PAIRS
                } else {
                    ActiveHooks::empty()
                },
            )
            .active_events(self.active_events())
            .contact_force_event_threshold(
                self.contact_force_events
                    .map_or(0.0, |s| s.min_force as Real),
            )
            .build()
    }
}

/// Materializes [`RigidBody2D`] + [`Collider2D`] descriptors into rapier objects.
///
/// **Deprecated:** Use [`SyncPhysicsBodies2D`](super::physics2d::SyncPhysicsBodies2D)
/// exclusive system instead, which automatically tracks spawns and despawns.
///
/// Creates a [`PhysicsWorld2D`] resource and iterates all entities that have
/// both descriptor components plus a [`Transform`](crate::Transform).
/// For each, builds a rapier rigid body and collider, and inserts a
/// [`RigidBody2DHandle`] component on the entity.
///
/// Call this once after spawning all physics entities in a scene.
/// Returns [`SystemError::InvalidConfiguration`](crate::SystemError::InvalidConfiguration)
/// for invalid descriptors or transforms, before replacing the physics resource.
#[deprecated(
    note = "Use `SyncPhysicsBodies2D` exclusive system instead, which automatically tracks spawns and despawns."
)]
pub fn build_physics_world_2d(world: &mut crate::World) -> Result<(), crate::SystemError> {
    for entity in world
        .iter_entities()
        .filter(|e| !world.is_excluded_from_game(*e))
    {
        if let Some(body) = world.get::<RigidBody2D>(entity) {
            body.validate(entity)?;
            if let Some(t) = world.get::<crate::Transform>(entity) {
                super::validation::transform(
                    entity,
                    t,
                    world.get::<crate::Parent>(entity).is_some(),
                )?;
            }
        }
        if let Some(collider) = world.get::<Collider2D>(entity) {
            collider.validate(entity)?;
        }
    }
    super::systems2d::prepare_mass_updates(
        world,
        &PhysicsWorld2D::default(),
        world.try_read::<RigidBody2D>().as_ref(),
        world.try_read::<Collider2D>().as_ref(),
        world.try_read::<ColliderBody2D>().as_ref(),
        world.try_read::<crate::Transform>().as_ref(),
    )?;
    world.insert_resource(PhysicsWorld2D::default());
    crate::ExclusiveSystem::run(&mut super::systems2d::SyncPhysicsBodies2D, world)
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::Vec3;

    #[test]
    fn rigid_body_2d_constructors() {
        assert_eq!(RigidBody2D::dynamic().body_type, RigidBodyType2D::Dynamic);
        assert_eq!(RigidBody2D::fixed().body_type, RigidBodyType2D::Fixed);
        assert_eq!(
            RigidBody2D::kinematic_position().body_type,
            RigidBodyType2D::KinematicPosition
        );
        assert_eq!(
            RigidBody2D::kinematic_velocity().body_type,
            RigidBodyType2D::KinematicVelocity
        );
    }

    #[test]
    fn collider_2d_constructors() {
        let ball = Collider2D::ball(1.0);
        assert!(matches!(ball.shape, ColliderShape2D::Ball { radius } if radius == 1.0));

        let cuboid = Collider2D::cuboid(1.0, 2.0);
        assert!(matches!(
            cuboid.shape,
            ColliderShape2D::Cuboid { half_extents } if half_extents == Vec2::new(1.0, 2.0)
        ));

        let capsule = Collider2D::capsule_y(0.5, 0.3);
        assert!(matches!(
            capsule.shape,
            ColliderShape2D::CapsuleY { half_height, radius } if half_height == 0.5 && radius == 0.3
        ));
    }

    #[test]
    fn collider_2d_builder_pattern() {
        let c = Collider2D::ball(0.5)
            .with_friction(0.8)
            .with_restitution(0.3)
            .with_density(2.0)
            .with_sensor(Some(SensorSettings::default()));
        assert_eq!(c.friction, 0.8);
        assert_eq!(c.restitution, 0.3);
        assert_eq!(c.density, 2.0);
        assert!(c.sensor.is_some());
    }

    #[test]
    #[allow(deprecated)]
    fn build_physics_world_2d_test() {
        let mut world = crate::World::new();
        world.register_component::<RigidBody2D>();
        world.register_component::<Collider2D>();
        world.register_component::<crate::Transform>();
        world.register_component::<RigidBody2DHandle>();

        let e = world.spawn();
        let _ = world.insert(e, RigidBody2D::dynamic());
        let _ = world.insert(e, Collider2D::ball(0.5));
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        );

        let g = world.spawn();
        let _ = world.insert(g, RigidBody2D::fixed());
        let _ = world.insert(g, Collider2D::cuboid(20.0, 0.1));
        let _ = world.insert(g, crate::Transform::IDENTITY);

        build_physics_world_2d(&mut world).unwrap();

        assert!(world.get::<RigidBody2DHandle>(e).is_some());
        assert!(world.get::<RigidBody2DHandle>(g).is_some());

        let physics = world.resource::<PhysicsWorld2D>();
        assert_eq!(physics.bodies.len(), 2);
        assert_eq!(physics.colliders.len(), 2);
    }
}

impl ColliderShape2D {
    pub(super) fn to_shared_shape(&self) -> SharedShape {
        match self {
            ColliderShape2D::Ball { radius } => SharedShape::ball(*radius as Real),
            ColliderShape2D::Cuboid { half_extents } => {
                SharedShape::cuboid(half_extents.x as Real, half_extents.y as Real)
            }
            ColliderShape2D::CapsuleY {
                half_height,
                radius,
            } => SharedShape::capsule_y(*half_height as Real, *radius as Real),
            ColliderShape2D::Compound { parts } => SharedShape::compound(
                parts
                    .iter()
                    .map(|part| {
                        (
                            part.local_pose.to_rapier(),
                            ColliderShape2D::from(part.shape.clone()).to_shared_shape(),
                        )
                    })
                    .collect(),
            ),
        }
    }
}

impl crate::ComponentField for ColliderPose2D {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.push_id(name, |ui| {
            ui.label(name);
            let mut edited = *self;
            let translation = self.translation.inspect_field("Translation", ui, ctx);
            let rotation = self.rotation.inspect_field("Rotation", ui, ctx);
            if let Some(v) = translation {
                edited.translation = v;
            }
            if let Some(v) = rotation {
                edited.rotation = v;
            }
            (translation.is_some() || rotation.is_some()).then_some(edited)
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

impl ColliderShape2D {
    /// Analytic primitive mass calculation; compounds combine part mass properties
    /// without constructing a temporary geometry/BVH for validation.
    pub(super) fn mass_properties_for_density(&self, density: Real) -> MassProperties {
        if density == 0.0 {
            return MassProperties::default();
        }
        match self {
            Self::Ball { radius } => Ball::new(*radius as Real).mass_properties(density),
            Self::Cuboid { half_extents: v } => {
                Cuboid::new(Vector::new(v.x as Real, v.y as Real)).mass_properties(density)
            }
            Self::CapsuleY {
                half_height,
                radius,
            } => Capsule::new_y(*half_height as Real, *radius as Real).mass_properties(density),
            Self::Compound { parts } => {
                parts.iter().fold(MassProperties::default(), |sum, part| {
                    sum + Self::from(part.shape.clone())
                        .mass_properties_for_density(density)
                        .transform_by(&part.local_pose.to_rapier())
                })
            }
        }
    }
}
