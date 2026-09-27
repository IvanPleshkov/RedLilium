//! 3D physics descriptor components.
//!
//! Components that describe rigid body, collider, and joint properties.
//! Use the [`SyncPhysicsBodies3D`](super::physics3d::SyncPhysicsBodies3D) and
//! [`SyncPhysicsJoints3D`](super::physics3d::SyncPhysicsJoints3D) systems
//! to automatically materialize these descriptors into rapier physics objects.

use super::{CollisionEventSettings, CollisionGroups, SensorSettings};
use redlilium_core::math::Vec3;

/// 3D collider shape.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ColliderShape3D {
    /// Sphere defined by radius.
    Ball { radius: f32 },
    /// Box defined by half extents along each axis.
    Cuboid { half_extents: Vec3 },
    /// Capsule (Y-axis) defined by half height and radius.
    CapsuleY { half_height: f32, radius: f32 },
    /// Cylinder (Y-axis) defined by half height and radius.
    Cylinder { half_height: f32, radius: f32 },
}

/// Rigid body type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum RigidBodyType {
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

/// Describes a 3D rigid body's type and physical properties.
///
/// Attach this component to an entity along with [`Collider3D`] and
/// [`Transform`](crate::Transform), then run
/// [`SyncPhysicsBodies3D`](super::physics3d::SyncPhysicsBodies3D).
/// Sync validates settings before creating or updating Rapier objects.
// Serialized: scenes/prefabs must carry physics authoring data (#101/#106).
// The rapier handle components stay unserialized — SyncPhysicsBodies3D
// rebuilds live bodies from these descriptors after a restore.
#[derive(Debug, Clone, PartialEq, crate::Component)]
pub struct RigidBody3D {
    /// Body type.
    pub body_type: RigidBodyType,
    /// Linear velocity damping.
    pub linear_damping: f32,
    /// Angular velocity damping.
    pub angular_damping: f32,
    /// Gravity multiplier (1.0 = normal, 0.0 = no gravity).
    pub gravity_scale: f32,
}

impl RigidBody3D {
    pub fn dynamic() -> Self {
        Self {
            body_type: RigidBodyType::Dynamic,
            ..Self::default()
        }
    }

    pub fn fixed() -> Self {
        Self {
            body_type: RigidBodyType::Fixed,
            ..Self::default()
        }
    }

    pub fn kinematic_position() -> Self {
        Self {
            body_type: RigidBodyType::KinematicPosition,
            ..Self::default()
        }
    }

    pub fn kinematic_velocity() -> Self {
        Self {
            body_type: RigidBodyType::KinematicVelocity,
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

    pub fn with_gravity_scale(mut self, v: f32) -> Self {
        self.gravity_scale = v;
        self
    }
}

impl Default for RigidBody3D {
    fn default() -> Self {
        Self {
            body_type: RigidBodyType::Dynamic,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
        }
    }
}

/// Describes a 3D collider's shape and material properties.
// Serialized: authoring data for scenes/prefabs, like RigidBody3D above.
#[derive(Debug, Clone, PartialEq, crate::Component)]
pub struct Collider3D {
    /// Collider shape.
    pub shape: ColliderShape3D,
    /// Friction coefficient.
    pub friction: f32,
    /// Restitution (bounciness, 0.0–1.0).
    pub restitution: f32,
    /// Mass density.
    pub density: f32,
    /// Whether this is a sensor/trigger (no contact forces).
    pub sensor: Option<SensorSettings>,
    /// Opt-in collision tracking. Register Events<CollisionEvent3D> before stepping.
    pub collision_events: Option<CollisionEventSettings>,
    /// Optional group filtering. None allows all groups; both sides must allow a pair.
    pub collision_groups: Option<CollisionGroups>,
}

impl Collider3D {
    pub fn ball(radius: f32) -> Self {
        Self {
            shape: ColliderShape3D::Ball { radius },
            ..Self::default()
        }
    }

    pub fn cuboid(hx: f32, hy: f32, hz: f32) -> Self {
        Self {
            shape: ColliderShape3D::Cuboid {
                half_extents: Vec3::new(hx, hy, hz),
            },
            ..Self::default()
        }
    }

    pub fn capsule_y(half_height: f32, radius: f32) -> Self {
        Self {
            shape: ColliderShape3D::CapsuleY {
                half_height,
                radius,
            },
            ..Self::default()
        }
    }

    pub fn cylinder(half_height: f32, radius: f32) -> Self {
        Self {
            shape: ColliderShape3D::Cylinder {
                half_height,
                radius,
            },
            ..Self::default()
        }
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

    /// Some enables sensor behavior; None restores a solid collider.
    pub fn with_sensor(mut self, v: Option<SensorSettings>) -> Self {
        self.sensor = v;
        self
    }

    /// Opts into pair transitions. Register Events<CollisionEvent3D> before stepping.
    pub fn with_collision_events(mut self, v: Option<CollisionEventSettings>) -> Self {
        self.collision_events = v;
        self
    }
}

impl Default for Collider3D {
    fn default() -> Self {
        Self {
            shape: ColliderShape3D::Ball { radius: 0.5 },
            friction: 0.5,
            restitution: 0.0,
            density: 1.0,
            sensor: None,
            collision_events: None,
            collision_groups: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Joint descriptor
// ---------------------------------------------------------------------------

/// 3D joint type descriptor.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum JointType3D {
    /// Ball-and-socket joint with anchor points on each body.
    Spherical { anchor1: Vec3, anchor2: Vec3 },
    /// Hinge joint around an axis.
    Revolute {
        /// Finite nonzero direction; normalized when applied to Rapier.
        axis: Vec3,
        anchor1: Vec3,
        anchor2: Vec3,
    },
    /// Rigid attachment (no relative movement).
    Fixed { anchor1: Vec3, anchor2: Vec3 },
    /// Sliding joint along an axis.
    Prismatic {
        /// Finite nonzero direction; normalized when applied to Rapier.
        axis: Vec3,
        anchor1: Vec3,
        anchor2: Vec3,
    },
}

/// Describes a 3D impulse joint between two rigid body entities.
///
/// Attach this component to a (possibly dedicated) entity to create a joint
/// constraint. The `body1` and `body2` fields reference entities that must
/// have [`RigidBody3D`] + [`Collider3D`] components.
///
/// Entity references are automatically remapped during prefab instantiation
/// via the `#[derive(Component)]` macro.
///
/// # Example
///
/// ```ignore
/// let joint_entity = world.spawn();
/// world.insert(joint_entity, ImpulseJoint3D::spherical(
///     body_a, body_b,
///     Vec3::new(1.0, 0.0, 0.0),
///     Vec3::new(-1.0, 0.0, 0.0),
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, crate::Component)]
pub struct ImpulseJoint3D {
    /// First body entity.
    pub body1: crate::Entity,
    /// Second body entity.
    pub body2: crate::Entity,
    /// Joint type and parameters.
    pub joint_type: JointType3D,
}

impl ImpulseJoint3D {
    /// Creates a spherical (ball-and-socket) joint.
    pub fn spherical(
        body1: crate::Entity,
        body2: crate::Entity,
        anchor1: Vec3,
        anchor2: Vec3,
    ) -> Self {
        Self {
            body1,
            body2,
            joint_type: JointType3D::Spherical { anchor1, anchor2 },
        }
    }

    /// Creates a revolute (hinge) joint around the given axis.
    pub fn revolute(
        body1: crate::Entity,
        body2: crate::Entity,
        axis: Vec3,
        anchor1: Vec3,
        anchor2: Vec3,
    ) -> Self {
        Self {
            body1,
            body2,
            joint_type: JointType3D::Revolute {
                axis,
                anchor1,
                anchor2,
            },
        }
    }

    /// Creates a fixed (rigid attachment) joint.
    pub fn fixed(body1: crate::Entity, body2: crate::Entity, anchor1: Vec3, anchor2: Vec3) -> Self {
        Self {
            body1,
            body2,
            joint_type: JointType3D::Fixed { anchor1, anchor2 },
        }
    }

    /// Creates a prismatic (slider) joint along the given axis.
    pub fn prismatic(
        body1: crate::Entity,
        body2: crate::Entity,
        axis: Vec3,
        anchor1: Vec3,
        anchor2: Vec3,
    ) -> Self {
        Self {
            body1,
            body2,
            joint_type: JointType3D::Prismatic {
                axis,
                anchor1,
                anchor2,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Build function — materializes descriptors into rapier objects
// ---------------------------------------------------------------------------

use super::rapier3d::prelude::*;
use super::world3d::{PhysicsWorld3D, RigidBody3DHandle};

// The descriptor is validated before conversion; rescaling handles very small
// and very large finite directions without changing their orientation.
fn joint_axis(axis: &redlilium_core::math::Vec3) -> Vector {
    let n = (axis / axis.amax()).normalize();
    Vector::new(n.x as Real, n.y as Real, n.z as Real)
}

impl ImpulseJoint3D {
    /// Convert this descriptor into a rapier `GenericJoint`.
    pub(crate) fn to_rapier_joint(&self) -> GenericJoint {
        use redlilium_core::math::Real;

        match &self.joint_type {
            JointType3D::Spherical { anchor1, anchor2 } => SphericalJointBuilder::new()
                .local_anchor1(Vector::new(
                    anchor1.x as Real,
                    anchor1.y as Real,
                    anchor1.z as Real,
                ))
                .local_anchor2(Vector::new(
                    anchor2.x as Real,
                    anchor2.y as Real,
                    anchor2.z as Real,
                ))
                .into(),
            JointType3D::Revolute {
                axis,
                anchor1,
                anchor2,
            } => RevoluteJointBuilder::new(joint_axis(axis))
                .local_anchor1(Vector::new(
                    anchor1.x as Real,
                    anchor1.y as Real,
                    anchor1.z as Real,
                ))
                .local_anchor2(Vector::new(
                    anchor2.x as Real,
                    anchor2.y as Real,
                    anchor2.z as Real,
                ))
                .into(),
            JointType3D::Fixed { anchor1, anchor2 } => FixedJointBuilder::new()
                .local_anchor1(Vector::new(
                    anchor1.x as Real,
                    anchor1.y as Real,
                    anchor1.z as Real,
                ))
                .local_anchor2(Vector::new(
                    anchor2.x as Real,
                    anchor2.y as Real,
                    anchor2.z as Real,
                ))
                .into(),
            JointType3D::Prismatic {
                axis,
                anchor1,
                anchor2,
            } => PrismaticJointBuilder::new(joint_axis(axis))
                .local_anchor1(Vector::new(
                    anchor1.x as Real,
                    anchor1.y as Real,
                    anchor1.z as Real,
                ))
                .local_anchor2(Vector::new(
                    anchor2.x as Real,
                    anchor2.y as Real,
                    anchor2.z as Real,
                ))
                .into(),
        }
    }
}

impl RigidBody3D {
    /// Convert this descriptor + transform into a rapier `RigidBody`.
    pub(crate) fn to_rigid_body(&self, transform: &crate::Transform) -> RigidBody {
        use super::rapier3d::prelude::Real;
        let pose = super::control3d::PhysicsPose3D::from_transform(transform).to_rapier();

        let builder = match self.body_type {
            RigidBodyType::Fixed => RigidBodyBuilder::fixed(),
            RigidBodyType::KinematicPosition => RigidBodyBuilder::kinematic_position_based(),
            RigidBodyType::KinematicVelocity => RigidBodyBuilder::kinematic_velocity_based(),
            RigidBodyType::Dynamic => RigidBodyBuilder::dynamic(),
        };

        builder
            .pose(pose)
            .linear_damping(self.linear_damping as Real)
            .angular_damping(self.angular_damping as Real)
            .gravity_scale(self.gravity_scale as Real)
            .build()
    }
}

impl Collider3D {
    /// Convert this descriptor into a rapier `Collider`.
    pub(crate) fn to_collider(&self) -> Collider {
        use redlilium_core::math::Real;

        let shared = match &self.shape {
            ColliderShape3D::Ball { radius } => SharedShape::ball(*radius as Real),
            ColliderShape3D::Cuboid { half_extents } => SharedShape::cuboid(
                half_extents.x as Real,
                half_extents.y as Real,
                half_extents.z as Real,
            ),
            ColliderShape3D::CapsuleY {
                half_height,
                radius,
            } => SharedShape::capsule_y(*half_height as Real, *radius as Real),
            ColliderShape3D::Cylinder {
                half_height,
                radius,
            } => SharedShape::cylinder(*half_height as Real, *radius as Real),
        };

        ColliderBuilder::new(shared)
            .friction(self.friction as Real)
            .restitution(self.restitution as Real)
            .density(self.density as Real)
            .sensor(self.sensor.is_some())
            .collision_groups(self.collision_groups.unwrap_or_default().into())
            .active_events(if self.collision_events.is_some() {
                ActiveEvents::COLLISION_EVENTS
            } else {
                ActiveEvents::empty()
            })
            .build()
    }
}

/// Materializes [`RigidBody3D`] + [`Collider3D`] descriptors into rapier objects.
///
/// **Deprecated:** Use [`SyncPhysicsBodies3D`](super::physics3d::SyncPhysicsBodies3D)
/// exclusive system instead, which automatically tracks spawns and despawns.
///
/// Creates a [`PhysicsWorld3D`] resource and iterates all entities that have
/// both descriptor components plus a [`Transform`](crate::Transform).
/// For each, builds a rapier rigid body and collider, and inserts a
/// [`RigidBody3DHandle`] component on the entity.
///
/// Call this once after spawning all physics entities in a scene.
/// Returns [`SystemError::InvalidConfiguration`](crate::SystemError::InvalidConfiguration)
/// for invalid descriptors or transforms, before replacing the physics resource.
///
/// # Example
///
/// ```ignore
/// let e = world.spawn();
/// world.insert(e, RigidBody3D::dynamic());
/// world.insert(e, Collider3D::ball(0.5).with_restitution(0.7));
/// world.insert(e, Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)));
/// world.insert(e, GlobalTransform::IDENTITY);
///
/// build_physics_world_3d(world).unwrap();
/// // Now the entity has a RigidBody3DHandle and a PhysicsWorld3D resource exists.
/// ```
#[deprecated(
    note = "Use `SyncPhysicsBodies3D` exclusive system instead, which automatically tracks spawns and despawns."
)]
pub fn build_physics_world_3d(world: &mut crate::World) -> Result<(), crate::SystemError> {
    // Phase 1: collect entity data (clone non-Copy components, copy the rest)
    let entities: Vec<_> = world
        .iter_entities()
        .filter_map(|entity| {
            let body = world.get::<RigidBody3D>(entity)?.clone();
            let collider = world.get::<Collider3D>(entity)?.clone();
            let transform = *world.get::<crate::Transform>(entity)?;
            Some((entity, body, collider, transform))
        })
        .collect();

    for (entity, body, collider, transform) in &entities {
        body.validate(*entity)?;
        collider.validate(*entity)?;
        super::validation::transform(
            *entity,
            transform,
            world.get::<crate::Parent>(*entity).is_some(),
        )?;
    }

    // Phase 2: build rapier world from descriptors
    let mut physics = PhysicsWorld3D::default();
    let mut handle_pairs = Vec::with_capacity(entities.len());

    for (entity, body_desc, collider_desc, transform) in &entities {
        let rapier_body = body_desc.to_rigid_body(transform);
        let body_handle = physics.add_body(rapier_body);

        let rapier_collider = collider_desc.to_collider();
        let collider_handle = physics.add_collider(rapier_collider, body_handle);
        physics.applied_bodies.insert(
            body_handle,
            (body_desc.clone(), collider_desc.clone(), collider_handle),
        );

        physics.entity_to_body.insert(*entity, body_handle);
        physics.body_to_entity.insert(body_handle, *entity);
        handle_pairs.push((*entity, body_handle));
    }

    // Phase 3: insert resource and handles
    world.insert_resource(physics);
    for (entity, handle) in handle_pairs {
        let _ = world.insert(entity, RigidBody3DHandle(handle));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::Vec3;

    #[test]
    fn rigid_body_constructors() {
        assert_eq!(RigidBody3D::dynamic().body_type, RigidBodyType::Dynamic);
        assert_eq!(RigidBody3D::fixed().body_type, RigidBodyType::Fixed);
        assert_eq!(
            RigidBody3D::kinematic_position().body_type,
            RigidBodyType::KinematicPosition
        );
        assert_eq!(
            RigidBody3D::kinematic_velocity().body_type,
            RigidBodyType::KinematicVelocity
        );
    }

    #[test]
    fn rigid_body_builder_pattern() {
        let rb = RigidBody3D::dynamic()
            .with_linear_damping(0.5)
            .with_angular_damping(0.3)
            .with_gravity_scale(2.0);
        assert_eq!(rb.linear_damping, 0.5);
        assert_eq!(rb.angular_damping, 0.3);
        assert_eq!(rb.gravity_scale, 2.0);
    }

    #[test]
    fn collider_constructors() {
        let ball = Collider3D::ball(1.0);
        assert!(matches!(ball.shape, ColliderShape3D::Ball { radius } if radius == 1.0));

        let cuboid = Collider3D::cuboid(1.0, 2.0, 3.0);
        assert!(matches!(
            cuboid.shape,
            ColliderShape3D::Cuboid { half_extents } if half_extents == Vec3::new(1.0, 2.0, 3.0)
        ));

        let capsule = Collider3D::capsule_y(0.5, 0.3);
        assert!(matches!(
            capsule.shape,
            ColliderShape3D::CapsuleY { half_height, radius } if half_height == 0.5 && radius == 0.3
        ));

        let cyl = Collider3D::cylinder(1.0, 0.5);
        assert!(matches!(
            cyl.shape,
            ColliderShape3D::Cylinder { half_height, radius } if half_height == 1.0 && radius == 0.5
        ));
    }

    #[test]
    fn collider_builder_pattern() {
        let c = Collider3D::ball(0.5)
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
    fn build_physics_world() {
        let mut world = crate::World::new();
        world.register_component::<RigidBody3D>();
        world.register_component::<Collider3D>();
        world.register_component::<crate::Transform>();
        world.register_component::<RigidBody3DHandle>();

        // Spawn a dynamic ball
        let e = world.spawn();
        let _ = world.insert(e, RigidBody3D::dynamic());
        let _ = world.insert(e, Collider3D::ball(0.5).with_restitution(0.7));
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        );

        // Spawn a fixed ground
        let g = world.spawn();
        let _ = world.insert(g, RigidBody3D::fixed());
        let _ = world.insert(g, Collider3D::cuboid(20.0, 0.1, 20.0));
        let _ = world.insert(g, crate::Transform::IDENTITY);

        build_physics_world_3d(&mut world).unwrap();

        // Check that handles were inserted
        assert!(world.get::<RigidBody3DHandle>(e).is_some());
        assert!(world.get::<RigidBody3DHandle>(g).is_some());

        // Check physics world resource
        let physics = world.resource::<PhysicsWorld3D>();
        assert_eq!(physics.bodies.len(), 2);
        assert_eq!(physics.colliders.len(), 2);
    }
}
