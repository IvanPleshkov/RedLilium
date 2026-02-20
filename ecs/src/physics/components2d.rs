//! 2D physics descriptor components.
//!
//! Components that describe rigid body, collider, and joint properties for 2D physics.
//! Use the [`SyncPhysicsBodies2D`](super::physics2d::SyncPhysicsBodies2D) and
//! [`SyncPhysicsJoints2D`](super::physics2d::SyncPhysicsJoints2D) systems
//! to automatically materialize these descriptors into rapier physics objects.

use redlilium_core::math::Vec2;

/// 2D collider shape.
#[derive(Debug, Clone, PartialEq)]
pub enum ColliderShape2D {
    /// Circle defined by radius.
    Ball { radius: f32 },
    /// Rectangle defined by half extents along each axis.
    Cuboid { half_extents: Vec2 },
    /// Capsule (Y-axis) defined by half height and radius.
    CapsuleY { half_height: f32, radius: f32 },
}

/// 2D rigid body type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
/// Attach this component to an entity along with [`Collider2D`] and
/// [`Transform`](crate::Transform), then call [`build_physics_world_2d`]
/// to create the corresponding rapier physics objects.
#[derive(Debug, Clone, PartialEq, crate::Component)]
#[skip_serialization]
pub struct RigidBody2D {
    /// Body type.
    pub body_type: RigidBodyType2D,
    /// Linear velocity damping.
    pub linear_damping: f32,
    /// Angular velocity damping.
    pub angular_damping: f32,
    /// Gravity multiplier (1.0 = normal, 0.0 = no gravity).
    pub gravity_scale: f32,
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
        }
    }
}

/// Describes a 2D collider's shape and material properties.
#[derive(Debug, Clone, PartialEq, crate::Component)]
#[skip_serialization]
pub struct Collider2D {
    /// Collider shape.
    pub shape: ColliderShape2D,
    /// Friction coefficient.
    pub friction: f32,
    /// Restitution (bounciness, 0.0–1.0).
    pub restitution: f32,
    /// Mass density.
    pub density: f32,
    /// Whether this is a sensor/trigger (no contact forces).
    pub is_sensor: bool,
}

impl Collider2D {
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

    pub fn with_sensor(mut self, v: bool) -> Self {
        self.is_sensor = v;
        self
    }
}

impl Default for Collider2D {
    fn default() -> Self {
        Self {
            shape: ColliderShape2D::Ball { radius: 0.5 },
            friction: 0.5,
            restitution: 0.0,
            density: 1.0,
            is_sensor: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Joint descriptor
// ---------------------------------------------------------------------------

/// 2D joint type descriptor.
#[derive(Debug, Clone, PartialEq)]
pub enum JointType2D {
    /// Hinge joint (rotation around Z axis). In 2D the rotation axis is implicit.
    Revolute { anchor1: Vec2, anchor2: Vec2 },
    /// Rigid attachment (no relative movement).
    Fixed { anchor1: Vec2, anchor2: Vec2 },
    /// Sliding joint along an axis.
    Prismatic {
        axis: Vec2,
        anchor1: Vec2,
        anchor2: Vec2,
    },
}

/// Describes a 2D impulse joint between two rigid body entities.
///
/// Attach this component to a (possibly dedicated) entity to create a joint
/// constraint. The `body1` and `body2` fields reference entities that must
/// have [`RigidBody2D`] + [`Collider2D`] components.
///
/// Entity references are automatically remapped during prefab instantiation
/// via the `#[derive(Component)]` macro.
#[derive(Debug, Clone, PartialEq, crate::Component)]
#[skip_serialization]
pub struct ImpulseJoint2D {
    /// First body entity.
    pub body1: crate::Entity,
    /// Second body entity.
    pub body2: crate::Entity,
    /// Joint type and parameters.
    pub joint_type: JointType2D,
}

impl ImpulseJoint2D {
    /// Creates a revolute (hinge) joint.
    pub fn revolute(
        body1: crate::Entity,
        body2: crate::Entity,
        anchor1: Vec2,
        anchor2: Vec2,
    ) -> Self {
        Self {
            body1,
            body2,
            joint_type: JointType2D::Revolute { anchor1, anchor2 },
        }
    }

    /// Creates a fixed (rigid attachment) joint.
    pub fn fixed(body1: crate::Entity, body2: crate::Entity, anchor1: Vec2, anchor2: Vec2) -> Self {
        Self {
            body1,
            body2,
            joint_type: JointType2D::Fixed { anchor1, anchor2 },
        }
    }

    /// Creates a prismatic (slider) joint along the given axis.
    pub fn prismatic(
        body1: crate::Entity,
        body2: crate::Entity,
        axis: Vec2,
        anchor1: Vec2,
        anchor2: Vec2,
    ) -> Self {
        Self {
            body1,
            body2,
            joint_type: JointType2D::Prismatic {
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

use super::physics2d::{PhysicsWorld2D, RigidBody2DHandle};
use super::rapier2d::prelude::*;

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

        builder
            .translation(translation)
            .linear_damping(self.linear_damping as Real)
            .angular_damping(self.angular_damping as Real)
            .gravity_scale(self.gravity_scale as Real)
            .build()
    }
}

impl Collider2D {
    /// Convert this descriptor into a rapier 2D `Collider`.
    pub(crate) fn to_collider(&self) -> Collider {
        use redlilium_core::math::Real;

        let shared = match &self.shape {
            ColliderShape2D::Ball { radius } => SharedShape::ball(*radius as Real),
            ColliderShape2D::Cuboid { half_extents } => {
                SharedShape::cuboid(half_extents.x as Real, half_extents.y as Real)
            }
            ColliderShape2D::CapsuleY {
                half_height,
                radius,
            } => SharedShape::capsule_y(*half_height as Real, *radius as Real),
        };

        ColliderBuilder::new(shared)
            .friction(self.friction as Real)
            .restitution(self.restitution as Real)
            .density(self.density as Real)
            .sensor(self.is_sensor)
            .build()
    }
}

impl ImpulseJoint2D {
    /// Convert this descriptor into a rapier `GenericJoint`.
    pub(crate) fn to_rapier_joint(&self) -> GenericJoint {
        use redlilium_core::math::Real;

        match &self.joint_type {
            JointType2D::Revolute { anchor1, anchor2 } => RevoluteJointBuilder::new()
                .local_anchor1(Vector::new(anchor1.x as Real, anchor1.y as Real))
                .local_anchor2(Vector::new(anchor2.x as Real, anchor2.y as Real))
                .into(),
            JointType2D::Fixed { anchor1, anchor2 } => FixedJointBuilder::new()
                .local_anchor1(Vector::new(anchor1.x as Real, anchor1.y as Real))
                .local_anchor2(Vector::new(anchor2.x as Real, anchor2.y as Real))
                .into(),
            JointType2D::Prismatic {
                axis,
                anchor1,
                anchor2,
            } => PrismaticJointBuilder::new(Vector::new(axis.x as Real, axis.y as Real))
                .local_anchor1(Vector::new(anchor1.x as Real, anchor1.y as Real))
                .local_anchor2(Vector::new(anchor2.x as Real, anchor2.y as Real))
                .into(),
        }
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
#[deprecated(
    note = "Use `SyncPhysicsBodies2D` exclusive system instead, which automatically tracks spawns and despawns."
)]
pub fn build_physics_world_2d(world: &mut crate::World) {
    // Phase 1: collect entity data (clone non-Copy components, copy the rest)
    let entities: Vec<_> = world
        .iter_entities()
        .filter_map(|entity| {
            let body = world.get::<RigidBody2D>(entity)?.clone();
            let collider = world.get::<Collider2D>(entity)?.clone();
            let transform = *world.get::<crate::Transform>(entity)?;
            Some((entity, body, collider, transform))
        })
        .collect();

    // Phase 2: build rapier world
    let mut physics = PhysicsWorld2D::default();
    let mut handle_pairs = Vec::with_capacity(entities.len());

    for (entity, body_desc, collider_desc, transform) in &entities {
        let rapier_body = body_desc.to_rigid_body(transform);
        let body_handle = physics.add_body(rapier_body);

        let rapier_collider = collider_desc.to_collider();
        physics.add_collider(rapier_collider, body_handle);

        handle_pairs.push((*entity, body_handle));
    }

    // Phase 3: insert resource and handles
    world.insert_resource(physics);
    for (entity, handle) in handle_pairs {
        let _ = world.insert(entity, RigidBody2DHandle(handle));
    }
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
            .with_sensor(true);
        assert_eq!(c.friction, 0.8);
        assert_eq!(c.restitution, 0.3);
        assert_eq!(c.density, 2.0);
        assert!(c.is_sensor);
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

        build_physics_world_2d(&mut world);

        assert!(world.get::<RigidBody2DHandle>(e).is_some());
        assert!(world.get::<RigidBody2DHandle>(g).is_some());

        let physics = world.resource::<PhysicsWorld2D>();
        assert_eq!(physics.bodies.len(), 2);
        assert_eq!(physics.colliders.len(), 2);
    }
}
