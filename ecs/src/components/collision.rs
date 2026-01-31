//! Collision components for physics and spatial queries.

use bevy_ecs::component::Component;
use glam::{Quat, Vec3};

/// Collision shape component for physics and spatial queries.
///
/// Defines the collision geometry for an entity. The shape is in local space
/// relative to the entity's [`Transform`].
///
/// # Example
///
/// ```
/// use redlilium_ecs::components::{Collider, ColliderShape};
///
/// // Create a sphere collider
/// let collider = Collider::new(ColliderShape::Sphere { radius: 1.0 });
///
/// // Create a box collider
/// let box_collider = Collider::new(ColliderShape::Box {
///     half_extents: glam::Vec3::new(1.0, 0.5, 2.0),
/// });
/// ```
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Collider {
    /// The collision shape.
    pub shape: ColliderShape,

    /// Local offset from entity transform.
    pub offset: Vec3,

    /// Local rotation offset from entity transform.
    pub rotation_offset: Quat,

    /// Collision layer this collider belongs to.
    pub collision_layer: CollisionLayer,

    /// Collision mask determining which layers this collider interacts with.
    pub collision_mask: CollisionLayer,

    /// Whether this collider is a trigger (non-solid, generates events only).
    pub is_trigger: bool,

    /// Whether this collider is enabled.
    pub enabled: bool,
}

impl Collider {
    /// Creates a new collider with the given shape.
    #[inline]
    pub fn new(shape: ColliderShape) -> Self {
        Self {
            shape,
            offset: Vec3::ZERO,
            rotation_offset: Quat::IDENTITY,
            collision_layer: CollisionLayer::DEFAULT,
            collision_mask: CollisionLayer::ALL,
            is_trigger: false,
            enabled: true,
        }
    }

    /// Creates a sphere collider.
    #[inline]
    pub fn sphere(radius: f32) -> Self {
        Self::new(ColliderShape::Sphere { radius })
    }

    /// Creates a box collider with the given half-extents.
    #[inline]
    pub fn cuboid(half_extents: Vec3) -> Self {
        Self::new(ColliderShape::Box { half_extents })
    }

    /// Creates a capsule collider.
    #[inline]
    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self::new(ColliderShape::Capsule {
            radius,
            half_height,
        })
    }

    /// Returns this collider with a local offset.
    #[inline]
    #[must_use]
    pub fn with_offset(mut self, offset: Vec3) -> Self {
        self.offset = offset;
        self
    }

    /// Returns this collider with a rotation offset.
    #[inline]
    #[must_use]
    pub fn with_rotation_offset(mut self, rotation: Quat) -> Self {
        self.rotation_offset = rotation;
        self
    }

    /// Returns this collider with the specified collision layer.
    #[inline]
    #[must_use]
    pub fn with_collision_layer(mut self, layer: CollisionLayer) -> Self {
        self.collision_layer = layer;
        self
    }

    /// Returns this collider with the specified collision mask.
    #[inline]
    #[must_use]
    pub fn with_collision_mask(mut self, mask: CollisionLayer) -> Self {
        self.collision_mask = mask;
        self
    }

    /// Returns this collider as a trigger.
    #[inline]
    #[must_use]
    pub fn as_trigger(mut self) -> Self {
        self.is_trigger = true;
        self
    }

    /// Checks if this collider can interact with another based on layers.
    #[inline]
    pub fn can_interact_with(&self, other: &Collider) -> bool {
        self.enabled
            && other.enabled
            && self.collision_mask.intersects(&other.collision_layer)
            && other.collision_mask.intersects(&self.collision_layer)
    }
}

/// Primitive collision shapes.
#[derive(Debug, Clone, PartialEq)]
pub enum ColliderShape {
    /// Sphere defined by radius.
    Sphere { radius: f32 },

    /// Axis-aligned box defined by half-extents.
    Box { half_extents: Vec3 },

    /// Capsule (cylinder with hemispherical caps) along the Y axis.
    Capsule { radius: f32, half_height: f32 },

    /// Cylinder along the Y axis.
    Cylinder { radius: f32, half_height: f32 },

    /// Infinite plane defined by normal (pointing up) and offset from origin.
    Plane { normal: Vec3, offset: f32 },

    /// Convex hull from a set of points.
    ConvexHull { points: Vec<Vec3> },

    /// Triangle mesh for complex static geometry.
    TriMesh { mesh_handle: u64 },

    /// Height field for terrain.
    HeightField {
        heights: Vec<f32>,
        rows: u32,
        columns: u32,
        scale: Vec3,
    },

    /// Compound shape composed of multiple child shapes.
    Compound { children: Vec<CompoundChild> },
}

/// A child shape within a compound collider.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundChild {
    /// The child shape.
    pub shape: Box<ColliderShape>,
    /// Local position offset.
    pub position: Vec3,
    /// Local rotation offset.
    pub rotation: Quat,
}

impl CompoundChild {
    /// Creates a new compound child with the given shape and transform.
    pub fn new(shape: ColliderShape, position: Vec3, rotation: Quat) -> Self {
        Self {
            shape: Box::new(shape),
            position,
            rotation,
        }
    }
}

/// Collision layer mask for filtering collision interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollisionLayer(pub u32);

impl CollisionLayer {
    /// Default collision layer (layer 0).
    pub const DEFAULT: Self = Self(1);

    /// All layers enabled.
    pub const ALL: Self = Self(u32::MAX);

    /// No layers enabled.
    pub const NONE: Self = Self(0);

    /// Creates a collision layer with a single layer enabled.
    #[inline]
    pub const fn layer(layer: u8) -> Self {
        Self(1 << (layer as u32 & 31))
    }

    /// Creates a collision layer from a bitmask.
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the underlying bitmask.
    #[inline]
    pub const fn bits(&self) -> u32 {
        self.0
    }

    /// Adds a layer to this mask.
    #[inline]
    #[must_use]
    pub const fn with_layer(self, layer: u8) -> Self {
        Self(self.0 | (1 << (layer as u32 & 31)))
    }

    /// Removes a layer from this mask.
    #[inline]
    #[must_use]
    pub const fn without_layer(self, layer: u8) -> Self {
        Self(self.0 & !(1 << (layer as u32 & 31)))
    }

    /// Checks if this mask contains a specific layer.
    #[inline]
    pub const fn contains_layer(&self, layer: u8) -> bool {
        (self.0 & (1 << (layer as u32 & 31))) != 0
    }

    /// Checks if this mask intersects with another (shares at least one layer).
    #[inline]
    pub const fn intersects(&self, other: &CollisionLayer) -> bool {
        (self.0 & other.0) != 0
    }
}

impl Default for CollisionLayer {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Rigid body component for physics simulation.
///
/// Determines how the entity participates in physics simulation.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct RigidBody {
    /// The type of rigid body.
    pub body_type: RigidBodyType,

    /// Linear velocity in world space.
    pub linear_velocity: Vec3,

    /// Angular velocity in world space (axis-angle, magnitude is radians/second).
    pub angular_velocity: Vec3,

    /// Mass of the body in kilograms.
    pub mass: f32,

    /// Linear damping factor (0 = no damping, 1 = full damping per second).
    pub linear_damping: f32,

    /// Angular damping factor.
    pub angular_damping: f32,

    /// Gravity scale (0 = no gravity, 1 = normal gravity).
    pub gravity_scale: f32,

    /// Whether continuous collision detection is enabled.
    pub ccd_enabled: bool,

    /// Whether the body is currently sleeping (inactive).
    pub sleeping: bool,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            body_type: RigidBodyType::Dynamic,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            ccd_enabled: false,
            sleeping: false,
        }
    }
}

impl RigidBody {
    /// Creates a new dynamic rigid body.
    #[inline]
    pub fn dynamic() -> Self {
        Self::default()
    }

    /// Creates a new kinematic rigid body.
    #[inline]
    pub fn kinematic() -> Self {
        Self {
            body_type: RigidBodyType::Kinematic,
            ..Default::default()
        }
    }

    /// Creates a new static rigid body.
    #[inline]
    pub fn fixed() -> Self {
        Self {
            body_type: RigidBodyType::Static,
            ..Default::default()
        }
    }

    /// Returns this rigid body with the specified mass.
    #[inline]
    #[must_use]
    pub fn with_mass(mut self, mass: f32) -> Self {
        self.mass = mass.max(0.0);
        self
    }

    /// Returns this rigid body with the specified linear damping.
    #[inline]
    #[must_use]
    pub fn with_linear_damping(mut self, damping: f32) -> Self {
        self.linear_damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Returns this rigid body with the specified angular damping.
    #[inline]
    #[must_use]
    pub fn with_angular_damping(mut self, damping: f32) -> Self {
        self.angular_damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Returns this rigid body with the specified gravity scale.
    #[inline]
    #[must_use]
    pub fn with_gravity_scale(mut self, scale: f32) -> Self {
        self.gravity_scale = scale;
        self
    }

    /// Returns this rigid body with CCD enabled.
    #[inline]
    #[must_use]
    pub fn with_ccd(mut self, enabled: bool) -> Self {
        self.ccd_enabled = enabled;
        self
    }
}

/// Type of rigid body determining physics behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RigidBodyType {
    /// Fully simulated body affected by forces and collisions.
    #[default]
    Dynamic,

    /// Body moved by code, affects dynamic bodies but not affected by them.
    Kinematic,

    /// Immovable body, affects dynamic bodies but never moves.
    Static,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collider_sphere() {
        let collider = Collider::sphere(1.0);
        assert!(matches!(collider.shape, ColliderShape::Sphere { radius } if radius == 1.0));
    }

    #[test]
    fn collider_interaction() {
        let a = Collider::sphere(1.0).with_collision_layer(CollisionLayer::layer(0));
        let b = Collider::sphere(1.0).with_collision_layer(CollisionLayer::layer(1));

        // Default mask is ALL, so they can interact
        assert!(a.can_interact_with(&b));

        // Restrict mask
        let c = Collider::sphere(1.0)
            .with_collision_layer(CollisionLayer::layer(0))
            .with_collision_mask(CollisionLayer::layer(0));

        assert!(!c.can_interact_with(&b));
    }

    #[test]
    fn rigid_body_types() {
        let dynamic = RigidBody::dynamic();
        assert_eq!(dynamic.body_type, RigidBodyType::Dynamic);

        let kinematic = RigidBody::kinematic();
        assert_eq!(kinematic.body_type, RigidBodyType::Kinematic);

        let fixed = RigidBody::fixed();
        assert_eq!(fixed.body_type, RigidBodyType::Static);
    }
}
