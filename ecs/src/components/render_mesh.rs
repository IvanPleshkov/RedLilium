//! Render mesh component for entities that should be rendered with geometry.

use bevy_ecs::component::Component;
use glam::Vec3;

/// Component that references a mesh for rendering.
///
/// This is a lightweight handle to mesh data stored elsewhere (e.g., GPU buffers
/// managed by the graphics system). Entities with this component and a [`Transform`]
/// will be rendered by the render system.
///
/// # Example
///
/// ```
/// use redlilium_ecs::components::{RenderMesh, MeshHandle};
///
/// // Reference a mesh by its handle
/// let mesh = RenderMesh::new(MeshHandle::new(42));
/// ```
#[derive(Component, Debug, Clone, PartialEq)]
pub struct RenderMesh {
    /// Handle to the mesh resource.
    pub mesh: MeshHandle,

    /// Axis-aligned bounding box for culling (local space).
    pub bounds: Option<Aabb>,

    /// Whether this mesh casts shadows.
    pub cast_shadows: bool,

    /// Whether this mesh receives shadows.
    pub receive_shadows: bool,

    /// Render layer mask for selective rendering.
    pub render_layers: RenderLayers,
}

impl RenderMesh {
    /// Creates a new render mesh component with the given mesh handle.
    #[inline]
    pub fn new(mesh: MeshHandle) -> Self {
        Self {
            mesh,
            bounds: None,
            cast_shadows: true,
            receive_shadows: true,
            render_layers: RenderLayers::default(),
        }
    }

    /// Returns this render mesh with the specified bounds.
    #[inline]
    #[must_use]
    pub fn with_bounds(mut self, bounds: Aabb) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Returns this render mesh with shadow casting enabled/disabled.
    #[inline]
    #[must_use]
    pub fn with_cast_shadows(mut self, cast_shadows: bool) -> Self {
        self.cast_shadows = cast_shadows;
        self
    }

    /// Returns this render mesh with shadow receiving enabled/disabled.
    #[inline]
    #[must_use]
    pub fn with_receive_shadows(mut self, receive_shadows: bool) -> Self {
        self.receive_shadows = receive_shadows;
        self
    }

    /// Returns this render mesh with the specified render layers.
    #[inline]
    #[must_use]
    pub fn with_render_layers(mut self, layers: RenderLayers) -> Self {
        self.render_layers = layers;
        self
    }
}

/// Handle to a mesh resource.
///
/// This is a lightweight identifier that references mesh data stored elsewhere
/// (e.g., in an asset manager or GPU resource pool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u64);

impl MeshHandle {
    /// Creates a new mesh handle from an ID.
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying ID.
    #[inline]
    pub const fn id(&self) -> u64 {
        self.0
    }

    /// Invalid/null mesh handle.
    pub const INVALID: Self = Self(u64::MAX);
}

impl Default for MeshHandle {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Axis-Aligned Bounding Box for spatial queries and culling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Minimum corner of the bounding box.
    pub min: Vec3,
    /// Maximum corner of the bounding box.
    pub max: Vec3,
}

impl Aabb {
    /// Creates a new AABB from min and max corners.
    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Creates an AABB from center and half-extents.
    #[inline]
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Returns the center point of the AABB.
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Returns the half-extents (half-size) of the AABB.
    #[inline]
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Returns the size (full extents) of the AABB.
    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Checks if a point is inside the AABB.
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Checks if this AABB intersects another AABB.
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Returns the union of this AABB with another.
    #[inline]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Creates a unit cube AABB centered at origin.
    pub const UNIT: Self = Self {
        min: Vec3::new(-0.5, -0.5, -0.5),
        max: Vec3::new(0.5, 0.5, 0.5),
    };
}

impl Default for Aabb {
    fn default() -> Self {
        Self::UNIT
    }
}

/// Render layer mask for selective rendering.
///
/// Entities are only rendered by cameras that share at least one render layer.
/// This allows for effects like split-screen, UI-only cameras, or debug visualization layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderLayers(pub u32);

impl RenderLayers {
    /// Default render layer (layer 0).
    pub const DEFAULT: Self = Self(1);

    /// All layers enabled.
    pub const ALL: Self = Self(u32::MAX);

    /// No layers enabled.
    pub const NONE: Self = Self(0);

    /// Creates a render layers mask with a single layer enabled.
    #[inline]
    pub const fn layer(layer: u8) -> Self {
        Self(1 << (layer as u32 & 31))
    }

    /// Creates a render layers mask from a bitmask.
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
    pub const fn intersects(&self, other: &RenderLayers) -> bool {
        (self.0 & other.0) != 0
    }
}

impl Default for RenderLayers {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_basics() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert_eq!(aabb.center(), Vec3::splat(0.5));
        assert_eq!(aabb.half_extents(), Vec3::splat(0.5));
        assert!(aabb.contains_point(Vec3::splat(0.5)));
        assert!(!aabb.contains_point(Vec3::splat(2.0)));
    }

    #[test]
    fn aabb_intersection() {
        let a = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let b = Aabb::new(Vec3::splat(0.5), Vec3::splat(1.5));
        let c = Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0));

        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn render_layers() {
        let layers = RenderLayers::layer(0).with_layer(2);
        assert!(layers.contains_layer(0));
        assert!(!layers.contains_layer(1));
        assert!(layers.contains_layer(2));

        let other = RenderLayers::layer(2);
        assert!(layers.intersects(&other));
    }
}
