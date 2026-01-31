//! ECS components for the RedLilium engine.
//!
//! This module contains all the core components used in the engine's ECS architecture.
//! Components are organized by functionality:
//!
//! - [`transform`]: Position, rotation, and scale components
//! - [`material`]: Surface appearance and rendering properties
//! - [`render_mesh`]: Mesh geometry references
//! - [`collision`]: Physics collision shapes and rigid bodies
//! - [`hierarchy`]: Parent-child entity relationships
//! - [`camera`]: Camera viewpoints and render targets

mod camera;
mod collision;
mod hierarchy;
mod material;
mod render_mesh;
mod transform;

// Re-export all components at the module level for convenience
pub use camera::{Camera, CameraProjection, CameraViewport, RenderTarget};
pub use collision::{
    Collider, ColliderShape, CollisionLayer, CompoundChild, RigidBody, RigidBodyType,
};
pub use hierarchy::{
    ChildOf, Children, HierarchyDepth, HierarchyRoot, PreviousParent, TransformDirty,
};
pub use material::{AlphaMode, Material, TextureHandle};
pub use render_mesh::{Aabb, MeshHandle, RenderLayers, RenderMesh};
pub use transform::{GlobalTransform, Transform};
