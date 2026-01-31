//! ECS (Entity Component System) module for the RedLilium engine.
//!
//! This crate provides the core ECS components and systems used throughout the engine.
//! It uses [bevy_ecs](https://docs.rs/bevy_ecs) as the underlying ECS framework.
//!
//! # Architecture
//!
//! The crate is organized following ECS principles:
//!
//! - **Components** ([`components`]): Data attached to entities
//! - **Systems** ([`systems`]): Logic that operates on components
//!
//! # Core Components
//!
//! ## Transform
//!
//! - [`Transform`](components::Transform): Local position, rotation, scale relative to parent
//! - [`GlobalTransform`](components::GlobalTransform): Computed world-space transform
//!
//! ## Hierarchy
//!
//! Entity hierarchies are built using Bevy's relationship system:
//!
//! - [`ChildOf`](components::ChildOf): Marks an entity as a child of another
//! - [`Children`](components::Children): Auto-populated list of child entities
//!
//! ## Rendering
//!
//! - [`Material`](components::Material): PBR material properties
//! - [`RenderMesh`](components::RenderMesh): Reference to mesh geometry
//!
//! ## Physics
//!
//! - [`Collider`](components::Collider): Collision shape definition
//! - [`RigidBody`](components::RigidBody): Physics body properties
//!
//! # Example
//!
//! ```
//! use redlilium_ecs::prelude::*;
//! use bevy_ecs::prelude::*;
//! use glam::Vec3;
//!
//! // Create a world
//! let mut world = World::new();
//!
//! // Spawn a root entity
//! let root = world.spawn((
//!     Transform::from_xyz(0.0, 0.0, 0.0),
//!     GlobalTransform::IDENTITY,
//! )).id();
//!
//! // Spawn a child entity
//! let child = world.spawn((
//!     Transform::from_xyz(5.0, 0.0, 0.0),
//!     GlobalTransform::IDENTITY,
//!     ChildOf(root),
//! )).id();
//! ```

pub mod components;
pub mod systems;

/// Convenient re-exports of commonly used types.
pub mod prelude {
    pub use crate::components::{
        // Mesh
        Aabb,
        // Material
        AlphaMode,
        // Camera
        Camera,
        CameraProjection,
        CameraViewport,
        // Hierarchy
        ChildOf,
        Children,
        // Collision
        Collider,
        ColliderShape,
        CollisionLayer,
        // Transform
        GlobalTransform,
        HierarchyDepth,
        HierarchyRoot,
        Material,
        MeshHandle,
        RenderLayers,
        RenderMesh,
        RenderTarget,
        RigidBody,
        RigidBodyType,
        TextureHandle,
        Transform,
    };

    pub use crate::systems::{
        propagate_transforms, run_transform_systems, sync_root_transforms, update_hierarchy_depth,
    };

    // Re-export glam for convenience
    pub use glam::{Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
}

// Re-export bevy_ecs for users who need direct access
pub use bevy_ecs;
