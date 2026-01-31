//! ECS systems for the RedLilium engine.
//!
//! Systems contain the logic that operates on components. They are organized by functionality:
//!
//! - [`transform_propagation`]: Computes world-space transforms from hierarchy
//!
//! # Running Systems
//!
//! For simple usage without a full Bevy scheduler, use the convenience function:
//!
//! ```
//! use redlilium_ecs::prelude::*;
//! use bevy_ecs::prelude::*;
//!
//! let mut world = World::new();
//! // ... spawn entities with Transform, GlobalTransform, ChildOf ...
//! run_transform_systems(&mut world);
//! ```

mod transform_propagation;

pub use transform_propagation::{
    propagate_transforms, run_transform_systems, sync_root_transforms, update_hierarchy_depth,
};
