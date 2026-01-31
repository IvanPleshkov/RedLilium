//! ECS systems for the RedLilium engine.
//!
//! Systems contain the logic that operates on components. They are organized by functionality:
//!
//! - [`transform_propagation`]: Computes world-space transforms from hierarchy

mod transform_propagation;

pub use transform_propagation::{
    propagate_transforms, sync_root_transforms, update_hierarchy_depth,
};
