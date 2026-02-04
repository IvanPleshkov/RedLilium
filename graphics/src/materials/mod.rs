//! Material system for the graphics engine.
//!
//! This module provides a two-level material abstraction:
//!
//! - [`Material`] - Defines the shader and binding layout (created by device)
//! - [`MaterialInstance`] - Contains actual bound resources for rendering
//!
//! # Efficient Batching via Arc Sharing
//!
//! Binding layouts and groups are wrapped in `Arc` to enable efficient batching.
//! The renderer can compare `Arc` pointers to group draw calls that share the same
//! layouts or bindings, minimizing GPU state changes.

mod bindings;
mod instance;
mod material;

pub use bindings::{BindingLayout, BindingLayoutEntry, BindingType, ShaderStageFlags};
pub use instance::{BindingGroup, BoundResource, MaterialInstance};
pub use material::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, Material, MaterialDescriptor,
    ShaderSource, ShaderStage,
};
