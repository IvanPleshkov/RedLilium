//! Material system for the graphics engine.
//!
//! This module provides a two-level material abstraction:
//!
//! - [`Material`] - Defines the shader and binding layout (created by device)
//! - [`MaterialInstance`] - Contains actual bound resources for rendering
//!
//! # Binding Frequency Optimization
//!
//! Bindings are organized by update frequency to minimize state changes:
//!
//! - **PerFrame** (Group 0) - Camera, lighting, time - shared across all objects
//! - **PerMaterial** (Group 1) - Material textures, properties - shared per material
//! - **PerObject** (Group 2) - Transform, object-specific data - per draw call
//!
//! This organization allows efficient batching: sort by material, bind once,
//! draw many objects with only per-object bindings changing.

mod bindings;
mod instance;
mod material;

pub use bindings::{BindingFrequency, BindingLayout, BindingLayoutEntry, BindingType};
pub use instance::{BoundResource, MaterialInstance};
pub use material::{Material, MaterialDescriptor, ShaderSource, ShaderStage};
