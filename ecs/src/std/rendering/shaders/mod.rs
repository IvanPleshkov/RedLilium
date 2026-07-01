//! Standard shader materials for the ECS rendering system.
//!
//! Each submodule provides a shader, uniform struct, and factory functions
//! for a common material type.

pub mod entity_index;
pub mod opaque_color;

pub use entity_index::{EntityIndexUniforms, create_entity_index_material};
pub use opaque_color::OpaqueColorUniforms;
