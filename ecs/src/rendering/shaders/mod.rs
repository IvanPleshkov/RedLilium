//! Standard shader materials for the ECS rendering system.
//!
//! Each submodule provides a shader, uniform struct, and factory functions
//! for a common material type.

pub mod opaque_color;

pub use opaque_color::{
    OpaqueColorUniforms, create_opaque_color_entity, create_opaque_color_material,
    update_opaque_color_uniforms,
};
