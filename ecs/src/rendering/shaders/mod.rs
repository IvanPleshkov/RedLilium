//! Standard shader materials for the ECS rendering system.
//!
//! Each submodule provides a shader, uniform struct, and factory functions
//! for a common material type.

pub mod entity_index;
pub mod opaque_color;

pub use entity_index::{
    EntityIndexUniforms, create_entity_index_instance, create_entity_index_material,
    update_entity_index_uniforms,
};
pub use opaque_color::{
    OpaqueColorUniforms, create_opaque_color_entity, create_opaque_color_entity_with_picking,
    create_opaque_color_material, update_opaque_color_uniforms,
};
