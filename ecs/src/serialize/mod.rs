//! Serialization and deserialization for ECS components and prefabs.
//!
//! This module provides:
//!
//! - [`SerializeContext`] / [`DeserializeContext`] — contexts for component
//!   serialization with Arc deduplication and entity remapping
//! - [`SerializeField`] / [`DeserializeField`] — field-level dispatch wrappers
//!   using the same method-resolution trick as [`Inspect`](crate::inspect::Inspect)
//! - [`Value`] — format-agnostic intermediate representation
//! - [`SerializedPrefab`] — on-disk entity tree representation
//! - [`Format`] / [`encode`] / [`decode`] — format-specific I/O (feature-gated)
//!
//! # Derive macro integration
//!
//! `#[derive(Component)]` generates `serialize_component` and
//! `deserialize_component` by default. Use `#[skip_serialization]` to
//! opt out for components with non-serializable fields (e.g., GPU resources).
//!
//! # Custom serialization
//!
//! Components that need resource-based serialization (e.g., serializing
//! `Arc<Texture>` as an asset path via `TextureManager`) should use
//! `#[skip_serialization]` and manually implement the trait methods,
//! accessing resources via `ctx.world()`.

mod context;
mod error;
pub mod field;
mod format;
// Natural-RON conversion needs the `ron` crate (also pulled in by `rendering`).
#[cfg(feature = "serialize-ron")]
pub mod natural;
mod prefab_io;
pub mod value;

pub use context::{DeserializeContext, SerializeContext, UnmappedEntityPolicy};
pub use error::{DeserializeError, SerializeError};
pub use field::{
    DeserializeField, DeserializeFieldFallback, SerializeField, SerializeFieldFallback,
};
pub use format::Format;
#[cfg(feature = "serialize-ron")]
pub use natural::{find_entity, natural_to_value, parse_entity_spec, value_to_natural};
pub use prefab_io::{SerializedComponent, SerializedEntity, SerializedPrefab};
pub use value::Value;

// Re-export format functions
pub use format::{decode, encode};
