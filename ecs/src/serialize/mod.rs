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
mod prefab_io;
pub mod value;

pub use context::{DeserializeContext, SerializeContext};
pub use error::{DeserializeError, SerializeError};
pub use field::{
    DeserializeField, DeserializeFieldFallback, SerializeField, SerializeFieldFallback,
};
pub use format::Format;
pub use prefab_io::{SerializedComponent, SerializedEntity, SerializedPrefab};
pub use value::Value;

// Re-export format functions
pub use format::{decode, encode};
