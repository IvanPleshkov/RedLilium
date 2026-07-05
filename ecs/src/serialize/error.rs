//! Error types for component serialization and deserialization.

use std::fmt;

/// Errors that can occur during component serialization.
#[derive(Debug)]
pub enum SerializeError {
    /// A field could not be converted to a [`Value`](super::Value).
    FieldError { field: String, message: String },
    /// The component does not support serialization.
    NotSerializable { component: &'static str },
    /// Format encoding error (RON/bincode).
    FormatError(String),
    /// A component references a live entity outside the prefab's scope.
    /// Prefab assets must be self-contained (see `World::serialize_prefab_asset`):
    /// include the target in the prefab or clear the link.
    ExternalEntityRef {
        /// The in-scope entity carrying the reference.
        entity: crate::Entity,
        /// Name of the component holding the reference.
        component: String,
        /// The live out-of-scope entity being referenced.
        target: crate::Entity,
    },
}

impl fmt::Display for SerializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FieldError { field, message } => {
                write!(f, "failed to serialize field '{field}': {message}")
            }
            Self::NotSerializable { component } => {
                write!(f, "component '{component}' does not support serialization")
            }
            Self::FormatError(msg) => write!(f, "format error: {msg}"),
            Self::ExternalEntityRef {
                entity,
                component,
                target,
            } => {
                write!(
                    f,
                    "component '{component}' on {entity} references {target}, which is alive \
                     but outside the prefab; prefab assets must be self-contained — include \
                     the target in the prefab or clear the link"
                )
            }
        }
    }
}

impl std::error::Error for SerializeError {}

/// Errors that can occur during component deserialization.
#[derive(Debug)]
pub enum DeserializeError {
    /// A required field was missing from the serialized data.
    MissingField { field: String, component: String },
    /// A field value had an unexpected type.
    TypeMismatch {
        field: String,
        expected: String,
        found: String,
    },
    /// The component type is not registered in the world.
    UnknownComponent { type_name: String },
    /// The component does not support deserialization.
    NotDeserializable { component: String },
    /// Format decoding error.
    FormatError(String),
    /// Arc reference ID not found in the deduplication cache.
    InvalidArcRef { id: u32 },
    /// An entity reference pointed outside the deserialized set and could not
    /// be remapped. Fabricating a handle from the source indices would alias an
    /// unrelated or dead entity, so this is rejected instead.
    UnmappedEntityReference {
        field: String,
        index: u32,
        spawn_tick: u64,
    },
}

impl fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField { field, component } => {
                write!(f, "missing field '{field}' in component '{component}'")
            }
            Self::TypeMismatch {
                field,
                expected,
                found,
            } => {
                write!(
                    f,
                    "type mismatch for field '{field}': expected {expected}, found {found}"
                )
            }
            Self::UnknownComponent { type_name } => {
                write!(f, "unknown component type '{type_name}'")
            }
            Self::NotDeserializable { component } => {
                write!(
                    f,
                    "component '{component}' does not support deserialization"
                )
            }
            Self::FormatError(msg) => write!(f, "format error: {msg}"),
            Self::InvalidArcRef { id } => {
                write!(f, "invalid Arc reference id {id}")
            }
            Self::UnmappedEntityReference {
                field,
                index,
                spawn_tick,
            } => {
                write!(
                    f,
                    "entity reference in field '{field}' ({index}@{spawn_tick}) points outside \
                     the deserialized set and cannot be remapped"
                )
            }
        }
    }
}

impl std::error::Error for DeserializeError {}
