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
        }
    }
}

impl std::error::Error for DeserializeError {}
