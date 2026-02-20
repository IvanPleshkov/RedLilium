//! Format-specific encoding and decoding (feature-gated).
//!
//! Provides [`encode`] and [`decode`] functions that convert between
//! serde-serializable types and byte buffers in RON or bincode format.

use super::error::{DeserializeError, SerializeError};

/// Supported serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// RON (Rusty Object Notation) — human-readable text format.
    #[cfg(feature = "serialize-ron")]
    Ron,
    /// Bincode — compact binary format.
    #[cfg(feature = "serialize-bincode")]
    Bincode,
}

/// Encode a serde-serializable value to bytes in the given format.
#[allow(unused_variables)]
pub fn encode<T: serde::Serialize>(value: &T, format: Format) -> Result<Vec<u8>, SerializeError> {
    match format {
        #[cfg(feature = "serialize-ron")]
        Format::Ron => ron::ser::to_string_pretty(value, ron::ser::PrettyConfig::default())
            .map(|s| s.into_bytes())
            .map_err(|e| SerializeError::FormatError(e.to_string())),
        #[cfg(feature = "serialize-bincode")]
        Format::Bincode => {
            bincode::serialize(value).map_err(|e| SerializeError::FormatError(e.to_string()))
        }
    }
}

/// Decode bytes in the given format to a serde-deserializable type.
#[allow(unused_variables)]
pub fn decode<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    format: Format,
) -> Result<T, DeserializeError> {
    match format {
        #[cfg(feature = "serialize-ron")]
        Format::Ron => {
            let s = std::str::from_utf8(bytes)
                .map_err(|e| DeserializeError::FormatError(e.to_string()))?;
            ron::from_str(s).map_err(|e| DeserializeError::FormatError(e.to_string()))
        }
        #[cfg(feature = "serialize-bincode")]
        Format::Bincode => {
            bincode::deserialize(bytes).map_err(|e| DeserializeError::FormatError(e.to_string()))
        }
    }
}
