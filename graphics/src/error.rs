//! Graphics error types.

use std::fmt;

/// Errors that can occur in the graphics system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphicsError {
    /// Failed to initialize the graphics system.
    InitializationFailed(String),
    /// Failed to create a resource.
    ResourceCreationFailed(String),
    /// A requested feature is not supported.
    FeatureNotSupported(String),
    /// Out of GPU memory.
    OutOfMemory,
    /// The GPU device was lost.
    DeviceLost,
    /// An invalid parameter was provided.
    InvalidParameter(String),
    /// An internal error occurred.
    Internal(String),
    /// The surface is outdated and needs to be reconfigured.
    SurfaceOutdated,
    /// The surface was lost and needs to be recreated.
    SurfaceLost,
}

impl fmt::Display for GraphicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitializationFailed(msg) => write!(f, "initialization failed: {msg}"),
            Self::ResourceCreationFailed(msg) => write!(f, "resource creation failed: {msg}"),
            Self::FeatureNotSupported(msg) => write!(f, "feature not supported: {msg}"),
            Self::OutOfMemory => write!(f, "out of GPU memory"),
            Self::DeviceLost => write!(f, "GPU device lost"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
            Self::SurfaceOutdated => write!(f, "surface outdated, needs reconfiguration"),
            Self::SurfaceLost => write!(f, "surface lost, needs recreation"),
        }
    }
}

impl std::error::Error for GraphicsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GraphicsError::OutOfMemory;
        assert_eq!(err.to_string(), "out of GPU memory");

        let err = GraphicsError::InitializationFailed("no GPU found".to_string());
        assert_eq!(err.to_string(), "initialization failed: no GPU found");
    }
}
