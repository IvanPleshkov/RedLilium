//! Backend error types.

/// Errors that can occur in backend operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// Failed to initialize the backend.
    InitializationFailed(String),
    /// Failed to create a resource.
    ResourceCreationFailed(String),
    /// The requested feature is not supported.
    FeatureNotSupported(String),
    /// Out of GPU memory.
    OutOfMemory,
    /// The device was lost.
    DeviceLost,
    /// Invalid parameter.
    InvalidParameter(String),
    /// Internal backend error.
    Internal(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InitializationFailed(msg) => write!(f, "backend initialization failed: {msg}"),
            Self::ResourceCreationFailed(msg) => write!(f, "resource creation failed: {msg}"),
            Self::FeatureNotSupported(msg) => write!(f, "feature not supported: {msg}"),
            Self::OutOfMemory => write!(f, "out of GPU memory"),
            Self::DeviceLost => write!(f, "GPU device lost"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::Internal(msg) => write!(f, "internal backend error: {msg}"),
        }
    }
}

impl std::error::Error for BackendError {}
