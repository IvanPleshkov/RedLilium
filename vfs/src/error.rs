use std::fmt;

/// Errors that can occur during virtual file system operations.
#[derive(Debug)]
pub enum VfsError {
    /// The requested path was not found in the provider.
    NotFound(String),
    /// An IO error occurred while accessing a provider.
    Io(std::io::Error),
    /// The path is invalid (empty, contains `..`, or other normalization failure).
    InvalidPath(String),
    /// No provider is mounted at the given source name.
    NoSuchSource(String),
    /// The provider does not support write operations.
    ReadOnly,
}

impl fmt::Display for VfsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VfsError::NotFound(path) => write!(f, "not found: {path}"),
            VfsError::Io(err) => write!(f, "IO error: {err}"),
            VfsError::InvalidPath(reason) => write!(f, "invalid path: {reason}"),
            VfsError::NoSuchSource(name) => write!(f, "no such source: {name}"),
            VfsError::ReadOnly => write!(f, "provider is read-only"),
        }
    }
}

impl std::error::Error for VfsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VfsError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VfsError {
    fn from(err: std::io::Error) -> Self {
        if err.kind() == std::io::ErrorKind::NotFound {
            VfsError::NotFound(err.to_string())
        } else {
            VfsError::Io(err)
        }
    }
}
