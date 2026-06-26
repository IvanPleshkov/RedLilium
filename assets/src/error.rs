//! Asset pipeline errors.

use redlilium_graphics::GraphicsError;

/// Error produced while loading an asset (read / decode / gpu stage).
#[derive(Debug, Clone)]
pub enum AssetError {
    /// A file-backed source had no resolved path (DB lookup failed / missing).
    NotResolved,
    /// I/O / VFS failure while reading the source or a referenced file.
    Io(String),
    /// Decode/parse failure (bad/unsupported format, corrupt data).
    Decode(String),
    /// GPU residency (upload) failed.
    Gpu(GraphicsError),
}

impl std::fmt::Display for AssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotResolved => write!(f, "asset source has no resolved path"),
            Self::Io(m) => write!(f, "asset I/O error: {m}"),
            Self::Decode(m) => write!(f, "asset decode error: {m}"),
            Self::Gpu(e) => write!(f, "asset gpu error: {e}"),
        }
    }
}

impl std::error::Error for AssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            _ => None,
        }
    }
}

impl From<GraphicsError> for AssetError {
    fn from(e: GraphicsError) -> Self {
        Self::Gpu(e)
    }
}
