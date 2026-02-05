//! Error types for glTF loading.

/// Errors that can occur during glTF loading.
#[derive(Debug)]
pub enum GltfError {
    /// Failed to parse the glTF document.
    Parse(gltf_dep::Error),
    /// Failed to decode an image.
    ImageDecode(String),
    /// Unsupported primitive topology.
    UnsupportedTopology(String),
    /// A primitive is missing position data.
    MissingPositions {
        /// Mesh index in the glTF document.
        mesh: usize,
        /// Primitive index within the mesh.
        primitive: usize,
    },
    /// Error reading accessor data.
    AccessorError(String),
    /// Error resolving buffer data.
    BufferError(String),
}

impl std::fmt::Display for GltfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "glTF parse error: {e}"),
            Self::ImageDecode(msg) => write!(f, "image decode error: {msg}"),
            Self::UnsupportedTopology(msg) => write!(f, "unsupported topology: {msg}"),
            Self::MissingPositions { mesh, primitive } => {
                write!(
                    f,
                    "mesh {mesh} primitive {primitive} has no POSITION attribute"
                )
            }
            Self::AccessorError(msg) => write!(f, "accessor error: {msg}"),
            Self::BufferError(msg) => write!(f, "buffer error: {msg}"),
        }
    }
}

impl std::error::Error for GltfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<gltf_dep::Error> for GltfError {
    fn from(e: gltf_dep::Error) -> Self {
        Self::Parse(e)
    }
}
