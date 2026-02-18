//! # RedLilium Engine Core
//!
//! Core crate for RedLilium Engine basic utilities.

pub mod abstract_editor;
pub mod compute;
#[cfg(feature = "gltf")]
pub mod gltf;
pub mod material;
pub mod math;
pub mod mesh;
pub mod pool;
pub mod profiling;
pub mod sampler;
pub mod scene;
pub mod texture;

/// Core library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder for future game loop implementation
pub fn init() {
    log::info!("RedLilium Core v{} initialized", VERSION);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
