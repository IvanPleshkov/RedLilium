//! # RedLilium Engine Core
//!
//! Core crate for RedLilium Engine basic utilities.

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
