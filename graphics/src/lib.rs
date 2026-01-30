//! # RedLilium Graphics
//!
//! Custom rendering engine for RedLilium.
//!

/// Graphics library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Placeholder for future renderer initialization
pub fn init() {
    log::info!("RedLilium Graphics v{} initialized", VERSION);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
