/// Debug name for an entity.
///
/// Stores an owned string. Use this to give entities meaningful labels
/// for debugging and editor display.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, crate::Component)]
pub struct Name(pub String);

impl Name {
    /// Create a new name from a string.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        let name = Name::default();
        assert!(name.as_str().is_empty());
    }

    #[test]
    fn display() {
        let name = Name::new("TestEntity");
        assert_eq!(format!("{name}"), "TestEntity");
    }

    #[test]
    fn from_str() {
        let name = Name::new("hello");
        assert_eq!(name.as_str(), "hello");
    }

    #[test]
    fn from_string() {
        let name = Name::new("world".to_string());
        assert_eq!(name.as_str(), "world");
    }
}
