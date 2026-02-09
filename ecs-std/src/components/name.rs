/// Debug name for an entity.
///
/// Useful for editor display, logging, and debugging.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Name(pub String);

impl Name {
    /// Create a new name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Borrow the name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for Name {
    fn from(s: String) -> Self {
        Self(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let name = Name::new("Player");
        assert_eq!(format!("{name}"), "Player");
    }

    #[test]
    fn from_str() {
        let name: Name = "Entity".into();
        assert_eq!(name.as_str(), "Entity");
    }

    #[test]
    fn from_string() {
        let name: Name = String::from("NPC").into();
        assert_eq!(name.as_str(), "NPC");
    }
}
