use redlilium_ecs::StringId;

/// Debug name for an entity.
///
/// Stores a [`StringId`] referencing an interned string in the world's
/// [`StringTable`](redlilium_ecs::StringTable). Use the table to resolve
/// the ID back to a string slice.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    bytemuck::Pod,
    bytemuck::Zeroable,
    redlilium_ecs::Component,
)]
#[repr(C)]
pub struct Name(pub StringId);

impl Name {
    /// Create a new name from a [`StringId`].
    pub fn new(id: StringId) -> Self {
        Self(id)
    }

    /// Get the [`StringId`].
    pub fn id(&self) -> StringId {
        self.0
    }
}

impl Default for Name {
    fn default() -> Self {
        Self(StringId::EMPTY)
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Name({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        let name = Name::default();
        assert_eq!(name.id(), StringId::EMPTY);
    }

    #[test]
    fn display() {
        let name = Name::new(StringId(42));
        assert_eq!(format!("{name}"), "Name(StringId(42))");
    }

    #[test]
    fn is_pod() {
        let name = Name::new(StringId(7));
        let bytes = bytemuck::bytes_of(&name);
        assert_eq!(bytes.len(), 4);
    }
}
