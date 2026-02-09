/// Whether an entity should be rendered.
///
/// When attached to an entity with a [`MeshRenderer`](crate::MeshRenderer),
/// controls whether it is included in render submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Visibility(pub bool);

impl Visibility {
    /// Entity is visible (rendered).
    pub const VISIBLE: Self = Self(true);
    /// Entity is hidden (not rendered).
    pub const HIDDEN: Self = Self(false);

    /// Returns whether the entity is visible.
    pub fn is_visible(self) -> bool {
        self.0
    }
}

impl Default for Visibility {
    fn default() -> Self {
        Self::VISIBLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_visible() {
        assert!(Visibility::default().is_visible());
    }

    #[test]
    fn hidden_is_not_visible() {
        assert!(!Visibility::HIDDEN.is_visible());
    }
}
