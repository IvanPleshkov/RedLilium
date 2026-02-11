/// Whether an entity should be rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, redlilium_ecs::Component)]
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

    #[test]
    fn visible_value() {
        let v = Visibility::VISIBLE;
        assert!(v.0);
        let h = Visibility::HIDDEN;
        assert!(!h.0);
    }
}
