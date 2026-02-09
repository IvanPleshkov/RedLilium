/// Whether an entity should be rendered.
///
/// Uses `u8` instead of `bool` for Pod compatibility.
/// `0` = hidden, non-zero = visible.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable, redlilium_ecs::Component,
)]
#[repr(C)]
pub struct Visibility(pub u8);

impl Visibility {
    /// Entity is visible (rendered).
    pub const VISIBLE: Self = Self(1);
    /// Entity is hidden (not rendered).
    pub const HIDDEN: Self = Self(0);

    /// Returns whether the entity is visible.
    pub fn is_visible(self) -> bool {
        self.0 != 0
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
    fn is_pod() {
        let v = Visibility::VISIBLE;
        let bytes = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 1);
    }
}
