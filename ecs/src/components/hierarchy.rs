use crate::Entity;

/// Marks an entity as a child of another entity.
///
/// When an entity has a `Parent`, its [`GlobalTransform`](super::GlobalTransform)
/// is computed relative to the parent's world transform.
///
/// Use [`set_parent`](crate::hierarchy::set_parent) to set up parent-child
/// relationships (it updates both `Parent` and [`Children`] components).
#[derive(Debug, Clone, Copy, PartialEq, Eq, crate::Component)]
pub struct Parent(pub Entity);

/// Stores the ordered list of child entities.
///
/// Automatically managed by [`set_parent`](crate::hierarchy::set_parent)
/// and [`remove_parent`](crate::hierarchy::remove_parent).
/// Do not modify directly — use the hierarchy functions instead.
#[derive(Debug, Clone, Default, PartialEq, crate::Component)]
pub struct Children(pub Vec<Entity>);

impl Children {
    /// Iterates over child entities.
    pub fn iter(&self) -> impl Iterator<Item = &Entity> {
        self.0.iter()
    }

    /// Returns the number of children.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether this entity has no children.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
