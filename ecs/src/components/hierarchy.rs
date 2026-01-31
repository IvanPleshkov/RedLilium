//! Hierarchy components for parent-child entity relationships.
//!
//! This module provides components for building entity hierarchies (scene graphs).
//! Child entities have their transforms relative to their parent.

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;

// Re-export Bevy's hierarchy components
pub use bevy_ecs::hierarchy::{ChildOf, Children};

/// Marker component for root entities in a hierarchy.
///
/// Root entities have no parent and their [`Transform`] is in world space.
/// This component is automatically managed by hierarchy systems.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct HierarchyRoot;

/// Marker component indicating this entity's transform tree has changed.
///
/// Used for optimization - transform propagation can skip subtrees
/// where no transforms have changed.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct TransformDirty;

/// Component storing the previous parent entity.
///
/// Used to detect parent changes and trigger re-parenting logic.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct PreviousParent(pub Option<Entity>);

/// Depth in the hierarchy tree, starting at 0 for root entities.
///
/// Used for ordering transform propagation and ensuring parents
/// are updated before children.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct HierarchyDepth(pub u32);

impl HierarchyDepth {
    /// Returns the depth value.
    #[inline]
    pub const fn depth(&self) -> u32 {
        self.0
    }

    /// Creates a new hierarchy depth.
    #[inline]
    pub const fn new(depth: u32) -> Self {
        Self(depth)
    }

    /// Returns the child depth (one level deeper).
    #[inline]
    pub const fn child_depth(&self) -> Self {
        Self(self.0 + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hierarchy_depth() {
        let root = HierarchyDepth::new(0);
        let child = root.child_depth();
        let grandchild = child.child_depth();

        assert_eq!(root.depth(), 0);
        assert_eq!(child.depth(), 1);
        assert_eq!(grandchild.depth(), 2);
    }
}
