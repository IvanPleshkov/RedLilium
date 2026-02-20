//! Editable targets and reversible editor actions.
//!
//! This module defines the core abstractions for an undo/redo editor system:
//!
//! - [`Editable`] — marker trait for types that can be edited
//! - [`EditAction`] — a reversible edit operation (Command pattern)
//! - [`EditActionError`] / [`EditActionResult`] — error handling for actions
//!
//! EditActions are self-contained: each implementation internally stores whatever
//! data it needs (target identifiers, old/new values, brush pixels, etc.).

use std::any::Any;
use std::fmt;

/// Helper trait for downcasting trait objects to concrete types.
///
/// Automatically implemented for all `'static` types. Used by
/// [`EditAction::merge`] to downcast `&dyn EditAction<T>` to the
/// concrete action type for merging.
pub trait AsAny: 'static {
    /// Returns a reference to `self` as `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Marker trait for types that serve as editing targets.
///
/// Implement this on any type that actions can operate on — an ECS world,
/// a scene graph, a texture editor, etc.
///
/// # Example
///
/// ```ignore
/// struct MyScene { /* ... */ }
/// impl Editable for MyScene {}
/// ```
pub trait Editable: 'static {}

/// Error type for action execution failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditActionError {
    /// The target object was not found.
    TargetNotFound(String),
    /// The target is in an invalid state for this action.
    InvalidState(String),
    /// A custom error with a description.
    Custom(String),
}

impl fmt::Display for EditActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TargetNotFound(msg) => write!(f, "target not found: {msg}"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            Self::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for EditActionError {}

/// Result type for action operations.
pub type EditActionResult<T = ()> = Result<T, EditActionError>;

/// A reversible editor action (Command pattern).
///
/// EditActions encapsulate a single logical edit and capture enough state to
/// undo the change and redo it. Each implementation stores its own data
/// internally — there is no prescribed property system.
///
/// # Merging
///
/// Actions that represent incremental changes (e.g. each mouse move during
/// a drag) can override [`merge`](Self::merge) so that consecutive actions
/// coalesce into one undo step. Use [`AsAny::as_any`] on the `other`
/// action to downcast it to the concrete type.
///
/// # Object Safety
///
/// This trait is dyn-compatible so that different action types can be stored
/// in a single [`EditActionHistory`](super::EditActionHistory) undo/redo stack as
/// `Box<dyn EditAction<T>>`.
///
/// # Example
///
/// ```ignore
/// #[derive(Debug)]
/// struct MoveEntity {
///     entity: Entity,
///     old_pos: Vec3,
///     new_pos: Vec3,
/// }
///
/// impl EditAction<World> for MoveEntity {
///     fn apply(&mut self, target: &mut World) -> EditActionResult {
///         target.set_position(self.entity, self.new_pos);
///         Ok(())
///     }
///
///     fn undo(&mut self, target: &mut World) -> EditActionResult {
///         target.set_position(self.entity, self.old_pos);
///         Ok(())
///     }
///
///     fn description(&self) -> &str {
///         "Move entity"
///     }
///
///     fn merge(
///         &mut self,
///         other: Box<dyn EditAction<World>>,
///     ) -> Option<Box<dyn EditAction<World>>> {
///         if let Some(other) = other.as_any().downcast_ref::<MoveEntity>() {
///             if self.entity == other.entity {
///                 self.new_pos = other.new_pos;
///                 return None; // consumed
///             }
///         }
///         Some(other) // not mergeable
///     }
/// }
/// ```
pub trait EditAction<T: Editable>: fmt::Debug + AsAny + Send {
    /// Applies the action to the target (forward / redo direction).
    ///
    /// Returns `Ok(())` on success, or an [`EditActionError`] if the action
    /// could not be applied.
    fn apply(&mut self, target: &mut T) -> EditActionResult;

    /// Reverses the action (undo direction).
    ///
    /// Must restore the target to the state before [`apply`](Self::apply)
    /// was called.
    fn undo(&mut self, target: &mut T) -> EditActionResult;

    /// A short, human-readable description for display in the edit menu.
    ///
    /// Examples: `"Move entity"`, `"Change base color"`, `"Brush stroke"`.
    fn description(&self) -> &str;

    /// Tries to merge `other` into `self`, taking ownership.
    ///
    /// If the actions are compatible (e.g. consecutive drags on the same
    /// entity), `self` absorbs `other`'s effect and returns `None`
    /// (the other action is consumed). Otherwise returns `Some(other)`
    /// back to the caller.
    ///
    /// Returns `Some(other)` by default (no merging).
    ///
    /// Use [`AsAny::as_any`] on `other` to downcast to the concrete type:
    ///
    /// ```ignore
    /// fn merge(
    ///     &mut self,
    ///     other: Box<dyn EditAction<MyTarget>>,
    /// ) -> Option<Box<dyn EditAction<MyTarget>>> {
    ///     if let Some(other) = other.as_any().downcast_ref::<Self>() {
    ///         // absorb other's data into self
    ///         return None; // consumed
    ///     }
    ///     Some(other) // not mergeable, return it back
    /// }
    /// ```
    fn merge(&mut self, other: Box<dyn EditAction<T>>) -> Option<Box<dyn EditAction<T>>> {
        Some(other)
    }

    /// Whether this action is recorded in the undo/redo history.
    ///
    /// Return `false` for transient operations that should not be undoable,
    /// such as editor camera movement, selection highlighting, or viewport
    /// adjustments.
    ///
    /// Default: `true`.
    fn is_recorded(&self) -> bool {
        true
    }

    /// Whether executing this action prevents the next recorded action
    /// from merging with the previous undo entry.
    ///
    /// Only meaningful for non-recorded actions (`is_recorded() == false`).
    /// A camera zoom that interrupts a drag sequence is an example of a
    /// non-recorded action that should break the merge chain.
    ///
    /// Default: `false`.
    fn breaks_merge(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Counter {
        value: i32,
    }

    impl Editable for Counter {}

    #[derive(Debug)]
    struct Add {
        amount: i32,
    }

    impl EditAction<Counter> for Add {
        fn apply(&mut self, target: &mut Counter) -> EditActionResult {
            target.value += self.amount;
            Ok(())
        }

        fn undo(&mut self, target: &mut Counter) -> EditActionResult {
            target.value -= self.amount;
            Ok(())
        }

        fn description(&self) -> &str {
            "Add"
        }
    }

    #[test]
    fn apply_modifies_target() {
        let mut counter = Counter { value: 0 };
        let mut action = Add { amount: 5 };
        action.apply(&mut counter).unwrap();
        assert_eq!(counter.value, 5);
    }

    #[test]
    fn undo_reverses_apply() {
        let mut counter = Counter { value: 0 };
        let mut action = Add { amount: 5 };
        action.apply(&mut counter).unwrap();
        action.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 0);
    }

    #[test]
    fn action_description() {
        let action = Add { amount: 1 };
        assert_eq!(action.description(), "Add");
    }

    #[test]
    fn action_error_display() {
        assert_eq!(
            EditActionError::TargetNotFound("entity 42".into()).to_string(),
            "target not found: entity 42"
        );
        assert_eq!(
            EditActionError::InvalidState("locked".into()).to_string(),
            "invalid state: locked"
        );
        assert_eq!(
            EditActionError::Custom("something went wrong".into()).to_string(),
            "something went wrong"
        );
    }

    #[test]
    fn action_is_dyn_compatible() {
        let mut counter = Counter { value: 0 };
        let mut boxed: Box<dyn EditAction<Counter>> = Box::new(Add { amount: 3 });
        boxed.apply(&mut counter).unwrap();
        assert_eq!(counter.value, 3);
        boxed.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 0);
    }

    #[test]
    fn default_is_recorded() {
        let action = Add { amount: 1 };
        assert!(action.is_recorded());
    }

    #[test]
    fn default_breaks_merge() {
        let action = Add { amount: 1 };
        assert!(!action.breaks_merge());
    }

    #[derive(Debug)]
    struct TransientAction;

    impl EditAction<Counter> for TransientAction {
        fn apply(&mut self, target: &mut Counter) -> EditActionResult {
            target.value += 100;
            Ok(())
        }

        fn undo(&mut self, _target: &mut Counter) -> EditActionResult {
            unreachable!("transient actions should never be undone");
        }

        fn description(&self) -> &str {
            "Transient"
        }

        fn is_recorded(&self) -> bool {
            false
        }
    }

    #[test]
    fn transient_action_not_recorded() {
        let action = TransientAction;
        assert!(!action.is_recorded());
        assert!(!action.breaks_merge());
    }

    #[derive(Debug)]
    struct MergeBreakingTransient;

    impl EditAction<Counter> for MergeBreakingTransient {
        fn apply(&mut self, _target: &mut Counter) -> EditActionResult {
            Ok(())
        }

        fn undo(&mut self, _target: &mut Counter) -> EditActionResult {
            unreachable!("transient actions should never be undone");
        }

        fn description(&self) -> &str {
            "Merge-breaking transient"
        }

        fn is_recorded(&self) -> bool {
            false
        }

        fn breaks_merge(&self) -> bool {
            true
        }
    }

    #[test]
    fn merge_breaking_transient() {
        let action = MergeBreakingTransient;
        assert!(!action.is_recorded());
        assert!(action.breaks_merge());
    }
}
