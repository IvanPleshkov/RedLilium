//! Undo/redo action history.
//!
//! [`EditActionHistory`] manages a linear undo/redo stack of [`EditAction`] trait
//! objects. When a new action is pushed after undoing, the redo stack is
//! cleared (standard editor behavior).

use std::collections::VecDeque;
use std::fmt;

use super::action::{EditAction, EditActionError, EditActionResult, Editable};

/// Default maximum number of undo steps.
pub const DEFAULT_MAX_UNDO: usize = 100;

/// Manages an undo/redo stack of editor actions.
///
/// The undo stack is a bounded [`VecDeque`] — when it exceeds `max_undo`,
/// the oldest action is dropped from the front. The redo stack is an
/// unbounded [`Vec`] (it can never grow larger than the undo stack was).
///
/// # Example
///
/// ```ignore
/// let mut history = EditActionHistory::new(50);
/// let mut target = MyScene::new();
///
/// // Execute and record an action
/// history.execute(Box::new(my_action), &mut target).unwrap();
///
/// // Undo the last action
/// history.undo(&mut target).unwrap();
///
/// // Redo it
/// history.redo(&mut target).unwrap();
/// ```
pub struct EditActionHistory<T: Editable> {
    undo_stack: VecDeque<Box<dyn EditAction<T>>>,
    redo_stack: Vec<Box<dyn EditAction<T>>>,
    max_undo: usize,
    merge_broken: bool,
    /// Tracks distance from the saved state.
    ///
    /// - `Some(0)` — the current state matches the last save.
    /// - `Some(n)` where `n > 0` — `n` undos needed to reach the saved state.
    /// - `Some(n)` where `n < 0` — `|n|` redos needed to reach the saved state.
    /// - `None` — never saved, or the save point is permanently unreachable
    ///   (e.g. after capacity overflow dropped it, or the redo branch was discarded).
    save_distance: Option<i64>,
}

impl<T: Editable> EditActionHistory<T> {
    /// Creates a new empty action history with the given maximum undo depth.
    ///
    /// When the undo stack exceeds `max_undo`, the oldest action is dropped.
    pub fn new(max_undo: usize) -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: Vec::new(),
            max_undo,
            merge_broken: false,
            save_distance: Some(0),
        }
    }

    /// Applies an action to the target and, if the action is
    /// [recorded](EditAction::is_recorded), pushes it onto the undo stack.
    ///
    /// **Recorded actions** (the default) clear the redo stack and attempt
    /// to [merge](EditAction::merge) with the top of the undo stack.
    ///
    /// **Non-recorded actions** (`is_recorded() == false`) are applied but
    /// never pushed onto either stack. If such an action also returns
    /// `true` from [`breaks_merge`](EditAction::breaks_merge), the next
    /// recorded action will not merge with the previous undo entry.
    ///
    /// If the action fails, it is not pushed onto the stack.
    pub fn execute(
        &mut self,
        mut action: Box<dyn EditAction<T>>,
        target: &mut T,
    ) -> EditActionResult {
        action.apply(target)?;

        if !action.is_recorded() {
            if action.breaks_merge() {
                self.merge_broken = true;
            }
            return Ok(());
        }

        let is_content = action.modifies_content();

        // Clearing the redo stack invalidates a save point that was in redo.
        self.redo_stack.clear();
        if is_content
            && let Some(d) = self.save_distance
            && d < 0
        {
            self.save_distance = None;
        }

        if !self.merge_broken
            && let Some(last) = self.undo_stack.back_mut()
        {
            match last.merge(action) {
                None => {
                    // Merged into the top entry — if that entry was the save
                    // point, the save is now invalidated (content changed).
                    if is_content && self.save_distance == Some(0) {
                        self.save_distance = None;
                    }
                    return Ok(());
                }
                Some(returned) => action = returned,
            }
        }
        self.merge_broken = false;

        // New entry pushed — save point moves one step further away.
        if is_content && let Some(d) = &mut self.save_distance {
            *d += 1;
        }

        self.undo_stack.push_back(action);
        if self.undo_stack.len() > self.max_undo {
            self.undo_stack.pop_front();
            // If the save point was beyond the oldest surviving entry, it's gone.
            if let Some(d) = self.save_distance
                && d > self.undo_stack.len() as i64
            {
                self.save_distance = None;
            }
        }
        Ok(())
    }

    /// Undoes the most recent action.
    ///
    /// Returns an error if the undo stack is empty or the undo failed.
    pub fn undo(&mut self, target: &mut T) -> EditActionResult {
        let mut action = self
            .undo_stack
            .pop_back()
            .ok_or_else(|| EditActionError::Custom("nothing to undo".into()))?;
        action.undo(target)?;
        let is_content = action.modifies_content();
        self.redo_stack.push(action);
        if is_content && let Some(d) = &mut self.save_distance {
            *d -= 1;
        }
        Ok(())
    }

    /// Redoes the most recently undone action.
    ///
    /// Returns an error if the redo stack is empty or the redo failed.
    pub fn redo(&mut self, target: &mut T) -> EditActionResult {
        let mut action = self
            .redo_stack
            .pop()
            .ok_or_else(|| EditActionError::Custom("nothing to redo".into()))?;
        action.apply(target)?;
        let is_content = action.modifies_content();
        self.undo_stack.push_back(action);
        if is_content && let Some(d) = &mut self.save_distance {
            *d += 1;
        }
        if self.undo_stack.len() > self.max_undo {
            self.undo_stack.pop_front();
            if let Some(d) = self.save_distance
                && d > self.undo_stack.len() as i64
            {
                self.save_distance = None;
            }
        }
        Ok(())
    }

    /// Returns `true` if there are actions that can be undone.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns `true` if there are actions that can be redone.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Returns an iterator over undo action descriptions, most recent first.
    pub fn undo_descriptions(&self) -> impl Iterator<Item = &str> {
        self.undo_stack.iter().rev().map(|a| a.description())
    }

    /// Returns an iterator over redo action descriptions, most recent first.
    pub fn redo_descriptions(&self) -> impl Iterator<Item = &str> {
        self.redo_stack.iter().rev().map(|a| a.description())
    }

    /// Returns the number of actions in the undo stack.
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Returns the number of actions in the redo stack.
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Returns the maximum undo depth.
    pub fn max_undo(&self) -> usize {
        self.max_undo
    }

    /// Records the current state as the saved state.
    ///
    /// After calling this, [`has_unsaved_changes`](Self::has_unsaved_changes)
    /// returns `false` until the history is modified by execute, undo, or redo.
    pub fn mark_saved(&mut self) {
        self.save_distance = Some(0);
    }

    /// Returns `true` if the current state differs from the last saved state.
    ///
    /// Returns `true` if [`mark_saved`](Self::mark_saved) has never been called,
    /// or if the history has been modified since the last save, or if the save
    /// point is permanently unreachable (e.g. dropped by capacity overflow or
    /// the redo branch was discarded).
    pub fn has_unsaved_changes(&self) -> bool {
        self.save_distance != Some(0)
    }

    /// Clears both undo and redo stacks and resets the merge-broken flag.
    ///
    /// If the current state was the saved state (`has_unsaved_changes` was
    /// `false`), it remains so after clearing. Otherwise the save point is
    /// permanently lost.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.merge_broken = false;
        // If we were at the save point, clearing history doesn't change
        // the target — we're still at the saved state. Otherwise the
        // save point is unreachable.
        if self.save_distance != Some(0) {
            self.save_distance = None;
        }
    }
}

impl<T: Editable> fmt::Debug for EditActionHistory<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EditActionHistory")
            .field("undo_count", &self.undo_stack.len())
            .field("redo_count", &self.redo_stack.len())
            .field("max_undo", &self.max_undo)
            .field("merge_broken", &self.merge_broken)
            .field("save_distance", &self.save_distance)
            .finish()
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

    /// A mergeable action: simulates dragging by setting a value.
    /// Consecutive SetValue actions merge into one (keeps first old_value,
    /// takes latest new_value).
    #[derive(Debug)]
    struct SetValue {
        old_value: i32,
        new_value: i32,
    }

    impl EditAction<Counter> for SetValue {
        fn apply(&mut self, target: &mut Counter) -> EditActionResult {
            target.value = self.new_value;
            Ok(())
        }

        fn undo(&mut self, target: &mut Counter) -> EditActionResult {
            target.value = self.old_value;
            Ok(())
        }

        fn description(&self) -> &str {
            "Set value"
        }

        fn merge(
            &mut self,
            other: Box<dyn EditAction<Counter>>,
        ) -> Option<Box<dyn EditAction<Counter>>> {
            if let Some(other) = other.as_any().downcast_ref::<SetValue>() {
                self.new_value = other.new_value;
                return None;
            }
            Some(other)
        }
    }

    #[derive(Debug)]
    struct FailingAction;

    impl EditAction<Counter> for FailingAction {
        fn apply(&mut self, _target: &mut Counter) -> EditActionResult {
            Err(EditActionError::Custom("always fails".into()))
        }

        fn undo(&mut self, _target: &mut Counter) -> EditActionResult {
            Err(EditActionError::Custom("always fails".into()))
        }

        fn description(&self) -> &str {
            "Failing"
        }
    }

    /// Non-recorded action that does NOT break the merge chain.
    #[derive(Debug)]
    struct CameraMove {
        offset: i32,
    }

    impl EditAction<Counter> for CameraMove {
        fn apply(&mut self, target: &mut Counter) -> EditActionResult {
            target.value += self.offset;
            Ok(())
        }

        fn undo(&mut self, _target: &mut Counter) -> EditActionResult {
            unreachable!("non-recorded actions should never be undone");
        }

        fn description(&self) -> &str {
            "Camera move"
        }

        fn is_recorded(&self) -> bool {
            false
        }
    }

    /// Non-recorded action that DOES break the merge chain.
    #[derive(Debug)]
    struct CameraZoom;

    impl EditAction<Counter> for CameraZoom {
        fn apply(&mut self, _target: &mut Counter) -> EditActionResult {
            Ok(())
        }

        fn undo(&mut self, _target: &mut Counter) -> EditActionResult {
            unreachable!("non-recorded actions should never be undone");
        }

        fn description(&self) -> &str {
            "Camera zoom"
        }

        fn is_recorded(&self) -> bool {
            false
        }

        fn breaks_merge(&self) -> bool {
            true
        }
    }

    #[test]
    fn execute_applies_and_pushes() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();

        assert_eq!(counter.value, 5);
        assert_eq!(history.undo_count(), 1);
        assert_eq!(history.redo_count(), 0);
    }

    #[test]
    fn undo_reverses_and_moves_to_redo() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();
        history.undo(&mut counter).unwrap();

        assert_eq!(counter.value, 0);
        assert_eq!(history.undo_count(), 0);
        assert_eq!(history.redo_count(), 1);
    }

    #[test]
    fn redo_reapplies_and_moves_to_undo() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();
        history.undo(&mut counter).unwrap();
        history.redo(&mut counter).unwrap();

        assert_eq!(counter.value, 5);
        assert_eq!(history.undo_count(), 1);
        assert_eq!(history.redo_count(), 0);
    }

    #[test]
    fn execute_clears_redo_stack() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();
        history.undo(&mut counter).unwrap();
        assert_eq!(history.redo_count(), 1);

        history
            .execute(Box::new(Add { amount: 3 }), &mut counter)
            .unwrap();
        assert_eq!(history.redo_count(), 0);
        assert_eq!(counter.value, 3);
    }

    #[test]
    fn undo_empty_returns_error() {
        let mut history = EditActionHistory::<Counter>::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        let result = history.undo(&mut counter);
        assert!(result.is_err());
    }

    #[test]
    fn redo_empty_returns_error() {
        let mut history = EditActionHistory::<Counter>::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        let result = history.redo(&mut counter);
        assert!(result.is_err());
    }

    #[test]
    fn capacity_drops_oldest() {
        let mut history = EditActionHistory::new(2);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history
            .execute(Box::new(Add { amount: 2 }), &mut counter)
            .unwrap();
        history
            .execute(Box::new(Add { amount: 3 }), &mut counter)
            .unwrap();

        assert_eq!(history.undo_count(), 2);
        assert_eq!(counter.value, 6);

        // Undo the two remaining actions (amount=3 and amount=2)
        history.undo(&mut counter).unwrap();
        history.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 1); // only amount=1 remains applied
        assert!(history.undo(&mut counter).is_err());
    }

    #[test]
    fn full_round_trip() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 10 };

        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();
        assert_eq!(counter.value, 15);

        history.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 10);

        history.redo(&mut counter).unwrap();
        assert_eq!(counter.value, 15);

        history.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 10);
    }

    #[test]
    fn descriptions() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        assert_eq!(history.undo_descriptions().count(), 0);
        assert_eq!(history.redo_descriptions().count(), 0);

        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history
            .execute(Box::new(Add { amount: 2 }), &mut counter)
            .unwrap();

        let undos: Vec<&str> = history.undo_descriptions().collect();
        assert_eq!(undos, vec!["Add", "Add"]);

        history.undo(&mut counter).unwrap();
        history.undo(&mut counter).unwrap();

        let redos: Vec<&str> = history.redo_descriptions().collect();
        assert_eq!(redos, vec!["Add", "Add"]);
    }

    #[test]
    fn can_undo_can_redo() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        assert!(!history.can_undo());
        assert!(!history.can_redo());

        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        assert!(history.can_undo());
        assert!(!history.can_redo());

        history.undo(&mut counter).unwrap();
        assert!(!history.can_undo());
        assert!(history.can_redo());
    }

    #[test]
    fn clear_empties_both_stacks() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history
            .execute(Box::new(Add { amount: 2 }), &mut counter)
            .unwrap();
        history.undo(&mut counter).unwrap();

        history.clear();
        assert_eq!(history.undo_count(), 0);
        assert_eq!(history.redo_count(), 0);
    }

    #[test]
    fn failed_execute_does_not_push() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        let result = history.execute(Box::new(FailingAction), &mut counter);
        assert!(result.is_err());
        assert_eq!(history.undo_count(), 0);
        assert_eq!(counter.value, 0);
    }

    #[test]
    fn debug_impl() {
        let history = EditActionHistory::<Counter>::new(DEFAULT_MAX_UNDO);
        let debug = format!("{history:?}");
        assert!(debug.contains("EditActionHistory"));
        assert!(debug.contains("undo_count"));
    }

    #[test]
    fn max_undo_accessor() {
        let history = EditActionHistory::<Counter>::new(42);
        assert_eq!(history.max_undo(), 42);
    }

    #[test]
    fn merge_coalesces_consecutive_actions() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        // Simulate a drag: three incremental SetValue actions
        history
            .execute(
                Box::new(SetValue {
                    old_value: 0,
                    new_value: 10,
                }),
                &mut counter,
            )
            .unwrap();
        history
            .execute(
                Box::new(SetValue {
                    old_value: 10,
                    new_value: 20,
                }),
                &mut counter,
            )
            .unwrap();
        history
            .execute(
                Box::new(SetValue {
                    old_value: 20,
                    new_value: 30,
                }),
                &mut counter,
            )
            .unwrap();

        assert_eq!(counter.value, 30);
        // All three merged into one undo entry
        assert_eq!(history.undo_count(), 1);

        // Single undo reverts to the original value
        history.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 0);
    }

    #[test]
    fn merge_does_not_merge_different_types() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(
                Box::new(SetValue {
                    old_value: 0,
                    new_value: 10,
                }),
                &mut counter,
            )
            .unwrap();
        // Add is a different action type — should not merge
        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();

        assert_eq!(counter.value, 15);
        assert_eq!(history.undo_count(), 2);
    }

    #[test]
    fn merge_after_undo_clears_redo() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(
                Box::new(SetValue {
                    old_value: 0,
                    new_value: 10,
                }),
                &mut counter,
            )
            .unwrap();
        history.undo(&mut counter).unwrap();
        assert_eq!(history.redo_count(), 1);

        // New action clears redo even when it would merge (stack is empty after undo)
        history
            .execute(
                Box::new(SetValue {
                    old_value: 0,
                    new_value: 5,
                }),
                &mut counter,
            )
            .unwrap();
        assert_eq!(history.redo_count(), 0);
        assert_eq!(history.undo_count(), 1);
    }

    #[test]
    fn non_recorded_action_applies_but_not_pushed() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(CameraMove { offset: 42 }), &mut counter)
            .unwrap();

        assert_eq!(counter.value, 42); // effect applied
        assert_eq!(history.undo_count(), 0); // not in history
        assert_eq!(history.redo_count(), 0);
    }

    #[test]
    fn non_recorded_action_does_not_clear_redo() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        // Build a redo entry
        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();
        history.undo(&mut counter).unwrap();
        assert_eq!(history.redo_count(), 1);

        // Non-recorded action should NOT clear redo
        history
            .execute(Box::new(CameraMove { offset: 1 }), &mut counter)
            .unwrap();
        assert_eq!(history.redo_count(), 1);
    }

    #[test]
    fn non_recorded_without_break_preserves_merge() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        // First drag step
        history
            .execute(
                Box::new(SetValue {
                    old_value: 0,
                    new_value: 10,
                }),
                &mut counter,
            )
            .unwrap();

        // Camera move (non-recorded, does NOT break merge)
        history
            .execute(Box::new(CameraMove { offset: 0 }), &mut counter)
            .unwrap();

        // Second drag step — should still merge with first
        history
            .execute(
                Box::new(SetValue {
                    old_value: 10,
                    new_value: 20,
                }),
                &mut counter,
            )
            .unwrap();

        assert_eq!(counter.value, 20);
        assert_eq!(history.undo_count(), 1); // merged into one entry
        history.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 0); // reverts to original
    }

    #[test]
    fn breaks_merge_prevents_coalescing() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        // First drag step
        history
            .execute(
                Box::new(SetValue {
                    old_value: 0,
                    new_value: 10,
                }),
                &mut counter,
            )
            .unwrap();

        // Camera zoom (non-recorded, BREAKS merge)
        history.execute(Box::new(CameraZoom), &mut counter).unwrap();

        // Second set — should NOT merge with the first
        history
            .execute(
                Box::new(SetValue {
                    old_value: 10,
                    new_value: 20,
                }),
                &mut counter,
            )
            .unwrap();

        assert_eq!(counter.value, 20);
        assert_eq!(history.undo_count(), 2); // two separate entries

        history.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 10);
        history.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 0);
    }

    #[test]
    fn merge_broken_resets_after_recorded_action() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        // Set initial value
        history
            .execute(
                Box::new(SetValue {
                    old_value: 0,
                    new_value: 10,
                }),
                &mut counter,
            )
            .unwrap();

        // Break merge
        history.execute(Box::new(CameraZoom), &mut counter).unwrap();

        // This recorded action resets the merge_broken flag
        history
            .execute(
                Box::new(SetValue {
                    old_value: 10,
                    new_value: 20,
                }),
                &mut counter,
            )
            .unwrap();
        assert_eq!(history.undo_count(), 2);

        // Next merge should work again (flag was reset)
        history
            .execute(
                Box::new(SetValue {
                    old_value: 20,
                    new_value: 30,
                }),
                &mut counter,
            )
            .unwrap();
        assert_eq!(history.undo_count(), 2); // merged with previous
    }

    #[test]
    fn failed_non_recorded_action_does_not_break_merge() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(
                Box::new(SetValue {
                    old_value: 0,
                    new_value: 10,
                }),
                &mut counter,
            )
            .unwrap();

        // Failing action — should not affect merge state
        let _ = history.execute(Box::new(FailingAction), &mut counter);

        // Should still merge
        history
            .execute(
                Box::new(SetValue {
                    old_value: 10,
                    new_value: 20,
                }),
                &mut counter,
            )
            .unwrap();
        assert_eq!(history.undo_count(), 1);
    }

    // --- Save tracking tests ---

    #[test]
    fn no_unsaved_changes_on_fresh_history() {
        let history = EditActionHistory::<Counter>::new(DEFAULT_MAX_UNDO);
        assert!(!history.has_unsaved_changes());
    }

    #[test]
    fn not_unsaved_after_mark_saved() {
        let mut history = EditActionHistory::<Counter>::new(DEFAULT_MAX_UNDO);
        history.mark_saved();
        assert!(!history.has_unsaved_changes());
    }

    #[test]
    fn unsaved_after_execute() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history.mark_saved();
        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        assert!(history.has_unsaved_changes());
    }

    #[test]
    fn not_unsaved_after_undo_to_save_point() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history.mark_saved();
        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history.undo(&mut counter).unwrap();
        assert!(!history.has_unsaved_changes());
    }

    #[test]
    fn not_unsaved_after_undo_then_redo_to_save_point() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history.mark_saved();
        history.undo(&mut counter).unwrap();
        assert!(history.has_unsaved_changes());
        history.redo(&mut counter).unwrap();
        assert!(!history.has_unsaved_changes());
    }

    #[test]
    fn unsaved_after_undo_past_save_point() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history.mark_saved();
        history.undo(&mut counter).unwrap();
        assert!(history.has_unsaved_changes());
    }

    #[test]
    fn save_lost_when_new_branch_after_undo() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history.mark_saved();
        // Undo to before save, then execute new action (clears redo with save)
        history.undo(&mut counter).unwrap();
        history
            .execute(Box::new(Add { amount: 2 }), &mut counter)
            .unwrap();
        // Save was in the redo branch that was discarded
        assert!(history.has_unsaved_changes());
        // Can't get back to saved state
        history.undo(&mut counter).unwrap();
        assert!(history.has_unsaved_changes());
    }

    #[test]
    fn save_lost_when_capacity_overflow() {
        let mut history = EditActionHistory::new(2);
        let mut counter = Counter { value: 0 };

        history.mark_saved();
        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history
            .execute(Box::new(Add { amount: 2 }), &mut counter)
            .unwrap();
        // Still reachable (2 undos, stack has 2 entries)
        assert!(history.has_unsaved_changes());
        history.undo(&mut counter).unwrap();
        history.undo(&mut counter).unwrap();
        assert!(!history.has_unsaved_changes());

        // Now overflow: push a third, dropping the oldest
        history.redo(&mut counter).unwrap();
        history.redo(&mut counter).unwrap();
        history
            .execute(Box::new(Add { amount: 3 }), &mut counter)
            .unwrap();
        // save_distance was 3 but stack has 2 entries → lost
        assert!(history.has_unsaved_changes());
    }

    #[test]
    fn merge_at_save_point_invalidates() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(
                Box::new(SetValue {
                    old_value: 0,
                    new_value: 10,
                }),
                &mut counter,
            )
            .unwrap();
        history.mark_saved();
        // Merge into the save point entry
        history
            .execute(
                Box::new(SetValue {
                    old_value: 10,
                    new_value: 20,
                }),
                &mut counter,
            )
            .unwrap();
        // The save entry was modified by merge → save lost
        assert!(history.has_unsaved_changes());
    }

    #[test]
    fn non_recorded_action_does_not_affect_save() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history.mark_saved();
        history
            .execute(Box::new(CameraMove { offset: 42 }), &mut counter)
            .unwrap();
        // Non-recorded action doesn't change save state
        assert!(!history.has_unsaved_changes());
    }

    #[test]
    fn clear_preserves_save_at_current_state() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history.mark_saved();
        history.clear();
        // Target state unchanged, still at save point
        assert!(!history.has_unsaved_changes());
    }

    #[test]
    fn clear_loses_unreachable_save() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history.mark_saved();
        history
            .execute(Box::new(Add { amount: 1 }), &mut counter)
            .unwrap();
        history.clear();
        // Target state differs from save, and history is gone
        assert!(history.has_unsaved_changes());
    }

    // --- Non-content (UI-state) action tests ---

    /// Recorded action that does NOT modify content (UI state only).
    #[derive(Debug)]
    struct SelectionAction {
        old_value: i32,
        new_value: i32,
    }

    impl EditAction<Counter> for SelectionAction {
        fn apply(&mut self, target: &mut Counter) -> EditActionResult {
            target.value = self.new_value;
            Ok(())
        }

        fn undo(&mut self, target: &mut Counter) -> EditActionResult {
            target.value = self.old_value;
            Ok(())
        }

        fn description(&self) -> &str {
            "Select"
        }

        fn modifies_content(&self) -> bool {
            false
        }
    }

    #[test]
    fn non_content_action_is_recorded_but_no_save_change() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history.mark_saved();
        history
            .execute(
                Box::new(SelectionAction {
                    old_value: 0,
                    new_value: 42,
                }),
                &mut counter,
            )
            .unwrap();
        assert_eq!(counter.value, 42);
        assert_eq!(history.undo_count(), 1); // recorded in undo stack
        assert!(!history.has_unsaved_changes()); // but no save distance change
    }

    #[test]
    fn non_content_action_is_undoable() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(
                Box::new(SelectionAction {
                    old_value: 0,
                    new_value: 42,
                }),
                &mut counter,
            )
            .unwrap();
        assert_eq!(counter.value, 42);
        history.undo(&mut counter).unwrap();
        assert_eq!(counter.value, 0);
    }

    #[test]
    fn non_content_action_undo_does_not_affect_save() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        // Execute content action, mark saved
        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();
        history.mark_saved();
        assert!(!history.has_unsaved_changes());

        // Execute non-content action
        history
            .execute(
                Box::new(SelectionAction {
                    old_value: 5,
                    new_value: 99,
                }),
                &mut counter,
            )
            .unwrap();
        assert!(!history.has_unsaved_changes()); // still saved

        // Undo the non-content action
        history.undo(&mut counter).unwrap();
        assert!(!history.has_unsaved_changes()); // still saved
    }

    #[test]
    fn non_content_action_redo_does_not_affect_save() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history.mark_saved();

        // Execute + undo non-content action
        history
            .execute(
                Box::new(SelectionAction {
                    old_value: 0,
                    new_value: 42,
                }),
                &mut counter,
            )
            .unwrap();
        history.undo(&mut counter).unwrap();
        assert!(!history.has_unsaved_changes());

        // Redo the non-content action
        history.redo(&mut counter).unwrap();
        assert!(!history.has_unsaved_changes());
    }

    #[test]
    fn mixed_content_and_non_content_save_tracking() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history.mark_saved();

        // Content action: save_distance becomes 1
        history
            .execute(Box::new(Add { amount: 10 }), &mut counter)
            .unwrap();
        assert!(history.has_unsaved_changes());

        // Non-content action: save_distance stays 1
        history
            .execute(
                Box::new(SelectionAction {
                    old_value: 10,
                    new_value: 99,
                }),
                &mut counter,
            )
            .unwrap();
        assert!(history.has_unsaved_changes());

        // Undo non-content: save_distance stays 1
        history.undo(&mut counter).unwrap();
        assert!(history.has_unsaved_changes());

        // Undo content: save_distance becomes 0
        history.undo(&mut counter).unwrap();
        assert!(!history.has_unsaved_changes());
    }

    #[test]
    fn non_content_action_clears_redo_stack() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        // Execute content action, then undo to put it in redo
        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();
        history.undo(&mut counter).unwrap();
        assert_eq!(history.redo_count(), 1);

        // Execute non-content action — still clears redo (linear history)
        history
            .execute(
                Box::new(SelectionAction {
                    old_value: 0,
                    new_value: 42,
                }),
                &mut counter,
            )
            .unwrap();
        assert_eq!(history.redo_count(), 0);
    }

    #[test]
    fn non_content_at_save_point_does_not_invalidate() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();
        history.mark_saved();

        // Non-content action at save point — should not invalidate
        history
            .execute(
                Box::new(SelectionAction {
                    old_value: 5,
                    new_value: 99,
                }),
                &mut counter,
            )
            .unwrap();
        assert!(!history.has_unsaved_changes());
    }

    #[test]
    fn mark_saved_after_execute_and_undo_round_trip() {
        let mut history = EditActionHistory::new(DEFAULT_MAX_UNDO);
        let mut counter = Counter { value: 0 };

        history
            .execute(Box::new(Add { amount: 5 }), &mut counter)
            .unwrap();
        history
            .execute(Box::new(Add { amount: 3 }), &mut counter)
            .unwrap();
        history.undo(&mut counter).unwrap();
        history.mark_saved(); // saved in the middle
        assert!(!history.has_unsaved_changes());

        // Undo further → unsaved
        history.undo(&mut counter).unwrap();
        assert!(history.has_unsaved_changes());

        // Redo back → saved again
        history.redo(&mut counter).unwrap();
        assert!(!history.has_unsaved_changes());

        // Redo past save → unsaved
        history.redo(&mut counter).unwrap();
        assert!(history.has_unsaved_changes());
    }
}
