//! Abstract editor framework for reversible editing operations.
//!
//! This module provides the foundational traits and types for building
//! an undo/redo-capable editor system. It is decoupled from specific
//! editable types (ECS entities, materials, scenes) so that higher-level
//! crates can implement concrete editors.
//!
//! - [`Editable`] — marker trait for types that can be edited
//! - [`EditAction`] — an edit operation (Command pattern)
//! - [`EditActionHistory`] — undo/redo stack managing action sequences
//! - [`ActionQueue`] — thread-safe queue for submitting actions from read-only systems
//!
//! # Recorded vs non-recorded actions
//!
//! By default, actions are **recorded** in the undo/redo history. Override
//! [`EditAction::is_recorded`] to return `false` for transient operations
//! like camera movement that should not be undoable.
//!
//! Recorded actions can optionally return `false` from
//! [`EditAction::modifies_content`] to indicate they represent UI state
//! changes (such as entity selection) rather than document edits. These
//! actions are fully undoable but do not affect the save-distance tracker,
//! so [`EditActionHistory::has_unsaved_changes`] ignores them.
//!
//! Non-recorded actions can optionally **break the merge chain** by
//! overriding [`EditAction::breaks_merge`] to return `true`. This
//! prevents the next recorded action from merging with the previous
//! undo entry — useful when a deliberate interruption (e.g. camera zoom)
//! should separate two otherwise-mergeable drag sequences.

mod action;
mod action_queue;
mod history;

pub use action::{AsAny, EditAction, EditActionError, EditActionResult, Editable};
pub use action_queue::ActionQueue;
pub use history::{DEFAULT_MAX_UNDO, EditActionHistory};
