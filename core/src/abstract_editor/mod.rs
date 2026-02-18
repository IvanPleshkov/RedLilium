//! Abstract editor framework for reversible editing operations.
//!
//! This module provides the foundational traits and types for building
//! an undo/redo-capable editor system. It is decoupled from specific
//! editable types (ECS entities, materials, scenes) so that higher-level
//! crates can implement concrete editors.
//!
//! - [`Editable`] — marker trait for types that can be edited
//! - [`EditAction`] — a reversible edit operation (Command pattern)
//! - [`EditActionHistory`] — undo/redo stack managing action sequences

mod action;
mod history;

pub use action::{AsAny, EditAction, EditActionError, EditActionResult, Editable};
pub use history::{DEFAULT_MAX_UNDO, EditActionHistory};
