//! ECS Inspector UI built on [egui](https://docs.rs/egui).
//!
//! Provides two main panels:
//!
//! - **World Inspector** ([`show_world_inspector`]) — displays all entities as a
//!   hierarchy tree. Entities with a [`Name`](crate::Name) component show their
//!   name; others show their Entity ID. Click to select.
//!
//! - **Component Inspector** ([`show_component_inspector`]) — for a selected entity,
//!   lists all attached components with reflected fields that can be edited inline.
//!   Supports removing existing and adding new components. Inspector metadata is
//!   stored directly in the [`World`](crate::World) via
//!   [`register_inspector`](crate::World::register_inspector) /
//!   [`register_inspector_default`](crate::World::register_inspector_default).
//!
//! # Usage
//!
//! ```ignore
//! use redlilium_ecs::ui::{InspectorState, show_world_inspector, show_component_inspector};
//!
//! // During setup — register_std_components stores inspector metadata in the World
//! redlilium_ecs::register_std_components(&mut world);
//!
//! // During frame, inside egui update:
//! show_world_inspector(ctx, &world, &mut state);
//! show_component_inspector(ctx, &mut world, &mut state);
//! ```

mod component_inspector;
mod world_inspector;

pub use component_inspector::show_component_inspector;
pub use world_inspector::show_world_inspector;

use crate::Entity;

/// Persistent UI state for the inspector panels.
pub struct InspectorState {
    /// Currently selected entity (if any).
    pub selected: Option<Entity>,
    /// Whether the world inspector window is open.
    pub world_inspector_open: bool,
    /// Whether the component inspector window is open.
    pub component_inspector_open: bool,
    /// Filter text for entity search.
    pub filter: String,
    /// Tracks which tree nodes are expanded (by entity index).
    expanded: std::collections::HashSet<u32>,
    /// Add-component popup state.
    pub(crate) add_component_open: bool,
}

impl InspectorState {
    pub fn new() -> Self {
        Self {
            selected: None,
            world_inspector_open: true,
            component_inspector_open: true,
            filter: String::new(),
            expanded: std::collections::HashSet::new(),
            add_component_open: false,
        }
    }

    pub(crate) fn is_expanded(&self, entity: Entity) -> bool {
        self.expanded.contains(&entity.index())
    }

    pub(crate) fn toggle_expanded(&mut self, entity: Entity) {
        let idx = entity.index();
        if self.expanded.contains(&idx) {
            self.expanded.remove(&idx);
        } else {
            self.expanded.insert(idx);
        }
    }
}

impl Default for InspectorState {
    fn default() -> Self {
        Self::new()
    }
}
