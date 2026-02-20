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
//! // During frame, render into any egui::Ui container:
//! show_world_inspector(ui, &world, &mut state);
//! show_component_inspector(ui, &mut world, &mut state);
//! ```

mod component_inspector;
mod world_inspector;

pub use component_inspector::show_component_inspector;
pub use world_inspector::show_world_inspector;

use crate::{Entity, World};

/// A deferred hierarchy action recorded during drag-and-drop in the world
/// inspector. Applied when `&mut World` becomes available.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PendingReparent {
    /// Set `entity` as a child of `new_parent`.
    SetParent { entity: Entity, new_parent: Entity },
    /// Remove `entity` from its current parent, making it a root.
    MakeRoot { entity: Entity },
}

/// Persistent UI state for the inspector panels.
pub struct InspectorState {
    /// Currently selected entity (if any).
    pub selected: Option<Entity>,
    /// Filter text for entity search.
    pub filter: String,
    /// Tracks which tree nodes are expanded (by entity index).
    expanded: std::collections::HashSet<u32>,
    /// Add-component popup state.
    pub(crate) add_component_open: bool,
    /// Deferred reparent action from drag-and-drop.
    pub(crate) pending_reparent: Option<PendingReparent>,
}

impl InspectorState {
    pub fn new() -> Self {
        Self {
            selected: None,
            filter: String::new(),
            expanded: std::collections::HashSet::new(),
            add_component_open: false,
            pending_reparent: None,
        }
    }

    /// Applies any deferred hierarchy actions (e.g. drag-and-drop reparenting).
    ///
    /// Called automatically at the start of [`show_component_inspector`]. Users
    /// who only render the world inspector should call this manually each frame.
    pub fn apply_pending_actions(&mut self, world: &mut World) {
        if let Some(action) = self.pending_reparent.take() {
            match action {
                PendingReparent::SetParent { entity, new_parent } => {
                    if world.is_alive(entity) && world.is_alive(new_parent) {
                        crate::std::hierarchy::set_parent(world, entity, new_parent);
                    }
                }
                PendingReparent::MakeRoot { entity } => {
                    if world.is_alive(entity) {
                        crate::std::hierarchy::remove_parent(world, entity);
                    }
                }
            }
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
