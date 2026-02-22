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

pub use component_inspector::{ImportComponentAction, show_component_inspector};
pub use world_inspector::{DeleteEntityAction, SpawnPrefabAction, show_world_inspector};

use redlilium_core::abstract_editor::{ActionQueue, EditAction, EditActionResult};

use crate::{Entity, World};

// ---------------------------------------------------------------------------
// Selection resource
// ---------------------------------------------------------------------------

/// ECS resource tracking which entities are currently selected in the editor.
///
/// Stored as a resource in [`World`]. The world inspector writes to it via
/// [`SelectAction`], and the component inspector reads from it to decide
/// what to display.
#[derive(Debug, Default, Clone)]
pub struct Selection {
    entities: Vec<Entity>,
}

impl Selection {
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
        }
    }

    /// The list of currently selected entities.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Returns the single selected entity, or `None` if zero or multiple are selected.
    pub fn single(&self) -> Option<Entity> {
        if self.entities.len() == 1 {
            Some(self.entities[0])
        } else {
            None
        }
    }

    pub fn is_selected(&self, entity: Entity) -> bool {
        self.entities.contains(&entity)
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn set(&mut self, entities: Vec<Entity>) {
        self.entities = entities;
    }

    pub fn clear(&mut self) {
        self.entities.clear();
    }
}

// ---------------------------------------------------------------------------
// Select action (non-recorded — selection is not undoable)
// ---------------------------------------------------------------------------

/// Undoable editor action that changes the current entity selection.
///
/// Pushed to the [`ActionQueue`] by the world inspector on click.
/// Returns `false` from [`modifies_content()`](EditAction::modifies_content)
/// so that selection changes are undoable but do not count as unsaved
/// document changes.
#[derive(Debug)]
pub struct SelectAction {
    new_selection: Vec<Entity>,
    old_selection: Vec<Entity>,
}

impl SelectAction {
    /// Select a single entity (replaces any existing selection).
    pub fn single(entity: Entity) -> Self {
        Self {
            new_selection: vec![entity],
            old_selection: Vec::new(),
        }
    }

    /// Set selection to a specific list of entities.
    pub fn set(entities: Vec<Entity>) -> Self {
        Self {
            new_selection: entities,
            old_selection: Vec::new(),
        }
    }

    /// Clear selection entirely.
    pub fn clear() -> Self {
        Self {
            new_selection: Vec::new(),
            old_selection: Vec::new(),
        }
    }
}

impl EditAction<World> for SelectAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let mut sel = world.resource_mut::<Selection>();
        self.old_selection = sel.entities().to_vec();
        sel.set(self.new_selection.clone());
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        let mut sel = world.resource_mut::<Selection>();
        sel.set(self.old_selection.clone());
        Ok(())
    }

    fn description(&self) -> &str {
        "Select"
    }

    fn modifies_content(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Drag-and-drop payloads (shared between inspector and editor)
// ---------------------------------------------------------------------------

/// Payload for dragging a component from the Component Inspector.
#[derive(Clone, Debug)]
pub struct ComponentDragPayload {
    pub entity: Entity,
    pub name: &'static str,
}

/// Payload for dragging a `.component` file from the Asset Browser.
#[derive(Clone, Debug)]
pub struct ComponentFileDragPayload {
    pub vfs_path: String,
}

/// Payload for dragging a `.prefab` file from the Asset Browser.
#[derive(Clone, Debug)]
pub struct PrefabFileDragPayload {
    pub vfs_path: String,
}

/// Fallback deferred reparent for when no [`ActionQueue`] resource is present.
/// Used only as a last resort — the preferred path pushes to the action queue.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingReparent {
    pub entity: Entity,
    /// `Some(parent)` to set parent, `None` to make root.
    pub new_parent: Option<Entity>,
}

/// Persistent UI state for the inspector panels.
pub struct InspectorState {
    /// Filter text for entity search.
    pub filter: String,
    /// Whether to show editor-only entities in the world inspector.
    /// Defaults to `false` — editor entities (camera, grid, gizmos) are hidden.
    pub show_editor_entities: bool,
    /// Tracks which tree nodes are expanded (by entity index).
    expanded: std::collections::HashSet<u32>,
    /// Add-component popup state.
    pub(crate) add_component_open: bool,
    /// Fallback deferred reparent (only used when no ActionQueue resource exists).
    pub(crate) pending_reparent: Option<PendingReparent>,
    /// Set when a `.component` file is dropped on the component inspector.
    /// Tuple: (vfs_path, target_entity). Consumed by the editor.
    pub pending_component_import: Option<(String, Entity)>,
    /// Set when a `.prefab` file is dropped on the world inspector.
    /// Tuple: (vfs_path, parent_entity_or_none). Consumed by the editor.
    pub pending_prefab_import: Option<(String, Option<Entity>)>,
    /// Fallback deferred delete (only used when no ActionQueue resource exists).
    pub(crate) pending_delete: Option<Entity>,
    /// Fallback deferred selection (only used when no ActionQueue resource exists).
    pub(crate) pending_selection: Option<Vec<Entity>>,
}

impl InspectorState {
    pub fn new() -> Self {
        Self {
            filter: String::new(),
            show_editor_entities: false,
            expanded: std::collections::HashSet::new(),
            add_component_open: false,
            pending_reparent: None,
            pending_component_import: None,
            pending_prefab_import: None,
            pending_delete: None,
            pending_selection: None,
        }
    }

    /// Applies any deferred hierarchy actions (e.g. drag-and-drop reparenting).
    ///
    /// This is only needed when no [`ActionQueue`](redlilium_core::abstract_editor::ActionQueue)
    /// resource is present in the World. When the action queue exists,
    /// reparent operations go through it instead and are handled by the
    /// editor's undo/redo system.
    ///
    /// Called automatically at the start of [`show_component_inspector`].
    pub fn apply_pending_actions(&mut self, world: &mut World) {
        if let Some(action) = self.pending_reparent.take()
            && world.is_alive(action.entity)
        {
            match action.new_parent {
                Some(parent) if world.is_alive(parent) => {
                    crate::std::hierarchy::set_parent(world, action.entity, parent);
                }
                None => {
                    crate::std::hierarchy::remove_parent(world, action.entity);
                }
                _ => {}
            }
        }

        if let Some(entity) = self.pending_delete.take()
            && world.is_alive(entity)
        {
            crate::std::hierarchy::despawn_recursive(world, entity);
        }

        if let Some(entities) = self.pending_selection.take() {
            ensure_selection_resource(world);
            let mut sel = world.resource_mut::<Selection>();
            sel.set(entities);
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

/// Ensure a [`Selection`] resource exists in the world, inserting a default if needed.
fn ensure_selection_resource(world: &mut World) {
    if !world.has_resource::<Selection>() {
        world.insert_resource(Selection::new());
    }
}

/// Read the current selection from the [`Selection`] resource.
/// Returns an empty slice if the resource doesn't exist.
fn read_selection(world: &World) -> Vec<Entity> {
    if world.has_resource::<Selection>() {
        world.resource::<Selection>().entities().to_vec()
    } else {
        Vec::new()
    }
}

/// Check if an entity is currently selected.
fn is_entity_selected(world: &World, entity: Entity) -> bool {
    if world.has_resource::<Selection>() {
        world.resource::<Selection>().is_selected(entity)
    } else {
        false
    }
}

/// Submit a selection change. If an [`ActionQueue`] resource is present,
/// pushes a [`SelectAction`]. Otherwise stores in [`InspectorState`] for
/// deferred application.
fn submit_selection(world: &World, state: &mut InspectorState, entities: Vec<Entity>) {
    if world.has_resource::<ActionQueue<World>>() {
        let queue = world.resource::<ActionQueue<World>>();
        queue.push(Box::new(SelectAction::set(entities)));
    } else {
        state.pending_selection = Some(entities);
    }
}
