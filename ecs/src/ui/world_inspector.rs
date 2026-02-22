//! Entity hierarchy tree view with drag-and-drop reparenting.

use redlilium_core::abstract_editor::{ActionQueue, EditAction, EditActionError, EditActionResult};

use crate::{Entity, World};

use crate::std::components::{Children, Name, Parent};

use super::{InspectorState, PrefabFileDragPayload};

/// Render the world inspector — a tree of all entities organized by hierarchy.
///
/// Root entities (those without a `Parent`) appear at the top level.
/// Children appear nested under their parent, expandable via a toggle.
///
/// Entities with a [`Name`] component display their name; others show `Entity(index:gen)`.
///
/// Drag an entity onto another to reparent it. Drop on the zone at the bottom
/// of the tree to make an entity a root. If an [`ActionQueue<World>`] resource is
/// present, reparent operations are pushed as undoable actions; otherwise they
/// are applied directly on the next [`InspectorState::apply_pending_actions`] call.
///
/// The caller is responsible for placing this in whatever container they want
/// (dock tab, side panel, window, etc.).
pub fn show_world_inspector(ui: &mut egui::Ui, world: &World, state: &mut InspectorState) {
    // Filter input
    ui.horizontal(|ui| {
        ui.label("Filter:");
        ui.text_edit_singleline(&mut state.filter);
    });

    ui.checkbox(&mut state.show_editor_entities, "Show Editor Entities");
    ui.separator();

    // Entity count
    ui.label(format!("Entities: {}", world.entity_count()));

    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Collect root entities (no Parent component)
            let roots = collect_roots(world);

            if state.show_editor_entities {
                // Show editor entities first, separated from scene entities
                let (editor_roots, scene_roots): (Vec<&Entity>, Vec<&Entity>) =
                    roots.iter().partition(|e| is_editor_entity(**e));

                if !editor_roots.is_empty() {
                    ui.label(
                        egui::RichText::new("Editor Entities")
                            .weak()
                            .italics()
                            .size(11.0),
                    );
                    for entity in &editor_roots {
                        show_entity_node(ui, world, **entity, state);
                    }
                    ui.separator();
                }

                for entity in &scene_roots {
                    show_entity_node(ui, world, **entity, state);
                }
            } else {
                for entity in &roots {
                    if !is_editor_entity(*entity) {
                        show_entity_node(ui, world, *entity, state);
                    }
                }
            }

            // "Drop here to unparent" zone — only visible while dragging an entity
            if egui::DragAndDrop::has_payload_of_type::<Entity>(ui.ctx()) {
                ui.separator();
                let resp = ui.label(
                    egui::RichText::new("  Drop here to unparent (make root)  ")
                        .italics()
                        .weak(),
                );

                if let Some(dragged) = resp.dnd_hover_payload::<Entity>()
                    && world.get::<Parent>(*dragged).is_some()
                {
                    ui.painter().rect_stroke(
                        resp.rect,
                        egui::CornerRadius::same(2),
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 200, 100)),
                        egui::StrokeKind::Outside,
                    );
                }

                if let Some(dragged) = resp.dnd_release_payload::<Entity>()
                    && world.get::<Parent>(*dragged).is_some()
                {
                    let old_parent = world.get::<Parent>(*dragged).map(|p| p.0);
                    submit_reparent(world, state, *dragged, old_parent, None);
                }
            }

            // "Drop here to spawn prefab" zone — visible while dragging a .prefab file
            if egui::DragAndDrop::has_payload_of_type::<PrefabFileDragPayload>(ui.ctx()) {
                ui.separator();
                let resp = ui.label(
                    egui::RichText::new("  Drop here to spawn as root  ")
                        .italics()
                        .weak(),
                );

                if resp.dnd_hover_payload::<PrefabFileDragPayload>().is_some() {
                    ui.painter().rect_stroke(
                        resp.rect,
                        egui::CornerRadius::same(2),
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 200, 100)),
                        egui::StrokeKind::Outside,
                    );
                }

                if let Some(payload) = resp.dnd_release_payload::<PrefabFileDragPayload>() {
                    state.pending_prefab_import = Some((payload.vfs_path.clone(), None));
                }
            }
        });
}

/// Returns `true` if the entity has the EDITOR or INHERITED_EDITOR flag.
fn is_editor_entity(entity: Entity) -> bool {
    entity.flags() & (Entity::EDITOR | Entity::INHERITED_EDITOR) != 0
}

/// Collect entities that have no Parent component (root entities).
fn collect_roots(world: &World) -> Vec<Entity> {
    let has_parent = world.with::<Parent>();
    let mut roots: Vec<Entity> = world
        .iter_entities()
        .filter(|e| !has_parent.matches(e.index()))
        .collect();
    roots.sort_by_key(|e| e.index());
    roots
}

/// Render a single entity node in the tree (recursive for children).
fn show_entity_node(ui: &mut egui::Ui, world: &World, entity: Entity, state: &mut InspectorState) {
    // Skip editor entities when the toggle is off
    if !state.show_editor_entities && is_editor_entity(entity) {
        return;
    }

    let label = entity_label(world, entity);

    // Apply filter
    if !state.filter.is_empty() && !label.to_lowercase().contains(&state.filter.to_lowercase()) {
        // If this entity doesn't match, still check children
        if let Some(children) = world.get::<Children>(entity) {
            for &child in children.0.iter() {
                show_entity_node(ui, world, child, state);
            }
        }
        return;
    }

    let has_children = world.get::<Children>(entity).is_some_and(|c| !c.is_empty());
    let is_selected = super::is_entity_selected(world, entity);

    let entity_button =
        egui::Button::selectable(is_selected, &label).sense(egui::Sense::click_and_drag());

    if has_children {
        // Expandable node
        let expanded = state.is_expanded(entity);
        ui.horizontal(|ui| {
            let toggle_text = if expanded { "v" } else { ">" };
            if ui.small_button(toggle_text).clicked() {
                state.toggle_expanded(entity);
            }
            let resp = ui.add(entity_button);
            // Detect clicks from raw pointer state: resp.clicked() can fail
            // on macOS when modifier keys (Cmd/Ctrl) are held together with
            // Sense::click_and_drag().
            if resp.clicked() {
                handle_entity_click(ui, world, state, entity);
            }
            resp.dnd_set_drag_payload(entity);
            handle_drop_target(ui, &resp, world, entity, state);
            show_entity_context_menu(&resp, world, entity, state);
        });

        if expanded || state.is_expanded(entity) {
            ui.indent(egui::Id::new(("entity_tree", entity.index())), |ui| {
                if let Some(children) = world.get::<Children>(entity) {
                    let child_list: Vec<Entity> = children.0.clone();
                    for child in child_list {
                        show_entity_node(ui, world, child, state);
                    }
                }
            });
        }
    } else {
        // Leaf node
        ui.horizontal(|ui| {
            ui.add_space(20.0); // indent to match toggle button width
            let resp = ui.add(entity_button);
            if resp.clicked() {
                handle_entity_click(ui, world, state, entity);
            }
            resp.dnd_set_drag_payload(entity);
            handle_drop_target(ui, &resp, world, entity, state);
            show_entity_context_menu(&resp, world, entity, state);
        });
    }
}

/// Handle drag-and-drop hover/release on an entity row.
///
/// Shows visual feedback (blue for valid, red for invalid) and submits a
/// reparent action on release.
fn handle_drop_target(
    ui: &egui::Ui,
    response: &egui::Response,
    world: &World,
    target_entity: Entity,
    state: &mut InspectorState,
) {
    // Check payload type with dnd_hover_payload (non-destructive) BEFORE calling
    // dnd_release_payload (destructive — removes payload from egui context
    // regardless of downcast success). Only call release for the matching type.
    let hovering_entity = response.dnd_hover_payload::<Entity>().is_some();
    let hovering_prefab = response
        .dnd_hover_payload::<PrefabFileDragPayload>()
        .is_some();

    if hovering_entity {
        let dragged = response.dnd_hover_payload::<Entity>().unwrap();
        let color = if is_valid_reparent(world, *dragged, target_entity) {
            egui::Color32::from_rgb(100, 160, 255) // blue — valid
        } else {
            egui::Color32::from_rgb(255, 80, 80) // red — invalid
        };
        ui.painter().rect_stroke(
            response.rect,
            egui::CornerRadius::same(2),
            egui::Stroke::new(2.0, color),
            egui::StrokeKind::Outside,
        );

        if let Some(dragged) = response.dnd_release_payload::<Entity>()
            && is_valid_reparent(world, *dragged, target_entity)
        {
            let old_parent = world.get::<Parent>(*dragged).map(|p| p.0);
            submit_reparent(world, state, *dragged, old_parent, Some(target_entity));
        }
    } else if hovering_prefab {
        // Accept .prefab file drops — spawn as child of this entity
        ui.painter().rect_stroke(
            response.rect,
            egui::CornerRadius::same(2),
            egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 200, 100)),
            egui::StrokeKind::Outside,
        );

        if let Some(payload) = response.dnd_release_payload::<PrefabFileDragPayload>() {
            state.pending_prefab_import = Some((payload.vfs_path.clone(), Some(target_entity)));
        }
    }
}

/// Submit a reparent operation. If an [`ActionQueue`] resource is present, the
/// action is pushed there for undo/redo support. Otherwise it is stored in
/// [`InspectorState`] for deferred application.
fn submit_reparent(
    world: &World,
    state: &mut InspectorState,
    entity: Entity,
    old_parent: Option<Entity>,
    new_parent: Option<Entity>,
) {
    if world.has_resource::<ActionQueue<World>>() {
        let queue = world.resource::<ActionQueue<World>>();
        queue.push(Box::new(ReparentAction {
            entity,
            old_parent,
            new_parent,
        }));
    } else {
        // Fallback: store for deferred application via apply_pending_actions
        state.pending_reparent = Some(super::PendingReparent { entity, new_parent });
    }
}

/// Returns `true` if reparenting `entity` under `target_parent` is valid.
///
/// Invalid cases:
/// - `entity == target_parent` (self-parenting)
/// - `target_parent` is a descendant of `entity` (would create a cycle)
fn is_valid_reparent(world: &World, entity: Entity, target_parent: Entity) -> bool {
    if entity == target_parent {
        return false;
    }

    // Walk up from target_parent through ancestors. If we hit `entity`, the
    // target is a descendant — reparenting would create a cycle.
    let mut current = target_parent;
    while let Some(parent) = world.get::<Parent>(current) {
        if parent.0 == entity {
            return false;
        }
        current = parent.0;
    }

    true
}

/// Handle click on an entity row.
///
/// - Plain click: select only this entity (replaces selection).
/// - Ctrl/Cmd + click: toggle this entity in the current selection.
fn handle_entity_click(ui: &egui::Ui, world: &World, state: &mut InspectorState, entity: Entity) {
    let modifiers = ui.input(|i| i.modifiers);
    if modifiers.mac_cmd || modifiers.ctrl {
        // Ctrl/Cmd+click: toggle entity in selection
        let mut current = super::read_selection(world);
        if let Some(pos) = current.iter().position(|&e| e == entity) {
            current.remove(pos);
        } else {
            current.push(entity);
        }
        super::submit_selection(world, state, current);
    } else {
        // Plain click: select only this entity
        super::submit_selection(world, state, vec![entity]);
    }
}

/// Generate a display label for an entity.
fn entity_label(world: &World, entity: Entity) -> String {
    if let Some(name) = world.get::<Name>(entity) {
        let s = name.as_str();
        if !s.is_empty() {
            return format!("{} [{}@{}]", s, entity.index(), entity.spawn_tick());
        }
    }
    format!("Entity({}@{})", entity.index(), entity.spawn_tick())
}

/// Show right-click context menu for an entity node.
fn show_entity_context_menu(
    response: &egui::Response,
    world: &World,
    entity: Entity,
    state: &mut InspectorState,
) {
    response.context_menu(|ui| {
        if ui.button("Delete Entity").clicked() {
            submit_delete(world, state, entity);
            // Clear selection if we're deleting a selected entity
            if super::is_entity_selected(world, entity) {
                let remaining: Vec<Entity> = super::read_selection(world)
                    .into_iter()
                    .filter(|&e| e != entity)
                    .collect();
                super::submit_selection(world, state, remaining);
            }
            ui.close();
        }
    });
}

/// Submit a delete-entity operation. If an [`ActionQueue`] resource is present,
/// the action is pushed there for undo/redo support. Otherwise the deletion is
/// stored in [`InspectorState`] for deferred application.
fn submit_delete(world: &World, state: &mut InspectorState, entity: Entity) {
    let parent = world.get::<Parent>(entity).map(|p| p.0);
    if world.has_resource::<ActionQueue<World>>() {
        let queue = world.resource::<ActionQueue<World>>();
        queue.push(Box::new(DeleteEntityAction {
            entity,
            parent,
            serialized: None,
        }));
    } else {
        state.pending_delete = Some(entity);
    }
}

// ---------------------------------------------------------------------------
// Undoable delete-entity action
// ---------------------------------------------------------------------------

/// Reversible action that deletes an entity and its entire subtree.
///
/// - `apply()`: serializes the entity tree as a prefab (for undo), then despawns recursively.
/// - `undo()`: deserializes the saved prefab to restore the entity tree, re-parents under
///   the original parent if it still exists.
pub struct DeleteEntityAction {
    entity: Entity,
    parent: Option<Entity>,
    serialized: Option<crate::serialize::SerializedPrefab>,
}

impl std::fmt::Debug for DeleteEntityAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeleteEntityAction")
            .field("entity", &self.entity)
            .field("parent", &self.parent)
            .field("has_backup", &self.serialized.is_some())
            .finish()
    }
}

impl EditAction<World> for DeleteEntityAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        // Serialize the entity subtree before despawning so we can restore on undo.
        self.serialized = Some(
            world
                .serialize_prefab(self.entity)
                .map_err(|e| EditActionError::Custom(e.to_string()))?,
        );
        crate::std::hierarchy::despawn_recursive(world, self.entity);
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        let serialized = self
            .serialized
            .as_ref()
            .ok_or_else(|| EditActionError::Custom("no backup to restore".into()))?;
        let entities = world
            .deserialize_prefab(serialized)
            .map_err(|e| EditActionError::Custom(e.to_string()))?;
        // Re-parent the root under the original parent if it's still alive.
        if let Some(parent) = self.parent
            && world.is_alive(parent)
            && !entities.is_empty()
        {
            crate::std::hierarchy::set_parent(world, entities[0], parent);
        }
        // Update stored entity to the newly spawned root so repeated undo/redo works.
        if let Some(&root) = entities.first() {
            self.entity = root;
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Delete entity"
    }
}

// ---------------------------------------------------------------------------
// Undoable reparent action
// ---------------------------------------------------------------------------

/// Reversible reparent action for the editor's undo/redo history.
///
/// - `new_parent: Some(e)` → `set_parent(entity, e)`
/// - `new_parent: None`    → `remove_parent(entity)` (make root)
#[derive(Debug)]
struct ReparentAction {
    entity: Entity,
    old_parent: Option<Entity>,
    new_parent: Option<Entity>,
}

impl EditAction<World> for ReparentAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        apply_parent(world, self.entity, self.new_parent)
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        apply_parent(world, self.entity, self.old_parent)
    }

    fn description(&self) -> &str {
        "Reparent entity"
    }
}

/// Set or remove parent for an entity.
fn apply_parent(world: &mut World, entity: Entity, parent: Option<Entity>) -> EditActionResult {
    match parent {
        Some(p) => {
            if !world.is_alive(p) {
                return Err(EditActionError::TargetNotFound("parent despawned".into()));
            }
            crate::std::hierarchy::set_parent(world, entity, p);
        }
        None => {
            crate::std::hierarchy::remove_parent(world, entity);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Undoable spawn-prefab action
// ---------------------------------------------------------------------------

/// Reversible action that spawns entities from a serialized prefab.
///
/// - `apply()`: deserializes the prefab into new entities, optionally parenting
///   the root under `parent`.
/// - `undo()`: despawns the root (and all children recursively).
pub struct SpawnPrefabAction {
    serialized: crate::serialize::SerializedPrefab,
    parent: Option<Entity>,
    spawned_entities: Vec<Entity>,
}

impl SpawnPrefabAction {
    pub fn new(serialized: crate::serialize::SerializedPrefab, parent: Option<Entity>) -> Self {
        Self {
            serialized,
            parent,
            spawned_entities: Vec::new(),
        }
    }
}

impl std::fmt::Debug for SpawnPrefabAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnPrefabAction")
            .field("parent", &self.parent)
            .field("spawned_count", &self.spawned_entities.len())
            .finish()
    }
}

impl EditAction<World> for SpawnPrefabAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let entities = world
            .deserialize_prefab(&self.serialized)
            .map_err(|e| EditActionError::Custom(e.to_string()))?;
        if let Some(parent) = self.parent
            && world.is_alive(parent)
            && !entities.is_empty()
        {
            crate::std::hierarchy::set_parent(world, entities[0], parent);
        }
        self.spawned_entities = entities;
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if let Some(&root) = self.spawned_entities.first()
            && world.is_alive(root)
        {
            crate::std::hierarchy::despawn_recursive(world, root);
        }
        self.spawned_entities.clear();
        Ok(())
    }

    fn description(&self) -> &str {
        "Spawn prefab"
    }
}
