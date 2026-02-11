//! Entity hierarchy tree view.

use redlilium_ecs::{Entity, World};

use crate::components::{Children, Name, Parent};

use super::InspectorState;

/// Show the world inspector window — a tree of all entities organized by hierarchy.
///
/// Root entities (those without a `Parent`) appear at the top level.
/// Children appear nested under their parent, expandable via a toggle.
///
/// Entities with a [`Name`] component display their name; others show `Entity(index:gen)`.
pub fn show_world_inspector(ctx: &egui::Context, world: &World, state: &mut InspectorState) {
    if !state.world_inspector_open {
        return;
    }

    let mut open = state.world_inspector_open;
    egui::Window::new("World Inspector")
        .default_pos([10.0, 250.0])
        .default_width(250.0)
        .resizable(true)
        .open(&mut open)
        .show(ctx, |ui| {
            // Filter input
            ui.horizontal(|ui| {
                ui.label("Filter:");
                ui.text_edit_singleline(&mut state.filter);
            });
            ui.separator();

            // Entity count
            ui.label(format!("Entities: {}", world.entity_count()));
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // Collect root entities (no Parent component)
                    let roots = collect_roots(world);

                    for entity in &roots {
                        show_entity_node(ui, world, *entity, state);
                    }
                });
        });
    state.world_inspector_open = open;
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
    let is_selected = state.selected == Some(entity);

    if has_children {
        // Expandable node
        let expanded = state.is_expanded(entity);
        let resp = ui.horizontal(|ui| {
            let toggle_text = if expanded { "v" } else { ">" };
            if ui.small_button(toggle_text).clicked() {
                state.toggle_expanded(entity);
            }
            let resp = ui.selectable_label(is_selected, &label);
            if resp.clicked() {
                state.selected = Some(entity);
            }
        });
        let _ = resp;

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
            let resp = ui.selectable_label(is_selected, &label);
            if resp.clicked() {
                state.selected = Some(entity);
            }
        });
    }
}

/// Generate a display label for an entity.
fn entity_label(world: &World, entity: Entity) -> String {
    if let Some(name) = world.get::<Name>(entity) {
        let s = name.as_str();
        if !s.is_empty() {
            return format!("{} [{}:{}]", s, entity.index(), entity.generation());
        }
    }
    format!("Entity({}:{})", entity.index(), entity.generation())
}
