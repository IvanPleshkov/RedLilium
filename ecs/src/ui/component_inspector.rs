//! Component inspector panel for a selected entity.

use crate::World;

use super::InspectorState;

/// Render the component inspector for the currently selected entity.
///
/// Lists all inspector-registered components attached to the entity, with
/// editable fields via the `Component` trait's `inspect_ui`. Each component
/// header has a right-click context menu to remove it. An "Add Component"
/// button at the bottom opens a popup listing components the entity doesn't
/// have yet.
///
/// The caller is responsible for placing this in whatever container they want
/// (dock tab, side panel, window, etc.).
pub fn show_component_inspector(ui: &mut egui::Ui, world: &mut World, state: &mut InspectorState) {
    // Apply deferred actions from world inspector (e.g. drag-and-drop reparenting)
    state.apply_pending_actions(world);

    let selected = match state.selected {
        Some(e) if world.is_alive(e) => e,
        _ => {
            ui.label("No entity selected.");
            ui.small("Select an entity in the World Inspector.");
            return;
        }
    };

    ui.heading(format!(
        "Entity({}@{})",
        selected.index(),
        selected.spawn_tick()
    ));

    // Enabled/Disabled toggle
    let is_disabled = world.is_disabled(selected);
    let mut enabled = !is_disabled;
    if ui.checkbox(&mut enabled, "Enabled").changed() {
        if enabled {
            crate::std::hierarchy::enable(world, selected);
        } else {
            crate::std::hierarchy::disable(world, selected);
        }
    }

    ui.separator();

    // Collect components this entity has
    let present: Vec<&str> = world
        .inspectable_components_of(selected)
        .into_iter()
        .collect();

    // Track which components to remove after iteration
    let mut to_remove: Vec<&str> = Vec::new();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if present.is_empty() {
                ui.label("No registered components.");
            }

            for comp_name in &present {
                let header = egui::CollapsingHeader::new(egui::RichText::new(*comp_name).strong())
                    .default_open(true);

                let header_resp = header.show(ui, |ui| {
                    world.inspect_by_name(selected, comp_name, ui);
                });

                header_resp.header_response.context_menu(|ui| {
                    if ui.button("Remove Component").clicked() {
                        to_remove.push(comp_name);
                        ui.close();
                    }
                });

                ui.separator();
            }

            // Add component button
            ui.horizontal(|ui| {
                if ui.button("+ Add Component").clicked() {
                    state.add_component_open = !state.add_component_open;
                }
            });

            if state.add_component_open {
                let missing = world.addable_components_of(selected);
                if missing.is_empty() {
                    ui.label("All components already attached.");
                } else {
                    egui::Frame::new()
                        .inner_margin(egui::Margin::same(4))
                        .corner_radius(egui::CornerRadius::same(4))
                        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
                        .show(ui, |ui| {
                            for name in &missing {
                                if ui.button(*name).clicked() {
                                    world.insert_default_by_name(selected, name);
                                    state.add_component_open = false;
                                }
                            }
                        });
                }
            }
        });

    // Apply deferred removals
    for name in to_remove {
        world.remove_by_name(selected, name);
    }
}
