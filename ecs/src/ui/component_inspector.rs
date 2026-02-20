//! Component inspector panel for a selected entity.

use redlilium_core::abstract_editor::{ActionQueue, EditAction, EditActionError, EditActionResult};

use crate::{Entity, World};

use super::InspectorState;

// ---------------------------------------------------------------------------
// Editor actions
// ---------------------------------------------------------------------------

/// Reversible action that toggles an entity's enabled/disabled state.
#[derive(Debug)]
struct SetEnabledAction {
    entity: Entity,
    enable: bool,
}

impl EditAction<World> for SetEnabledAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        if self.enable {
            crate::std::hierarchy::enable(world, self.entity);
        } else {
            crate::std::hierarchy::disable(world, self.entity);
        }
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        if self.enable {
            crate::std::hierarchy::disable(world, self.entity);
        } else {
            crate::std::hierarchy::enable(world, self.entity);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        if self.enable {
            "Enable entity"
        } else {
            "Disable entity"
        }
    }
}

/// Reversible action that toggles an entity's static flag.
#[derive(Debug)]
struct SetStaticAction {
    entity: Entity,
    mark_static: bool,
}

impl EditAction<World> for SetStaticAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        if self.mark_static {
            crate::std::hierarchy::mark_static(world, self.entity);
        } else {
            crate::std::hierarchy::unmark_static(world, self.entity);
        }
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        if self.mark_static {
            crate::std::hierarchy::unmark_static(world, self.entity);
        } else {
            crate::std::hierarchy::mark_static(world, self.entity);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        if self.mark_static {
            "Mark entity static"
        } else {
            "Unmark entity static"
        }
    }
}

/// Reversible action that adds a default component to an entity.
#[derive(Debug)]
struct AddComponentAction {
    entity: Entity,
    name: &'static str,
}

impl EditAction<World> for AddComponentAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        world.insert_default_by_name(self.entity, self.name);
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        world.remove_by_name(self.entity, self.name);
        Ok(())
    }

    fn description(&self) -> &str {
        self.name
    }
}

/// Reversible action that removes a component from an entity.
///
/// On apply, extracts (clones) the component value into `saved` before removing.
/// On undo, restores the saved value back onto the entity.
struct RemoveComponentAction {
    entity: Entity,
    name: &'static str,
    saved: Option<Box<dyn crate::prefab::ComponentBag>>,
}

impl std::fmt::Debug for RemoveComponentAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoveComponentAction")
            .field("name", &self.name)
            .finish()
    }
}

impl EditAction<World> for RemoveComponentAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        self.saved = world.extract_by_name(self.entity, self.name);
        world.remove_by_name(self.entity, self.name);
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }
        if let Some(bag) = self.saved.take() {
            let restore = bag.clone_box();
            self.saved = Some(bag);
            world.insert_bag(self.entity, restore);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        self.name
    }
}

// ---------------------------------------------------------------------------
// Component inspector UI
// ---------------------------------------------------------------------------

/// Render the component inspector for the currently selected entity.
///
/// Lists all inspector-registered components attached to the entity, with
/// editable fields rendered from an immutable component reference. All
/// operations (field edits, enable/disable, static toggle, add/remove
/// component) produce [`EditAction`]s pushed to the [`ActionQueue<World>`]
/// resource (if present) for undo/redo support. When no action queue is
/// present, actions are applied directly.
///
/// Each component header has a right-click context menu to remove it.
/// An "Add Component" button at the bottom opens a popup listing components
/// the entity doesn't have yet.
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

    // Collect editor actions produced by all inspector interactions
    let mut actions: Vec<Box<dyn EditAction<World>>> = Vec::new();

    // Enabled/Disabled toggle
    let is_disabled = world.is_disabled(selected);
    let mut enabled = !is_disabled;
    if ui.checkbox(&mut enabled, "Enabled").changed() {
        actions.push(Box::new(SetEnabledAction {
            entity: selected,
            enable: enabled,
        }));
    }

    // Static toggle
    let is_static = world.is_static(selected);
    let mut static_val = is_static;
    if ui.checkbox(&mut static_val, "Static").changed() {
        actions.push(Box::new(SetStaticAction {
            entity: selected,
            mark_static: static_val,
        }));
    }

    ui.separator();

    // Collect inspectable components (have full UI) and all components on this entity
    let inspectable: Vec<&str> = world
        .inspectable_components_of(selected)
        .into_iter()
        .collect();
    let all_type_names: Vec<&str> = world.all_component_names_of(selected);

    // Non-inspectable: components present on the entity but not registered with inspector UI
    let non_inspectable: Vec<&str> = all_type_names
        .iter()
        .copied()
        .filter(|full_name| {
            let short = short_type_name(full_name);
            !inspectable.contains(&short)
        })
        .collect();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if inspectable.is_empty() && non_inspectable.is_empty() {
                ui.label("No components.");
            }

            // Inspectable components — full UI
            for comp_name in &inspectable {
                let header = egui::CollapsingHeader::new(egui::RichText::new(*comp_name).strong())
                    .default_open(true);

                let header_resp = header.show(ui, |ui| {
                    // Inspect with &World — returns actions if fields were edited
                    if let Some(mut comp_actions) = world.inspect_by_name(selected, comp_name, ui) {
                        actions.append(&mut comp_actions);
                    }
                });

                header_resp.header_response.context_menu(|ui| {
                    if ui.button("Remove Component").clicked() {
                        actions.push(Box::new(RemoveComponentAction {
                            entity: selected,
                            name: comp_name,
                            saved: None,
                        }));
                        ui.close();
                    }
                });

                ui.separator();
            }

            // Non-inspectable components — header only (no editable UI)
            for full_name in &non_inspectable {
                let short = short_type_name(full_name);
                egui::CollapsingHeader::new(egui::RichText::new(short).weak())
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("No inspector UI available.")
                                .weak()
                                .italics(),
                        );
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
                                    actions.push(Box::new(AddComponentAction {
                                        entity: selected,
                                        name,
                                    }));
                                    state.add_component_open = false;
                                }
                            }
                        });
                }
            }
        });

    // Dispatch actions: push to ActionQueue if present, otherwise apply directly
    if !actions.is_empty() {
        if world.has_resource::<ActionQueue<World>>() {
            let queue = world.resource::<ActionQueue<World>>();
            for action in actions {
                queue.push(action);
            }
        } else {
            for mut action in actions {
                let _ = action.apply(world);
            }
        }
    }
}

/// Extracts the short type name from a fully-qualified Rust type path.
///
/// e.g. `"redlilium_graphics::material::MaterialInstance"` → `"MaterialInstance"`
fn short_type_name(full: &str) -> &str {
    full.rsplit("::").next().unwrap_or(full)
}
