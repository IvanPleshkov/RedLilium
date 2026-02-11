//! Component registry for the inspector UI.
//!
//! Stores metadata about component types so the inspector can enumerate,
//! add, and remove components on entities at runtime.

use std::any::TypeId;
use std::collections::BTreeMap;

use redlilium_ecs::{Entity, World};

/// Type-erased operations for a single component type.
struct ComponentEntry {
    /// Check if an entity has this component.
    has: fn(&World, Entity) -> bool,
    /// Remove this component from an entity. Returns true if removed.
    remove: fn(&mut World, Entity) -> bool,
    /// Insert a default instance of this component on an entity (None if no Default).
    insert_default: Option<fn(&mut World, Entity)>,
    /// Show this component's fields in egui. Returns true if a remove was requested.
    inspect: fn(&mut World, Entity, &mut egui::Ui) -> bool,
}

/// Registry of inspectable component types.
///
/// Register component types during setup. The inspector uses this to
/// enumerate which components an entity has, render their fields,
/// and provide add/remove functionality.
pub struct ComponentRegistry {
    /// Ordered by name for consistent UI display.
    entries: BTreeMap<&'static str, ComponentEntry>,
    /// Maps TypeId → name for quick lookup.
    type_to_name: std::collections::HashMap<TypeId, &'static str>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            type_to_name: std::collections::HashMap::new(),
        }
    }

    /// Register a component type that implements the `Component` reflection trait
    /// and `Default`.
    ///
    /// The component must already be registered in the World via
    /// `world.register_component::<T>()`.
    pub fn register<T>(&mut self, name: &'static str)
    where
        T: redlilium_ecs::Component + Default + Send + Sync + 'static,
    {
        let entry = ComponentEntry {
            has: |world, entity| world.get::<T>(entity).is_some(),
            remove: |world, entity| world.remove::<T>(entity).is_some(),
            insert_default: Some(|world: &mut World, entity: Entity| {
                let _ = world.insert(entity, T::default());
            }),
            inspect: |world, entity, ui| inspect_component::<T>(world, entity, ui),
        };
        self.type_to_name.insert(TypeId::of::<T>(), name);
        self.entries.insert(name, entry);
    }

    /// Register a component type for inspection only (no Default required).
    ///
    /// The component can be viewed and removed, but cannot be added via the UI.
    pub fn register_readonly<T>(&mut self, name: &'static str)
    where
        T: redlilium_ecs::Component + Send + Sync + 'static,
    {
        let entry = ComponentEntry {
            has: |world, entity| world.get::<T>(entity).is_some(),
            remove: |world, entity| world.remove::<T>(entity).is_some(),
            insert_default: None,
            inspect: |world, entity, ui| inspect_component::<T>(world, entity, ui),
        };
        self.type_to_name.insert(TypeId::of::<T>(), name);
        self.entries.insert(name, entry);
    }

    /// Iterate over all registered component names.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.entries.keys().copied()
    }

    /// Check if entity has a registered component by name.
    pub fn has(&self, world: &World, entity: Entity, name: &str) -> bool {
        self.entries
            .get(name)
            .is_some_and(|e| (e.has)(world, entity))
    }

    /// Get all component names that this entity has.
    pub fn components_of(&self, world: &World, entity: Entity) -> Vec<&'static str> {
        self.entries
            .iter()
            .filter(|(_, e)| (e.has)(world, entity))
            .map(|(name, _)| *name)
            .collect()
    }

    /// Get all component names that this entity does NOT have and can be added.
    pub fn missing_components_of(&self, world: &World, entity: Entity) -> Vec<&'static str> {
        self.entries
            .iter()
            .filter(|(_, e)| e.insert_default.is_some() && !(e.has)(world, entity))
            .map(|(name, _)| *name)
            .collect()
    }

    /// Remove a component by name from an entity.
    pub fn remove_by_name(&self, world: &mut World, entity: Entity, name: &str) -> bool {
        self.entries
            .get(name)
            .is_some_and(|e| (e.remove)(world, entity))
    }

    /// Insert a default component by name on an entity.
    pub fn insert_default_by_name(&self, world: &mut World, entity: Entity, name: &str) {
        if let Some(e) = self.entries.get(name)
            && let Some(insert_fn) = e.insert_default
        {
            insert_fn(world, entity);
        }
    }

    /// Show the inspector UI for a component by name. Returns true if removal was requested.
    pub fn inspect_by_name(
        &self,
        world: &mut World,
        entity: Entity,
        name: &str,
        ui: &mut egui::Ui,
    ) -> bool {
        if let Some(e) = self.entries.get(name) {
            (e.inspect)(world, entity, ui)
        } else {
            false
        }
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Inspect a single component on an entity. Returns true if removal was requested.
fn inspect_component<T>(world: &mut World, entity: Entity, ui: &mut egui::Ui) -> bool
where
    T: redlilium_ecs::Component + 'static,
{
    let mut remove_requested = false;

    let Some(comp) = world.get_mut::<T>(entity) else {
        return false;
    };

    let infos = comp.field_infos().to_vec();
    let comp_name = comp.component_name();

    let header =
        egui::CollapsingHeader::new(egui::RichText::new(comp_name).strong()).default_open(true);

    let header_resp = header.show(ui, |ui| {
        for info in &infos {
            // Re-borrow mutably for each field edit
            let Some(comp) = world.get_mut::<T>(entity) else {
                return;
            };
            show_field(comp, info, ui);
        }
    });

    // Right-click context menu on the header for removal
    header_resp.header_response.context_menu(|ui| {
        if ui.button("Remove Component").clicked() {
            remove_requested = true;
            ui.close();
        }
    });

    remove_requested
}

/// Render a single field editor based on its FieldKind.
fn show_field<T: redlilium_ecs::Component>(
    comp: &mut T,
    info: &redlilium_ecs::FieldInfo,
    ui: &mut egui::Ui,
) {
    use redlilium_ecs::FieldKind;

    match info.kind {
        FieldKind::F32 => {
            if let Some(val) = comp
                .field_mut(info.name)
                .and_then(|f| f.downcast_mut::<f32>())
            {
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    ui.add(egui::DragValue::new(val).speed(0.01));
                });
            }
        }
        FieldKind::U8 => {
            if let Some(val) = comp
                .field_mut(info.name)
                .and_then(|f| f.downcast_mut::<u8>())
            {
                let mut v = *val as i32;
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    if ui
                        .add(egui::DragValue::new(&mut v).range(0..=255))
                        .changed()
                    {
                        *val = v as u8;
                    }
                });
            }
        }
        FieldKind::U32 => {
            if let Some(val) = comp
                .field_mut(info.name)
                .and_then(|f| f.downcast_mut::<u32>())
            {
                let mut v = *val as i64;
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    if ui
                        .add(egui::DragValue::new(&mut v).range(0..=u32::MAX as i64))
                        .changed()
                    {
                        *val = v as u32;
                    }
                });
            }
        }
        FieldKind::I32 => {
            if let Some(val) = comp
                .field_mut(info.name)
                .and_then(|f| f.downcast_mut::<i32>())
            {
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    ui.add(egui::DragValue::new(val));
                });
            }
        }
        FieldKind::Vec2 => {
            show_vec2_field(comp, info, ui);
        }
        FieldKind::Vec3 => {
            show_vec3_field(comp, info, ui);
        }
        FieldKind::Vec4 => {
            show_vec4_field(comp, info, ui);
        }
        FieldKind::Quat => {
            show_quat_field(comp, info, ui);
        }
        FieldKind::Mat4 => {
            // Mat4 shown as read-only
            ui.horizontal(|ui| {
                ui.label(info.name);
                ui.label("(matrix)");
            });
        }
        FieldKind::StringId => {
            if let Some(val) = comp
                .field(info.name)
                .and_then(|f| f.downcast_ref::<redlilium_ecs::StringId>())
            {
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    ui.label(format!("StringId({})", val.0));
                });
            }
        }
        FieldKind::Bool => {
            if let Some(val) = comp
                .field_mut(info.name)
                .and_then(|f| f.downcast_mut::<bool>())
            {
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    ui.checkbox(val, "");
                });
            }
        }
        FieldKind::F64 => {
            if let Some(val) = comp
                .field_mut(info.name)
                .and_then(|f| f.downcast_mut::<f64>())
            {
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    ui.add(egui::DragValue::new(val).speed(0.01));
                });
            }
        }
        FieldKind::U64 => {
            if let Some(val) = comp
                .field_mut(info.name)
                .and_then(|f| f.downcast_mut::<u64>())
            {
                // Display as read-only for values > i64::MAX, editable otherwise
                let mut v = *val as i64;
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    if ui
                        .add(egui::DragValue::new(&mut v).range(0..=i64::MAX))
                        .changed()
                    {
                        *val = v as u64;
                    }
                });
            }
        }
        FieldKind::Usize => {
            if let Some(val) = comp
                .field_mut(info.name)
                .and_then(|f| f.downcast_mut::<usize>())
            {
                let mut v = *val as i64;
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    if ui
                        .add(egui::DragValue::new(&mut v).range(0..=i64::MAX))
                        .changed()
                    {
                        *val = v as usize;
                    }
                });
            }
        }
        FieldKind::String => {
            if let Some(val) = comp
                .field_mut(info.name)
                .and_then(|f| f.downcast_mut::<String>())
            {
                ui.horizontal(|ui| {
                    ui.label(info.name);
                    ui.text_edit_singleline(val);
                });
            }
        }
        FieldKind::Opaque => {
            ui.horizontal(|ui| {
                ui.label(info.name);
                ui.weak(format!("({})", info.type_name));
            });
        }
    }
}

fn show_vec2_field<T: redlilium_ecs::Component>(
    comp: &mut T,
    info: &redlilium_ecs::FieldInfo,
    ui: &mut egui::Ui,
) {
    // nalgebra Vector2<f32> is [f32; 2] under the hood but we access via Any
    // The type is nalgebra::Vector2<f32> which we can try to downcast
    use redlilium_core::math::Vec2;
    if let Some(val) = comp
        .field_mut(info.name)
        .and_then(|f| f.downcast_mut::<Vec2>())
    {
        ui.horizontal(|ui| {
            ui.label(info.name);
            ui.add(egui::DragValue::new(&mut val.x).speed(0.01).prefix("x: "));
            ui.add(egui::DragValue::new(&mut val.y).speed(0.01).prefix("y: "));
        });
    }
}

fn show_vec3_field<T: redlilium_ecs::Component>(
    comp: &mut T,
    info: &redlilium_ecs::FieldInfo,
    ui: &mut egui::Ui,
) {
    use redlilium_core::math::Vec3;
    if let Some(val) = comp
        .field_mut(info.name)
        .and_then(|f| f.downcast_mut::<Vec3>())
    {
        ui.horizontal(|ui| {
            ui.label(info.name);
            ui.add(egui::DragValue::new(&mut val.x).speed(0.01).prefix("x: "));
            ui.add(egui::DragValue::new(&mut val.y).speed(0.01).prefix("y: "));
            ui.add(egui::DragValue::new(&mut val.z).speed(0.01).prefix("z: "));
        });
    }
}

fn show_vec4_field<T: redlilium_ecs::Component>(
    comp: &mut T,
    info: &redlilium_ecs::FieldInfo,
    ui: &mut egui::Ui,
) {
    use redlilium_core::math::Vec4;
    if let Some(val) = comp
        .field_mut(info.name)
        .and_then(|f| f.downcast_mut::<Vec4>())
    {
        ui.horizontal(|ui| {
            ui.label(info.name);
            ui.add(egui::DragValue::new(&mut val.x).speed(0.01).prefix("x: "));
            ui.add(egui::DragValue::new(&mut val.y).speed(0.01).prefix("y: "));
            ui.add(egui::DragValue::new(&mut val.z).speed(0.01).prefix("z: "));
            ui.add(egui::DragValue::new(&mut val.w).speed(0.01).prefix("w: "));
        });
    }
}

fn show_quat_field<T: redlilium_ecs::Component>(
    comp: &mut T,
    info: &redlilium_ecs::FieldInfo,
    ui: &mut egui::Ui,
) {
    use redlilium_core::math::Quat;
    if let Some(val) = comp
        .field_mut(info.name)
        .and_then(|f| f.downcast_mut::<Quat>())
    {
        ui.horizontal(|ui| {
            ui.label(info.name);
            // Show as xyzw (display only — editing quaternions directly is fragile)
            ui.label(format!(
                "[{:.3}, {:.3}, {:.3}, {:.3}]",
                val.coords.x, val.coords.y, val.coords.z, val.coords.w
            ));
        });
    }
}
