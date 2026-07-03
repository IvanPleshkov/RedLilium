//! Inspector UI for the [`MeshRenderer`] component.
//!
//! Lists each primitive with its mesh and bound material instance; both rows
//! are drop targets — a matching asset dragged from the browser replaces the
//! reference (an undoable component edit; the new ref loads asynchronously).
//! Editing a material instance's property values is done through the asset
//! browser (the instance record's settings) — see `docs/MATERIAL_ASSETS.md`.

use redlilium_assets::{AssetRef, AssetRefSource};

use super::MeshRenderer;
use super::asset_drag::asset_drop_target;
use super::loaders::{MaterialInstanceSource, MeshSource};
use crate::{Entity, InspectResult, World};

/// Custom inspector for [`MeshRenderer`]: each primitive's mesh and
/// material-instance references, editable by asset drag-and-drop.
pub(crate) fn inspect_mesh_renderer_ui(
    world: &World,
    entity: Entity,
    ui: &mut egui::Ui,
) -> InspectResult {
    let renderer = world.get::<MeshRenderer>(entity)?;
    let single = renderer.primitives.len() == 1;

    let mut edited = renderer.clone();
    let mut changed = false;
    for (index, primitive) in edited.primitives.iter_mut().enumerate() {
        egui::CollapsingHeader::new(format!("Primitive {index}"))
            .default_open(single)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("mesh");
                    let label = match primitive.mesh().and_then(|m| m.label().map(str::to_owned)) {
                        Some(label) => label,
                        None => format!("{:?}", primitive.mesh.source()),
                    };
                    let loading = primitive.mesh().is_none();
                    if let Some(guid) =
                        asset_drop_target(ui, &label, loading, <MeshSource as AssetRefSource>::KIND)
                    {
                        primitive.mesh = AssetRef::new(MeshSource::File(guid));
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("material");
                    let guid = primitive.material.source().guid;
                    let loading = primitive.material().is_none();
                    if let Some(dropped) = asset_drop_target(
                        ui,
                        &format!("{guid:?}"),
                        loading,
                        <MaterialInstanceSource as AssetRefSource>::KIND,
                    ) {
                        primitive.material =
                            AssetRef::new(MaterialInstanceSource { guid: dropped });
                        changed = true;
                    }
                });
            });
    }

    if changed {
        crate::set_component_actions(entity, renderer.clone(), edited)
    } else {
        None
    }
}
