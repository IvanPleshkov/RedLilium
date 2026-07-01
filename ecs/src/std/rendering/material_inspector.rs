//! Inspector UI for the [`MeshRenderer`] component.
//!
//! Lists each primitive with its mesh label and the bound material-instance asset
//! (read-only). Editing a material instance's property values is done through the
//! asset browser (the instance record's settings), the same way vertex layouts are
//! edited — see `docs/MATERIAL_ASSETS.md`.

use super::MeshRenderer;
use crate::{Entity, InspectResult, World};

/// Custom inspector for [`MeshRenderer`]: lists each primitive with its mesh label
/// and the bound material-instance asset guid (read-only).
pub(crate) fn inspect_mesh_renderer_ui(
    world: &World,
    entity: Entity,
    ui: &mut egui::Ui,
) -> InspectResult {
    let renderer = world.get::<MeshRenderer>(entity)?;
    let single = renderer.primitives.len() == 1;

    for (index, primitive) in renderer.primitives.iter().enumerate() {
        egui::CollapsingHeader::new(format!("Primitive {index}"))
            .default_open(single)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("mesh");
                    match primitive.mesh().and_then(|m| m.label().map(str::to_owned)) {
                        Some(label) => ui.label(format!("Mesh: {label}")),
                        None => ui.weak("Mesh (loading…)"),
                    };
                });
                ui.horizontal(|ui| {
                    ui.label("material");
                    let guid = primitive.material_source.guid;
                    match primitive.material() {
                        Some(_) => ui.label(format!("{guid:?}")),
                        None => ui.weak(format!("{guid:?} (loading…)")),
                    };
                });
            });
    }

    // Property editing lives in the asset browser (the instance record's settings),
    // so this component inspector produces no edit actions.
    None
}
