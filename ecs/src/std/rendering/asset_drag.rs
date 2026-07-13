//! Drag-and-drop of assets from the browser into reference fields.
//!
//! One mechanic for every field shape: the browser sets an [`AssetDragPayload`]
//! when a registered asset file is dragged, and any reference field renders
//! itself through [`asset_drop_target`] — a component's
//! [`AssetRef`](redlilium_assets::AssetRef) field (via `ComponentField`), a
//! material's texture property, a record reference. The payload rides in the
//! egui context, so field widgets need no World/DB plumbing; kind checking is
//! against the DB record kind carried in the payload.

use redlilium_assets::Guid;

/// Payload set by the asset browser when a file is dragged. `asset` is present
/// only for files with a DB record — unregistered files still drag (for
/// directory moves) but no reference field accepts them.
#[derive(Clone, Debug)]
pub struct AssetDragPayload {
    /// Full VFS path (`mount/rel`) — directory-move drops resolve it.
    pub vfs_path: String,
    /// The registered asset's identity: guid + record kind.
    pub asset: Option<(Guid, String)>,
}

/// A reference-field widget that accepts a dragged asset of `accept_kind`:
/// shows `current` (weak + "loading…" while unresolved), highlights while a
/// matching asset hovers (error-colored for a kind mismatch), and returns the
/// dropped guid on release.
pub fn asset_drop_target(
    ui: &mut egui::Ui,
    current: &str,
    loading: bool,
    accept_kind: &str,
) -> Option<Guid> {
    let response = egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(4, 2))
        .show(ui, |ui| {
            if loading {
                ui.weak(format!("{current} (loading…)"));
            } else {
                ui.monospace(current);
            }
        })
        .response;

    let payload = response.dnd_hover_payload::<AssetDragPayload>()?;
    let accepted = payload
        .asset
        .as_ref()
        .is_some_and(|(_, kind)| kind == accept_kind);
    if ui.input(|i| i.pointer.any_released()) {
        log::debug!(
            "dnd: release over target accepting '{accept_kind}': payload {:?} (accepted: {accepted})",
            payload.asset
        );
    }
    let stroke_color = if accepted {
        ui.visuals().selection.stroke.color
    } else {
        ui.visuals().error_fg_color
    };
    ui.painter().rect_stroke(
        response.rect,
        2.0,
        egui::Stroke::new(2.0_f32, stroke_color),
        egui::StrokeKind::Outside,
    );
    if !accepted {
        return None;
    }
    response
        .dnd_release_payload::<AssetDragPayload>()
        .and_then(|p| p.asset.as_ref().map(|(guid, _)| *guid))
}
