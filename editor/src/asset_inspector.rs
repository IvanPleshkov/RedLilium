//! Inspector for the asset selected in the browser: shows its DB record and
//! delegates editing of the per-kind `settings` to ECS
//! ([`inspect_asset_settings`](redlilium_ecs::inspect_asset_settings)) — the
//! editor knows nothing about specific asset kinds. Returns `true` when the
//! in-memory DB was edited (the caller persists the mount).

use redlilium_assets::{AssetDb, AssetPath};
use redlilium_ecs::World;

/// Render the asset at `source/path`. Returns whether the DB was edited.
pub fn show_asset_inspector(
    ui: &mut egui::Ui,
    world: &mut World,
    source: &str,
    path: &str,
) -> bool {
    let asset_path = AssetPath::new(source, path);
    let Some(guid) = world.resource::<AssetDb>().guid_of(&asset_path) else {
        ui.heading(path);
        ui.weak("Not a registered asset (no DB record).");
        return false;
    };

    // Snapshot the record fields so we don't hold the DB borrow while editing.
    let (kind, settings) = {
        let db = world.resource::<AssetDb>();
        let record = db.record(&guid).expect("guid resolved above");
        (record.kind.clone(), record.settings.clone())
    };

    ui.heading(path);
    ui.horizontal(|ui| {
        ui.label("kind");
        ui.monospace(&kind);
    });
    ui.horizontal(|ui| {
        ui.label("guid");
        ui.monospace(guid.0.to_string());
    });
    ui.separator();

    // Per-kind editing lives in ECS (next to the loaders); the editor is generic.
    match redlilium_ecs::inspect_asset_settings(&kind, settings.as_deref(), ui, world) {
        Some(new_settings) => {
            world
                .resource_mut::<AssetDb>()
                .set_settings(&guid, Some(new_settings));
            true
        }
        None => false,
    }
}
