//! Inspector for the asset selected in the browser: shows its DB record and
//! delegates editing of the per-kind `settings` to ECS
//! ([`inspect_asset_settings`](redlilium_ecs::inspect_asset_settings)) — the
//! editor knows nothing about specific asset kinds.
//!
//! Edits are never applied directly: they become
//! [`SetAssetSettingsAction`](redlilium_ecs::SetAssetSettingsAction) /
//! [`SetAssetReferenceAction`](redlilium_ecs::SetAssetReferenceAction)s pushed
//! to the [`ActionQueue`] — the same undo/redo path component edits take. The
//! actions feed hot reload and mark the mount dirty for persistence.

use redlilium_assets::{AssetDb, AssetPath};
use redlilium_core::abstract_editor::{ActionQueue, EditAction};
use redlilium_ecs::{SetAssetReferenceAction, SetAssetSettingsAction, World};

/// Render the asset at `source/path`.
pub fn show_asset_inspector(ui: &mut egui::Ui, world: &mut World, source: &str, path: &str) {
    let asset_path = AssetPath::new(source, path);
    let Some(guid) = world.resource::<AssetDb>().guid_of(&asset_path) else {
        ui.heading(path);
        ui.weak("Not a registered asset (no DB record).");
        return;
    };

    // Snapshot the record fields so we don't hold the DB borrow while editing.
    let (kind, settings, references) = {
        let db = world.resource::<AssetDb>();
        let record = db.record(&guid).expect("guid resolved above");
        (
            record.kind.clone(),
            record.settings.clone(),
            record.references.clone(),
        )
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

    let mut actions: Vec<Box<dyn EditAction<World>>> = Vec::new();

    // Named record references (e.g. a mesh's "layout") — drop targets where the
    // role has a known accepted kind (knowledge lives in ECS next to the loaders).
    for (role, target) in &references {
        ui.horizontal(|ui| {
            ui.label(format!("ref · {role}"));
            let display = world
                .resource::<AssetDb>()
                .record(target)
                .map(|r| r.path.path.clone())
                .unwrap_or_else(|| format!("{target:?}"));
            match redlilium_ecs::reference_accepted_kind(&kind, role) {
                Some(accept) => {
                    if let Some(dropped) =
                        redlilium_ecs::asset_drop_target(ui, &display, false, accept)
                    {
                        actions.push(Box::new(SetAssetReferenceAction {
                            guid,
                            role: role.clone(),
                            old: Some(*target),
                            new: Some(dropped),
                        }));
                    }
                }
                None => {
                    ui.monospace(display);
                }
            }
        });
    }
    ui.separator();

    // Per-kind editing lives in ECS (next to the loaders); the editor is generic.
    if let Some(new_settings) =
        redlilium_ecs::inspect_asset_settings(&kind, settings.as_deref(), ui, world)
    {
        actions.push(Box::new(SetAssetSettingsAction {
            guid,
            old: settings,
            new: Some(new_settings),
        }));
    }

    // Dispatch through the ActionQueue → history (undoable), mirroring the
    // component inspector; apply directly only if no queue exists.
    if !actions.is_empty() {
        if world.has_resource::<ActionQueue<World>>() {
            let queue = world.resource::<ActionQueue<World>>();
            for action in actions {
                queue.push(action);
            }
        } else {
            for mut action in actions {
                if let Err(e) = action.apply(world) {
                    log::warn!("asset edit failed: {e}");
                }
            }
        }
    }
}
