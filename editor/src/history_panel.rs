use redlilium_core::abstract_editor::EditActionHistory;
use redlilium_ecs::World;

/// Displays the undo/redo action history for debugging.
pub fn show_history(ui: &mut egui::Ui, history: &EditActionHistory<World>) {
    let undo_count = history.undo_count();
    let redo_count = history.redo_count();

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("Undo: {undo_count}"))
                .monospace()
                .color(if history.can_undo() {
                    egui::Color32::from_rgb(120, 220, 120)
                } else {
                    egui::Color32::from_gray(100)
                }),
        );
        ui.separator();
        ui.label(
            egui::RichText::new(format!("Redo: {redo_count}"))
                .monospace()
                .color(if history.can_redo() {
                    egui::Color32::from_rgb(100, 180, 255)
                } else {
                    egui::Color32::from_gray(100)
                }),
        );
    });

    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Redo stack (top — these would be re-applied next)
            if redo_count > 0 {
                // Show redo entries in reverse so the next-to-redo is closest to the cursor
                let redo_descs: Vec<_> = history.redo_descriptions().collect();
                for desc in redo_descs.iter().rev() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.label(
                            egui::RichText::new("REDO")
                                .monospace()
                                .color(egui::Color32::from_rgb(100, 180, 255)),
                        );
                        ui.label(
                            egui::RichText::new(*desc)
                                .monospace()
                                .color(egui::Color32::from_gray(120)),
                        );
                    });
                }
            }

            // Current position marker
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("▸ current")
                        .monospace()
                        .color(egui::Color32::from_rgb(255, 200, 60)),
                );
            });

            // Undo stack (bottom — most recent action first)
            for desc in history.undo_descriptions() {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.label(
                        egui::RichText::new("UNDO")
                            .monospace()
                            .color(egui::Color32::from_rgb(120, 220, 120)),
                    );
                    ui.label(egui::RichText::new(desc).monospace());
                });
            }
        });
}
