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
                    crate::theme::SUCCESS
                } else {
                    crate::theme::TEXT_MUTED
                }),
        );
        ui.separator();
        ui.label(
            egui::RichText::new(format!("Redo: {redo_count}"))
                .monospace()
                .color(if history.can_redo() {
                    crate::theme::INFO
                } else {
                    crate::theme::TEXT_MUTED
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
                                .color(crate::theme::INFO),
                        );
                        ui.label(
                            egui::RichText::new(*desc)
                                .monospace()
                                .color(crate::theme::TEXT_SECONDARY),
                        );
                    });
                }
            }

            // Current position marker
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("▸ current")
                        .monospace()
                        .color(crate::theme::WARNING),
                );
            });

            // Undo stack (bottom — most recent action first)
            for desc in history.undo_descriptions() {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.label(
                        egui::RichText::new("UNDO")
                            .monospace()
                            .color(crate::theme::SUCCESS),
                    );
                    ui.label(egui::RichText::new(desc).monospace());
                });
            }
        });
}
