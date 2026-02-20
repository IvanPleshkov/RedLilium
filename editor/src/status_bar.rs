/// Draw the bottom status bar strip.
pub fn draw_status_bar(ctx: &egui::Context, fps: f32) {
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("{fps:.0} FPS"));
            });
        });
    });
}
