/// Draw the bottom status bar strip. `tool` is the active viewport tool's
/// `(label, hint)` — the mode indicator that tells the user the editor is in
/// a tool mode and how to leave it.
pub fn draw_status_bar(ctx: &egui::Context, fps: f32, tool: Option<(String, String)>) {
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if let Some((label, hint)) = tool {
                ui.colored_label(crate::theme::ACCENT, format!("● {label}"));
                if !hint.is_empty() {
                    ui.weak(hint);
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("{fps:.0} FPS"));
            });
        });
    });
}
