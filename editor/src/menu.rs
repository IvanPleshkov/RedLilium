/// Draw the main menu bar.
pub fn draw_menu_bar(ctx: &egui::Context) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("RedLilium", |ui| {
                if ui.button("About RedLilium Editor").clicked() {
                    ui.close();
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui
                    .add(egui::Button::new("Undo").shortcut_text("Ctrl+Z"))
                    .clicked()
                {
                    ui.close();
                }
                if ui
                    .add(egui::Button::new("Redo").shortcut_text("Ctrl+Shift+Z"))
                    .clicked()
                {
                    ui.close();
                }
            });
        });
    });
}
