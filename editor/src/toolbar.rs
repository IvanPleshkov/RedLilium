/// Editor play state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayState {
    Editing,
    Playing,
    Paused,
}

/// Draw the top toolbar strip. Returns the updated play state.
pub fn draw_toolbar(ctx: &egui::Context, play_state: PlayState) -> PlayState {
    let mut new_state = play_state;

    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        ui.horizontal_centered(|ui| {
            // Center the buttons
            let available = ui.available_width();
            ui.add_space((available / 2.0 - 60.0).max(0.0));

            match play_state {
                PlayState::Editing => {
                    if ui.button("▶ Play").clicked() {
                        new_state = PlayState::Playing;
                    }
                }
                PlayState::Playing => {
                    if ui.button("⏸ Pause").clicked() {
                        new_state = PlayState::Paused;
                    }
                    if ui.button("⏹ Stop").clicked() {
                        new_state = PlayState::Editing;
                    }
                }
                PlayState::Paused => {
                    if ui.button("▶ Resume").clicked() {
                        new_state = PlayState::Playing;
                    }
                    if ui.button("⏹ Stop").clicked() {
                        new_state = PlayState::Editing;
                    }
                }
            }
        });
    });

    new_state
}
