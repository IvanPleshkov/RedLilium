/// Editor play state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayState {
    Editing,
    Playing,
    Paused,
}

/// Draw the play/pause/stop controls inline in a horizontal UI region.
///
/// Used inside the titlebar / menu bar. Returns the updated play state.
pub fn draw_play_controls(ui: &mut egui::Ui, play_state: PlayState) -> PlayState {
    let mut new_state = play_state;

    match play_state {
        PlayState::Editing => {
            if ui.button("\u{25B6} Play").clicked() {
                new_state = PlayState::Playing;
            }
        }
        PlayState::Playing => {
            if ui.button("\u{23F8} Pause").clicked() {
                new_state = PlayState::Paused;
            }
            if ui.button("\u{23F9} Stop").clicked() {
                new_state = PlayState::Editing;
            }
        }
        PlayState::Paused => {
            if ui.button("\u{25B6} Resume").clicked() {
                new_state = PlayState::Playing;
            }
            if ui.button("\u{23F9} Stop").clicked() {
                new_state = PlayState::Editing;
            }
        }
    }

    new_state
}
