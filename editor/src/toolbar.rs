/// Editor play state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayState {
    Editing,
    Playing,
    Paused,
}

/// Action requested by the play controls UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayAction {
    Play,
    Pause,
    Resume,
    Stop,
}

/// Draw the play/pause/stop controls inline in a horizontal UI region.
///
/// Used inside the titlebar / menu bar. Returns an action if a button was clicked,
/// or None if no button was clicked.
/// If `paused_due_to_panic` is true, the Resume button is disabled.
pub fn draw_play_controls(
    ui: &mut egui::Ui,
    play_state: PlayState,
    paused_due_to_panic: bool,
) -> Option<PlayAction> {
    match play_state {
        PlayState::Editing => {
            if ui.button("\u{25B6} Play").clicked() {
                Some(PlayAction::Play)
            } else {
                None
            }
        }
        PlayState::Playing => {
            if ui.button("\u{23F8} Pause").clicked() {
                return Some(PlayAction::Pause);
            }
            if ui.button("\u{23F9} Stop").clicked() {
                return Some(PlayAction::Stop);
            }
            None
        }
        PlayState::Paused => {
            let resume_btn =
                ui.add_enabled(!paused_due_to_panic, egui::Button::new("\u{25B6} Resume"));
            if resume_btn.clicked() {
                return Some(PlayAction::Resume);
            }
            if ui.button("\u{23F9} Stop").clicked() {
                return Some(PlayAction::Stop);
            }
            None
        }
    }
}
