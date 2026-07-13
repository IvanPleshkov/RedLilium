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

/// Draw play mode state indicator badge (showing "PLAYING" or "PAUSED").
///
/// Returns nothing; for display only. Empty during Editing mode.
pub fn draw_play_mode_indicator(ui: &mut egui::Ui, play_state: PlayState) {
    match play_state {
        PlayState::Editing => {
            // No indicator during Edit mode
        }
        PlayState::Playing => {
            ui.colored_label(egui::Color32::from_rgb(200, 50, 50), "● PLAYING");
        }
        PlayState::Paused => {
            ui.colored_label(egui::Color32::from_rgb(200, 180, 50), "⏸ PAUSED");
        }
    }
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

/// Draw the gizmo mode switch (W/E/R shortcuts do the same); returns the
/// newly selected mode when the user clicks a different one.
pub fn draw_gizmo_mode_controls(
    ui: &mut egui::Ui,
    current: redlilium_gizmo::GizmoMode,
) -> Option<redlilium_gizmo::GizmoMode> {
    use redlilium_gizmo::GizmoMode;
    let mut selected = None;
    for (label, mode, hint) in [
        ("Move", GizmoMode::Translate, "Translate (W)"),
        ("Rot", GizmoMode::Rotate, "Rotate (E)"),
        ("Scale", GizmoMode::Scale, "Scale (R)"),
    ] {
        let active = current == mode;
        let button = egui::Button::new(label).selected(active);
        if ui.add(button).on_hover_text(hint).clicked() && !active {
            selected = Some(mode);
        }
    }
    selected
}
