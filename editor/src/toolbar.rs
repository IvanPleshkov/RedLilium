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
/// Used inside the titlebar / menu bar. Returns an action if a button was
/// clicked, or None if no button was clicked.
pub fn draw_play_controls(ui: &mut egui::Ui, play_state: PlayState) -> Option<PlayAction> {
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
            if ui.button("\u{25B6} Resume").clicked() {
                return Some(PlayAction::Resume);
            }
            if ui.button("\u{23F9} Stop").clicked() {
                return Some(PlayAction::Stop);
            }
            None
        }
    }
}

/// Tier-1 build status for the titlebar indicator (ADR-037) — a compact
/// mirror of `remote_commands::GameStatus`'s behavior-reload fields.
///
/// Consumed only by the macOS custom-titlebar toolbar, so it reads as dead
/// code on other platforms; allowed there rather than `cfg`-gated so the item
/// stays available if the indicator moves cross-platform.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub struct GameBuildStatus {
    pub stale: bool,
    pub rebuilding: bool,
    pub restart_required: bool,
    pub schema_diverged: bool,
}

/// What the user asked for through the game-build indicator.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameBuildAction {
    /// Rebuild the game cdylib (Tier 1).
    Rebuild,
    /// Exec-restart the editor with session carry (Tier 2).
    Restart,
}

/// Draw the game-build indicator: a rebuild button when sources went stale,
/// progress while cargo runs, a restart button when only a process restart
/// helps, and the schema-divergence warning.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn draw_game_build_indicator(
    ui: &mut egui::Ui,
    status: GameBuildStatus,
) -> Option<GameBuildAction> {
    if status.restart_required {
        if ui
            .button("\u{27F3} Restart editor")
            .on_hover_text(
                "The rebuilt game was compiled against a changed engine (fingerprint \
                 mismatch) — restart the editor to pick everything up (scene and camera \
                 pose carry over, undo history is cleared)",
            )
            .clicked()
        {
            return Some(GameBuildAction::Restart);
        }
        return None;
    }
    if status.rebuilding {
        ui.spinner();
        ui.colored_label(egui::Color32::from_rgb(200, 180, 50), "building game…");
        return None;
    }
    let mut action = None;
    if status.stale
        && ui
            .button("\u{27F3} Rebuild game")
            .on_hover_text("Game sources changed — rebuild the cdylib for the next Play")
            .clicked()
    {
        action = Some(GameBuildAction::Rebuild);
    }
    if status.schema_diverged {
        ui.colored_label(egui::Color32::from_rgb(200, 180, 50), "\u{26A0} schemas")
            .on_hover_text(
                "Behavior module's component schemas diverged from the editor's static image: \
                 play runs the new code, authoring the changed fields needs an editor restart",
            );
    }
    action
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
