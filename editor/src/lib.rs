//! # RedLilium Editor
//!
//! The editor as a **library** (ADR-037): a game project owns its editor
//! binary and launches it with [`run`], statically linking its
//! [`Plugin`](redlilium_runtime::Plugin). The editing world hosts the game's
//! type registrations from the static image, so authoring (inspector, scene
//! save/load, undo) never depends on a dylib; Play boots a separate game
//! world from the same plugin (ADR-036).
//!
//! ```ignore
//! // the game's editor binary, in its entirety
//! fn main() {
//!     redlilium_editor::run(CarGamePlugin);
//! }
//! ```
//!
//! The engine repo keeps a plain `redlilium-editor` binary ([`run_without_game`])
//! for engine development; it can still host a foreign game cdylib via the
//! `REDLILIUM_GAME=<path>` override (ADR-020 hosting, fingerprint-gated).
//!
//! `REDLILIUM_HEADLESS=1` selects the headless shell (no window/swapchain,
//! frames tick on demand from the remote channel — docs/REMOTE.md) for either
//! entry point.

// The trait solver recurses deeply proving Send/Sync for resources holding
// wgpu-backed types (e.g. the texture manager's resident cache); the default
// limit of 128 overflows.
#![recursion_limit = "256"]

mod asset_browser;
mod asset_inspector;
mod background_vfs;
mod behavior_reload;
mod console;
mod core;
mod dock;
mod editor;
mod fs_watcher;
mod game_host;
mod gestures;
mod gizmo_system;
#[cfg(test)]
mod golden;
mod gpu_stats_panel;
mod headless;
mod history_panel;
mod log_capture;
mod menu;
mod play;
mod project;
mod remote_commands;
mod scene_view;
mod session;
mod status_bar;
mod theme;
mod tool_system;
mod toolbar;

use redlilium_app::{App, AppArgs, DefaultAppArgs};

/// Launch the editor with `game` statically hosted: its `register_types`
/// feed the editing world (authoring), its full composition boots play
/// worlds (ADR-036). Blocks until the editor exits.
///
/// Shorthand for [`hosting`]`(game).run()` — use [`hosting`] to opt into
/// Tier-1 behavior reload (ADR-037).
pub fn run(game: impl redlilium_runtime::Plugin + 'static) {
    hosting(game).run();
}

/// Launch the editor with no statically linked game (engine development).
/// A game cdylib may still be hosted via `REDLILIUM_GAME=<path>`.
pub fn run_without_game() {
    launch(None, None);
}

/// Configure the editor around a statically hosted `game` (ADR-037).
pub fn hosting(game: impl redlilium_runtime::Plugin + 'static) -> EditorLaunch {
    EditorLaunch {
        game: Box::new(game),
        behavior_reload: None,
    }
}

/// Builder for a game-owned editor binary — see [`hosting`].
pub struct EditorLaunch {
    game: Box<dyn redlilium_runtime::Plugin>,
    behavior_reload: Option<behavior_reload::BehaviorReloadSpec>,
}

impl EditorLaunch {
    /// Enable Tier-1 behavior reload (ADR-037): the editor watches the game
    /// package's sources (stale marker), rebuilds its cdylib on request with
    /// the drift-free fixed invocation, and hot-swaps it for play worlds.
    /// `game_package` is the cargo package name (e.g. `"car-game"`); the
    /// package must keep `crate-type = ["cdylib", "rlib"]` and cargo's
    /// default lib-name transform.
    pub fn behavior_reload(mut self, game_package: impl Into<String>) -> Self {
        self.behavior_reload = Some(behavior_reload::BehaviorReloadSpec {
            game_package: game_package.into(),
        });
        self
    }

    /// Launch (blocks until the editor exits).
    pub fn run(self) {
        launch(Some(self.game), self.behavior_reload);
    }
}

fn launch(
    game: Option<Box<dyn redlilium_runtime::Plugin>>,
    behavior: Option<behavior_reload::BehaviorReloadSpec>,
) {
    log_capture::install();
    // Headless shell: no window/swapchain, frames tick on demand from the
    // remote channel (docs/REMOTE.md).
    if std::env::var("REDLILIUM_HEADLESS").is_ok_and(|v| v == "1") {
        headless::run(game, behavior);
    } else {
        let args = DefaultAppArgs::parse()
            .with_title_str("RedLilium Editor")
            .with_custom_titlebar(true);
        App::run(editor::Editor::with_game(game, behavior), args);
    }
    // Tier-2 exec-restart (ADR-037): by now the loop has exited and every
    // world, GPU resource, and mapped game image is torn down — the one
    // safe point to replace the process image. The successor picks up the
    // session carry written by the shell.
    if session::restart_requested() {
        session::exec_restart();
    }
}
