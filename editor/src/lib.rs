//! # RedLilium Editor
//!
//! The editor as a **library** (ADR-033): a game project owns its editor
//! binary and launches it with [`run`], statically linking its
//! [`Plugin`](redlilium_runtime::Plugin). The editing world hosts the game's
//! type registrations from the static image, so authoring (inspector, scene
//! save/load, undo) never depends on a dylib; Play boots a separate game
//! world from the same plugin (ADR-032).
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
mod console;
mod core;
mod dock;
mod editor;
mod fs_watcher;
mod game_host;
mod gestures;
mod gizmo_system;
mod gpu_stats_panel;
mod headless;
mod history_panel;
mod log_capture;
mod menu;
mod play;
mod project;
mod remote_commands;
mod scene_view;
mod status_bar;
mod theme;
mod toolbar;

use redlilium_app::{App, AppArgs, DefaultAppArgs};

/// Launch the editor with `game` statically hosted: its `register_types`
/// feed the editing world (authoring), its full composition boots play
/// worlds (ADR-032). Blocks until the editor exits.
pub fn run(game: impl redlilium_runtime::Plugin + 'static) {
    launch(Some(Box::new(game)));
}

/// Launch the editor with no statically linked game (engine development).
/// A game cdylib may still be hosted via `REDLILIUM_GAME=<path>`.
pub fn run_without_game() {
    launch(None);
}

fn launch(game: Option<Box<dyn redlilium_runtime::Plugin>>) {
    log_capture::install();
    // Headless shell: no window/swapchain, frames tick on demand from the
    // remote channel (docs/REMOTE.md).
    if std::env::var("REDLILIUM_HEADLESS").is_ok_and(|v| v == "1") {
        headless::run(game);
        return;
    }
    let args = DefaultAppArgs::parse()
        .with_title_str("RedLilium Editor")
        .with_custom_titlebar(true);
    App::run(editor::Editor::with_game(game), args);
}
