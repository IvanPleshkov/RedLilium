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
mod headless;
mod history_panel;
mod log_capture;
mod menu;
mod project;
mod remote_commands;
mod scene_view;
mod status_bar;
mod theme;
mod toolbar;

use redlilium_app::{App, AppArgs, DefaultAppArgs};

fn main() {
    log_capture::install();
    // Headless shell: no window/swapchain, frames tick on demand from the
    // remote channel (docs/REMOTE.md).
    if std::env::var("REDLILIUM_HEADLESS").is_ok_and(|v| v == "1") {
        headless::run();
        return;
    }
    let args = DefaultAppArgs::parse()
        .with_title_str("RedLilium Editor")
        .with_custom_titlebar(true);
    App::run(editor::Editor::new(), args);
}
