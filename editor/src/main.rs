mod asset_browser;
mod background_vfs;
mod dock;
mod editor;
mod fs_watcher;
mod menu;
mod project;
mod scene_view;
mod status_bar;
mod toolbar;

use redlilium_app::{App, AppArgs, DefaultAppArgs};

fn main() {
    let args = DefaultAppArgs::parse().with_title_str("RedLilium Editor");
    App::run(editor::Editor::new(), args);
}
