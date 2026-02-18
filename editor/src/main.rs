mod dock;
mod editor;
mod menu;
mod scene_view;
mod toolbar;

use redlilium_app::{App, AppArgs, DefaultAppArgs};

fn main() {
    let args = DefaultAppArgs::parse().with_title_str("RedLilium Editor");
    App::run(editor::Editor::new(), args);
}
