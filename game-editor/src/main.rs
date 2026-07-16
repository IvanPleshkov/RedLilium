//! The car game's editor binary (ADR-033): the game owns the editor, not the
//! other way around. The plugin is statically linked, so the editing world
//! always knows the game's types (authoring never depends on a dylib), and
//! Play boots the full game composition from the same plugin (ADR-032).
//!
//! ```sh
//! cargo run -p car-game-editor                       # windowed
//! REDLILIUM_HEADLESS=1 cargo run -p car-game-editor  # headless (docs/REMOTE.md)
//! ```

fn main() {
    redlilium_editor::run(car_game::CarGamePlugin);
}
