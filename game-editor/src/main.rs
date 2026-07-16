//! The car game's editor binary (ADR-037): the game owns the editor, not the
//! other way around. The plugin is statically linked, so the editing world
//! always knows the game's types (authoring never depends on a dylib), and
//! Play boots the full game composition from the same plugin (ADR-036).
//!
//! Tier-1 behavior reload is on: the editor watches `game/src`, rebuilds the
//! `car-game` cdylib on request (`reload_game` / the titlebar button), and
//! hot-swaps it for play worlds — authoring always stays on the static image.
//!
//! ```sh
//! cargo run -p car-game-editor                       # windowed
//! REDLILIUM_HEADLESS=1 cargo run -p car-game-editor  # headless (docs/REMOTE.md)
//! ```

fn main() {
    redlilium_editor::hosting(car_game::CarGamePlugin)
        .behavior_reload("car-game")
        .run();
}
