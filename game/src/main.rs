//! Standalone entry point for the car game (#103).
//!
//! Controls: right-drag to look, WASD to fly (placeholder free-fly camera).
#![recursion_limit = "256"]

use car_game::CarGamePlugin;
use redlilium_runtime::GameConfig;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    // eprintln, not log: visible regardless of RUST_LOG, on log's stream.
    eprintln!("car-game {}", car_game::BUILD_STAMP);

    redlilium_runtime::run(
        GameConfig {
            title: "Car Game".to_string(),
            mounts: vec![("std", "std-assets"), ("game", "game/assets")],
            // Dev aid: CAR_GAME_SCENE=scenes/level1.scene skips the menu.
            // A host parameter, not game logic — the plugin always starts
            // at the menu; the override rides App::boot's start_scene.
            start_scene: std::env::var("CAR_GAME_SCENE").ok(),
            ..Default::default()
        },
        CarGamePlugin,
    );
}
