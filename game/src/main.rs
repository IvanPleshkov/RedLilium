//! Standalone entry point for the car game (#103).
//!
//! Controls: right-drag to look, WASD to fly (placeholder free-fly camera).
#![recursion_limit = "256"]

use car_game::CarGamePlugin;
use redlilium_runtime::GameConfig;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    redlilium_runtime::run(
        GameConfig {
            title: "Car Game".to_string(),
            mounts: vec![("std", "std-assets"), ("game", "game/assets")],
            ..Default::default()
        },
        CarGamePlugin,
    );
}
