//! # Plugin Demo
//!
//! Runs [`redlilium_demos::SpinDemo`] — game code authored as a
//! `redlilium-runtime` [`Plugin`](redlilium_runtime::Plugin) (ADR-020, #44/#45)
//! — as a shipped static binary. The same plugin is also exported from the
//! `redlilium-demos` cdylib as a loadable game module the editor can
//! warm-restart-reload.
//!
//! Controls: right-drag to look, WASD to fly (std `FreeFlyCamera`).
#![recursion_limit = "256"]

use redlilium_demos::SpinDemo;
use redlilium_runtime::GameConfig;

fn main() {
    redlilium_runtime::run(
        GameConfig {
            title: "RedLilium Plugin Demo".to_string(),
            ..Default::default()
        },
        SpinDemo,
    );
}
