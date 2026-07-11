#![recursion_limit = "256"]

//! # RedLilium Runtime
//!
//! The game runtime: hosts user game code written as [`Plugin`]s and drives
//! the frame loop that glues the window/GPU layer (`redlilium-app`), the ECS
//! (`redlilium-ecs`), and the asset system (`redlilium-assets`).
//!
//! This is the static layer of ADR-020 (`docs/DECISIONS.md`): a game is a
//! [`Plugin`] that registers components/resources and adds systems to named
//! schedules through the [`App`] builder. The runtime owns `main()`'s event
//! loop, the `World`, the `Schedules`, and the render bracket — game code
//! never touches the frame loop directly.
//!
//! ```ignore
//! use redlilium_runtime::{App, GameConfig, Plugin};
//!
//! struct MyGame;
//!
//! impl Plugin for MyGame {
//!     fn build(&self, app: &mut App) {
//!         // Registration only: components, resources, events, systems.
//!         app.register_component::<MyComponent>();
//!         app.add_system::<redlilium_ecs::Update, _>(MySystem);
//!     }
//!     fn spawn_scene(&self, app: &mut App) {
//!         // Populate the initial scene through app.world_mut().
//!     }
//! }
//!
//! fn main() {
//!     redlilium_runtime::run(GameConfig::default(), MyGame);
//! }
//! ```
//!
//! Persistent engine state (GPU device, asset database, GPU resource
//! managers) lives in the host-owned [`EngineContext`] and is injected into
//! the world as shared resources — the prerequisite for warm-restart reload
//! and Play-mode world swaps (#44, #45).

mod abi;
mod app;
mod blit;
// Std-assets compiled into the wasm binary (no local disk in a browser, #33).
#[cfg(target_arch = "wasm32")]
mod embedded_assets;
mod engine_context;
mod handler;

pub use abi::{
    ABI_FINGERPRINT, ABI_FINGERPRINT_SYMBOL, AbiFingerprintFn, GAME_MODULE_SYMBOL, GameModule,
    GameModuleError, GameModuleFn, LOGGER_INIT_SYMBOL, LoggerInitFn, abi_fingerprint,
};
pub use app::App;
pub use engine_context::EngineContext;

use redlilium_app::AppArgs;

/// User game code, structured as a plugin.
///
/// The contract deliberately splits *registration* from *initial-scene spawn*,
/// because warm-restart reload (ADR-020, #45) re-runs registration against a
/// fresh world but takes the scene from a snapshot, not from a fresh spawn:
///
/// - [`build`](Plugin::build) — register components, resources, events, and add
///   systems to schedules. It runs **once per world generation**: on first boot
///   and again after every reload. Keep it idempotent and free of scene
///   population — anything spawned here would be duplicated by the restored
///   snapshot on reload.
/// - [`spawn_scene`](Plugin::spawn_scene) — populate the initial scene through
///   [`App::world_mut`]. It runs **only when there is no snapshot to restore**
///   (first boot); a reload restores entities from the captured snapshot
///   instead. Default: no-op.
/// - [`on_stop`](Plugin::on_stop) — cleanup when transitioning from Play to Stop.
///   Runs **before** engine-side despawn and snapshot restore, while game world is
///   still live. Default: no-op. See ADR-024.
/// - [`on_unload`](Plugin::on_unload) — cleanup before dylib unload during reload.
///   Runs after scene serialization but **before** World drop, the last chance to
///   access game state. Default: no-op. Use for: joining tasks, flushing I/O.
///
/// The host calls `build` after the graphics device exists and before the first
/// frame. Plugins can compose other plugins via [`App::add_plugin`].
pub trait Plugin {
    /// Register components, resources, events, and systems. Runs once per world
    /// generation (first boot and every reload); must not spawn scene entities.
    fn build(&self, app: &mut App);

    /// Populate the initial scene. Called only on first boot — a reload
    /// restores entities from a snapshot instead. Default: no-op.
    fn spawn_scene(&self, app: &mut App) {
        let _ = app;
    }

    /// Stop cleanup: called when Play → Stop transition fires.
    ///
    /// Runs **before** engine-side hooks (despawn, restore, etc.). Use for:
    /// - Muting audio
    /// - Canceling async tasks
    /// - Flushing pending game state
    ///
    /// **Invariant:** The game world is still live and consistent;
    /// entity despawn and snapshot restore happen after this returns.
    /// Default: no-op.
    ///
    /// See `docs/DESIGN_45_PLUGIN_LIFECYCLE.md` for lifecycle details.
    fn on_stop(&self, app: &mut App) {
        let _ = app;
    }

    /// Reload cleanup: called before World drop during reload.
    ///
    /// Runs after scene serialization but **before** World drop and before dylib
    /// unload. This is the last chance to access game state. Use for:
    /// - Joining long-lived async tasks
    /// - Flushing external resources (files, network sockets)
    /// - Saving state to persistent storage
    ///
    /// **Invariant:** World is available but about to be dropped. This callback
    /// runs while the dylib is still mapped; ensure all spawned tasks complete
    /// or are canceled before returning, or dylib unload will UB.
    /// Default: no-op.
    ///
    /// See `docs/DESIGN_45_PLUGIN_LIFECYCLE.md` for lifecycle details.
    fn on_unload(&self, app: &mut App) {
        let _ = app;
    }
}

/// Host configuration for [`run`].
pub struct GameConfig {
    /// Window title.
    pub title: String,
    /// Local asset-pack mounts `(name, dir)`; each directory's `assets.db`
    /// is loaded into the merged in-memory database at startup. Paths are
    /// relative to the working directory.
    pub mounts: Vec<(&'static str, &'static str)>,
    /// Clear color for the main camera target and the swapchain.
    pub clear_color: [f32; 4],
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            title: "RedLilium Game".to_string(),
            mounts: vec![("std", "std-assets"), ("project", "project-assets")],
            clear_color: [0.02, 0.02, 0.03, 1.0],
        }
    }
}

/// Run a game: create the window and GPU device, build the world through the
/// plugin, then drive `Schedules::run_frame` + the `Render` schedule until
/// the window closes.
pub fn run<P: Plugin + 'static>(config: GameConfig, plugin: P) {
    let args = redlilium_app::DefaultAppArgs::parse().with_title_str(config.title.clone());
    redlilium_app::App::run(handler::RuntimeHandler::new(config, plugin), args);
}
