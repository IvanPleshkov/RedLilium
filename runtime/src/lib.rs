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
//!     fn register_types(&self, world: &mut redlilium_ecs::World) {
//!         // Type registrations only — runs in every hosting world
//!         // (game AND editor), so scenes with game components can be
//!         // inspected and serialized anywhere.
//!         world.register_inspector::<MyComponent>();
//!     }
//!     fn build(&self, app: &mut App) {
//!         // Resources, events, systems — game worlds only.
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
mod ui;

pub use abi::{
    ABI_FINGERPRINT, ABI_FINGERPRINT_SYMBOL, AbiFingerprintFn, EngineTypeIdProbe,
    GAME_MODULE_SYMBOL, GameModule, GameModuleError, GameModuleFn, LOGGER_INIT_SYMBOL,
    LoggerInitFn, TYPEID_PROBE_SYMBOL, TypeIdProbeFn, abi_fingerprint, engine_typeid_probe,
};
pub use app::App;
pub use engine_context::EngineContext;
pub use ui::{AppControl, GameUi};

use redlilium_app::AppArgs;

/// User game code, structured as a plugin.
///
/// The contract splits three obligations with different audiences:
///
/// - [`register_types`](Plugin::register_types) — make the game's *types*
///   known to a world: component registrations (inspection, serialization)
///   and similar type-level metadata. This is the only part of the plugin
///   that runs against **every** world that must understand game data — the
///   game world itself, and the editor's editing world (which authors scenes
///   containing game components but hosts none of the game's systems).
/// - [`build`](Plugin::build) — resources, events, and systems: what makes
///   the game *run*. It executes only in a **game world** (the standalone
///   build and the editor's Play), once per world generation: on first boot
///   and again after every reload. Keep it idempotent and free of scene
///   population — anything spawned here would be duplicated by the restored
///   snapshot on reload (ADR-020, #45).
/// - [`spawn_scene`](Plugin::spawn_scene) — populate the initial scene through
///   [`App::world_mut`]. It runs **only when there is no snapshot to restore**
///   (first boot); a reload restores entities from the captured snapshot
///   instead. Default: no-op.
/// - [`on_unload`](Plugin::on_unload) — cleanup before dylib unload during reload.
///   Runs after scene serialization but **before** World drop, the last chance to
///   access game state. Default: no-op. Use for: joining tasks, flushing I/O.
///
/// The host calls `build` after the graphics device exists and before the first
/// frame. Plugins can compose other plugins via [`App::add_plugin`].
pub trait Plugin {
    /// Register the game's types (components, inspectors) with a world.
    ///
    /// Called for **both** game worlds and editing worlds — every host that
    /// needs to inspect, serialize, or author the game's components. Must not
    /// add systems, insert resources, or spawn entities: a world receiving
    /// only `register_types` understands the game's data without running any
    /// of it. Hosts standing up a *game* world ([`App::boot`], [`App::reload`],
    /// [`App::add_plugin`]) call this automatically before [`build`](Plugin::build).
    /// Default: no-op.
    fn register_types(&self, world: &mut redlilium_ecs::World) {
        let _ = world;
    }

    /// Register resources, events, and systems. Runs only in a game world,
    /// once per world generation (first boot and every reload); must not
    /// spawn scene entities.
    fn build(&self, app: &mut App);

    /// Populate the initial scene. Called only on first boot — a reload
    /// restores entities from a snapshot instead. Default: no-op.
    fn spawn_scene(&self, app: &mut App) {
        let _ = app;
    }

    /// Extend the *editing world's* view layer: preview/overlay systems,
    /// viewport context-menu operations, and viewport tools for the plugin's
    /// own components. Called by editor hosts right after
    /// [`register_types`](Plugin::register_types), and again after every
    /// reload against a fresh [`EditingView`] (so generations never stack).
    /// Unlike [`build`](Plugin::build) this runs in a world the plugin does
    /// not own: systems added here must be read-only observers, and every
    /// mutation — from menu ops and tools alike — goes through the
    /// `ActionQueue` (the contexts don't even expose a mutable world).
    /// Default: no-op.
    fn build_editing_view(&self, view: &mut EditingView<'_>) {
        let _ = view;
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

/// Mutable view of the editing world's extension points, handed to
/// [`Plugin::build_editing_view`]. The editor owns the underlying storage
/// (schedules + world resources); this struct only groups the borrows so the
/// hook signature can grow without breaking implementors again.
pub struct EditingView<'a> {
    /// The editing world's schedule graph (add read-only view systems).
    pub schedules: &'a mut redlilium_ecs::Schedules,
    /// Right-click viewport menu operations.
    pub ops: &'a mut redlilium_ecs::ui::ViewportOps,
    /// Interactive viewport tools (activated from ops or the shell).
    pub tools: &'a mut redlilium_ecs::ui::ViewportTools,
}

/// One embedded asset pack: `(pack-relative path, verbatim bytes)` for every
/// file, exactly as it sits in the mount's source directory. Verbatim matters:
/// `.slang` bytes are hashed to look up baked WGSL, so embedded content must
/// be byte-identical to what `xtask bake-shaders` hashed (see #33).
pub type EmbeddedPack = &'static [(&'static str, &'static [u8])];

/// Host configuration for [`run`].
pub struct GameConfig {
    /// Window title.
    pub title: String,
    /// Local asset-pack mounts `(name, dir)`; each directory's `assets.db`
    /// is loaded into the merged in-memory database at startup. A relative
    /// dir is resolved against the executable's directory first (so a dist
    /// folder runs from any cwd, #132), then against the working directory
    /// (dev runs from the workspace root).
    pub mounts: Vec<(&'static str, &'static str)>,
    /// Embedded packs for targets without a local disk (wasm), keyed by the
    /// mount *source dir* from [`mounts`](Self::mounts). On wasm a mount with
    /// an entry here is served from memory (its `assets.db` included); without
    /// one it falls back to the built-in `std-assets` embed or an empty
    /// provider. Ignored on native (filesystem mounts). Games typically fill
    /// this from a build-script-generated table (#108) — see `game/build.rs`.
    pub embedded_packs: Vec<(&'static str, EmbeddedPack)>,
    /// Clear color for the main camera target and the swapchain.
    pub clear_color: [f32; 4],
    /// Host override of the start scene (mount-relative path): `Some(path)`
    /// supersedes whatever scene the game's `spawn_scene` requested (see
    /// [`App::boot`]). The game binary typically fills this from a CLI flag
    /// or env var; `None` lets the game decide.
    pub start_scene: Option<String>,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            title: "RedLilium Game".to_string(),
            mounts: vec![("std", "std-assets"), ("project", "project-assets")],
            embedded_packs: Vec::new(),
            clear_color: [0.02, 0.02, 0.03, 1.0],
            start_scene: None,
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
