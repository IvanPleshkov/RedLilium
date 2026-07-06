//! # Warm-restart reload harness (#45)
//!
//! End-to-end check of the static-linked reload path against a **real** game
//! cdylib: load the module, boot a game, capture a snapshot, drop the world,
//! reload from a freshly loaded module, and assert the scene survived. Proves
//! `GameModule::load` → `App::boot` → `App::capture` → `App::reload` on a genuine
//! `dlopen`ed cdylib (not a statically linked test plugin).
//!
//! Run by hand (needs a GPU adapter):
//!
//! ```text
//! cargo build -p redlilium-demos
//! cargo run -p redlilium-demos --bin reload_harness -- target/debug/libredlilium_demos.dylib
//! ```
#![recursion_limit = "256"]

use redlilium_graphics::GraphicsInstance;
use redlilium_runtime::{App, EngineContext, GameModule};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: reload_harness <path-to-game-cdylib>");

    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    // The persistent engine state that must survive the reload. No mounts: the
    // harness does not render, so unresolved asset handles are harmless.
    let engine = EngineContext::new(device, &[]);

    // --- First boot from the loaded module ---
    // SAFETY: the cdylib was built in this same `cargo build` as this harness
    // (matching fingerprint, same engine rlibs → same TypeIds/layout, default
    // allocator both sides), so the plugin handoff is sound; the module is held
    // alive past every App it builds (see GameModule docs).
    let module = unsafe { GameModule::load(&path) }.expect("load game module");
    let app = App::boot(&engine, module.plugin(), 1.0);
    let booted = app.world().iter_entities().count();
    println!("booted:   {booted} entities (build + spawn_scene)");
    assert!(booted > 0, "SpinDemo should spawn a scene");

    // --- Capture, tear down the old world AND module, then reload fresh ---
    let snapshot = app.capture().expect("capture snapshot");
    drop(app);
    // Drop the first module before loading again: `dlopen` is path-keyed, so
    // loading the same path while the old handle is alive would just alias the
    // same mapped image (refcount bump) — not a fresh load. The snapshot is
    // fully owned data, safe to hold across the unload.
    drop(module);

    // Load the "rebuilt" module from a unique temp copy, modeling the editor
    // reload flow (a genuinely different image, sidestepping dyld path caching).
    let fresh_path = copy_to_temp(&path);
    let module2 = unsafe { GameModule::load(&fresh_path) }.expect("reload game module");
    let reloaded = App::reload(&engine, module2.plugin(), 1.0, &snapshot).expect("reload");
    let after = reloaded.world().iter_entities().count();
    println!("reloaded: {after} entities (build + restored snapshot, spawn_scene skipped)");
    assert_eq!(
        booted, after,
        "scene must survive the reload with no duplication"
    );

    // The module outlives every App its plugin touched.
    drop(reloaded);
    drop(module2);
    let _ = std::fs::remove_file(&fresh_path);
    println!("OK: warm reload preserved the scene");
}

/// Copy the cdylib to a unique temp path so the second load maps a fresh image.
fn copy_to_temp(src: &str) -> std::path::PathBuf {
    let stem = std::path::Path::new(src)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("game");
    let dst = std::env::temp_dir().join(format!(
        "redlilium_reload_{}_{stem}.dylib",
        std::process::id()
    ));
    std::fs::copy(src, &dst).expect("copy cdylib to temp");
    dst
}
