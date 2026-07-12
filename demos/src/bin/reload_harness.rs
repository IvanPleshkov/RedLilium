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

use std::sync::Mutex;

use redlilium_ecs::EcsRunner;
use redlilium_graphics::GraphicsInstance;
use redlilium_runtime::{App, EngineContext, GameModule};

/// Captures every record routed through the host logger, so the harness can
/// assert that game-cdylib `log` calls actually arrive (#56 handoff) and that
/// a contained game panic was reported (#57).
struct CaptureLogger(Mutex<Vec<String>>);

impl log::Log for CaptureLogger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        let line = format!("[{}] {}", record.level(), record.args());
        println!("{line}");
        self.0.lock().unwrap().push(line);
    }
    fn flush(&self) {}
}

static LOGGER: CaptureLogger = CaptureLogger(Mutex::new(Vec::new()));

fn captured_contains(needle: &str) -> bool {
    LOGGER.0.lock().unwrap().iter().any(|l| l.contains(needle))
}

fn main() {
    log::set_logger(&LOGGER).expect("logger installed once");
    log::set_max_level(log::LevelFilter::Info);

    let path = std::env::args()
        .nth(1)
        .expect("usage: reload_harness <path-to-game-cdylib>");

    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    // The persistent engine state that must survive the reload. No mounts: the
    // harness does not render, so unresolved asset handles are harmless.
    let engine = EngineContext::new(device, &[]);

    // Child mode: run only the #57 panic phase (see the end of main).
    if std::env::args().any(|a| a == "--panic-child") {
        run_panic_child(&path, &engine);
    }

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

    // #56: the game's `log::info!` in `build` (executed just above, inside
    // the cdylib image) must have arrived through the HOST's logger — proof
    // the load-time logger handoff installed it in the cdylib's `log` static.
    assert!(
        captured_contains("SpinDemo::build"),
        "game-module log output must reach the host logger (#56 handoff)"
    );
    println!("OK: game log routed through the host logger (#56)");

    // --- Capture, tear down the old world AND module, then reload fresh ---
    let snapshot = app.capture().expect("capture snapshot");
    drop(app);
    // Drop the first module before loading again: `dlopen` is path-keyed, so
    // loading the same path while the old handle is alive would just alias the
    // same mapped image (refcount bump) — not a fresh load. The snapshot is
    // fully owned data, safe to hold across the unload.
    drop(module);

    // Load the "rebuilt" module from a unique temp copy — the same loader the
    // editor reload uses (#59): a genuinely different image, sidestepping
    // dlopen path-keying and dyld caching.
    let (module2, fresh_path) =
        unsafe { GameModule::load_fresh_copy(&path) }.expect("reload game module");
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

    // --- #57: cross-image panic behavior (verified via a subprocess) ---
    // EMPIRICAL FINDING: a panic raised inside the game cdylib's image CANNOT
    // be caught by the host's catch_unwind. Each image embeds its own libstd;
    // std stamps every panic payload with the address of a per-image canary
    // static, and a catch in the *other* image detects the mismatch and
    // deliberately aborts: "Rust cannot catch foreign exceptions". So the
    // runner's per-system containment holds only for same-image panics; a
    // game-cdylib panic today terminates the host (see abi.rs docs and the
    // in-image-shield follow-up issue).
    //
    // The child process runs the panic phase and must abort with exactly that
    // diagnostic — pinning the current behavior so the follow-up fix flips
    // this assertion to "exit 0, panic contained".
    let child = std::process::Command::new(std::env::current_exe().expect("self path"))
        .arg(&path)
        .arg("--panic-child")
        .output()
        .expect("spawn panic-phase child");
    let stderr = String::from_utf8_lossy(&child.stderr);
    assert!(
        !child.status.success(),
        "cross-image panic is expected to abort the host today (see #57); \
         if this now exits cleanly, the in-image shield landed — flip this check"
    );
    assert!(
        stderr.contains("cannot catch foreign exceptions"),
        "expected the std foreign-exception abort, got:\n{stderr}"
    );
    println!("OK: cross-image panic behavior pinned (abort, not containment) (#57)");

    println!("OK: all harness phases passed");
}

/// The #57 panic phase, run in a child process (it aborts today, by std
/// design — see main). Boots the module with `REDLILIUM_DEMO_PANIC=1` so the
/// game's `PanicOnce` system panics inside the cdylib image on frame 1.
fn run_panic_child(path: &str, engine: &EngineContext) -> ! {
    // SAFETY: single-threaded at this point — no concurrent env access.
    unsafe { std::env::set_var("REDLILIUM_DEMO_PANIC", "1") };
    let (module3, path3) =
        unsafe { GameModule::load_fresh_copy(path) }.expect("load panic-test module");
    let app3 = App::boot(engine, module3.plugin(), 1.0);
    let (mut world, mut schedules) = app3.into_parts();

    let runner = EcsRunner::multi_thread(2);
    // PanicOnce fires inside the cdylib image here. Today this aborts the
    // process before the calls below run; once the in-image shield lands it
    // will be contained and the clean-exit path takes over.
    schedules.run_frame(&mut world, &runner, 1.0 / 60.0);

    // Containment path (future): next frame clean, host bookkeeping sane.
    schedules.run_frame(&mut world, &runner, 1.0 / 60.0);
    assert!(!std::thread::panicking());
    drop(world);
    drop(schedules);
    drop(module3);
    let _ = std::fs::remove_file(&path3);
    println!("panic-child: contained (in-image shield active)");
    std::process::exit(0);
}
