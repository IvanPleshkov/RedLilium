//! Tier-1 behavior reload (ADR-037, #127): the editor rebuilds the game
//! cdylib **itself** and hot-swaps it for play worlds, while authoring stays
//! on the statically linked plugin.
//!
//! The invocation-discipline failure class (feature unification drifting
//! between separately issued `cargo build`s) disappears here by construction:
//! the editor always builds `-p <its-own-package> -p <game-package>`, and the
//! game is already inside the editor binary's dependency tree, so unification
//! matches the running binary. Note the same compilation emits both the
//! cdylib and the rlib linked into the editor binary — so the binary on disk
//! relinking after a game edit is *normal* and proves nothing (no mtime
//! heuristics here). Two outcomes remain, told apart by the **fingerprint
//! gates at load time** ([`apply_behavior_build`]):
//!
//! - the cdylib carries this running binary's engine build id → swap it in
//!   for the next play world (Tier 1);
//! - it carries a different one → engine/editor sources changed since this
//!   binary was built; the load is refused and "restart required" (Tier 2)
//!   is reported.
//!
//! Everything here is deliberately asynchronous and *marking, not acting*
//! (ADR-037 §3): the source watcher only sets a `stale` flag; the rebuild
//! runs on an explicit request; a red build leaves the old module running.
//! Threads report through mpsc channels drained once per frame by the shell.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;

/// What the owning binary configures (see `redlilium_editor::hosting`).
pub struct BehaviorReloadSpec {
    /// The game's cargo **package** name (e.g. `"car-game"`). The cdylib
    /// artifact name is derived by cargo's default `-` → `_` transform; a
    /// game overriding `[lib] name` away from that is not supported here.
    pub game_package: String,
}

/// A finished green build.
pub struct BuildOutcome {
    /// The freshly built game cdylib.
    pub dylib: PathBuf,
}

enum Msg {
    SourceDirResolved(Result<PathBuf, String>),
    SourceChanged,
    BuildFinished(Box<Result<BuildOutcome, String>>),
}

/// Per-shell driver: owns the source watcher, the background build, and the
/// status flags surfaced in the UI and the remote `state` response.
pub struct BehaviorReload {
    spec: BehaviorReloadSpec,
    tx: mpsc::Sender<Msg>,
    rx: mpsc::Receiver<Msg>,
    /// Kept alive for the watch to stay registered.
    watcher: Option<notify::RecommendedWatcher>,
    stale: bool,
    rebuilding: bool,
    restart_required: bool,
    /// The authoring (static) schemas diverged from the loaded behavior
    /// dylib: play runs new code, the inspector can't author its new fields.
    schema_diverged: bool,
}

impl BehaviorReload {
    /// Start the driver: resolves the game package's source directory in the
    /// background (`cargo pkgid`) and begins watching it once resolved.
    pub fn new(spec: BehaviorReloadSpec) -> Self {
        let (tx, rx) = mpsc::channel();
        let resolve_tx = tx.clone();
        let package = spec.game_package.clone();
        std::thread::spawn(move || {
            let _ = resolve_tx.send(Msg::SourceDirResolved(resolve_package_dir(&package)));
        });
        Self {
            spec,
            tx,
            rx,
            watcher: None,
            stale: false,
            rebuilding: false,
            restart_required: false,
            schema_diverged: false,
        }
    }

    /// Drain background messages. Returns a finished build, if one landed
    /// this frame — the shell applies it (stop play session, swap behavior
    /// module) between frames.
    pub fn poll(&mut self) -> Option<Result<BuildOutcome, String>> {
        let mut finished = None;
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::SourceDirResolved(Ok(dir)) => self.start_watching(&dir),
                Msg::SourceDirResolved(Err(e)) => {
                    log::warn!(
                        "behavior reload: could not resolve source dir of '{}' ({e}); \
                         stale marking disabled, explicit rebuilds still work",
                        self.spec.game_package
                    );
                }
                Msg::SourceChanged => self.stale = true,
                Msg::BuildFinished(result) => {
                    self.rebuilding = false;
                    finished = Some(*result);
                }
            }
        }
        finished
    }

    /// Kick off `cargo build -p <editor-package> -p <game-package>` in the
    /// background. Returns `false` if a build is already running.
    pub fn request_rebuild(&mut self) -> bool {
        if self.rebuilding {
            return false;
        }
        self.rebuilding = true;
        let tx = self.tx.clone();
        let game_package = self.spec.game_package.clone();
        std::thread::spawn(move || {
            let result = run_build(&game_package);
            let _ = tx.send(Msg::BuildFinished(Box::new(result)));
        });
        true
    }

    /// A green build was applied: the behavior module now matches the
    /// sources (as of the build), with the given schema verdict.
    pub fn applied(&mut self, schema_diverged: bool) {
        self.stale = false;
        self.schema_diverged = schema_diverged;
    }

    /// The build relinked the editor binary — only a process restart helps.
    pub fn require_restart(&mut self) {
        self.restart_required = true;
    }

    pub fn stale(&self) -> bool {
        self.stale
    }

    pub fn rebuilding(&self) -> bool {
        self.rebuilding
    }

    pub fn restart_required(&self) -> bool {
        self.restart_required
    }

    pub fn schema_diverged(&self) -> bool {
        self.schema_diverged
    }

    fn start_watching(&mut self, package_dir: &Path) {
        use notify::Watcher;
        let src = package_dir.join("src");
        let tx = self.tx.clone();
        let watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
            if let Ok(event) = event
                && matches!(
                    event.kind,
                    notify::EventKind::Create(_)
                        | notify::EventKind::Modify(_)
                        | notify::EventKind::Remove(_)
                )
            {
                let _ = tx.send(Msg::SourceChanged);
            }
        });
        match watcher {
            Ok(mut w) => match w.watch(&src, notify::RecursiveMode::Recursive) {
                Ok(()) => {
                    log::info!("behavior reload: watching {src:?} for staleness");
                    self.watcher = Some(w);
                }
                Err(e) => log::warn!("behavior reload: watch on {src:?} failed: {e}"),
            },
            Err(e) => log::warn!("behavior reload: watcher creation failed: {e}"),
        }
    }
}

/// Apply a finished Tier-1 build (both shells, between frames): a red build
/// swaps nothing; a green one stops the play session (the old behavior image
/// must have no live worlds) and loads the fresh cdylib as the behavior
/// module. Whether the swap is sound is decided by the **fingerprint gates
/// inside the load** — a cdylib carrying a different engine build id than
/// this running binary is refused, and that refusal IS the Tier-2 signal
/// ("engine changed — restart"). Note the game's rlib is linked into the
/// editor binary from the same compilation, so the binary on disk relinking
/// after a game-only edit is *normal* and proves nothing — which is why no
/// mtime heuristic is used here. Returns whether a swap happened.
pub fn apply_behavior_build(
    result: Result<BuildOutcome, String>,
    reload: &mut BehaviorReload,
    host: &mut crate::game_host::GameHost,
    play: &mut Option<crate::play::PlaySession>,
) -> bool {
    match result {
        Err(e) => {
            log::error!("behavior rebuild: {e}");
            false
        }
        Ok(outcome) => {
            if play.take().is_some() {
                log::info!("behavior swap: play session stopped");
            }
            // SAFETY: the cdylib was built by `run_build`'s fixed invocation;
            // the fingerprint + probe gates inside refuse anything that was
            // not built against this running engine.
            match unsafe { host.load_behavior(&outcome.dylib) } {
                Ok(verdict) => {
                    let diverged = matches!(verdict, crate::game_host::SchemaVerdict::Diverged(_));
                    reload.applied(diverged);
                    if diverged {
                        log::warn!(
                            "behavior swap: component schemas diverged — play runs the new \
                             code, authoring the changed fields needs an editor restart"
                        );
                    }
                    true
                }
                Err(
                    e @ (redlilium_runtime::GameModuleError::FingerprintMismatch { .. }
                    | redlilium_runtime::GameModuleError::EngineMetadataDrift),
                ) => {
                    reload.require_restart();
                    log::warn!(
                        "behavior rebuild: the fresh cdylib was built against a different \
                         engine than this running binary ({e}) — engine/editor sources \
                         changed; restart the editor to pick everything up (Tier 2)"
                    );
                    false
                }
                Err(e) => {
                    log::error!("behavior swap failed (old module kept): {e}");
                    false
                }
            }
        }
    }
}

/// The game package's directory, from `cargo pkgid` (dependency-free JSON
/// avoidance): `path+file:///…/game#car-game@0.1.0` → `/…/game`.
fn resolve_package_dir(package: &str) -> Result<PathBuf, String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let out = Command::new(cargo)
        .args(["pkgid", "-p", package])
        .output()
        .map_err(|e| format!("cargo pkgid failed to run: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo pkgid -p {package}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let id = String::from_utf8_lossy(&out.stdout);
    let id = id.trim();
    let path = id
        .strip_prefix("path+file://")
        .and_then(|rest| rest.split('#').next())
        .ok_or_else(|| format!("unexpected pkgid format: {id}"))?;
    Ok(PathBuf::from(path))
}

/// The fixed build invocation (module docs): both packages, one command,
/// profile matching the running binary. Blocking — runs on a worker thread.
fn run_build(game_package: &str) -> Result<BuildOutcome, String> {
    let editor_exe =
        std::env::current_exe().map_err(|e| format!("current_exe unavailable: {e}"))?;
    // `target/<profile>/<bin>` — the artifact dir both for the profile flag
    // and for locating the freshly built cdylib next to the binary.
    let artifact_dir = editor_exe
        .parent()
        .ok_or("editor binary has no parent dir")?
        .to_path_buf();
    let editor_package = editor_exe
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("editor binary has no utf-8 file stem")?
        .to_owned();
    let release = artifact_dir.file_name().is_some_and(|n| n == "release");

    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.args(["build", "-p", &editor_package, "-p", game_package]);
    if release {
        cmd.arg("--release");
    }
    log::info!("behavior reload: cargo build -p {editor_package} -p {game_package} …");
    let out = cmd
        .output()
        .map_err(|e| format!("cargo build failed to run: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        // The interesting part of a rustc failure is at the end.
        let tail: String = stderr
            .lines()
            .rev()
            .take(30)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!("build failed (old module kept):\n{tail}"));
    }

    let dylib = artifact_dir.join(dylib_file_name(game_package));
    if !dylib.exists() {
        return Err(format!(
            "build succeeded but {dylib:?} is missing — does '{game_package}' \
             declare crate-type [\"cdylib\", …]?"
        ));
    }
    Ok(BuildOutcome { dylib })
}

/// Cargo's default artifact name for a cdylib of `package` on this platform.
fn dylib_file_name(package: &str) -> String {
    let lib = package.replace('-', "_");
    #[cfg(target_os = "macos")]
    return format!("lib{lib}.dylib");
    #[cfg(all(unix, not(target_os = "macos")))]
    return format!("lib{lib}.so");
    #[cfg(windows)]
    return format!("{lib}.dll");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkgid_parse_shape() {
        // The parser itself, without invoking cargo.
        let id = "path+file:///home/u/ws/game#car-game@0.1.0";
        let path = id
            .strip_prefix("path+file://")
            .and_then(|rest| rest.split('#').next())
            .unwrap();
        assert_eq!(path, "/home/u/ws/game");
    }

    #[test]
    fn dylib_name_follows_cargo_transform() {
        let name = dylib_file_name("car-game");
        assert!(name.contains("car_game"), "dash→underscore: {name}");
    }
}
