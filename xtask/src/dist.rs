//! Desktop distribution (#107): package `car-game` into a runnable folder.
//!
//! `cargo xtask dist --target desktop` produces
//!
//! ```text
//! target/dist/car-game-desktop/
//!   car-game            release binary (run it from THIS folder)
//!   std-assets/         engine asset pack (assets.db + sources)
//!   game/assets/        game asset pack (assets.db + scenes)
//! target/dist/car-game-desktop.zip
//! ```
//!
//! The folder layout mirrors the workspace because `GameConfig::mounts` holds
//! *relative* directories compiled into the binary (`std-assets`,
//! `game/assets`) that the runtime resolves against the working directory —
//! so the game must be launched with the dist folder as cwd. Packs are copied
//! verbatim (v1); pruning to referenced-assets-only is a later optimization.
//! Shaders need no extra files: the default (Slang-off) build embeds the
//! baked WGSL table in the binary.
//!
//! `--target web` is #108 and not implemented yet.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The asset packs a shipped build needs, as workspace-relative directories.
/// Source path == destination path: the layout must match the mount dirs in
/// `game/src/main.rs` (`GameConfig::mounts`), which the runtime resolves
/// against the working directory.
const PACKS: &[&str] = &["std-assets", "game/assets"];

const GAME_PACKAGE: &str = "car-game";
const GAME_BIN: &str = "car-game";
const DIST_NAME: &str = "car-game-desktop";

pub fn run() {
    let target = parse_target();
    match target.as_str() {
        "desktop" => {
            if let Err(e) = dist_desktop() {
                eprintln!("dist failed: {e}");
                std::process::exit(1);
            }
        }
        "web" => {
            eprintln!("dist --target web is not implemented yet (#108)");
            std::process::exit(2);
        }
        other => {
            eprintln!("unknown dist target {other:?}; available: desktop");
            std::process::exit(2);
        }
    }
}

/// `--target <name>` from the args after `dist` (defaults to `desktop`).
fn parse_target() -> String {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--target" {
            return iter.next().cloned().unwrap_or_else(|| {
                eprintln!("--target needs a value (e.g. desktop)");
                std::process::exit(2);
            });
        }
    }
    "desktop".to_string()
}

fn dist_desktop() -> Result<(), String> {
    let root = workspace_root();

    // 1. Release build of the game binary. One plain cargo invocation — the
    //    dist build must not inherit xtask's feature set.
    println!("dist: building {GAME_PACKAGE} (release)...");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let status = Command::new(&cargo)
        .current_dir(&root)
        .args(["build", "--release", "-p", GAME_PACKAGE, "--bin", GAME_BIN])
        .status()
        .map_err(|e| format!("failed to run cargo: {e}"))?;
    if !status.success() {
        return Err("release build failed".to_string());
    }

    // 2. Fresh dist folder.
    let dist_root = root.join("target/dist");
    let dist = dist_root.join(DIST_NAME);
    if dist.exists() {
        std::fs::remove_dir_all(&dist).map_err(|e| format!("cleaning {dist:?}: {e}"))?;
    }
    std::fs::create_dir_all(&dist).map_err(|e| format!("creating {dist:?}: {e}"))?;

    // 3. The binary.
    let bin_name = format!("{GAME_BIN}{}", std::env::consts::EXE_SUFFIX);
    let bin_src = root.join("target/release").join(&bin_name);
    let bin_dst = dist.join(&bin_name);
    std::fs::copy(&bin_src, &bin_dst).map_err(|e| format!("copying {bin_src:?}: {e}"))?;

    // 4. Asset packs, verbatim, mirroring the workspace-relative mount dirs.
    for pack in PACKS {
        let src = root.join(pack);
        let dst = dist.join(pack);
        let files = copy_dir(&src, &dst)?;
        println!("dist: {pack} — {files} files");
    }

    // 5. Zip the folder (best effort: the folder itself is the deliverable;
    //    a missing archiver downgrades to a warning, not a failure).
    let zip_name = format!("{DIST_NAME}.zip");
    let zip_path = dist_root.join(&zip_name);
    if zip_path.exists() {
        std::fs::remove_file(&zip_path).map_err(|e| format!("removing old {zip_name}: {e}"))?;
    }
    match zip_dir(&dist_root, DIST_NAME, &zip_name) {
        Ok(()) => println!("dist: wrote {}", zip_path.display()),
        Err(e) => eprintln!("dist: zip skipped ({e}); the folder is still complete"),
    }

    println!(
        "dist: done — run the game with the dist folder as cwd:\n  cd {} && ./{bin_name}",
        dist.display()
    );
    Ok(())
}

/// `xtask/..` — the workspace root, independent of the invoking cwd.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent dir")
        .to_path_buf()
}

/// Recursively copy `src` into `dst`, skipping dotfiles (`.DS_Store` and
/// friends are OS noise, not assets). Returns the number of files copied.
fn copy_dir(src: &Path, dst: &Path) -> Result<usize, String> {
    std::fs::create_dir_all(dst).map_err(|e| format!("creating {dst:?}: {e}"))?;
    let mut copied = 0;
    let entries = std::fs::read_dir(src).map_err(|e| format!("reading {src:?}: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("reading {src:?}: {e}"))?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        let ty = entry
            .file_type()
            .map_err(|e| format!("stat {from:?}: {e}"))?;
        if ty.is_dir() {
            copied += copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).map_err(|e| format!("copying {from:?}: {e}"))?;
            copied += 1;
        }
    }
    Ok(copied)
}

/// Archive `dir_name` (relative to `cwd`) as `zip_name` using the platform's
/// stock archiver — `zip` on unix, `Compress-Archive` on Windows — so xtask
/// carries no archive dependency.
fn zip_dir(cwd: &Path, dir_name: &str, zip_name: &str) -> Result<(), String> {
    let status = if cfg!(windows) {
        Command::new("powershell")
            .current_dir(cwd)
            .args([
                "-NoProfile",
                "-Command",
                &format!("Compress-Archive -Path {dir_name} -DestinationPath {zip_name} -Force"),
            ])
            .status()
    } else {
        Command::new("zip")
            .current_dir(cwd)
            .args(["-rq", zip_name, dir_name])
            .status()
    };
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("archiver exited with {s}")),
        Err(e) => Err(format!("archiver not runnable: {e}")),
    }
}
