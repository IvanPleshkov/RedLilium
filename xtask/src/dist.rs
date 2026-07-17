//! Desktop distribution (#107, #133): package `car-game` into a native,
//! per-host-platform artifact.
//!
//! `cargo xtask dist --target desktop` builds with `--profile dist` (fat LTO,
//! codegen-units=1, stripped) and then packages for the *host* OS:
//!
//! **macOS** — a `.app` bundle plus a `.dmg` (#133):
//!
//! ```text
//! target/dist/car-game-desktop/
//!   Car Game.app/Contents/
//!     Info.plist
//!     MacOS/car-game                 shipping binary
//!     Resources/icon.icns            from game/icon (scripts/gen-game-icon.py)
//!     Resources/std-assets/          engine asset pack
//!     Resources/game/assets/         game asset pack
//! target/dist/car-game-desktop.dmg
//! ```
//!
//! Asset packs live in `Contents/Resources` (data belongs there for
//! codesigning); the runtime's mount resolution tries the exe dir first and
//! `../Resources` second (#132/#133), so the bundle is self-contained.
//! Optional signing is gated on env vars: `REDLILIUM_SIGN_IDENTITY` runs
//! `codesign` on the bundle, `REDLILIUM_NOTARY_PROFILE` submits the dmg via
//! `xcrun notarytool` (requires a signed bundle) and staples the ticket.
//!
//! **Linux / Windows** — the flat folder from #107, zipped:
//!
//! ```text
//! target/dist/car-game-desktop/
//!   car-game            shipping binary (Windows: icon + version resource
//!                       embedded by game/build.rs via winresource)
//!   std-assets/         engine asset pack (assets.db + sources)
//!   game/assets/        game asset pack (assets.db + scenes)
//!   car-game.desktop    Linux only: XDG launcher template (+ car-game.png);
//!                       Exec/Icon assume the binary lands on PATH
//! target/dist/car-game-desktop.zip
//! ```
//!
//! The flat layout mirrors the workspace because `GameConfig::mounts` holds
//! *relative* directories compiled into the binary (`std-assets`,
//! `game/assets`); the runtime resolves them against the executable's
//! directory first (#132), so the folder runs from any cwd — double-click
//! included. Packs are copied verbatim (v1); pruning to
//! referenced-assets-only is #134. Shaders need no extra files: the default
//! (Slang-off) build embeds the baked WGSL table in the binary.
//!
//! Note cargo's `--profile dist` build artifacts also land under
//! `target/dist/` (the artifact dir is named after the profile); the packaged
//! folders live alongside them, which is harmless.
//!
//! `cargo xtask dist --target web` (#108) produces
//!
//! ```text
//! target/dist/car-game-web/
//!   index.html          copied from game/web/index.html
//!   pkg/                wasm-pack output (car_game.js, car_game_bg.wasm, ...)
//! target/dist/car-game-web.zip
//! ```
//!
//! Assets need no folder on web: both packs are compiled into the wasm binary
//! (std-assets by the runtime, game/assets via `game/build.rs` →
//! `GameConfig::embedded_packs`). Serve the folder from any static file
//! server (`python3 -m http.server`) — `file://` won't do, wasm modules
//! require HTTP.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The asset packs a shipped build needs, as workspace-relative directories.
/// Source path == destination path: the layout must match the mount dirs in
/// `game/src/main.rs` (`GameConfig::mounts`), which the runtime resolves
/// exe-dir-first (#132; on macOS also `Contents/Resources`, #133).
const PACKS: &[&str] = &["std-assets", "game/assets"];

const GAME_PACKAGE: &str = "car-game";
const GAME_BIN: &str = "car-game";
const GAME_DISPLAY_NAME: &str = "Car Game";
const DIST_NAME: &str = "car-game-desktop";
const BUNDLE_ID: &str = "com.redlilium.car-game";

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
            if let Err(e) = dist_web() {
                eprintln!("dist failed: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown dist target {other:?}; available: desktop, web");
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

    // 1. Shipping build of the game binary (`[profile.dist]`: fat LTO,
    //    codegen-units=1, stripped). One plain cargo invocation — the dist
    //    build must not inherit xtask's feature set.
    println!("dist: building {GAME_PACKAGE} (--profile dist)...");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let status = Command::new(&cargo)
        .current_dir(&root)
        .args([
            "build",
            "--profile",
            "dist",
            "-p",
            GAME_PACKAGE,
            "--bin",
            GAME_BIN,
        ])
        .status()
        .map_err(|e| format!("failed to run cargo: {e}"))?;
    if !status.success() {
        return Err("dist build failed".to_string());
    }

    // 2. Fresh dist folder.
    let dist_root = root.join("target/dist");
    let dist = dist_root.join(DIST_NAME);
    if dist.exists() {
        std::fs::remove_dir_all(&dist).map_err(|e| format!("cleaning {dist:?}: {e}"))?;
    }
    std::fs::create_dir_all(&dist).map_err(|e| format!("creating {dist:?}: {e}"))?;

    // 3. The binary (cargo names the artifact dir after the profile).
    let bin_name = format!("{GAME_BIN}{}", std::env::consts::EXE_SUFFIX);
    let bin_src = root.join("target/dist").join(&bin_name);

    // 4. Platform packaging (#133): the artifact is native to the *host* OS —
    //    dist builds for the platform it runs on.
    match std::env::consts::OS {
        "macos" => package_macos(&root, &dist_root, &dist, &bin_src),
        other => package_flat(&root, &dist_root, &dist, &bin_src, &bin_name, other),
    }
}

/// macOS (#133): assemble `Car Game.app`, optionally codesign it, wrap the
/// dist folder in a `.dmg` (the native download format — replaces the zip),
/// and optionally notarize + staple the dmg.
fn package_macos(root: &Path, dist_root: &Path, dist: &Path, bin_src: &Path) -> Result<(), String> {
    let app = dist.join(format!("{GAME_DISPLAY_NAME}.app"));
    let contents = app.join("Contents");
    let macos_dir = contents.join("MacOS");
    let resources = contents.join("Resources");
    std::fs::create_dir_all(&macos_dir).map_err(|e| format!("creating {macos_dir:?}: {e}"))?;
    std::fs::create_dir_all(&resources).map_err(|e| format!("creating {resources:?}: {e}"))?;

    let bin_dst = macos_dir.join(GAME_BIN);
    std::fs::copy(bin_src, &bin_dst).map_err(|e| format!("copying {bin_src:?}: {e}"))?;

    // Asset packs go into Resources (data belongs there for codesigning); the
    // runtime finds them via the `../Resources` mount candidate (#132/#133).
    for pack in PACKS {
        let files = copy_dir(&root.join(pack), &resources.join(pack))?;
        println!("dist: {pack} — {files} files");
    }

    let icns_src = root.join("game/icon/icon.icns");
    std::fs::copy(&icns_src, resources.join("icon.icns"))
        .map_err(|e| format!("copying {icns_src:?} (regenerate: scripts/gen-game-icon.py): {e}"))?;

    let version = env!("CARGO_PKG_VERSION"); // workspace version == game version
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>{GAME_DISPLAY_NAME}</string>
    <key>CFBundleDisplayName</key><string>{GAME_DISPLAY_NAME}</string>
    <key>CFBundleIdentifier</key><string>{BUNDLE_ID}</string>
    <key>CFBundleVersion</key><string>{version}</string>
    <key>CFBundleShortVersionString</key><string>{version}</string>
    <key>CFBundleExecutable</key><string>{GAME_BIN}</string>
    <key>CFBundleIconFile</key><string>icon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"#
    );
    std::fs::write(contents.join("Info.plist"), plist)
        .map_err(|e| format!("writing Info.plist: {e}"))?;

    // Optional signing, gated on an identity in the environment. Unsigned
    // bundles still run locally (Gatekeeper only quarantines downloads).
    if let Ok(identity) = std::env::var("REDLILIUM_SIGN_IDENTITY") {
        println!("dist: codesigning ({identity})...");
        run_tool(
            Command::new("codesign")
                .args([
                    "--force",
                    "--options",
                    "runtime",
                    "--timestamp",
                    "--sign",
                    &identity,
                ])
                .arg(&app),
            "codesign",
        )?;
    }

    // The dmg wraps the dist folder, so the volume shows just the .app.
    let dmg_name = format!("{DIST_NAME}.dmg");
    let dmg_path = dist_root.join(&dmg_name);
    if dmg_path.exists() {
        std::fs::remove_file(&dmg_path).map_err(|e| format!("removing old {dmg_name}: {e}"))?;
    }
    let dmg = Command::new("hdiutil")
        .args([
            "create",
            "-volname",
            GAME_DISPLAY_NAME,
            "-format",
            "UDZO",
            "-srcfolder",
        ])
        .arg(dist)
        .arg(&dmg_path)
        .status();
    match dmg {
        Ok(s) if s.success() => {
            println!("dist: wrote {}", dmg_path.display());
            // Notarization needs a signed bundle and a stored notarytool
            // keychain profile (`xcrun notarytool store-credentials`).
            if let Ok(profile) = std::env::var("REDLILIUM_NOTARY_PROFILE") {
                println!("dist: notarizing ({profile})...");
                run_tool(
                    Command::new("xcrun")
                        .args([
                            "notarytool",
                            "submit",
                            "--wait",
                            "--keychain-profile",
                            &profile,
                        ])
                        .arg(&dmg_path),
                    "notarytool",
                )?;
                run_tool(
                    Command::new("xcrun")
                        .args(["stapler", "staple"])
                        .arg(&dmg_path),
                    "stapler",
                )?;
            }
        }
        Ok(s) => {
            eprintln!("dist: dmg skipped (hdiutil exited with {s}); the .app is still complete")
        }
        Err(e) => {
            eprintln!("dist: dmg skipped (hdiutil not runnable: {e}); the .app is still complete")
        }
    }

    println!(
        "dist: done — self-contained bundle (assets in Contents/Resources), run from anywhere:\n  open \"{}\"",
        app.display()
    );
    Ok(())
}

/// Linux/Windows: the flat folder layout from #107 — binary next to the asset
/// packs — zipped. Linux additionally gets an XDG launcher template + icon;
/// the Windows binary carries its icon internally (game/build.rs).
fn package_flat(
    root: &Path,
    dist_root: &Path,
    dist: &Path,
    bin_src: &Path,
    bin_name: &str,
    os: &str,
) -> Result<(), String> {
    let bin_dst = dist.join(bin_name);
    std::fs::copy(bin_src, &bin_dst).map_err(|e| format!("copying {bin_src:?}: {e}"))?;

    for pack in PACKS {
        let files = copy_dir(&root.join(pack), &dist.join(pack))?;
        println!("dist: {pack} — {files} files");
    }

    if os == "linux" {
        // A template, not an installed entry: Exec/Icon are bare names that
        // resolve once the binary/icon are placed on PATH / hicolor.
        let desktop = format!(
            "[Desktop Entry]\nType=Application\nName={GAME_DISPLAY_NAME}\n\
             Comment=Arcade car built on the RedLilium engine\n\
             Exec={GAME_BIN}\nIcon={GAME_BIN}\nTerminal=false\nCategories=Game;\n"
        );
        std::fs::write(dist.join(format!("{GAME_BIN}.desktop")), desktop)
            .map_err(|e| format!("writing .desktop: {e}"))?;
        let icon_src = root.join("game/icon/icon.png");
        std::fs::copy(&icon_src, dist.join(format!("{GAME_BIN}.png")))
            .map_err(|e| format!("copying {icon_src:?}: {e}"))?;
    }

    // Zip the folder (best effort: the folder itself is the deliverable;
    // a missing archiver downgrades to a warning, not a failure).
    let zip_name = format!("{DIST_NAME}.zip");
    let zip_path = dist_root.join(&zip_name);
    if zip_path.exists() {
        std::fs::remove_file(&zip_path).map_err(|e| format!("removing old {zip_name}: {e}"))?;
    }
    match zip_dir(dist_root, DIST_NAME, &zip_name) {
        Ok(()) => println!("dist: wrote {}", zip_path.display()),
        Err(e) => eprintln!("dist: zip skipped ({e}); the folder is still complete"),
    }

    println!(
        "dist: done — the folder is self-contained (assets resolve next to the \
         binary), run from anywhere:\n  {}/{bin_name}",
        dist.display()
    );
    Ok(())
}

/// Run an external packaging tool, turning a non-zero exit into an error.
fn run_tool(cmd: &mut Command, name: &str) -> Result<(), String> {
    let status = cmd
        .status()
        .map_err(|e| format!("failed to run {name}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{name} exited with {status}"))
    }
}

fn dist_web() -> Result<(), String> {
    let root = workspace_root();
    let dist_root = root.join("target/dist");
    let dist_name = "car-game-web";
    let dist = dist_root.join(dist_name);
    if dist.exists() {
        std::fs::remove_dir_all(&dist).map_err(|e| format!("cleaning {dist:?}: {e}"))?;
    }
    std::fs::create_dir_all(&dist).map_err(|e| format!("creating {dist:?}: {e}"))?;

    // 1. wasm-pack release build straight into the dist folder. Both asset
    //    packs ride inside the wasm binary (embedded mounts), so pkg/ + the
    //    HTML shell is the whole game.
    println!("dist: building {GAME_PACKAGE} (wasm-pack, release)...");
    let status = Command::new("wasm-pack")
        .current_dir(&root)
        .args(["build", "game", "--target", "web", "--out-dir"])
        .arg(dist.join("pkg"))
        .status()
        .map_err(|e| {
            format!("failed to run wasm-pack: {e} (install: https://rustwasm.github.io/wasm-pack/)")
        })?;
    if !status.success() {
        return Err("wasm-pack build failed".to_string());
    }
    // wasm-pack emits npm-packaging noise the static site doesn't serve.
    for junk in ["package.json", ".gitignore", "README.md"] {
        let _ = std::fs::remove_file(dist.join("pkg").join(junk));
    }

    // 2. The HTML shell.
    let html_src = root.join("game/web/index.html");
    std::fs::copy(&html_src, dist.join("index.html"))
        .map_err(|e| format!("copying {html_src:?}: {e}"))?;

    // 3. Zip (best effort, same as desktop).
    let zip_name = format!("{dist_name}.zip");
    let zip_path = dist_root.join(&zip_name);
    if zip_path.exists() {
        std::fs::remove_file(&zip_path).map_err(|e| format!("removing old {zip_name}: {e}"))?;
    }
    match zip_dir(&dist_root, dist_name, &zip_name) {
        Ok(()) => println!("dist: wrote {}", zip_path.display()),
        Err(e) => eprintln!("dist: zip skipped ({e}); the folder is still complete"),
    }

    println!(
        "dist: done — serve the folder over HTTP (wasm won't load from file://):\n  \
         cd {} && python3 -m http.server 8080\n  \
         then open http://localhost:8080/",
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
