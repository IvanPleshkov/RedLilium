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
//! included.
//!
//! Packs are **pruned to the referenced closure** (#134): the dependency
//! walk starts from the entry scenes in `game/dist-manifest.ron`, follows
//! guid references through the merged `assets.db` + asset source text
//! (`redlilium_assets::AssetDb::dependency_closure`), and ships only
//! reachable records plus the manifest's `keep` prefixes (assets loaded by
//! code paths — today the whole std pack). Each pack's `assets.db` is
//! regenerated to the shipped subset; pruned files are printed so a
//! missing-asset regression is diagnosable. The wasm embed (`game/build.rs`)
//! is not pruned — currently the whole game pack is reachable anyway.
//! Shaders need no extra files: the default (Slang-off) build embeds the
//! baked WGSL table in the binary.
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

/// The asset packs a shipped build needs: `(mount name, workspace-relative
/// dir)`. Both must match `GameConfig::mounts` in `game/src/main.rs`: the
/// mount names key the DB walk and the manifest, and the dist artifact
/// mirrors the dirs because the runtime resolves them exe-dir-first (#132;
/// on macOS also `Contents/Resources`, #133).
const MOUNTS: &[(&str, &str)] = &[("std", "std-assets"), ("game", "game/assets")];

/// Prune manifest (#134): closure roots + code-referenced keep prefixes.
const MANIFEST: &str = "game/dist-manifest.ron";

const GAME_PACKAGE: &str = "car-game";
const GAME_BIN: &str = "car-game";
const GAME_DISPLAY_NAME: &str = "Car Game";
const DIST_NAME: &str = "car-game-desktop";
const BUNDLE_ID: &str = "com.redlilium.car-game";

pub fn run() {
    // Build stamp (#135) — the same version+hash the binary prints at startup
    // (workspace version == game version), so dist output identifies the build.
    println!(
        "dist: {GAME_PACKAGE} {}+{}",
        env!("CARGO_PKG_VERSION"),
        git_hash()
    );
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

    // 4. Prune the asset packs to the referenced closure (#134).
    let plans = plan_packs(&root)?;

    // 5. Platform packaging (#133): the artifact is native to the *host* OS —
    //    dist builds for the platform it runs on.
    match std::env::consts::OS {
        "macos" => package_macos(&root, &dist_root, &dist, &bin_src, &plans),
        other => package_flat(&root, &dist_root, &dist, &bin_src, &bin_name, other, &plans),
    }
}

/// macOS (#133): assemble `Car Game.app`, optionally codesign it, wrap the
/// dist folder in a `.dmg` (the native download format — replaces the zip),
/// and optionally notarize + staple the dmg.
fn package_macos(
    root: &Path,
    dist_root: &Path,
    dist: &Path,
    bin_src: &Path,
    plans: &[PackPlan],
) -> Result<(), String> {
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
    for plan in plans {
        write_pack(root, plan, &resources)?;
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
#[allow(clippy::too_many_arguments)]
fn package_flat(
    root: &Path,
    dist_root: &Path,
    dist: &Path,
    bin_src: &Path,
    bin_name: &str,
    os: &str,
    plans: &[PackPlan],
) -> Result<(), String> {
    let bin_dst = dist.join(bin_name);
    std::fs::copy(bin_src, &bin_dst).map_err(|e| format!("copying {bin_src:?}: {e}"))?;

    for plan in plans {
        write_pack(root, plan, dist)?;
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

/// Short git hash with a `-dirty` marker, `unknown` without git — mirrors the
/// stamp `game/build.rs` embeds into the binary (#135).
fn git_hash() -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .current_dir(workspace_root())
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    match git(&["rev-parse", "--short=9", "HEAD"]) {
        Some(hash) => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
                .is_none_or(|s| !s.is_empty());
            if dirty { format!("{hash}-dirty") } else { hash }
        }
        None => "unknown".to_string(),
    }
}

/// `xtask/..` — the workspace root, independent of the invoking cwd.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent dir")
        .to_path_buf()
}

/// Parsed `game/dist-manifest.ron` (#134).
#[derive(serde::Deserialize)]
struct DistManifest {
    /// Closure roots, `"mount:path"` — the scenes the game can load.
    entries: Vec<String>,
    /// Always-shipped `"mount:path-prefix"` matches — assets that code loads
    /// by hardcoded path, invisible to the data-driven walk.
    keep: Vec<String>,
}

/// What ships for one asset pack after pruning (#134): the selected files
/// (mount-relative, forward slashes) and the regenerated `assets.db` text.
struct PackPlan {
    dir: &'static str,
    files: Vec<String>,
    db_ron: String,
}

/// Compute the shipped subset of every pack (#134): merge the packs'
/// `assets.db` files, walk the dependency closure from the manifest's entry
/// scenes, and keep reachable records plus manifest `keep` prefixes. Prints
/// one report line per pack — a pruned file that should have shipped is
/// diagnosable from here.
fn plan_packs(root: &Path) -> Result<Vec<PackPlan>, String> {
    let mut db = redlilium_assets::AssetDb::new();
    for &(mount, dir) in MOUNTS {
        let path = root.join(dir).join("assets.db");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("reading {path:?}: {e}"))?;
        let conflicts = db
            .merge_ron(mount, &text)
            .map_err(|e| format!("parsing {path:?}: {e}"))?;
        if !conflicts.is_empty() {
            return Err(format!("{path:?}: {} guid/path conflicts", conflicts.len()));
        }
    }

    let manifest_path = root.join(MANIFEST);
    let manifest: DistManifest = ron::from_str(
        &std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("reading {manifest_path:?}: {e}"))?,
    )
    .map_err(|e| format!("parsing {manifest_path:?}: {e}"))?;

    let dir_of = |mount: &str| {
        MOUNTS
            .iter()
            .find(|&&(m, _)| m == mount)
            .map(|&(_, dir)| dir)
    };
    let split = |s: &str| -> Result<(String, String), String> {
        match s.split_once(':') {
            Some((mount, path)) if dir_of(mount).is_some() => {
                Ok((mount.to_string(), path.to_string()))
            }
            _ => Err(format!(
                "manifest entry {s:?} is not \"mount:path\" with a known mount"
            )),
        }
    };

    // Closure roots must resolve — a typo here would silently ship nothing.
    let mut roots = Vec::new();
    for entry in &manifest.entries {
        let (mount, path) = split(entry)?;
        let guid = db
            .guid_of(&redlilium_assets::AssetPath::new(&mount, &path))
            .ok_or_else(|| format!("manifest entry {entry:?} is not in any assets.db"))?;
        roots.push(guid);
    }
    let keep: Vec<(String, String)> = manifest
        .keep
        .iter()
        .map(|s| split(s))
        .collect::<Result<_, _>>()?;

    // Walk guid references through the DB and through source text (scene
    // files store asset references as guid strings; binary sources fail the
    // UTF-8 read and are skipped — their references live in the DB record).
    let reached = db.dependency_closure(roots, |record| {
        let dir = dir_of(&record.path.mount)?;
        std::fs::read_to_string(root.join(dir).join(&record.path.path)).ok()
    });

    let mut plans = Vec::new();
    for &(mount, dir) in MOUNTS {
        let kept_by_prefix = |path: &str| {
            keep.iter()
                .any(|(m, p)| m == mount && path.starts_with(p.as_str()))
        };

        // The pack's pruned DB: reachable or kept records only.
        let mut pack_db = redlilium_assets::AssetDb::new();
        for (guid, record) in db.to_records() {
            if record.path.mount == mount
                && (reached.contains(&guid) || kept_by_prefix(&record.path.path))
            {
                pack_db.insert(guid, record).map_err(|e| e.to_string())?;
            }
        }
        let db_ron = pack_db
            .to_ron_for_mount(mount)
            .map_err(|e| format!("serializing {mount} assets.db: {e}"))?;

        let mut all = Vec::new();
        collect_files(&root.join(dir), String::new(), &mut all)?;
        let (mut files, mut pruned) = (Vec::new(), Vec::new());
        for rel in all {
            if rel == "assets.db" {
                continue; // regenerated from the pruned DB
            }
            let selected = kept_by_prefix(&rel)
                || db
                    .guid_of(&redlilium_assets::AssetPath::new(mount, &rel))
                    .is_some_and(|g| reached.contains(&g));
            if selected {
                files.push(rel);
            } else {
                pruned.push(rel);
            }
        }
        if pruned.is_empty() {
            println!("dist: {dir} — {} files, nothing pruned", files.len() + 1);
        } else {
            println!(
                "dist: {dir} — {} files, pruned {}: {}",
                files.len() + 1,
                pruned.len(),
                pruned.join(", ")
            );
        }
        plans.push(PackPlan { dir, files, db_ron });
    }
    Ok(plans)
}

/// Materialize one pack plan under `dst_root`: the selected files plus the
/// regenerated (pruned) `assets.db`.
fn write_pack(root: &Path, plan: &PackPlan, dst_root: &Path) -> Result<(), String> {
    let dst_dir = dst_root.join(plan.dir);
    for rel in &plan.files {
        let from = root.join(plan.dir).join(rel);
        let to = dst_dir.join(rel);
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("creating {parent:?}: {e}"))?;
        }
        std::fs::copy(&from, &to).map_err(|e| format!("copying {from:?}: {e}"))?;
    }
    std::fs::create_dir_all(&dst_dir).map_err(|e| format!("creating {dst_dir:?}: {e}"))?;
    std::fs::write(dst_dir.join("assets.db"), &plan.db_ron)
        .map_err(|e| format!("writing {} assets.db: {e}", plan.dir))?;
    Ok(())
}

/// Collect pack-relative (forward-slash) paths of every non-dotfile under
/// `dir` (`.DS_Store` and friends are OS noise, not assets).
fn collect_files(dir: &Path, prefix: String, out: &mut Vec<String>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("reading {dir:?}: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("reading {dir:?}: {e}"))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let rel = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}/{name}")
        };
        let ty = entry
            .file_type()
            .map_err(|e| format!("stat {:?}: {e}", entry.path()))?;
        if ty.is_dir() {
            collect_files(&entry.path(), rel, out)?;
        } else {
            out.push(rel);
        }
    }
    Ok(())
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
