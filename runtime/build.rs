//! Captures, into env vars the ABI fingerprint (see `src/abi.rs`) bakes in, the
//! two things that must match for a game cdylib to be loadable (ADR-020, #45):
//!
//! - `REDLILIUM_RUSTC_VERSION` — the compiling toolchain. Rust has no stable
//!   ABI, so a toolchain change silently alters layout, vtable order, and
//!   `TypeId` hashes.
//! - `REDLILIUM_BUILD_ID` — a content id for the **engine** source: git tree
//!   hashes of the engine crate dirs a game module actually links (plus the
//!   workspace `Cargo.lock` — dependency bumps change layouts too), plus,
//!   when any of them are dirty, a hash of their uncommitted diff. Version +
//!   rustc alone are necessary but NOT sufficient — two builds at the same
//!   version and toolchain differ in layout after any source edit without a
//!   version bump, and such an edit keeps the same `TypeId` (so the
//!   `QualifiedTypeId` fail-fast can't catch it). Hashing the diff makes an
//!   **unstaged** layout-changing edit shift the id. It stays deterministic:
//!   identical engine content → identical id.
//!
//!   Scoped to engine dirs deliberately (ADR-037): a commit touching only
//!   `editor/`, `demos/`, or a game crate does NOT shift the id — the engine
//!   rlibs inside an existing game cdylib are still the ones the host runs,
//!   so invalidating the module would be a false positive (this was the
//!   "every editor commit stales the game" failure mode).
//!
//! Caveat that still stands: within a *single* `cargo build`, the build-script
//! output is computed once and shared, so the host and the game cdylib always
//! get the *same* id and are mutually consistent (this is what makes the reload
//! harness sound). The id only discriminates across *separate* builds — the
//! editor reload flow — which is exactly where a stale game must be rejected.
//! For that to work the script must rerun when engine sources change; the
//! `rerun-if-changed` lines below cover the engine crate `src/` trees plus the
//! git refs. A brand-new engine crate must be added to that list.

use std::process::Command;

/// Engine crate `src/` directories (relative to this build script) whose
/// contents feed the build id — edits here must re-trigger it.
const ENGINE_SRC_DIRS: &[&str] = &[
    "src",
    "../core/src",
    "../ecs/src",
    "../ecs-macro/src",
    "../graphics/src",
    "../app/src",
    "../assets/src",
    "../vfs/src",
    "../debug_drawer/src",
];

/// Additional files feeding the build id: each engine crate's manifest
/// (feature/edition changes) and the workspace lockfile (dependency bumps
/// change layouts).
const ENGINE_MANIFESTS: &[&str] = &[
    "Cargo.toml",
    "../core/Cargo.toml",
    "../ecs/Cargo.toml",
    "../ecs-macro/Cargo.toml",
    "../graphics/Cargo.toml",
    "../app/Cargo.toml",
    "../assets/Cargo.toml",
    "../vfs/Cargo.toml",
    "../debug_drawer/Cargo.toml",
    "../Cargo.lock",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");
    // Rerun when any engine source changes, so an *unstaged* edit refreshes the
    // build id rather than baking a stale one (the F1 hole: git refs don't move
    // on unstaged edits, and recompiling a crate does not by itself rerun its
    // build script).
    for path in ENGINE_SRC_DIRS.iter().chain(ENGINE_MANIFESTS) {
        println!("cargo:rerun-if-changed={path}");
    }

    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = Command::new(&rustc)
        .arg("-vV")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|text| {
            let mut release = None;
            let mut commit = None;
            for line in text.lines() {
                if let Some(v) = line.strip_prefix("release: ") {
                    release = Some(v.trim().to_owned());
                }
                if let Some(v) = line.strip_prefix("commit-hash: ") {
                    commit = Some(v.trim().to_owned());
                }
            }
            release.map(|r| match commit {
                Some(c) => format!("{r} ({c})"),
                None => r,
            })
        })
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=REDLILIUM_RUSTC_VERSION={version}");

    println!("cargo:rustc-env=REDLILIUM_BUILD_ID={}", build_id());
}

/// A content id for the **engine** source (module docs): `eng-<hash>` over
/// the git tree/blob hashes of the engine dirs + manifests at `HEAD`, with
/// `-dirty.<diff-hash>` appended when any of them carry uncommitted changes.
/// Falls back to `nogit` outside a git checkout. Deterministic: identical
/// engine content yields an identical id; commits elsewhere in the repo
/// (editor, demos, game crates) leave it unchanged.
fn build_id() -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
    };
    // Repo-relative form of the engine paths ("src" → "runtime/src",
    // "../core/src" → "core/src") for `rev-parse HEAD:<path>`.
    let repo_relative: Vec<String> = ENGINE_SRC_DIRS
        .iter()
        .chain(ENGINE_MANIFESTS)
        .map(|p| match p.strip_prefix("../") {
            Some(rest) => rest.to_owned(),
            None => format!("runtime/{p}"),
        })
        .collect();
    if git(&["rev-parse", "HEAD"]).is_none() {
        return "nogit".to_owned();
    }
    // Object hash per path at HEAD (tree for dirs, blob for files). A path
    // missing from HEAD (brand-new crate not yet committed) contributes only
    // its name — the dirty diff below carries its actual content.
    let mut content = String::new();
    for path in &repo_relative {
        if let Some(hash) = git(&["rev-parse", &format!("HEAD:{path}")]) {
            content.push_str(hash.trim());
        }
        content.push('|');
        content.push_str(path);
        content.push('\n');
    }
    let id = format!("eng-{:016x}", fnv1a(content.as_bytes()));
    // Uncommitted changes (staged + unstaged) restricted to the engine
    // paths. Empty when only non-engine parts of the tree are dirty.
    let mut diff_args = vec!["diff", "HEAD", "--"];
    let engine_paths: Vec<&str> = ENGINE_SRC_DIRS
        .iter()
        .chain(ENGINE_MANIFESTS)
        .copied()
        .collect();
    diff_args.extend(engine_paths);
    let diff = git(&diff_args).unwrap_or_default();
    if diff.trim().is_empty() {
        id
    } else {
        format!("{id}-dirty.{:016x}", fnv1a(diff.as_bytes()))
    }
}

/// FNV-1a 64-bit hash — a small, dependency-free, deterministic content hash.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
