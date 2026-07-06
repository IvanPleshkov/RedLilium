//! Captures, into env vars the ABI fingerprint (see `src/abi.rs`) bakes in, the
//! two things that must match for a game cdylib to be loadable (ADR-020, #45):
//!
//! - `REDLILIUM_RUSTC_VERSION` — the compiling toolchain. Rust has no stable
//!   ABI, so a toolchain change silently alters layout, vtable order, and
//!   `TypeId` hashes.
//! - `REDLILIUM_BUILD_ID` — a content id for the engine source: the `HEAD`
//!   revision plus, when the working tree is dirty, a hash of the full
//!   uncommitted diff. Version + rustc alone are necessary but NOT sufficient —
//!   two builds at the same version and toolchain differ in layout after any
//!   source edit without a version bump, and such an edit keeps the same
//!   `TypeId` (so the `QualifiedTypeId` fail-fast can't catch it). Hashing the
//!   diff makes an **unstaged** layout-changing edit shift the id, so a game
//!   built against edited engine sources no longer matches a host built against
//!   different ones. It stays deterministic: identical source content → identical
//!   id, so a clean rebuild of the same tree still matches (no spurious game
//!   rebuilds).
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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");
    // Rerun when any engine source changes, so an *unstaged* edit refreshes the
    // build id rather than baking a stale one (the F1 hole: git refs don't move
    // on unstaged edits, and recompiling a crate does not by itself rerun its
    // build script).
    for dir in ENGINE_SRC_DIRS {
        println!("cargo:rerun-if-changed={dir}");
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

/// A content id for the engine source: `<short-hash>` on a clean tree, or
/// `<short-hash>-dirty.<diff-hash>` when there are uncommitted changes, falling
/// back to `nogit` outside a git checkout. Deterministic: identical content
/// yields an identical id, and any uncommitted edit shifts `diff-hash`.
fn build_id() -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
    };
    let Some(rev) = git(&["rev-parse", "--short", "HEAD"]) else {
        return "nogit".to_owned();
    };
    let rev = rev.trim();
    // The full uncommitted diff (staged + unstaged). Empty on a clean tree.
    let diff = git(&["diff", "HEAD"]).unwrap_or_default();
    if diff.trim().is_empty() {
        rev.to_owned()
    } else {
        format!("{rev}-dirty.{:016x}", fnv1a(diff.as_bytes()))
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
