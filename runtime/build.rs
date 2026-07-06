//! Captures, into env vars the ABI fingerprint (see `src/abi.rs`) bakes in, the
//! two things that must match for a game cdylib to be loadable under ADR-020's
//! shared-engine-dylib contract:
//!
//! - `REDLILIUM_RUSTC_VERSION` — the compiling toolchain. Rust has no stable
//!   ABI, so a toolchain change silently alters layout, vtable order, and
//!   `TypeId` hashes.
//! - `REDLILIUM_BUILD_ID` — the engine source revision (git rev + dirty flag).
//!   Version + rustc alone are necessary but NOT sufficient: two builds at the
//!   same version and toolchain can differ in layout after any source edit
//!   without a version bump. A per-commit id makes such a rebuild change the
//!   fingerprint, so a stale game (built against an older engine revision) is
//!   rejected. It is deterministic — identical commits produce identical ids —
//!   so a clean rebuild of the same revision still matches (no spurious game
//!   rebuilds). A `-dirty` suffix marks an uncommitted working tree; two
//!   distinct dirty builds share the `-dirty` id and so are NOT distinguished —
//!   do not reload across dirty builds.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Refresh the build id when HEAD moves (commit/checkout) or the index
    // changes (staging). Uncommitted unstaged edits only flip the dirty flag.
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");

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

/// The engine source revision: `<short-hash>` or `<short-hash>-dirty`, falling
/// back to `nogit` outside a git checkout. Deterministic for a given commit.
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
    // `--porcelain` prints one line per change; any output means dirty.
    let dirty = git(&["status", "--porcelain"])
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if dirty {
        format!("{rev}-dirty")
    } else {
        rev.to_owned()
    }
}
