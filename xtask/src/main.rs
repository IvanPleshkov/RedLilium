//! RedLilium build tooling.
//!
//! ```text
//! cargo xtask dist [--target desktop|web]                      # package a shippable build (#107, #108)
//! cargo run -p xtask --features slang -- bake-shaders          # regenerate baked shaders
//! cargo run -p xtask --features slang -- bake-shaders --check  # staleness gate
//! cargo run -p xtask -- bake-ibl                               # regenerate baked IBL KTX2 set (#137)
//! ```
//!
//! `dist` needs no extra SDK (see [`dist`] for the folder layout it produces).
//! `bake-shaders` needs the native Slang compiler and is gated behind the
//! `slang` feature (off by default so a plain `cargo build --workspace` —
//! preflight, fresh clones — needs no Slang SDK); see [`bake`].
//! `bake-ibl` needs the pinned source HDRI (scripts/fetch-hdri.sh) and
//! self-skips when it is absent; see [`ibl`].

#[cfg(feature = "slang")]
mod bake;
mod dist;
mod ibl;

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "dist" => dist::run(),
        "bake-ibl" => ibl::run(),
        "bake-shaders" => {
            #[cfg(feature = "slang")]
            bake::run();
            #[cfg(not(feature = "slang"))]
            {
                eprintln!(
                    "xtask `bake-shaders` needs the Slang SDK and the `slang` feature.\n\
                     Provision the SDK (scripts/fetch-slang.sh) and run:\n  \
                     cargo run -p xtask --features slang -- bake-shaders [--check]"
                );
                std::process::exit(2);
            }
        }
        other => {
            eprintln!(
                "unknown task {other:?}; available: dist [--target desktop|web], \
                 bake-shaders [--check], bake-ibl"
            );
            std::process::exit(2);
        }
    }
}
