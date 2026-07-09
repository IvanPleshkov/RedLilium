//! RedLilium build tooling.
//!
//! The only task is `bake-shaders`, which needs the native Slang compiler and is
//! therefore gated behind the `slang` feature (off by default so a plain
//! `cargo build --workspace` — preflight, fresh clones — needs no Slang SDK). See
//! [`bake`] for the bake itself.
//!
//! ```text
//! cargo run -p xtask --features slang -- bake-shaders          # regenerate
//! cargo run -p xtask --features slang -- bake-shaders --check  # staleness gate
//! ```

#[cfg(feature = "slang")]
mod bake;

fn main() {
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
