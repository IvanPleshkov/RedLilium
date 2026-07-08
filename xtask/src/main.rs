//! RedLilium build tooling.
//!
//! `bake-shaders`: compile every shader permutation the runtime uses from Slang
//! to WGSL offline, and emit `graphics/src/shader/baked_generated.rs`. The
//! browser has no Slang compiler (#33), so on wasm the wgpu backend looks WGSL
//! up in that table instead of compiling at runtime.
//!
//! Run: `SLANG_DIR=/path/to/slang cargo run -p xtask -- bake-shaders`
//! CI/preflight re-runs this and `git diff --exit-code`s the generated file, so
//! a forgotten rebake fails the build.
//!
//! **Determinism:** Slang's WGSL emission is stable for a fixed Slang version
//! but not guaranteed across versions — pin the Slang SDK used here (documented
//! beside `SLANG_DIR`); a version bump shows as a reviewed diff, not surprise
//! churn. Entries are emitted sorted by key so the file order never depends on
//! iteration order.

use std::collections::BTreeMap;
use std::path::PathBuf;

use redlilium_graphics::shader::{ShaderLibrary, SlangCompiler, baked};

/// One shader source and the exact `(entry_point, defines)` permutations the
/// runtime compiles it with. This registry is the reviewed source of truth —
/// permutations cannot be auto-discovered offline, so a new define-gated variant
/// must be added here or it will miss loudly at runtime on wasm.
struct ShaderSpec {
    /// Human name for diagnostics.
    name: &'static str,
    /// Workspace-relative path to the `.slang` source (must be the *same bytes*
    /// the runtime hashes — verbatim `include_str!`/embedded VFS content).
    path: &'static str,
    /// Entry points to bake (each becomes its own WGSL output).
    entry_points: &'static [&'static str],
    /// Define sets; one WGSL output per (entry_point × define set).
    define_sets: &'static [&'static [(&'static str, &'static str)]],
}

const NO_DEFINES: &[&[(&str, &str)]] = &[&[]];
// egui varies on the surface color space (see egui/renderer.rs): HDR, sRGB, or
// neither (linear non-sRGB). Bake all three.
const EGUI_DEFINES: &[&[(&str, &str)]] = &[&[], &[("HDR_OUTPUT", "")], &[("SRGB_FRAMEBUFFER", "")]];

const REGISTRY: &[ShaderSpec] = &[
    ShaderSpec {
        name: "egui",
        path: "shaders/library/egui.slang",
        entry_points: &["vs_main", "fs_main"],
        define_sets: EGUI_DEFINES,
    },
    ShaderSpec {
        name: "blit",
        path: "runtime/shaders/blit.slang",
        entry_points: &["vs_main", "fs_main"],
        define_sets: NO_DEFINES,
    },
    ShaderSpec {
        name: "entity_index",
        path: "std-assets/shaders/entity_index.slang",
        entry_points: &["vs_main", "fs_main"],
        define_sets: NO_DEFINES,
    },
    ShaderSpec {
        name: "opaque_color",
        path: "std-assets/shaders/opaque_color.slang",
        entry_points: &["vs_main", "fs_main"],
        define_sets: NO_DEFINES,
    },
    ShaderSpec {
        name: "opaque_textured",
        path: "std-assets/shaders/opaque_textured.slang",
        entry_points: &["vs_main", "fs_main"],
        define_sets: NO_DEFINES,
    },
];

fn workspace_root() -> PathBuf {
    // xtask/ lives directly under the workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent dir")
        .to_path_buf()
}

fn defines_label(defines: &[(&str, &str)]) -> String {
    if defines.is_empty() {
        "[]".to_string()
    } else {
        let inner: Vec<String> = defines
            .iter()
            .map(|(k, v)| {
                if v.is_empty() {
                    k.to_string()
                } else {
                    format!("{k}={v}")
                }
            })
            .collect();
        format!("[{}]", inner.join(","))
    }
}

fn bake_shaders() -> Result<(), String> {
    let root = workspace_root();

    // key -> (wgsl, name), BTreeMap so emission is sorted by key.
    let mut table: BTreeMap<u64, (String, String)> = BTreeMap::new();

    for spec in REGISTRY {
        let full = root.join(spec.path);
        let source =
            std::fs::read_to_string(&full).map_err(|e| format!("read {}: {e}", full.display()))?;

        for &entry in spec.entry_points {
            for defines in spec.define_sets {
                // Fresh compiler + library modules per compile, mirroring the
                // runtime (graphics `compile_to_wgsl`), so baked WGSL matches
                // what native produces live.
                let compiler = SlangCompiler::new()
                    .map_err(|e| format!("SlangCompiler::new (set SLANG_DIR?): {e:?}"))?;
                compiler
                    .write_library_modules(&ShaderLibrary::standard_slang())
                    .map_err(|e| format!("write_library_modules: {e:?}"))?;

                let wgsl = compiler
                    .compile_to_wgsl(&source, entry, &[], defines)
                    .map_err(|e| {
                        format!(
                            "compile {} / {} / {}: {e:?}",
                            spec.name,
                            entry,
                            defines_label(defines)
                        )
                    })?;

                let key = baked::shader_key(&source, entry, defines);
                let name = format!("{} / {} / {}", spec.name, entry, defines_label(defines));

                if let Some((_, prev)) = table.get(&key) {
                    return Err(format!(
                        "key collision {key:#018x}: '{name}' vs '{prev}' — two permutations \
                         hashed identically (bug in the registry or shader_key)"
                    ));
                }
                table.insert(key, (wgsl, name));
            }
        }
    }

    let out = render_generated(&table);
    let dest = root.join("graphics/src/shader/baked_generated.rs");
    std::fs::write(&dest, out).map_err(|e| format!("write {}: {e}", dest.display()))?;
    eprintln!(
        "baked {} shader permutations -> {}",
        table.len(),
        dest.display()
    );
    Ok(())
}

fn render_generated(table: &BTreeMap<u64, (String, String)>) -> String {
    let mut s = String::new();
    s.push_str(
        "//! @generated by `xtask bake-shaders` — DO NOT EDIT BY HAND.\n\
         //!\n\
         //! Regenerate with `cargo run -p xtask -- bake-shaders` (needs SLANG_DIR).\n\
         //! CI/preflight re-bakes and `git diff --exit-code`s this file, so a stale entry\n\
         //! fails the build. Entries are sorted by key for a stable, review-friendly diff.\n\
         //!\n\
         //! `BAKED_WGSL`: key -> compiled WGSL. `BAKED_NAMES`: key -> `shader / entry /\n\
         //! defines` (diagnostics on a miss). Both sorted ascending by key for binary search.\n\n",
    );

    s.push_str("/// key -> compiled WGSL, sorted ascending by key.\n");
    s.push_str("pub static BAKED_WGSL: &[(u64, &str)] = &[\n");
    for (key, (wgsl, _)) in table {
        s.push_str(&format!("    ({key:#018x}, {wgsl:?}),\n"));
    }
    s.push_str("];\n\n");

    s.push_str("/// key -> human-readable `shader / entry / defines`, sorted ascending by key.\n");
    s.push_str("pub static BAKED_NAMES: &[(u64, &str)] = &[\n");
    for (key, (_, name)) in table {
        s.push_str(&format!("    ({key:#018x}, {name:?}),\n"));
    }
    s.push_str("];\n");
    s
}

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "bake-shaders" => {
            if let Err(e) = bake_shaders() {
                eprintln!("bake-shaders failed: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown task {other:?}; available: bake-shaders");
            std::process::exit(2);
        }
    }
}
