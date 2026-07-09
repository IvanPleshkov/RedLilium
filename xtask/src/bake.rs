//! Offline shader bake (needs the Slang SDK; gated behind the `slang` feature).
//!
//! `bake-shaders`: compile every shader permutation the runtime uses from Slang
//! to WGSL offline, and emit `graphics/src/shader/baked_generated.rs`. The
//! browser has no Slang compiler (#33), and — since #33's follow-up — neither
//! does the *default* native build (Slang is off unless the `slang-shaders`
//! feature is on), so the wgpu backend looks WGSL and binding reflection up in
//! that table instead of compiling at runtime.
//!
//! Run: `cargo run -p xtask --features slang -- bake-shaders` (needs `SLANG_DIR`;
//! see `scripts/fetch-slang.sh`). Preflight runs `bake-shaders --check` the same
//! way when the SDK is present: it re-bakes into memory and byte-compares against
//! the committed file WITHOUT writing, so a forgotten rebake fails the build
//! without dirtying the working tree.
//!
//! **Determinism:** Slang's WGSL emission is stable for a fixed Slang version
//! but not guaranteed across versions — pin the Slang SDK used here (documented
//! beside `SLANG_DIR`); a version bump shows as a reviewed diff, not surprise
//! churn. Entries are emitted sorted by key so the file order never depends on
//! iteration order.

use std::collections::BTreeMap;
use std::path::PathBuf;

use redlilium_graphics::ShaderStage;
use redlilium_graphics::shader::{ShaderLibrary, SlangCompiler, baked};

/// One shader source and the exact `(entry_point, stage, defines)` permutations
/// the runtime compiles it with. This registry is the reviewed source of truth —
/// permutations cannot be auto-discovered offline, so a new define-gated variant
/// must be added here or it will miss loudly at runtime on wasm (both the WGSL
/// and the binding-layout reflection).
struct ShaderSpec {
    /// Human name for diagnostics.
    name: &'static str,
    /// Workspace-relative path to the `.slang` source (must be the *same bytes*
    /// the runtime hashes — verbatim `include_str!`/embedded VFS content).
    path: &'static str,
    /// Entry points to bake with their stage. Each becomes its own WGSL output;
    /// together (per define set) they form one material shader SET whose binding
    /// reflection is baked under [`baked::shader_set_key`].
    entry_points: &'static [(&'static str, ShaderStage)],
    /// Define sets; one WGSL output per (entry_point × define set) and one
    /// reflection entry per (spec × define set).
    define_sets: &'static [&'static [(&'static str, &'static str)]],
}

const NO_DEFINES: &[&[(&str, &str)]] = &[&[]];
// egui varies on the surface color space (see egui/renderer.rs): HDR, sRGB, or
// neither (linear non-sRGB). Bake all three.
const EGUI_DEFINES: &[&[(&str, &str)]] = &[&[], &[("HDR_OUTPUT", "")], &[("SRGB_FRAMEBUFFER", "")]];
// The deferred resolve pass `#ifdef HDR_OUTPUT`s its tonemap; the demo may run
// against either a linear-HDR or an sRGB/LDR target, so bake both.
const HDR_DEFINES: &[&[(&str, &str)]] = &[&[], &[("HDR_OUTPUT", "")]];

const VS_FS: &[(&str, ShaderStage)] = &[
    ("vs_main", ShaderStage::Vertex),
    ("fs_main", ShaderStage::Fragment),
];

const REGISTRY: &[ShaderSpec] = &[
    ShaderSpec {
        name: "egui",
        path: "shaders/library/egui.slang",
        entry_points: VS_FS,
        define_sets: EGUI_DEFINES,
    },
    ShaderSpec {
        name: "blit",
        path: "runtime/shaders/blit.slang",
        entry_points: VS_FS,
        define_sets: NO_DEFINES,
    },
    ShaderSpec {
        name: "entity_index",
        path: "std-assets/shaders/entity_index.slang",
        entry_points: VS_FS,
        define_sets: NO_DEFINES,
    },
    ShaderSpec {
        name: "opaque_color",
        path: "std-assets/shaders/opaque_color.slang",
        entry_points: VS_FS,
        define_sets: NO_DEFINES,
    },
    ShaderSpec {
        name: "opaque_textured",
        path: "std-assets/shaders/opaque_textured.slang",
        entry_points: VS_FS,
        define_sets: NO_DEFINES,
    },
    // Debug-draw lines/shapes — used by the editor's gizmos (debug_drawer), so it
    // must be baked or the default (Slang-off) editor build misses at runtime.
    ShaderSpec {
        name: "debug_draw",
        path: "shaders/standard/debug_draw.slang",
        entry_points: VS_FS,
        define_sets: NO_DEFINES,
    },
    // Demo shaders — baked so the demos render on the default Slang-off build too.
    ShaderSpec {
        name: "skybox",
        path: "demos/shaders/skybox.slang",
        entry_points: VS_FS,
        define_sets: NO_DEFINES,
    },
    ShaderSpec {
        name: "deferred_gbuffer",
        path: "demos/shaders/deferred_gbuffer.slang",
        entry_points: VS_FS,
        define_sets: NO_DEFINES,
    },
    ShaderSpec {
        name: "deferred_resolve",
        path: "demos/shaders/deferred_resolve.slang",
        entry_points: VS_FS,
        define_sets: HDR_DEFINES,
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

/// key -> (compiled WGSL, human-readable `shader / entry / defines`), ordered by
/// key so emission is deterministic.
type BakedTable = BTreeMap<u64, (String, String)>;

/// shader-set key -> (rendered `&[BakedGroup]` literal, human-readable name),
/// ordered by key. One entry per (spec × define set) — the material shader SET,
/// not a single WGSL permutation.
type ReflectionTable = BTreeMap<u64, (String, String)>;

/// Everything a bake produces: the WGSL table, the binding-layout reflection
/// table, and the Slang build tag.
struct Baked {
    wgsl: BakedTable,
    reflection: ReflectionTable,
    slang_tag: String,
}

/// Compile every registered permutation to WGSL, reflect each material shader
/// set's binding layouts, and capture the Slang build tag. Shared by `bake`
/// (writes the file) and `check` (compares against the committed file), so both
/// produce byte-identical output from the same compile.
fn bake_table() -> Result<Baked, String> {
    let root = workspace_root();

    // BTreeMaps so emission is sorted by key.
    let mut wgsl: BakedTable = BTreeMap::new();
    let mut reflection: ReflectionTable = BTreeMap::new();
    let mut slang_tag: Option<String> = None;

    // Fresh compiler + library modules per reflect/compile, mirroring the runtime
    // (graphics `compile_to_wgsl` / `create_material`), so baked output matches
    // what native produces live.
    let fresh_compiler = |slang_tag: &mut Option<String>| -> Result<SlangCompiler, String> {
        let compiler = SlangCompiler::new()
            .map_err(|e| format!("SlangCompiler::new (set SLANG_DIR?): {e:?}"))?;
        if slang_tag.is_none() {
            *slang_tag = Some(compiler.build_tag().to_string());
        }
        compiler
            .write_library_modules(&ShaderLibrary::standard_slang())
            .map_err(|e| format!("write_library_modules: {e:?}"))?;
        Ok(compiler)
    };

    for spec in REGISTRY {
        let full = root.join(spec.path);
        let source =
            std::fs::read_to_string(&full).map_err(|e| format!("read {}: {e}", full.display()))?;

        for defines in spec.define_sets {
            let defines: &[(&str, &str)] = defines;

            // One WGSL output per entry point.
            for &(entry, _stage) in spec.entry_points {
                let compiler = fresh_compiler(&mut slang_tag)?;
                let compiled = compiler
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
                if let Some((_, prev)) = wgsl.get(&key) {
                    return Err(format!(
                        "WGSL key collision {key:#018x}: '{name}' vs '{prev}' — two permutations \
                         hashed identically (bug in the registry or shader_key)"
                    ));
                }
                wgsl.insert(key, (compiled, name));
            }

            // One reflection entry per (spec × define set): the whole shader SET
            // reflected together, exactly as `create_material` reflects it.
            let set: Vec<baked::ShaderSetInput<'_>> = spec
                .entry_points
                .iter()
                .map(|&(entry, stage)| (source.as_str(), entry, stage, defines))
                .collect();
            let compiler = fresh_compiler(&mut slang_tag)?;
            let (layouts, rates) = compiler.reflect_all_bindings(&set).map_err(|e| {
                format!("reflect {} / {}: {e:?}", spec.name, defines_label(defines))
            })?;

            let key = baked::shader_set_key(&set);
            let name = format!("{} / {}", spec.name, defines_label(defines));
            if let Some((_, prev)) = reflection.get(&key) {
                return Err(format!(
                    "reflection key collision {key:#018x}: '{name}' vs '{prev}' — two shader sets \
                     hashed identically (bug in the registry or shader_set_key)"
                ));
            }
            reflection.insert(key, (render_groups(&layouts, &rates), name));
        }
    }

    // "unknown" only if the registry is empty (no compile ran) — a real bake
    // always sets it. Kept explicit so `--check` never panics on an empty table.
    let slang_tag = slang_tag.unwrap_or_else(|| "unknown".to_string());
    Ok(Baked {
        wgsl,
        reflection,
        slang_tag,
    })
}

/// Render an `Option<String>` as a Rust `Option<&'static str>` literal.
fn render_opt_str(o: &Option<String>) -> String {
    match o {
        Some(x) => format!("Some({x:?})"),
        None => "None".to_string(),
    }
}

/// Render reflected binding layouts + update rates as a `&[BakedGroup]` literal
/// (fully-qualified paths so the generated file compiles inside the graphics
/// crate). rustfmt normalizes the spacing afterwards, so this stays compact.
fn render_groups(
    layouts: &[redlilium_graphics::BindingLayout],
    rates: &[Option<redlilium_graphics::UpdateRate>],
) -> String {
    let mut s = String::from("&[");
    for (layout, rate) in layouts.iter().zip(rates) {
        let rate_lit = match rate {
            Some(r) => format!("Some(crate::materials::UpdateRate::{r:?})"),
            None => "None".to_string(),
        };
        s.push_str(&format!(
            "crate::shader::baked::BakedGroup {{ rate: {}, label: {}, entries: &[",
            rate_lit,
            render_opt_str(&layout.label),
        ));
        for e in &layout.entries {
            s.push_str(&format!(
                "crate::shader::baked::BakedEntry {{ binding: {}, ty: crate::materials::BindingType::{:?}, vis: {}, label: {} }},",
                e.binding,
                e.binding_type,
                e.visibility.bits(),
                render_opt_str(&e.label),
            ));
        }
        s.push_str("] },");
    }
    s.push(']');
    s
}

fn baked_dest() -> PathBuf {
    workspace_root().join("graphics/src/shader/baked_generated.rs")
}

/// Run generated Rust through `rustfmt` so the emitted file is byte-identical to
/// what `cargo fmt --all` would produce. Without this, preflight's `cargo fmt`
/// reflows the long WGSL tuples into multi-line form and `--check`'s compact
/// render would never match the committed (fmt'd) file. rustfmt is always
/// present here (preflight runs `cargo fmt`). Edition pinned to the workspace's.
fn rustfmt(src: &str) -> Result<String, String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = Command::new("rustfmt")
        .arg("--edition")
        .arg("2024")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn rustfmt: {e}"))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(src.as_bytes())
        .map_err(|e| format!("write rustfmt stdin: {e}"))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("wait rustfmt: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "rustfmt failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("rustfmt output not utf8: {e}"))
}

fn bake_shaders() -> Result<(), String> {
    let baked = bake_table()?;
    let out = rustfmt(&render_generated(&baked))?;
    let dest = baked_dest();
    std::fs::write(&dest, out).map_err(|e| format!("write {}: {e}", dest.display()))?;
    eprintln!(
        "baked {} WGSL permutations + {} reflection sets (slang {}) -> {}",
        baked.wgsl.len(),
        baked.reflection.len(),
        baked.slang_tag,
        dest.display()
    );
    Ok(())
}

/// Staleness gate (#33 "1e-6"): re-bake into memory and byte-compare against the
/// committed `baked_generated.rs`, never writing — so a failing preflight leaves
/// the working tree clean. Exits nonzero (via `Err`) if they differ, with a
/// message that distinguishes a Slang-version drift from real source edits.
fn check_shaders() -> Result<(), String> {
    let baked = bake_table()?;
    let expected = rustfmt(&render_generated(&baked))?;
    let dest = baked_dest();
    let actual =
        std::fs::read_to_string(&dest).map_err(|e| format!("read {}: {e}", dest.display()))?;

    if actual == expected {
        eprintln!(
            "baked shaders are up to date ({} WGSL + {} reflection sets, slang {})",
            baked.wgsl.len(),
            baked.reflection.len(),
            baked.slang_tag
        );
        return Ok(());
    }

    // Diagnose: a Slang upgrade reformats/renames WGSL wholesale, so lead with
    // that rather than an opaque byte-diff if the committed tag differs.
    match committed_slang_tag(&actual) {
        Some(committed) if committed != baked.slang_tag => Err(format!(
            "baked_generated.rs is stale: it was baked with slang '{committed}', but your slang \
             reports '{slang_tag}'. Slang's WGSL emission is not stable across versions — align \
             your Slang SDK with the pinned one, or rebake with `cargo run -p xtask \
             --features slang -- bake-shaders` and review the diff.",
            slang_tag = baked.slang_tag,
        )),
        _ => Err(
            "baked_generated.rs is out of date with the shader sources — rebake with \
                  `cargo run -p xtask --features slang -- bake-shaders` and commit the result."
                .to_string(),
        ),
    }
}

/// Extract the `pub static BAKED_SLANG_TAG: &str = "...";` value from a committed
/// generated file, for the version-drift diagnosis in [`check_shaders`].
fn committed_slang_tag(file: &str) -> Option<String> {
    let line = file.lines().find(|l| l.contains("BAKED_SLANG_TAG"))?;
    let start = line.find('"')? + 1;
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

fn render_generated(baked: &Baked) -> String {
    let mut s = String::new();
    s.push_str(
        "//! @generated by `xtask bake-shaders` — DO NOT EDIT BY HAND.\n\
         //!\n\
         //! Regenerate with `cargo run -p xtask --features slang -- bake-shaders`\n\
         //! (needs the Slang SDK; see scripts/fetch-slang.sh).\n\
         //! CI/preflight re-bakes and `git diff --exit-code`s this file, so a stale entry\n\
         //! fails the build. Entries are sorted by key for a stable, review-friendly diff.\n\
         //!\n\
         //! `BAKED_WGSL`: key -> compiled WGSL. `BAKED_NAMES`: key -> `shader / entry /\n\
         //! defines` (diagnostics on a miss). `BAKED_REFLECTION`: shader-set key -> baked\n\
         //! binding-layout reflection (the runtime uses it on wasm, where Slang can't run).\n\
         //! All sorted ascending by key for binary search.\n\n",
    );

    s.push_str(&format!(
        "/// Slang build tag this table was baked with. `xtask bake-shaders --check` compares\n\
         /// this first, so a compiler upgrade reads as an explicit version message rather than\n\
         /// an opaque WGSL byte-diff. Not referenced by the runtime.\n\
         #[allow(dead_code)]\n\
         pub static BAKED_SLANG_TAG: &str = {:?};\n\n",
        baked.slang_tag
    ));

    s.push_str("/// key -> compiled WGSL, sorted ascending by key.\n");
    s.push_str("pub static BAKED_WGSL: &[(u64, &str)] = &[\n");
    for (key, (wgsl, _)) in &baked.wgsl {
        s.push_str(&format!("    ({key:#018x}, {wgsl:?}),\n"));
    }
    s.push_str("];\n\n");

    s.push_str("/// key -> human-readable `shader / entry / defines`, sorted ascending by key.\n");
    s.push_str("pub static BAKED_NAMES: &[(u64, &str)] = &[\n");
    for (key, (_, name)) in &baked.wgsl {
        s.push_str(&format!("    ({key:#018x}, {name:?}),\n"));
    }
    s.push_str("];\n\n");

    s.push_str("/// shader-set key -> baked binding-layout reflection, sorted ascending by key.\n");
    s.push_str("pub static BAKED_REFLECTION: &[(u64, &[crate::shader::baked::BakedGroup])] = &[\n");
    for (key, (groups, _)) in &baked.reflection {
        s.push_str(&format!("    ({key:#018x}, {groups}),\n"));
    }
    s.push_str("];\n");
    s
}

pub fn run() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let check = std::env::args().nth(2).as_deref() == Some("--check");
    match cmd.as_str() {
        "bake-shaders" => {
            let result = if check {
                check_shaders()
            } else {
                bake_shaders()
            };
            if let Err(e) = result {
                let task = if check {
                    "bake-shaders --check"
                } else {
                    "bake-shaders"
                };
                eprintln!("{task} failed: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown task {other:?}; available: bake-shaders [--check]");
            std::process::exit(2);
        }
    }
}
