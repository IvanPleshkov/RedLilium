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
use redlilium_graphics::shader::{ShaderLibrary, ShaderVariantSpace, SlangCompiler, baked};

/// One shader source and the entry points the runtime compiles it with. The
/// registry lists the *shaders*; their define permutations are no longer
/// hand-written — each shader declares its variant space with
/// `//#pragma variant` / `//#pragma variant_system` lines in the source
/// (#6, Decision 5), and the bake enumerates the full cartesian product via
/// [`ShaderVariantSpace::enumerate_all`]. A shader missing from this registry
/// still misses loudly at runtime on wasm (both the WGSL and the
/// binding-layout reflection); a *variant* can no longer be forgotten.
struct ShaderSpec {
    /// Human name for diagnostics.
    name: &'static str,
    /// Workspace-relative path to the `.slang` source (must be the *same bytes*
    /// the runtime hashes — verbatim `include_str!`/embedded VFS content).
    path: &'static str,
    /// Entry points to bake with their stage. Each becomes its own WGSL output;
    /// together (per variant) they form one material shader SET whose binding
    /// reflection is baked under [`baked::shader_set_key`].
    entry_points: &'static [(&'static str, ShaderStage)],
    /// Bake every entry as SPIR-V instead of WGSL (#117), for sources Slang's
    /// WGSL target cannot load (e.g. unbounded bindless arrays). Specs with
    /// task/mesh entries force this implicitly (#111).
    force_spirv: bool,
    /// Skip the binding-layout reflection bake (#117): sources whose bindings
    /// reflection cannot express (bindless runtime arrays) pair with
    /// materials that declare explicit binding layouts instead.
    skip_reflection: bool,
}

const VS_FS: &[(&str, ShaderStage)] = &[
    ("vs_main", ShaderStage::Vertex),
    ("fs_main", ShaderStage::Fragment),
];

/// A plain vertex+fragment spec with default artifact/reflection handling.
const fn vs_fs_spec(name: &'static str, path: &'static str) -> ShaderSpec {
    ShaderSpec {
        name,
        path,
        entry_points: VS_FS,
        force_spirv: false,
        skip_reflection: false,
    }
}

const REGISTRY: &[ShaderSpec] = &[
    vs_fs_spec("egui", "shaders/library/egui.slang"),
    vs_fs_spec("blit", "runtime/shaders/blit.slang"),
    vs_fs_spec("entity_index", "std-assets/shaders/entity_index.slang"),
    vs_fs_spec("opaque_color", "std-assets/shaders/opaque_color.slang"),
    vs_fs_spec(
        "opaque_textured",
        "std-assets/shaders/opaque_textured.slang",
    ),
    // Depth-only pass shader (#129): vertex stage only — zero color
    // attachments make the fragment stage unnecessary. Reflection still
    // bakes (the draw path classifies its camera/model sets by rate).
    ShaderSpec {
        name: "depth_only",
        path: "std-assets/shaders/depth_only.slang",
        entry_points: &[("vs_main", ShaderStage::Vertex)],
        force_spirv: false,
        skip_reflection: false,
    },
    // Debug-draw lines/shapes — used by the editor's gizmos (debug_drawer), so it
    // must be baked or the default (Slang-off) editor build misses at runtime.
    vs_fs_spec("debug_draw", "shaders/standard/debug_draw.slang"),
    // Standard deferred PBR/IBL path (#144) — the G-buffer shader behind the
    // `pbr` shading model plus the pipeline's own fullscreen passes.
    vs_fs_spec(
        "std_deferred_gbuffer",
        "std-assets/shaders/deferred_gbuffer.slang",
    ),
    vs_fs_spec(
        "std_deferred_gbuffer_textured",
        "std-assets/shaders/deferred_gbuffer_textured.slang",
    ),
    vs_fs_spec(
        "std_deferred_resolve",
        "std-assets/shaders/deferred_resolve.slang",
    ),
    vs_fs_spec("std_skybox", "std-assets/shaders/skybox.slang"),
    // Display-output pass (#142): scene-referred -> display-referred
    // (exposure + tonemap + encode variants).
    vs_fs_spec(
        "std_display_output",
        "std-assets/shaders/display_output.slang",
    ),
    // TAA resolve (#148): history accumulation with variance clipping.
    vs_fs_spec("std_taa_resolve", "std-assets/shaders/taa_resolve.slang"),
    // Velocity completion (#149): fills camera-only motion into the background
    // of GBUFFER_VELOCITY so motion blur (and TAA) see the sky move.
    vs_fs_spec(
        "std_velocity_complete",
        "std-assets/shaders/velocity_complete.slang",
    ),
    // Demo shaders — baked so the demos render on the default Slang-off build
    // too. (The pbr_ibl demo's own shaders retired with #144 — it consumes
    // the std deferred path now.)
    vs_fs_spec("compressed_quad", "demos/shaders/compressed_quad.slang"),
    // Meshlet demo (#111): the task/mesh entries force the whole set to
    // SPIR-V (no WGSL form); reflection still bakes (the material relies on
    // it).
    ShaderSpec {
        name: "meshlet",
        path: "demos/shaders/meshlet.slang",
        entry_points: &[
            ("ts_main", ShaderStage::Task),
            ("ms_main", ShaderStage::Mesh),
            ("fs_main", ShaderStage::Fragment),
        ],
        force_spirv: false,
        skip_reflection: false,
    },
    // Bindless demo (#117): unbounded descriptor arrays — no WGSL form and
    // no reflectable layout; the material declares explicit binding layouts.
    ShaderSpec {
        name: "bindless",
        path: "demos/shaders/bindless.slang",
        entry_points: VS_FS,
        force_spirv: true,
        skip_reflection: true,
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

/// key -> (compiled SPIR-V bytes, human-readable name), ordered by key. The
/// artifact kind for task/mesh entry points (#111): naga cannot express mesh
/// shading, so there is no WGSL form to bake — the runtime consumes Slang's
/// SPIR-V directly (Vulkan is the only backend with mesh shaders).
type BakedSpirvTable = BTreeMap<u64, (Vec<u8>, String)>;

/// shader-set key -> (rendered `&[BakedGroup]` literal, human-readable name),
/// ordered by key. One entry per (spec × define set) — the material shader SET,
/// not a single WGSL permutation.
type ReflectionTable = BTreeMap<u64, (String, String)>;

/// Everything a bake produces: the WGSL table, the binding-layout reflection
/// table, and the Slang build tag.
struct Baked {
    wgsl: BakedTable,
    spirv: BakedSpirvTable,
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
    let mut spirv: BakedSpirvTable = BTreeMap::new();
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

        // The shader's declared variant space (#6) drives the permutations:
        // full cartesian product, capped by MAX_VARIANTS_PER_SHADER. A shader
        // with no `//#pragma variant` lines yields exactly the empty set.
        let space = ShaderVariantSpace::parse(&source)
            .map_err(|e| format!("variant pragmas of {}: {e}", spec.name))?;
        let variants = space
            .enumerate_all()
            .map_err(|e| format!("variant space of {}: {e}", spec.name))?;

        for variant in &variants {
            let defines_owned: Vec<(&str, &str)> = variant
                .defines()
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            let defines: &[(&str, &str)] = &defines_owned;

            // One artifact per entry point: WGSL for classic stages, SPIR-V
            // for task/mesh stages (#111 — naga has no mesh-shading support,
            // so there is no WGSL form; only the Vulkan backend runs these).
            // A spec containing ANY task/mesh entry bakes ALL its entries to
            // SPIR-V: Slang's WGSL target refuses to even load a module with
            // mesh-shading entry points, and such materials are Vulkan-only
            // anyway. `force_spirv` opts in explicitly for the same reason
            // (e.g. bindless runtime arrays, #117).
            let spirv_spec = spec.force_spirv
                || spec
                    .entry_points
                    .iter()
                    .any(|&(_, s)| matches!(s, ShaderStage::Task | ShaderStage::Mesh));
            for &(entry, _stage) in spec.entry_points {
                let key = baked::shader_key(&source, entry, defines);
                let name = format!("{} / {} / {}", spec.name, entry, defines_label(defines));
                let prev = wgsl
                    .get(&key)
                    .map(|(_, n)| n)
                    .or_else(|| spirv.get(&key).map(|(_, n)| n));
                if let Some(prev) = prev {
                    return Err(format!(
                        "key collision {key:#018x}: '{name}' vs '{prev}' — two permutations \
                         hashed identically (bug in the registry or shader_key)"
                    ));
                }

                let compiler = fresh_compiler(&mut slang_tag)?;
                if spirv_spec {
                    let compiled = compiler
                        .compile_to_spirv(&source, entry, &[], defines)
                        .map_err(|e| {
                            format!(
                                "compile {} / {} / {}: {e:?}",
                                spec.name,
                                entry,
                                defines_label(defines)
                            )
                        })?;
                    spirv.insert(key, (compiled, name));
                } else {
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
                    wgsl.insert(key, (compiled, name));
                }
            }

            // Reflection is skipped for specs whose bindings it cannot
            // express (#117): their materials declare explicit layouts, so a
            // baked entry would be dead weight at best and wrong at worst.
            if spec.skip_reflection {
                continue;
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
        spirv,
        reflection,
        slang_tag,
    })
}

/// Render SPIR-V bytes as a Rust byte-string literal (`b"..."`). Printable
/// ASCII stays literal, everything else becomes `\xNN` — roughly 3.5 chars
/// per byte, on a single line rustfmt will not reflow (it never splits
/// string literals), unlike a `&[u8]` array literal which it would explode
/// to one element per line.
fn render_byte_string(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 4 + 3);
    s.push_str("b\"");
    for &b in bytes {
        match b {
            b'"' => s.push_str("\\\""),
            b'\\' => s.push_str("\\\\"),
            0x20..=0x7e => s.push(b as char),
            _ => {
                let _ = write!(s, "\\x{b:02x}");
            }
        }
    }
    s.push('"');
    s
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
        "baked {} WGSL + {} SPIR-V permutations + {} reflection sets (slang {}) -> {}",
        baked.wgsl.len(),
        baked.spirv.len(),
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
            "baked shaders are up to date ({} WGSL + {} SPIR-V + {} reflection sets, slang {})",
            baked.wgsl.len(),
            baked.spirv.len(),
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
         //! `BAKED_WGSL`: key -> compiled WGSL. `BAKED_SPIRV`: key -> compiled SPIR-V\n\
         //! (task/mesh stages, #111). `BAKED_NAMES`: key -> `shader / entry / defines`\n\
         //! (diagnostics on a miss). `BAKED_REFLECTION`: shader-set key -> baked\n\
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

    s.push_str(
        "/// key -> compiled SPIR-V (LE bytes), sorted ascending by key. Task/mesh\n\
         /// entry points only (#111) — they have no WGSL form.\n",
    );
    s.push_str("pub static BAKED_SPIRV: &[(u64, &[u8])] = &[\n");
    for (key, (bytes, _)) in &baked.spirv {
        s.push_str(&format!(
            "    ({key:#018x}, {}),\n",
            render_byte_string(bytes)
        ));
    }
    s.push_str("];\n\n");

    s.push_str("/// key -> human-readable `shader / entry / defines`, sorted ascending by key.\n");
    s.push_str("pub static BAKED_NAMES: &[(u64, &str)] = &[\n");
    let names = baked
        .wgsl
        .iter()
        .map(|(k, (_, n))| (*k, n))
        .chain(baked.spirv.iter().map(|(k, (_, n))| (*k, n)))
        .collect::<BTreeMap<u64, &String>>();
    for (key, name) in names {
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
