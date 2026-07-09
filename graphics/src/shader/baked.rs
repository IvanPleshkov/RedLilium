//! Offline-baked slang→WGSL lookup (#33).
//!
//! The Slang compiler is a native C++ library and cannot run in a browser, so
//! on wasm the engine cannot compile `.slang` at runtime. Instead the `xtask
//! bake-shaders` tool compiles every shader permutation to WGSL offline and
//! emits [`baked_generated`] — a table keyed by [`shader_key`]. At runtime the
//! wgpu backend's `compile_to_wgsl` looks the WGSL up here instead of invoking
//! Slang.
//!
//! **Keying must match bake-time and run-time exactly.** Both sides call
//! [`shader_key`] over the *normalized* source ([`normalize_source`]) so a CRLF
//! checkout (Windows / `core.autocrlf`) cannot silently miss the bake. A miss is
//! reported loudly with the shader's name via [`name_for_key`] — never a silent
//! wrong/stale shader.

use crate::materials::{
    BindingLayout, BindingLayoutEntry, BindingType, ShaderStage, ShaderStageFlags, UpdateRate,
};

#[path = "baked_generated.rs"]
mod baked_generated;

/// Normalize shader source for stable, cross-platform hashing: convert CRLF/CR
/// to LF and strip trailing whitespace on each line. Line-ending drift between a
/// bake host and a browser checkout would otherwise change the hash and miss the
/// baked entry (there is no runtime Slang fallback on wasm).
pub fn normalize_source(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for (i, line) in src.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(line.trim_end());
    }
    out
}

/// FNV-1a 64-bit — a small, dependency-free, deterministic hash (stable across
/// platforms and toolchains, unlike `std`'s `DefaultHasher`), so the committed
/// baked table's keys match what the runtime computes.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Stable key for a baked WGSL entry: hash of the normalized source plus the
/// entry point and the sorted defines. Sorting the defines makes the key
/// independent of the order the caller listed them in.
pub fn shader_key(source: &str, entry_point: &str, defines: &[(&str, &str)]) -> u64 {
    let normalized = normalize_source(source);
    let mut sorted: Vec<(&str, &str)> = defines.to_vec();
    sorted.sort_unstable();

    let mut buf = Vec::with_capacity(normalized.len() + entry_point.len() + 16);
    buf.extend_from_slice(normalized.as_bytes());
    buf.push(0);
    buf.extend_from_slice(entry_point.as_bytes());
    buf.push(0);
    for (k, v) in sorted {
        buf.extend_from_slice(k.as_bytes());
        buf.push(b'=');
        buf.extend_from_slice(v.as_bytes());
        buf.push(0);
    }
    fnv1a(&buf)
}

/// Look up baked WGSL for `(source, entry_point, defines)`. `None` means the
/// permutation was never baked — the caller must fail loudly (see
/// [`name_for_key`]); on wasm there is no Slang fallback.
pub fn lookup(source: &str, entry_point: &str, defines: &[(&str, &str)]) -> Option<&'static str> {
    let key = shader_key(source, entry_point, defines);
    baked_generated::BAKED_WGSL
        .binary_search_by_key(&key, |(k, _)| *k)
        .ok()
        .map(|i| baked_generated::BAKED_WGSL[i].1)
}

/// Human-readable name (`shader / entry / defines`) for a key, for diagnostics
/// on a miss. Present for every baked permutation so a miss can name the shader
/// that needs (re)baking rather than a bare hex hash.
pub fn name_for_key(
    source: &str,
    entry_point: &str,
    defines: &[(&str, &str)],
) -> Option<&'static str> {
    let key = shader_key(source, entry_point, defines);
    baked_generated::BAKED_NAMES
        .binary_search_by_key(&key, |(k, _)| *k)
        .ok()
        .map(|i| baked_generated::BAKED_NAMES[i].1)
}

/// Stage discriminator for [`shader_set_key`], so two stages that share an
/// entry-point name cannot collide in the folded key.
fn stage_tag(stage: ShaderStage) -> u8 {
    match stage {
        ShaderStage::Vertex => 0,
        ShaderStage::Fragment => 1,
        ShaderStage::Compute => 2,
    }
}

/// One shader in a set for [`shader_set_key`]: `(source, entry_point, stage,
/// defines)` — the same shape as `ShaderReflectInput`, but defined here so it is
/// available on wasm (where the Slang compiler module is absent).
pub type ShaderSetInput<'a> = (&'a str, &'a str, ShaderStage, &'a [(&'a str, &'a str)]);

/// Stable key for a baked **reflection** entry — a material's whole shader SET
/// (all stages together), not a single WGSL permutation (#33).
///
/// `reflect_all_bindings` merges the stages order-independently, so this folds
/// the per-shader [`shader_key`]s after sorting: the key is independent of the
/// order the caller lists the shaders in. `xtask bake-shaders` and
/// [`GraphicsDevice::create_material`](crate::GraphicsDevice::create_material)
/// must compute it identically, or the wasm lookup misses.
pub fn shader_set_key(shaders: &[ShaderSetInput<'_>]) -> u64 {
    let mut subs: Vec<(u8, u64)> = shaders
        .iter()
        .map(|(src, entry, stage, defs)| (stage_tag(*stage), shader_key(src, entry, defs)))
        .collect();
    subs.sort_unstable();

    let mut buf = Vec::with_capacity(subs.len() * 9);
    for (tag, key) in subs {
        buf.push(tag);
        buf.extend_from_slice(&key.to_le_bytes());
    }
    fnv1a(&buf)
}

/// One binding entry in a baked group — the `'static` form of
/// [`BindingLayoutEntry`], emitted by `xtask bake-shaders`.
pub struct BakedEntry {
    /// Binding index within the group.
    pub binding: u32,
    /// Resource type expected at this binding.
    pub ty: BindingType,
    /// [`ShaderStageFlags`] bits (stage visibility).
    pub vis: u32,
    /// Optional debug label.
    pub label: Option<&'static str>,
}

/// One binding group in a baked reflection — the `'static` form of
/// [`BindingLayout`] plus its [`UpdateRate`] (parallel to the layout, exactly as
/// native `reflect_all_bindings` returns them).
pub struct BakedGroup {
    /// Update-frequency class of the whole set (`None` for legacy sets).
    pub rate: Option<UpdateRate>,
    /// Optional debug label.
    pub label: Option<&'static str>,
    /// Binding entries in this group.
    pub entries: &'static [BakedEntry],
}

/// Look up baked binding-layout reflection for a shader set (keyed by
/// [`shader_set_key`]).
///
/// Returns the densified per-group layouts and their update rates exactly as
/// native `reflect_all_bindings` produced them at bake time — **post-densify,
/// pre-reclassification** (the Dynamic/External→`DynamicUniformBuffer` step and
/// the legacy `dynamic_uniforms` step run afterward, in `create_material`, on
/// both native and wasm so they can't drift). `None` means the set was never
/// baked; the caller must fail loudly (there is no runtime Slang on wasm).
pub fn lookup_reflection(key: u64) -> Option<(Vec<BindingLayout>, Vec<Option<UpdateRate>>)> {
    let groups = baked_generated::BAKED_REFLECTION
        .binary_search_by_key(&key, |(k, _)| *k)
        .ok()
        .map(|i| baked_generated::BAKED_REFLECTION[i].1)?;

    let mut layouts = Vec::with_capacity(groups.len());
    let mut rates = Vec::with_capacity(groups.len());
    for g in groups {
        layouts.push(BindingLayout {
            entries: g
                .entries
                .iter()
                .map(|e| BindingLayoutEntry {
                    binding: e.binding,
                    binding_type: e.ty,
                    visibility: ShaderStageFlags::from_bits_truncate(e.vis),
                    label: e.label.map(str::to_string),
                })
                .collect(),
            label: g.label.map(str::to_string),
        });
        rates.push(g.rate);
    }
    Some((layouts, rates))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_collapses_crlf_and_trailing_ws() {
        assert_eq!(normalize_source("a \r\nb\t\r\nc"), "a\nb\nc");
        assert_eq!(normalize_source("x\ny"), "x\ny");
    }

    #[test]
    fn key_is_line_ending_and_define_order_independent() {
        let lf = shader_key("void main(){}\n", "main", &[("A", ""), ("B", "1")]);
        let crlf = shader_key("void main(){}\r\n", "main", &[("B", "1"), ("A", "")]);
        assert_eq!(lf, crlf);
    }

    #[test]
    fn key_separates_entry_and_defines() {
        let a = shader_key("s", "vs_main", &[]);
        let b = shader_key("s", "fs_main", &[]);
        let c = shader_key("s", "vs_main", &[("HDR_OUTPUT", "")]);
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    /// Every offline-baked WGSL entry (`xtask bake-shaders`) must parse and
    /// validate through naga — the same frontend wgpu uses. This exercises the
    /// exact bytes shipped to wasm on a platform we CAN run (there is no browser
    /// here), so a bad bake fails `cargo test`, not only in a tab.
    #[test]
    fn all_baked_wgsl_parses_and_validates() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        for (key, wgsl) in super::baked_generated::BAKED_WGSL {
            let module = naga::front::wgsl::parse_str(wgsl)
                .unwrap_or_else(|e| panic!("baked WGSL {key:#018x} failed to parse: {e}"));
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("baked WGSL {key:#018x} failed naga validation: {e:?}"));
        }
    }

    /// The generated table must be sorted ascending by key — `lookup`/`name_for_key`
    /// binary-search it, and sorted emission keeps the committed diff stable.
    #[test]
    fn baked_table_is_sorted_by_key() {
        assert!(
            super::baked_generated::BAKED_WGSL
                .windows(2)
                .all(|w| w[0].0 < w[1].0),
            "BAKED_WGSL not strictly ascending by key"
        );
        assert!(
            super::baked_generated::BAKED_NAMES
                .windows(2)
                .all(|w| w[0].0 < w[1].0),
            "BAKED_NAMES not strictly ascending by key"
        );
    }
}
