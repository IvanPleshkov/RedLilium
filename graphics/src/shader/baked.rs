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
