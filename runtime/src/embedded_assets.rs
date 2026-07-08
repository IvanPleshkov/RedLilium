//! Embedded std-assets for wasm (#33).
//!
//! A browser has no local disk, so `EngineContext::new` cannot mount the
//! `std-assets` pack off the filesystem. Instead the pack is compiled into the
//! wasm binary here (owner's choice: embed over HTTP-fetch) and served through a
//! [`MemoryProvider`]. `include_bytes!` stores every file **verbatim**, so the
//! `.slang` bytes the runtime hashes to look up baked WGSL are byte-identical to
//! the on-disk sources the `xtask bake-shaders` tool hashes — the two cannot
//! drift (see `redlilium_graphics::shader::baked`).
//!
//! Paths are pack-relative (forward slashes, no leading slash) to match the VFS
//! request paths the asset system issues.

use redlilium_vfs::MemoryProvider;

/// `(pack-relative path, verbatim bytes)` for every file in `std-assets`.
/// Keep in sync with the on-disk pack; the asset DB references these by path.
static STD_ASSETS: &[(&str, &[u8])] = &[
    ("assets.db", include_bytes!("../../std-assets/assets.db")),
    (
        "textures/checker.png",
        include_bytes!("../../std-assets/textures/checker.png"),
    ),
    (
        "materials/default.matinst",
        include_bytes!("../../std-assets/materials/default.matinst"),
    ),
    (
        "materials/opaque.material",
        include_bytes!("../../std-assets/materials/opaque.material"),
    ),
    (
        "materials/textured.matinst",
        include_bytes!("../../std-assets/materials/textured.matinst"),
    ),
    (
        "materials/textured.material",
        include_bytes!("../../std-assets/materials/textured.material"),
    ),
    (
        "meshes/cube.rmesh",
        include_bytes!("../../std-assets/meshes/cube.rmesh"),
    ),
    (
        "meshes/sphere.rmesh",
        include_bytes!("../../std-assets/meshes/sphere.rmesh"),
    ),
    (
        "layouts/position_normal.vlayout",
        include_bytes!("../../std-assets/layouts/position_normal.vlayout"),
    ),
    (
        "shaders/entity_index.slang",
        include_bytes!("../../std-assets/shaders/entity_index.slang"),
    ),
    (
        "shaders/opaque_textured.slang",
        include_bytes!("../../std-assets/shaders/opaque_textured.slang"),
    ),
    (
        "shaders/opaque_color.slang",
        include_bytes!("../../std-assets/shaders/opaque_color.slang"),
    ),
];

fn pack_for(dir: &str) -> Option<&'static [(&'static str, &'static [u8])]> {
    match dir {
        "std-assets" => Some(STD_ASSETS),
        _ => None,
    }
}

/// A [`MemoryProvider`] populated with the embedded pack for `dir`, or `None` if
/// no pack is embedded for that mount (e.g. an empty `project-assets`). `dir` is
/// the mount's source directory as configured in `GameConfig::mounts`.
pub fn provider_for(dir: &str) -> Option<MemoryProvider> {
    let files = pack_for(dir)?;
    let provider = MemoryProvider::new();
    for &(path, bytes) in files {
        provider.insert(path, bytes.to_vec());
    }
    Some(provider)
}

/// The embedded `assets.db` (RON text) for `dir`, or `None` if not embedded.
/// Read directly (not through the async VFS) so `EngineContext::new` can merge
/// the database synchronously, mirroring the native `std::fs` path.
pub fn assets_db_text(dir: &str) -> Option<String> {
    let files = pack_for(dir)?;
    let bytes = files.iter().find(|(p, _)| *p == "assets.db").map(|(_, b)| *b)?;
    Some(String::from_utf8_lossy(bytes).into_owned())
}
