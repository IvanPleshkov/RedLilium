//! Generator for the engine `std` asset pack. Run to (re)generate `std-assets/`:
//!   cargo run -p redlilium-editor --bin gen_std_assets
//!
//! Writes the mesh blobs + the (empty) settings-carrying files + `assets.db`, and
//! records the (pre-existing, committed) shader `.slang` and texture files. Guids
//! are **deterministic** (derived from the mount-relative path via `Guid::stable`)
//! and each record's `source_hash` is the FNV-1a of the source file bytes — the
//! same hash the asset scanner computes — so regenerating this pack reproduces
//! the committed DB byte-for-byte (no churn), and references survive.

use std::collections::BTreeMap;

use redlilium_assets::{AssetDb, AssetPath, AssetRecord, Guid};
use redlilium_core::mesh::{CpuMeshData, VertexLayout, generators};
use redlilium_ecs::std::rendering::{MaterialData, MaterialInstanceData, PropValue, TextureSource};

/// FNV-1a 64-bit of the source file bytes — the exact change-detection hash the
/// asset scanner persists (`assets::scan`), so a regenerated pack matches a
/// scanned one. A settings-only asset has an empty source file, so its hash is
/// the FNV offset basis.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Read a committed source file (shader / texture) whose content this generator
/// records but does not author.
fn read(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

/// Insert a record with a deterministic guid keyed by its mount-relative path and
/// a `source_hash` over the given source bytes (empty for settings-only assets).
fn add(
    db: &mut AssetDb,
    path: &str,
    kind: &str,
    source: &[u8],
    settings: Option<String>,
    references: BTreeMap<String, Guid>,
) -> Guid {
    let guid = Guid::stable(path);
    db.insert(
        guid,
        AssetRecord {
            path: AssetPath::new("std", path),
            kind: kind.to_owned(),
            source_hash: fnv1a(source),
            settings,
            references,
        },
    )
    .expect("insert std record");
    guid
}

/// Add a `material` record (settings-only, empty file) and return its guid.
fn material(db: &mut AssetDb, path: &str, data: &MaterialData) -> Guid {
    let guid = add(
        db,
        path,
        "material",
        &[],
        Some(ron::to_string(data).expect("material -> ron")),
        BTreeMap::new(),
    );
    std::fs::write(format!("std-assets/{path}"), b"").unwrap();
    guid
}

/// Add a `material_instance` record (settings-only, empty file) parented to `parent`.
fn instance(db: &mut AssetDb, path: &str, parent: Guid) {
    let data = MaterialInstanceData {
        parent,
        overrides: Vec::new(),
    };
    add(
        db,
        path,
        "material_instance",
        &[],
        Some(ron::to_string(&data).expect("instance -> ron")),
        BTreeMap::new(),
    );
    std::fs::write(format!("std-assets/{path}"), b"").unwrap();
}

/// Repack position_normal_uv (stride 32) -> position_normal (stride 24): drop UVs
/// so the cube and sphere share one layout (the demo shader uses only pos+normal).
fn drop_uvs(mut data: CpuMeshData) -> CpuMeshData {
    let src = &data.vertex_buffers[0];
    let mut pn = Vec::with_capacity(data.vertex_count as usize * 24);
    for v in 0..data.vertex_count as usize {
        let off = v * 32;
        pn.extend_from_slice(&src[off..off + 24]);
    }
    data.vertex_buffers = vec![pn];
    data
}

fn main() {
    std::fs::create_dir_all("std-assets/meshes").unwrap();
    std::fs::create_dir_all("std-assets/layouts").unwrap();
    let mut db = AssetDb::new();

    // --- Vertex layout: parameters in the record's settings, empty file. ---
    let pn = (*VertexLayout::position_normal()).clone();
    let pn_guid = add(
        &mut db,
        "layouts/position_normal.vlayout",
        "vertex_layout",
        &[],
        Some(ron::to_string(&pn).expect("layout -> ron")),
        BTreeMap::new(),
    );
    std::fs::write("std-assets/layouts/position_normal.vlayout", b"").unwrap();

    // --- Meshes: bincode blob + shared layout reference. ---
    let layout_ref = |g: Guid| BTreeMap::from([("layout".to_owned(), g)]);
    let cube = CpuMeshData::from_cpu_mesh(&generators::generate_cube(0.5));
    let sphere = drop_uvs(CpuMeshData::from_cpu_mesh(&generators::generate_sphere(
        0.5, 32, 16,
    )));
    let cube_blob = bincode::serialize(&cube).unwrap();
    let sphere_blob = bincode::serialize(&sphere).unwrap();
    std::fs::write("std-assets/meshes/cube.rmesh", &cube_blob).unwrap();
    std::fs::write("std-assets/meshes/sphere.rmesh", &sphere_blob).unwrap();
    add(
        &mut db,
        "meshes/cube.rmesh",
        "mesh",
        &cube_blob,
        None,
        layout_ref(pn_guid),
    );
    add(
        &mut db,
        "meshes/sphere.rmesh",
        "mesh",
        &sphere_blob,
        None,
        layout_ref(pn_guid),
    );

    // --- Shaders + texture: committed content this tool records but does not
    // author. Hash their on-disk bytes so the record matches a scan. ---
    for slang in [
        "opaque_color",
        "entity_index",
        "opaque_textured",
        "depth_only",
        "layered_decal",
    ] {
        let path = format!("shaders/{slang}.slang");
        add(
            &mut db,
            &path,
            "shader",
            &read(&format!("std-assets/{path}")),
            None,
            BTreeMap::new(),
        );
    }
    let checker_guid = add(
        &mut db,
        "textures/checker.png",
        "texture",
        &read("std-assets/textures/checker.png"),
        None,
        BTreeMap::new(),
    );

    // --- Materials: a surface (shading model + values) + a bindable instance.
    // The material references no shader — the shading model (engine code) owns
    // that. Data lives in the record settings; the files are empty. ---
    std::fs::create_dir_all("std-assets/materials").unwrap();
    let opaque_guid = material(
        &mut db,
        "materials/opaque.material",
        &MaterialData {
            shading_model: "opaque".to_owned(),
            properties: vec![(
                "base_color".to_owned(),
                PropValue::Vec4([0.6, 0.6, 0.65, 1.0]),
            )],
            features: Vec::new(),
        },
    );
    instance(&mut db, "materials/default.matinst", opaque_guid);

    let textured_guid = material(
        &mut db,
        "materials/textured.material",
        &MaterialData {
            shading_model: "opaque_textured".to_owned(),
            properties: vec![(
                "base_texture".to_owned(),
                PropValue::Texture(TextureSource::File(checker_guid)),
            )],
            features: Vec::new(),
        },
    );
    instance(&mut db, "materials/textured.matinst", textured_guid);

    // Baked-decal channel (procedural: design/decals-design.md). Instances are
    // published programmatically by the preview with per-layer overrides; the record
    // only needs to name the shading model so the defaults (strength 0 = no layers)
    // make a decal-less instance render as the base.
    material(
        &mut db,
        "materials/layered_decal.material",
        &MaterialData {
            shading_model: "layered_decal".to_owned(),
            properties: Vec::new(),
            features: Vec::new(),
        },
    );

    std::fs::write(
        "std-assets/assets.db",
        db.to_ron_for_mount("std").expect("db -> ron"),
    )
    .unwrap();
    println!("generated std-assets/ ({} records)", db.len());
}
