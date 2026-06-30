//! Generator for the engine `std` asset pack. Run to (re)generate `std-assets/`:
//!   cargo run -p redlilium-editor --bin gen_std_assets
//!
//! Writes the mesh blobs + the (empty) vertex-layout file + `assets.db`, and
//! records the (pre-existing, committed) shader `.slang` files. Guids are
//! **deterministic** (derived from the mount-relative path via `Guid::stable`),
//! so regeneration produces the same DB — no churn, and references survive.

use std::collections::BTreeMap;

use redlilium_assets::{AssetDb, AssetPath, AssetRecord, Guid};
use redlilium_core::mesh::{CpuMeshData, VertexLayout, generators};
use redlilium_ecs::std::rendering::{MaterialData, MaterialInstanceData, PropValue};

/// Insert a record with a deterministic guid keyed by its mount-relative path.
fn add(
    db: &mut AssetDb,
    path: &str,
    kind: &str,
    settings: Option<String>,
    references: BTreeMap<String, Guid>,
) -> Guid {
    let guid = Guid::stable(path);
    db.insert(
        guid,
        AssetRecord {
            path: AssetPath::new("std", path),
            kind: kind.to_owned(),
            source_hash: 0,
            settings,
            references,
        },
    )
    .expect("insert std record");
    guid
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
    std::fs::write(
        "std-assets/meshes/cube.rmesh",
        bincode::serialize(&cube).unwrap(),
    )
    .unwrap();
    std::fs::write(
        "std-assets/meshes/sphere.rmesh",
        bincode::serialize(&sphere).unwrap(),
    )
    .unwrap();
    add(
        &mut db,
        "meshes/cube.rmesh",
        "mesh",
        None,
        layout_ref(pn_guid),
    );
    add(
        &mut db,
        "meshes/sphere.rmesh",
        "mesh",
        None,
        layout_ref(pn_guid),
    );

    // --- Shaders: the `.slang` files are committed content; just record them. ---
    add(
        &mut db,
        "shaders/opaque_color.slang",
        "shader",
        None,
        BTreeMap::new(),
    );
    add(
        &mut db,
        "shaders/entity_index.slang",
        "shader",
        None,
        BTreeMap::new(),
    );

    // --- Materials: a surface (shading model + values) + a bindable instance. ---
    // The material references no shader — the `opaque` shading model (engine code)
    // owns that. Data lives in the record settings; the files are empty.
    std::fs::create_dir_all("std-assets/materials").unwrap();
    let mat_data = MaterialData {
        shading_model: "opaque".to_owned(),
        properties: vec![(
            "base_color".to_owned(),
            PropValue::Vec4([0.6, 0.6, 0.65, 1.0]),
        )],
    };
    let mat_guid = add(
        &mut db,
        "materials/opaque.material",
        "material",
        Some(ron::to_string(&mat_data).expect("material -> ron")),
        BTreeMap::new(),
    );
    std::fs::write("std-assets/materials/opaque.material", b"").unwrap();

    let inst_data = MaterialInstanceData {
        parent: mat_guid,
        overrides: Vec::new(),
    };
    add(
        &mut db,
        "materials/default.matinst",
        "material_instance",
        Some(ron::to_string(&inst_data).expect("instance -> ron")),
        BTreeMap::new(),
    );
    std::fs::write("std-assets/materials/default.matinst", b"").unwrap();

    std::fs::write(
        "std-assets/assets.db",
        db.to_ron_for_mount("std").expect("db -> ron"),
    )
    .unwrap();
    println!("generated std-assets/ ({} records)", db.len());
}
