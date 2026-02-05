//! Integration test: load the ToyCar.glb sample model.

use crate::gltf::load_gltf;

const TOY_CAR_GLB: &[u8] = include_bytes!("ToyCar.glb");

#[test]
fn test_load_toy_car() {
    let doc = load_gltf(TOY_CAR_GLB, &[]).expect("failed to load ToyCar.glb");

    println!("Loaded {} meshes", doc.meshes.len());
    for (i, mesh) in doc.meshes.iter().enumerate() {
        println!(
            "  mesh {}: vertices={}, indices={}, layout_buffers={}, material={:?}",
            i,
            mesh.vertex_count(),
            mesh.index_count(),
            mesh.layout().buffer_count(),
            mesh.material(),
        );
    }

    // ToyCar has meshes
    assert!(!doc.meshes.is_empty(), "expected at least one mesh");

    // Every mesh must have vertex data
    for (i, mesh) in doc.meshes.iter().enumerate() {
        assert!(mesh.vertex_count() > 0, "mesh {i} has no vertices");
        assert!(
            mesh.vertex_buffer_data(0).is_some(),
            "mesh {i} has no vertex buffer data"
        );
        let data = mesh.vertex_buffer_data(0).unwrap();
        assert!(!data.is_empty(), "mesh {i} vertex buffer is empty");
    }
}

#[test]
fn test_toy_car_materials_on_meshes() {
    let doc = load_gltf(TOY_CAR_GLB, &[]).expect("failed to load ToyCar.glb");

    assert!(!doc.materials.is_empty(), "expected materials");

    // Every mesh should have a material assigned
    for (i, mesh) in doc.meshes.iter().enumerate() {
        let mat_idx = mesh.material();
        assert!(mat_idx.is_some(), "mesh {i} has no material");
        assert!(
            mat_idx.unwrap() < doc.materials.len(),
            "mesh {i} material index out of range"
        );
    }

    for (i, mat) in doc.materials.iter().enumerate() {
        println!(
            "  material {}: name={:?}, base_color={:?}, metallic={}, roughness={}",
            i, mat.name, mat.base_color_factor, mat.metallic_factor, mat.roughness_factor,
        );
    }
}

#[test]
fn test_toy_car_has_textures_and_images() {
    let doc = load_gltf(TOY_CAR_GLB, &[]).expect("failed to load ToyCar.glb");

    assert!(!doc.textures.is_empty(), "expected textures");
    assert!(!doc.images.is_empty(), "expected images");

    for (i, img) in doc.images.iter().enumerate() {
        println!(
            "  image {}: name={:?}, {}x{}, {} bytes",
            i,
            img.name,
            img.width,
            img.height,
            img.data.len(),
        );
        // RGBA8: 4 bytes per pixel
        assert_eq!(
            img.data.len(),
            (img.width * img.height * 4) as usize,
            "image {i} data size doesn't match RGBA8 dimensions"
        );
    }
}

#[test]
fn test_toy_car_has_scene() {
    let doc = load_gltf(TOY_CAR_GLB, &[]).expect("failed to load ToyCar.glb");

    assert!(!doc.scenes.is_empty(), "expected at least one scene");
    assert!(doc.default_scene.is_some(), "expected a default scene");

    let scene = &doc.scenes[doc.default_scene.unwrap()];
    assert!(!scene.nodes.is_empty(), "expected nodes in default scene");

    // Count total nodes recursively
    fn count_nodes(nodes: &[crate::gltf::GltfNode]) -> usize {
        nodes.iter().map(|n| 1 + count_nodes(&n.children)).sum()
    }
    let total = count_nodes(&scene.nodes);
    println!("Default scene has {} total nodes", total);
    assert!(total > 0);

    // Verify nodes reference valid mesh indices
    fn check_mesh_refs(nodes: &[crate::gltf::GltfNode], mesh_count: usize) {
        for node in nodes {
            for &idx in &node.meshes {
                assert!(idx < mesh_count, "node mesh index {idx} out of range");
            }
            check_mesh_refs(&node.children, mesh_count);
        }
    }
    check_mesh_refs(&scene.nodes, doc.meshes.len());
}

#[test]
fn test_toy_car_layout_sharing() {
    // Load with no shared layouts — all layouts should be new
    let doc1 = load_gltf(TOY_CAR_GLB, &[]).expect("failed to load ToyCar.glb");
    assert!(
        !doc1.new_layouts.is_empty(),
        "expected new layouts when none shared"
    );

    // Load again with the previously created layouts as shared
    let doc2 = load_gltf(TOY_CAR_GLB, &doc1.new_layouts).expect("failed to load ToyCar.glb");
    assert!(
        doc2.new_layouts.is_empty(),
        "expected no new layouts when all are shared, got {}",
        doc2.new_layouts.len()
    );

    // Verify that meshes in doc2 point to layouts from doc1
    for mesh in &doc2.meshes {
        let layout = mesh.layout();
        let shared = doc1
            .new_layouts
            .iter()
            .any(|l| std::sync::Arc::ptr_eq(l, layout));
        assert!(shared, "mesh layout should be a shared Arc from doc1");
    }
}
