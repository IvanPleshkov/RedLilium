//! Integration test: load the ToyCar.glb sample model.

use crate::gltf::load_gltf;

const TOY_CAR_GLB: &[u8] = include_bytes!("ToyCar.glb");

/// Helper: get the default scene from a loaded document.
fn default_scene(doc: &crate::gltf::GltfDocument) -> &crate::scene::Scene {
    let idx = doc.default_scene.expect("expected a default scene");
    &doc.scenes[idx]
}

#[test]
fn test_load_toy_car() {
    let doc = load_gltf(TOY_CAR_GLB, &[]).expect("failed to load ToyCar.glb");
    let scene = default_scene(&doc);

    println!("Loaded {} meshes", scene.meshes.len());
    for (i, mesh) in scene.meshes.iter().enumerate() {
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
    assert!(!scene.meshes.is_empty(), "expected at least one mesh");

    // Every mesh must have vertex data
    for (i, mesh) in scene.meshes.iter().enumerate() {
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
    let scene = default_scene(&doc);

    assert!(!doc.materials.is_empty(), "expected materials");

    // Every mesh should have a material assigned
    for (i, mesh) in scene.meshes.iter().enumerate() {
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
fn test_toy_car_has_textures() {
    let doc = load_gltf(TOY_CAR_GLB, &[]).expect("failed to load ToyCar.glb");

    assert!(!doc.textures.is_empty(), "expected textures");

    for (i, tex) in doc.textures.iter().enumerate() {
        println!(
            "  texture {}: name={:?}, {}x{}, format={:?}, {} bytes",
            i,
            tex.name,
            tex.width,
            tex.height,
            tex.format,
            tex.data.len(),
        );
        // RGBA8: 4 bytes per pixel
        assert_eq!(
            tex.data.len(),
            (tex.width * tex.height * 4) as usize,
            "texture {i} data size doesn't match RGBA8 dimensions"
        );
    }
}

#[test]
fn test_toy_car_has_scene() {
    let doc = load_gltf(TOY_CAR_GLB, &[]).expect("failed to load ToyCar.glb");

    assert!(!doc.scenes.is_empty(), "expected at least one scene");
    let scene = default_scene(&doc);
    assert!(!scene.nodes.is_empty(), "expected nodes in default scene");

    // Count total nodes recursively
    fn count_nodes(nodes: &[crate::scene::SceneNode]) -> usize {
        nodes.iter().map(|n| 1 + count_nodes(&n.children)).sum()
    }
    let total = count_nodes(&scene.nodes);
    println!("Default scene has {} total nodes", total);
    assert!(total > 0);

    // Verify nodes reference valid mesh indices
    fn check_mesh_refs(nodes: &[crate::scene::SceneNode], mesh_count: usize) {
        for node in nodes {
            for &idx in &node.meshes {
                assert!(idx < mesh_count, "node mesh index {idx} out of range");
            }
            check_mesh_refs(&node.children, mesh_count);
        }
    }
    check_mesh_refs(&scene.nodes, scene.meshes.len());
}
