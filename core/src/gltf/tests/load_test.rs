//! Integration test: load the ToyCar.glb sample model.

use std::sync::Arc;

use crate::gltf::load_gltf;
use crate::material::CpuMaterialInstance;

use super::default_pbr_material;

const TOY_CAR_GLB: &[u8] = include_bytes!("ToyCar.glb");

/// Helper: get the default scene from a loaded document.
fn default_scene(doc: &crate::gltf::GltfDocument) -> &crate::scene::Scene {
    let idx = doc.default_scene.expect("expected a default scene");
    &doc.scenes[idx]
}

/// Collect all unique material instances from a document's scenes.
fn collect_instances(doc: &crate::gltf::GltfDocument) -> Vec<Arc<CpuMaterialInstance>> {
    let mut instances: Vec<Arc<CpuMaterialInstance>> = Vec::new();
    for scene in &doc.scenes {
        for inst in &scene.materials {
            if !instances.iter().any(|m| Arc::ptr_eq(m, inst)) {
                instances.push(Arc::clone(inst));
            }
        }
    }
    instances
}

#[test]
fn test_load_toy_car() {
    let doc =
        load_gltf(TOY_CAR_GLB, &[], &[], default_pbr_material).expect("failed to load ToyCar.glb");
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
    let doc =
        load_gltf(TOY_CAR_GLB, &[], &[], default_pbr_material).expect("failed to load ToyCar.glb");
    let scene = default_scene(&doc);

    let instances = collect_instances(&doc);
    assert!(!instances.is_empty(), "expected material instances");

    // Every mesh should have a material assigned
    for (i, mesh) in scene.meshes.iter().enumerate() {
        assert!(mesh.material().is_some(), "mesh {i} has no material");
    }

    for (i, inst) in instances.iter().enumerate() {
        let base_color = inst.get_vec4("base_color");
        let metallic = inst.get_float("metallic");
        let roughness = inst.get_float("roughness");
        println!(
            "  instance {}: name={:?}, base_color={:?}, metallic={:?}, roughness={:?}",
            i, inst.name, base_color, metallic, roughness,
        );
    }
}

#[test]
fn test_toy_car_has_textures() {
    use crate::material::TextureSource;

    let doc =
        load_gltf(TOY_CAR_GLB, &[], &[], default_pbr_material).expect("failed to load ToyCar.glb");

    let instances = collect_instances(&doc);

    // Collect all textures referenced by material instances
    let mut texture_count = 0;
    for inst in &instances {
        for tex_ref in inst.textures() {
            if let TextureSource::Cpu(cpu_tex) = &tex_ref.texture {
                println!(
                    "  texture in {:?}: name={:?}, {}x{}, format={:?}, {} bytes",
                    inst.name,
                    cpu_tex.name,
                    cpu_tex.width,
                    cpu_tex.height,
                    cpu_tex.format,
                    cpu_tex.data.len(),
                );
                // RGBA8: 4 bytes per pixel
                assert_eq!(
                    cpu_tex.data.len(),
                    (cpu_tex.width * cpu_tex.height * 4) as usize,
                    "texture data size doesn't match RGBA8 dimensions"
                );
                texture_count += 1;
            }
        }
    }
    assert!(texture_count > 0, "expected textures in materials");

    // Verify base color textures are accessible via get_texture
    for inst in &instances {
        if let Some(tex_ref) = inst.get_texture("base_color_texture") {
            assert!(
                matches!(&tex_ref.texture, TextureSource::Cpu(_)),
                "expected Cpu texture source"
            );
        }
    }
}

#[test]
fn test_toy_car_has_scene() {
    let doc =
        load_gltf(TOY_CAR_GLB, &[], &[], default_pbr_material).expect("failed to load ToyCar.glb");

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
