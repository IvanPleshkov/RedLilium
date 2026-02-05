//! Integration test: load the ToyCar.glb sample model.

use std::sync::Arc;

use crate::gltf::load_gltf;
use crate::material::CpuMaterial;

const TOY_CAR_GLB: &[u8] = include_bytes!("ToyCar.glb");

/// Helper: get the default scene from a loaded document.
fn default_scene(doc: &crate::gltf::GltfDocument) -> &crate::scene::Scene {
    let idx = doc.default_scene.expect("expected a default scene");
    &doc.scenes[idx]
}

/// Collect all unique materials from a document's meshes.
fn collect_materials(doc: &crate::gltf::GltfDocument) -> Vec<Arc<CpuMaterial>> {
    let mut materials: Vec<Arc<CpuMaterial>> = Vec::new();
    for scene in &doc.scenes {
        for mesh in &scene.meshes {
            if let Some(mat) = mesh.material()
                && !materials.iter().any(|m| Arc::ptr_eq(m, mat))
            {
                materials.push(Arc::clone(mat));
            }
        }
    }
    materials
}

#[test]
fn test_load_toy_car() {
    let doc = load_gltf(TOY_CAR_GLB, &[], &[], &[]).expect("failed to load ToyCar.glb");
    let scene = default_scene(&doc);

    println!("Loaded {} meshes", scene.meshes.len());
    for (i, mesh) in scene.meshes.iter().enumerate() {
        println!(
            "  mesh {}: vertices={}, indices={}, layout_buffers={}, material={:?}",
            i,
            mesh.vertex_count(),
            mesh.index_count(),
            mesh.layout().buffer_count(),
            mesh.material().map(|m| m.name.as_deref()),
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
    let doc = load_gltf(TOY_CAR_GLB, &[], &[], &[]).expect("failed to load ToyCar.glb");
    let scene = default_scene(&doc);

    let materials = collect_materials(&doc);
    assert!(!materials.is_empty(), "expected materials");

    // Every mesh should have a material assigned
    for (i, mesh) in scene.meshes.iter().enumerate() {
        assert!(mesh.material().is_some(), "mesh {i} has no material");
    }

    for (i, mat) in materials.iter().enumerate() {
        use crate::material::MaterialSemantic;
        let base_color = mat.get_vec4(&MaterialSemantic::BaseColorFactor);
        let metallic = mat.get_float(&MaterialSemantic::MetallicFactor);
        let roughness = mat.get_float(&MaterialSemantic::RoughnessFactor);
        println!(
            "  material {}: name={:?}, base_color={:?}, metallic={:?}, roughness={:?}",
            i, mat.name, base_color, metallic, roughness,
        );
    }
}

#[test]
fn test_toy_car_has_textures() {
    use crate::material::{MaterialSemantic, MaterialValue, TextureSource};

    let doc = load_gltf(TOY_CAR_GLB, &[], &[], &[]).expect("failed to load ToyCar.glb");

    let materials = collect_materials(&doc);

    // Collect all textures referenced by materials
    let mut texture_count = 0;
    for mat in &materials {
        for prop in &mat.properties {
            if let MaterialValue::Texture(tex_ref) = &prop.value
                && let TextureSource::Cpu(cpu_tex) = &tex_ref.texture
            {
                println!(
                    "  texture in {:?}: name={:?}, {}x{}, format={:?}, {} bytes",
                    prop.semantic,
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
    for mat in &materials {
        if let Some(tex_ref) = mat.get_texture(&MaterialSemantic::BaseColorTexture) {
            assert!(
                matches!(&tex_ref.texture, TextureSource::Cpu(_)),
                "expected Cpu texture source"
            );
        }
    }
}

#[test]
fn test_toy_car_has_scene() {
    let doc = load_gltf(TOY_CAR_GLB, &[], &[], &[]).expect("failed to load ToyCar.glb");

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
