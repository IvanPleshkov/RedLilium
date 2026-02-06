//! Roundtrip test: load → export → reload and verify equality.

use std::sync::Arc;

use crate::gltf::{load_gltf, save_gltf};
use crate::material::{CpuMaterialInstance, TextureSource};
use crate::mesh::VertexLayout;
use crate::sampler::CpuSampler;
use crate::scene::SceneNode;

const TOY_CAR_GLB: &[u8] = include_bytes!("ToyCar.glb");

/// Collect all unique Arc<VertexLayout> from a document's scenes.
fn collect_layouts(doc: &crate::gltf::GltfDocument) -> Vec<Arc<VertexLayout>> {
    let mut layouts: Vec<Arc<VertexLayout>> = Vec::new();
    for scene in &doc.scenes {
        for mesh in &scene.meshes {
            let layout = mesh.layout().clone();
            if !layouts.iter().any(|l| Arc::ptr_eq(l, &layout)) {
                layouts.push(layout);
            }
        }
    }
    layouts
}

/// Collect all unique Arc<CpuSampler> from a document's mesh material instances.
fn collect_samplers(doc: &crate::gltf::GltfDocument) -> Vec<Arc<CpuSampler>> {
    let mut samplers: Vec<Arc<CpuSampler>> = Vec::new();
    for inst in &collect_instances(doc) {
        for tex_ref in inst.textures() {
            if let Some(s) = &tex_ref.sampler
                && !samplers.iter().any(|existing| Arc::ptr_eq(existing, s))
            {
                samplers.push(Arc::clone(s));
            }
        }
    }
    samplers
}

/// Collect all unique Arc<CpuMaterialInstance> from a document's meshes.
fn collect_instances(doc: &crate::gltf::GltfDocument) -> Vec<Arc<CpuMaterialInstance>> {
    let mut instances: Vec<Arc<CpuMaterialInstance>> = Vec::new();
    for scene in &doc.scenes {
        for mesh in &scene.meshes {
            if let Some(inst) = mesh.material()
                && !instances.iter().any(|m| Arc::ptr_eq(m, inst))
            {
                instances.push(Arc::clone(inst));
            }
        }
    }
    instances
}

fn count_nodes(nodes: &[SceneNode]) -> usize {
    nodes.iter().map(|n| 1 + count_nodes(&n.children)).sum()
}

#[test]
fn test_roundtrip_toy_car() {
    // Step 1: Load the original document
    let original =
        load_gltf(TOY_CAR_GLB, &[], &[], &[]).expect("failed to load original ToyCar.glb");

    // Step 2: Export to GLB bytes
    let scene_refs: Vec<&_> = original.scenes.iter().collect();
    let glb_bytes = save_gltf(&scene_refs, original.default_scene).expect("failed to export glb");

    // Verify GLB header
    assert!(glb_bytes.len() > 12, "GLB too small");
    assert_eq!(&glb_bytes[0..4], &0x46546C67u32.to_le_bytes(), "bad magic");
    assert_eq!(&glb_bytes[4..8], &2u32.to_le_bytes(), "bad version");

    // Step 3: Collect shared resources from the original for reuse
    let shared_layouts = collect_layouts(&original);
    let shared_samplers = collect_samplers(&original);
    let shared_instances = collect_instances(&original);

    // Step 4: Reload from exported bytes, passing shared resources
    let reloaded = load_gltf(
        &glb_bytes,
        &shared_layouts,
        &shared_samplers,
        &shared_instances,
    )
    .expect("failed to reload exported glb");

    // -- Structural equality checks --

    // Scene count
    assert_eq!(
        original.scenes.len(),
        reloaded.scenes.len(),
        "scene count mismatch"
    );
    assert_eq!(
        original.default_scene, reloaded.default_scene,
        "default scene mismatch"
    );

    // Material instance count
    let orig_instances = collect_instances(&original);
    let re_instances = collect_instances(&reloaded);
    assert_eq!(
        orig_instances.len(),
        re_instances.len(),
        "material instance count mismatch"
    );

    // Per-scene checks
    for (si, (orig_scene, re_scene)) in original
        .scenes
        .iter()
        .zip(reloaded.scenes.iter())
        .enumerate()
    {
        assert_eq!(
            orig_scene.meshes.len(),
            re_scene.meshes.len(),
            "scene {si}: mesh count mismatch"
        );

        assert_eq!(
            orig_scene.cameras.len(),
            re_scene.cameras.len(),
            "scene {si}: camera count mismatch"
        );

        assert_eq!(
            orig_scene.animations.len(),
            re_scene.animations.len(),
            "scene {si}: animation count mismatch"
        );

        // Node structure
        assert_eq!(
            count_nodes(&orig_scene.nodes),
            count_nodes(&re_scene.nodes),
            "scene {si}: total node count mismatch"
        );

        // Per-mesh checks
        for (mi, (orig_mesh, re_mesh)) in orig_scene
            .meshes
            .iter()
            .zip(re_scene.meshes.iter())
            .enumerate()
        {
            assert_eq!(
                orig_mesh.vertex_count(),
                re_mesh.vertex_count(),
                "scene {si} mesh {mi}: vertex count mismatch"
            );
            assert_eq!(
                orig_mesh.index_count(),
                re_mesh.index_count(),
                "scene {si} mesh {mi}: index count mismatch"
            );
            assert_eq!(
                orig_mesh.topology(),
                re_mesh.topology(),
                "scene {si} mesh {mi}: topology mismatch"
            );
            assert_eq!(
                orig_mesh.index_format(),
                re_mesh.index_format(),
                "scene {si} mesh {mi}: index format mismatch"
            );

            // Arc::ptr_eq for material instance — shared instances should be reused
            match (orig_mesh.material(), re_mesh.material()) {
                (Some(om), Some(rm)) => {
                    assert!(
                        Arc::ptr_eq(om, rm),
                        "scene {si} mesh {mi}: CpuMaterialInstance Arc not shared (ptr_eq failed)"
                    );
                }
                (None, None) => {}
                _ => panic!("scene {si} mesh {mi}: material presence mismatch"),
            }

            // Vertex data byte equality
            let orig_vtx = orig_mesh.vertex_buffer_data(0).unwrap_or(&[]);
            let re_vtx = re_mesh.vertex_buffer_data(0).unwrap_or(&[]);
            assert_eq!(
                orig_vtx, re_vtx,
                "scene {si} mesh {mi}: vertex data mismatch"
            );

            // Index data byte equality
            let orig_idx = orig_mesh.index_data().unwrap_or(&[]);
            let re_idx = re_mesh.index_data().unwrap_or(&[]);
            assert_eq!(
                orig_idx, re_idx,
                "scene {si} mesh {mi}: index data mismatch"
            );

            // Arc::ptr_eq for VertexLayout — shared layouts should be reused
            assert!(
                Arc::ptr_eq(orig_mesh.layout(), re_mesh.layout()),
                "scene {si} mesh {mi}: VertexLayout Arc not shared (ptr_eq failed)"
            );
        }
    }

    // Per-instance checks (via collected instances)
    for (mi, (orig_inst, re_inst)) in orig_instances.iter().zip(re_instances.iter()).enumerate() {
        // Arc::ptr_eq should hold since we passed shared_instances
        assert!(
            Arc::ptr_eq(orig_inst, re_inst),
            "instance {mi}: Arc not shared (ptr_eq failed)"
        );

        // Sampler Arc::ptr_eq for each texture
        let texture_names = [
            "base_color_texture",
            "metallic_roughness_texture",
            "normal_texture",
            "occlusion_texture",
            "emissive_texture",
        ];
        for name in &texture_names {
            let orig_tex = orig_inst.get_texture(name);
            let re_tex = re_inst.get_texture(name);
            match (orig_tex, re_tex) {
                (Some(o), Some(r)) => {
                    assert_eq!(
                        o.tex_coord, r.tex_coord,
                        "instance {mi} {name}: tex_coord mismatch"
                    );

                    match (&o.sampler, &r.sampler) {
                        (Some(os), Some(rs)) => {
                            assert!(
                                Arc::ptr_eq(os, rs),
                                "instance {mi} {name}: CpuSampler Arc not shared (ptr_eq failed)"
                            );
                        }
                        (None, None) => {}
                        _ => panic!("instance {mi} {name}: sampler presence mismatch"),
                    }

                    match (&o.texture, &r.texture) {
                        (TextureSource::Cpu(_), TextureSource::Cpu(_)) => {}
                        (TextureSource::Named(a), TextureSource::Named(b)) => {
                            assert_eq!(a, b, "instance {mi} {name}: named texture mismatch");
                        }
                        _ => panic!("instance {mi} {name}: texture source type mismatch"),
                    }
                }
                (None, None) => {}
                _ => panic!("instance {mi} {name}: texture presence mismatch"),
            }
        }
    }
}

#[test]
fn test_roundtrip_node_transforms() {
    let original = load_gltf(TOY_CAR_GLB, &[], &[], &[]).expect("failed to load");

    let scene_refs: Vec<&_> = original.scenes.iter().collect();
    let glb_bytes = save_gltf(&scene_refs, original.default_scene).expect("export");

    let shared_layouts = collect_layouts(&original);
    let shared_samplers = collect_samplers(&original);
    let shared_instances = collect_instances(&original);
    let reloaded = load_gltf(
        &glb_bytes,
        &shared_layouts,
        &shared_samplers,
        &shared_instances,
    )
    .expect("reload");

    fn compare_nodes(orig: &[SceneNode], re: &[SceneNode], path: &str) {
        assert_eq!(orig.len(), re.len(), "{path}: child count mismatch");
        for (i, (o, r)) in orig.iter().zip(re.iter()).enumerate() {
            let node_path = format!("{path}/{}", o.name.as_deref().unwrap_or(&format!("{i}")));
            assert_eq!(o.name, r.name, "{node_path}: name mismatch");
            assert_eq!(
                o.transform.translation, r.transform.translation,
                "{node_path}: translation mismatch"
            );
            assert_eq!(
                o.transform.rotation, r.transform.rotation,
                "{node_path}: rotation mismatch"
            );
            assert_eq!(
                o.transform.scale, r.transform.scale,
                "{node_path}: scale mismatch"
            );
            assert_eq!(o.meshes, r.meshes, "{node_path}: mesh refs mismatch");
            assert_eq!(o.camera, r.camera, "{node_path}: camera ref mismatch");
            compare_nodes(&o.children, &r.children, &node_path);
        }
    }

    for (si, (os, rs)) in original
        .scenes
        .iter()
        .zip(reloaded.scenes.iter())
        .enumerate()
    {
        compare_nodes(&os.nodes, &rs.nodes, &format!("scene{si}"));
    }
}

#[test]
fn test_roundtrip_animations() {
    let original = load_gltf(TOY_CAR_GLB, &[], &[], &[]).expect("failed to load");

    let scene_refs: Vec<&_> = original.scenes.iter().collect();
    let glb_bytes = save_gltf(&scene_refs, original.default_scene).expect("export");

    let shared_layouts = collect_layouts(&original);
    let shared_samplers = collect_samplers(&original);
    let shared_instances = collect_instances(&original);
    let reloaded = load_gltf(
        &glb_bytes,
        &shared_layouts,
        &shared_samplers,
        &shared_instances,
    )
    .expect("reload");

    for (si, (os, rs)) in original
        .scenes
        .iter()
        .zip(reloaded.scenes.iter())
        .enumerate()
    {
        for (ai, (oa, ra)) in os.animations.iter().zip(rs.animations.iter()).enumerate() {
            assert_eq!(oa.name, ra.name, "scene {si} anim {ai}: name mismatch");
            assert_eq!(
                oa.channels.len(),
                ra.channels.len(),
                "scene {si} anim {ai}: channel count mismatch"
            );
            for (ci, (oc, rc)) in oa.channels.iter().zip(ra.channels.iter()).enumerate() {
                assert_eq!(
                    oc.target_node, rc.target_node,
                    "scene {si} anim {ai} ch {ci}: target_node mismatch"
                );
                assert_eq!(
                    oc.property, rc.property,
                    "scene {si} anim {ai} ch {ci}: property mismatch"
                );
                assert_eq!(
                    oc.interpolation, rc.interpolation,
                    "scene {si} anim {ai} ch {ci}: interpolation mismatch"
                );
                assert_eq!(
                    oc.timestamps, rc.timestamps,
                    "scene {si} anim {ai} ch {ci}: timestamps mismatch"
                );
                assert_eq!(
                    oc.values, rc.values,
                    "scene {si} anim {ai} ch {ci}: values mismatch"
                );
            }
        }
    }
}
