use std::sync::Arc;

use redlilium_core::mesh::CpuMesh;
use redlilium_core::scene::{Scene, SceneNode};
use redlilium_ecs::{Entity, World};

use crate::components::{Camera, GlobalTransform, MeshRenderer, Name, Transform, Visibility};

/// Spawns all entities from a loaded [`Scene`] into the ECS [`World`].
///
/// Recursively walks the scene tree and creates entities with appropriate
/// components. Returns the list of root entities.
///
/// # Components assigned per node
///
/// - **Transform** + **GlobalTransform** — always (from NodeTransform)
/// - **Name** — if the node has a name
/// - **Visibility** — always (default: visible)
/// - **MeshRenderer** — one per mesh on the node (extra meshes get separate entities)
/// - **Camera** — if the node has a camera reference
///
/// # Hierarchy limitation
///
/// Without Parent/Children components (ECS Phase 3), all spawned entities
/// are flat. Nested scene transforms are NOT propagated — each entity gets
/// only its local transform. This will be corrected when hierarchy is added.
pub fn spawn_scene(world: &mut World, scene: &Scene) -> Vec<Entity> {
    let mesh_arcs: Vec<Arc<CpuMesh>> = scene.meshes.iter().map(|m| Arc::new(m.clone())).collect();

    scene
        .nodes
        .iter()
        .map(|node| spawn_node(world, node, scene, &mesh_arcs))
        .collect()
}

fn spawn_node(
    world: &mut World,
    node: &SceneNode,
    scene: &Scene,
    mesh_arcs: &[Arc<CpuMesh>],
) -> Entity {
    let entity = world.spawn();

    let transform = Transform::from(node.transform);
    world.insert(entity, transform);
    world.insert(entity, GlobalTransform(transform.to_matrix()));
    world.insert(entity, Visibility::VISIBLE);

    if let Some(name) = &node.name {
        world.insert(entity, Name::new(name.clone()));
    }

    if let Some(camera_idx) = node.camera
        && let Some(scene_camera) = scene.cameras.get(camera_idx)
    {
        world.insert(
            entity,
            Camera::from_projection(scene_camera.projection.clone()),
        );
    }

    for (i, &mesh_idx) in node.meshes.iter().enumerate() {
        let Some(mesh_arc) = mesh_arcs.get(mesh_idx) else {
            continue;
        };
        let material_idx = mesh_arc.material().unwrap_or(0);
        let Some(material_arc) = scene.materials.get(material_idx) else {
            continue;
        };

        if i == 0 {
            world.insert(
                entity,
                MeshRenderer::new(Arc::clone(mesh_arc), Arc::clone(material_arc)),
            );
        } else {
            let child = world.spawn();
            world.insert(child, transform);
            world.insert(child, GlobalTransform(transform.to_matrix()));
            world.insert(child, Visibility::VISIBLE);
            world.insert(
                child,
                MeshRenderer::new(Arc::clone(mesh_arc), Arc::clone(material_arc)),
            );
            if let Some(name) = &node.name {
                world.insert(child, Name::new(format!("{}_mesh_{}", name, i)));
            }
        }
    }

    for child_node in &node.children {
        spawn_node(world, child_node, scene, mesh_arcs);
    }

    entity
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::scene::{CameraProjection, NodeTransform, Scene, SceneCamera, SceneNode};

    #[test]
    fn spawn_empty_scene() {
        let mut world = World::new();
        let scene = Scene::new();
        let roots = spawn_scene(&mut world, &scene);
        assert!(roots.is_empty());
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn spawn_single_node_with_name() {
        let mut world = World::new();
        let scene = Scene::new().with_nodes(vec![
            SceneNode::new()
                .with_name("TestNode")
                .with_transform(NodeTransform::IDENTITY.with_translation([1.0, 2.0, 3.0])),
        ]);

        let roots = spawn_scene(&mut world, &scene);
        assert_eq!(roots.len(), 1);
        assert_eq!(world.entity_count(), 1);

        let e = roots[0];

        // Check Transform
        let t = world.get::<Transform>(e).unwrap();
        assert!((t.translation - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);

        // Check GlobalTransform
        let gt = world.get::<GlobalTransform>(e).unwrap();
        assert!((gt.translation() - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);

        // Check Visibility
        let v = world.get::<Visibility>(e).unwrap();
        assert!(v.is_visible());

        // Check Name
        let n = world.get::<Name>(e).unwrap();
        assert_eq!(n.as_str(), "TestNode");
    }

    #[test]
    fn spawn_node_with_camera() {
        let mut world = World::new();
        let scene = Scene::new()
            .with_cameras(vec![SceneCamera {
                name: Some("MainCam".to_string()),
                projection: CameraProjection::Perspective {
                    yfov: 1.0,
                    aspect: Some(16.0 / 9.0),
                    znear: 0.1,
                    zfar: Some(100.0),
                },
            }])
            .with_nodes(vec![SceneNode::new().with_camera(0)]);

        let roots = spawn_scene(&mut world, &scene);
        let e = roots[0];

        let cam = world.get::<Camera>(e).unwrap();
        assert!(cam.active);
        assert!(matches!(
            cam.projection,
            CameraProjection::Perspective { .. }
        ));
    }

    #[test]
    fn spawn_nested_nodes() {
        let mut world = World::new();
        let scene =
            Scene::new().with_nodes(vec![SceneNode::new().with_name("parent").with_children(
                vec![
                    SceneNode::new().with_name("child_a"),
                    SceneNode::new().with_name("child_b"),
                ],
            )]);

        let roots = spawn_scene(&mut world, &scene);
        assert_eq!(roots.len(), 1);
        // Parent + 2 children = 3 entities
        assert_eq!(world.entity_count(), 3);
    }
}
