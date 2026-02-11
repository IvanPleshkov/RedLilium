use redlilium_core::scene::{CameraProjection, Scene, SceneNode};
use redlilium_ecs::{Entity, StringTable, World};

use crate::components::{Camera, GlobalTransform, Name, Transform, Visibility};
use crate::hierarchy::set_parent;

/// Spawns all entities from a loaded [`Scene`] into the ECS [`World`].
///
/// Recursively walks the scene tree and creates entities with appropriate
/// components. Parent-child relationships are established via [`Parent`](crate::Parent)
/// and [`Children`](crate::Children) components, enabling hierarchy-aware
/// transform propagation.
///
/// Returns the list of root entities.
///
/// # Components assigned per node
///
/// - **Transform** + **GlobalTransform** — always (from NodeTransform)
/// - **Name** — if the node has a name (interned via `string_table`)
/// - **Visibility** — always (default: visible)
/// - **Camera** — if the node has a camera reference
/// - **Parent** / **Children** — for nested nodes
pub fn spawn_scene(
    world: &mut World,
    scene: &Scene,
    string_table: &mut StringTable,
) -> Vec<Entity> {
    scene
        .nodes
        .iter()
        .map(|node| spawn_node(world, node, scene, string_table, None))
        .collect()
}

fn spawn_node(
    world: &mut World,
    node: &SceneNode,
    scene: &Scene,
    string_table: &mut StringTable,
    parent_entity: Option<Entity>,
) -> Entity {
    let entity = world.spawn();

    let transform = Transform::from(node.transform);
    world
        .insert(entity, transform)
        .expect("Transform not registered");
    world
        .insert(entity, GlobalTransform(transform.to_matrix()))
        .expect("GlobalTransform not registered");
    world
        .insert(entity, Visibility::VISIBLE)
        .expect("Visibility not registered");

    if let Some(name) = &node.name {
        let id = string_table.intern(name);
        world
            .insert(entity, Name::new(id))
            .expect("Name not registered");
    }

    if let Some(camera_idx) = node.camera
        && let Some(scene_camera) = scene.cameras.get(camera_idx)
    {
        let camera = match scene_camera.projection {
            CameraProjection::Perspective {
                yfov,
                aspect,
                znear,
                zfar,
            } => {
                let aspect = aspect.unwrap_or(16.0 / 9.0);
                let zfar = zfar.unwrap_or(1000.0);
                Camera::perspective(yfov, aspect, znear, zfar)
            }
            CameraProjection::Orthographic {
                xmag,
                ymag,
                znear,
                zfar,
            } => Camera::orthographic(xmag, ymag, znear, zfar),
        };
        world.insert(entity, camera).expect("Camera not registered");
    }

    if let Some(parent) = parent_entity {
        set_parent(world, entity, parent);
    }

    for child_node in &node.children {
        spawn_node(world, child_node, scene, string_table, Some(entity));
    }

    entity
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::scene::{CameraProjection, NodeTransform, SceneCamera, SceneNode};

    #[test]
    fn spawn_empty_scene() {
        let mut world = World::new();
        let mut strings = StringTable::new();
        let scene = Scene::new();
        let roots = spawn_scene(&mut world, &scene, &mut strings);
        assert!(roots.is_empty());
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn spawn_single_node_with_name() {
        let mut world = World::new();
        crate::register_std_components(&mut world);
        let mut strings = StringTable::new();
        let scene = Scene::new().with_nodes(vec![
            SceneNode::new()
                .with_name("TestNode")
                .with_transform(NodeTransform::IDENTITY.with_translation([1.0, 2.0, 3.0])),
        ]);

        let roots = spawn_scene(&mut world, &scene, &mut strings);
        assert_eq!(roots.len(), 1);
        assert_eq!(world.entity_count(), 1);

        let e = roots[0];

        // Check Transform
        let t = world.get::<Transform>(e).unwrap();
        assert!((t.translation - redlilium_core::math::Vec3::new(1.0, 2.0, 3.0)).norm() < 1e-6);

        // Check GlobalTransform
        let gt = world.get::<GlobalTransform>(e).unwrap();
        assert!((gt.translation() - redlilium_core::math::Vec3::new(1.0, 2.0, 3.0)).norm() < 1e-6);

        // Check Visibility
        let v = world.get::<Visibility>(e).unwrap();
        assert!(v.is_visible());

        // Check Name resolves via StringTable
        let n = world.get::<Name>(e).unwrap();
        assert_eq!(strings.get(n.id()), "TestNode");
    }

    #[test]
    fn spawn_node_with_camera() {
        let mut world = World::new();
        crate::register_std_components(&mut world);
        let mut strings = StringTable::new();
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

        let roots = spawn_scene(&mut world, &scene, &mut strings);
        let e = roots[0];

        let cam = world.get::<Camera>(e).unwrap();
        // Projection should be computed eagerly (not identity)
        assert_ne!(
            cam.projection_matrix,
            redlilium_core::math::Mat4::identity()
        );
    }

    #[test]
    fn spawn_nested_nodes() {
        let mut world = World::new();
        crate::register_std_components(&mut world);
        let mut strings = StringTable::new();
        let scene =
            Scene::new().with_nodes(vec![SceneNode::new().with_name("parent").with_children(
                vec![
                    SceneNode::new().with_name("child_a"),
                    SceneNode::new().with_name("child_b"),
                ],
            )]);

        let roots = spawn_scene(&mut world, &scene, &mut strings);
        assert_eq!(roots.len(), 1);
        // Parent + 2 children = 3 entities
        assert_eq!(world.entity_count(), 3);

        // Verify parent-child relationships
        let parent_entity = roots[0];
        let children = world.get::<crate::Children>(parent_entity).unwrap();
        assert_eq!(children.len(), 2);

        // Children should have Parent pointing back
        for &child in children.0.iter() {
            let p = world.get::<crate::Parent>(child).unwrap();
            assert_eq!(p.0, parent_entity);
        }
    }

    #[test]
    fn spawn_deep_hierarchy() {
        let mut world = World::new();
        crate::register_std_components(&mut world);
        let mut strings = StringTable::new();
        let scene =
            Scene::new().with_nodes(vec![SceneNode::new().with_name("root").with_children(
                vec![SceneNode::new()
                .with_name("mid")
                .with_children(vec![SceneNode::new().with_name("leaf")])],
            )]);

        let roots = spawn_scene(&mut world, &scene, &mut strings);
        assert_eq!(roots.len(), 1);
        assert_eq!(world.entity_count(), 3);

        // Root should have no parent
        let root = roots[0];
        assert!(world.get::<crate::Parent>(root).is_none());

        // Mid is child of root
        let root_children = world.get::<crate::Children>(root).unwrap();
        assert_eq!(root_children.len(), 1);
        let mid = root_children.0[0];
        assert_eq!(world.get::<crate::Parent>(mid).unwrap().0, root);

        // Leaf is child of mid
        let mid_children = world.get::<crate::Children>(mid).unwrap();
        assert_eq!(mid_children.len(), 1);
        let leaf = mid_children.0[0];
        assert_eq!(world.get::<crate::Parent>(leaf).unwrap().0, mid);
    }
}
