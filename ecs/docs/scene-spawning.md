# Scene Spawning

The scene spawning system converts loaded `Scene` data (from glTF or other sources) into live ECS entities with proper component assignment and hierarchy.

## Overview

`spawn_scene()` recursively walks a scene tree and creates entities with appropriate components. It returns the root entities.

```rust
use redlilium_ecs::spawn_scene;
use redlilium_core::scene::Scene;

let mut world = World::new();
register_std_components(&mut world);

let scene: Scene = load_my_scene();
let roots = spawn_scene(&mut world, &scene);

println!("Spawned {} root entities", roots.len());
```

## Components Assigned Per Node

Each scene node becomes an entity with:

| Component | Condition | Source |
|-----------|-----------|--------|
| `Transform` | Always | From `NodeTransform` |
| `GlobalTransform` | Always | Computed from local transform |
| `Visibility` | Always | Default: `VISIBLE` |
| `Name` | If node has a name | From `SceneNode.name` |
| `Camera` | If node references a camera | From `SceneCamera` |
| `Parent` / `Children` | For nested nodes | From scene hierarchy |

## Scene Data Types (from redlilium-core)

The `Scene` struct comes from `redlilium_core::scene`:

```rust
// Scene contains:
// - nodes: Vec<SceneNode>     (top-level nodes)
// - meshes: Vec<CpuMesh>      (referenced by index)
// - cameras: Vec<SceneCamera>  (referenced by index)
// - skins, animations, etc.

// SceneNode contains:
// - name: Option<String>
// - transform: NodeTransform
// - camera: Option<usize>     (index into scene.cameras)
// - children: Vec<SceneNode>  (nested nodes)
```

## Example: Single Node

```rust
use redlilium_core::scene::{Scene, SceneNode, NodeTransform};

let scene = Scene::new().with_nodes(vec![
    SceneNode::new()
        .with_name("Player")
        .with_transform(NodeTransform::IDENTITY.with_translation([1.0, 2.0, 3.0])),
]);

let roots = spawn_scene(&mut world, &scene);
let entity = roots[0];

// Verify components
let t = world.get::<Transform>(entity).unwrap();
assert_eq!(t.translation, Vec3::new(1.0, 2.0, 3.0));

let name = world.get::<Name>(entity).unwrap();
assert_eq!(name.as_str(), "Player");

let vis = world.get::<Visibility>(entity).unwrap();
assert!(vis.is_visible());
```

## Example: Hierarchy

```rust
let scene = Scene::new().with_nodes(vec![
    SceneNode::new()
        .with_name("Root")
        .with_children(vec![
            SceneNode::new().with_name("Child_A"),
            SceneNode::new().with_name("Child_B")
                .with_children(vec![
                    SceneNode::new().with_name("Grandchild"),
                ]),
        ]),
]);

let roots = spawn_scene(&mut world, &scene);
// Creates 4 entities with proper Parent/Children relationships:
//
// Root
// ├── Child_A
// └── Child_B
//     └── Grandchild

assert_eq!(world.entity_count(), 4);

let root = roots[0];
let children = world.get::<Children>(root).unwrap();
assert_eq!(children.len(), 2);
```

## Example: Camera Node

```rust
use redlilium_core::scene::{SceneCamera, CameraProjection};

let scene = Scene::new()
    .with_cameras(vec![SceneCamera {
        name: Some("MainCamera".to_string()),
        projection: CameraProjection::Perspective {
            yfov: 1.0,
            aspect: Some(16.0 / 9.0),
            znear: 0.1,
            zfar: Some(100.0),
        },
    }])
    .with_nodes(vec![
        SceneNode::new()
            .with_name("CameraNode")
            .with_camera(0), // Index into scene.cameras
    ]);

let roots = spawn_scene(&mut world, &scene);
let cam_entity = roots[0];

// Camera component is attached
let cam = world.get::<Camera>(cam_entity).unwrap();
// Projection matrix is computed
```

## NodeTransform Conversion

`NodeTransform` (from core) uses plain arrays and converts to `Transform` (ECS):

```rust
// NodeTransform uses [f32; 3] / [f32; 4]
let node_transform = NodeTransform {
    translation: [1.0, 2.0, 3.0],
    rotation: [0.0, 0.0, 0.0, 1.0],
    scale: [1.0, 1.0, 1.0],
};

// Converts to Transform with Vec3/Quat
let transform = Transform::from(node_transform);
```

## Public API

| Function | Description |
|----------|-------------|
| `spawn_scene(&mut world, &scene)` | Spawn all nodes, returns root entities |
