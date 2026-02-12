# Transform System

The transform system propagates local transforms through the entity hierarchy to compute world-space matrices. It consists of two built-in systems that run in sequence each frame.

## Systems

### UpdateGlobalTransforms

Reads each entity's local `Transform` and writes its `GlobalTransform` (world-space matrix).

For entities without a parent, the global transform equals the local transform's matrix:
```
GlobalTransform = Transform.to_matrix()
```

For entities with a parent, the global transform combines the parent's global with the local:
```
GlobalTransform = parent.GlobalTransform * Transform.to_matrix()
```

### UpdateCameraMatrices

Reads each entity's `GlobalTransform` and updates the `Camera` component's view matrix. The view matrix is the inverse of the global transform — it represents "looking from" that position.

This system must run **after** `UpdateGlobalTransforms`.

## Setup

```rust
use redlilium_ecs::{
    SystemsContainer, Edge,
    UpdateGlobalTransforms, UpdateCameraMatrices,
};

let mut systems = SystemsContainer::new();

// Register transform systems
systems.add(UpdateGlobalTransforms);
systems.add(UpdateCameraMatrices);

// Camera depends on transforms
systems.add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>().unwrap();

// Your game systems should run before transforms
systems.add(MovementSystem);
systems.add_edge::<MovementSystem, UpdateGlobalTransforms>().unwrap();
```

## Execution Order

```
Game Logic (modify Transform)
    ↓
UpdateGlobalTransforms (compute GlobalTransform from Transform + hierarchy)
    ↓
UpdateCameraMatrices (compute Camera.view_matrix from GlobalTransform)
    ↓
Rendering (use GlobalTransform + Camera matrices)
```

## Example: Moving an Entity

```rust
// Setup
let mut world = World::new();
register_std_components(&mut world);

let entity = world.spawn();
world.insert(entity, Transform::from_translation(Vec3::new(0.0, 0.0, 0.0))).unwrap();
world.insert(entity, GlobalTransform::default()).unwrap();

// Game loop
world.get_mut::<Transform>(entity).unwrap().translation.x += 1.0;

// After running UpdateGlobalTransforms:
// GlobalTransform now reflects the new position
```

## Example: Hierarchy Transforms

```rust
let mut world = World::new();
register_std_components(&mut world);

let parent = world.spawn();
world.insert(parent, Transform::from_translation(Vec3::new(10.0, 0.0, 0.0))).unwrap();
world.insert(parent, GlobalTransform::default()).unwrap();

let child = world.spawn();
world.insert(child, Transform::from_translation(Vec3::new(0.0, 5.0, 0.0))).unwrap();
world.insert(child, GlobalTransform::default()).unwrap();
set_parent(&mut world, child, parent);

// After UpdateGlobalTransforms:
// parent.GlobalTransform = translate(10, 0, 0)
// child.GlobalTransform  = translate(10, 5, 0)  (parent + local)
```

## Example: Camera Setup

```rust
let camera_entity = world.spawn();
world.insert(camera_entity, Transform::from_translation(Vec3::new(0.0, 5.0, 10.0))).unwrap();
world.insert(camera_entity, GlobalTransform::default()).unwrap();
world.insert(camera_entity, Camera::perspective(
    std::f32::consts::FRAC_PI_4,
    16.0 / 9.0,
    0.1,
    1000.0,
)).unwrap();

// After UpdateGlobalTransforms + UpdateCameraMatrices:
// camera.view_matrix = inverse of GlobalTransform
// camera.view_projection() = view_matrix * projection_matrix
```

## How Transform.to_matrix() Works

The local transform produces a 4x4 matrix combining translation, rotation, and scale:

```
Matrix = Translation * Rotation * Scale
```

This follows the standard TRS convention used in game engines and glTF.
