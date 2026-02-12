# Standard Components

The ECS provides a set of built-in components for common game engine functionality. Register them all at once with `register_std_components(&mut world)`.

## Transform

Local transform relative to the parent (or world origin if no parent).

```rust
use redlilium_ecs::Transform;

// Identity (no translation, rotation, or scale)
let t = Transform::IDENTITY;

// From translation
let t = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0));

// From rotation
let t = Transform::from_rotation(Quat::from_axis_angle(Vec3::Y, 0.5));

// Full constructor
let t = Transform::new(
    Vec3::new(1.0, 2.0, 3.0),  // translation
    Quat::identity(),            // rotation
    Vec3::new(1.0, 1.0, 1.0),  // scale
);

// Convert to 4x4 matrix
let matrix = t.to_matrix();
```

Fields: `translation: Vec3`, `rotation: Quat`, `scale: Vec3`

Implements `Pod + Zeroable` (bytemuck) for GPU upload and serialization.

## GlobalTransform

World-space transform matrix, computed from the local Transform and parent hierarchy.

```rust
use redlilium_ecs::GlobalTransform;

// Wraps a Mat4
let gt = GlobalTransform(Mat4::identity());

// Extract world-space translation
let world_pos = gt.translation();
```

Updated automatically by the `UpdateGlobalTransforms` system.

## Camera

Perspective or orthographic camera with view and projection matrices.

```rust
use redlilium_ecs::Camera;

// Perspective camera
let cam = Camera::perspective(
    std::f32::consts::FRAC_PI_4, // fov_y (45 degrees)
    16.0 / 9.0,                   // aspect ratio
    0.1,                           // near plane
    1000.0,                        // far plane
);

// Orthographic camera
let cam = Camera::orthographic(
    10.0,   // x magnitude
    10.0,   // y magnitude
    0.1,    // near plane
    100.0,  // far plane
);

// Access matrices
let vp = cam.view_projection(); // combined view * projection
```

The `UpdateCameraMatrices` system computes the view matrix from `GlobalTransform`.

## Visibility

Controls whether an entity is visible for rendering.

```rust
use redlilium_ecs::Visibility;

let v = Visibility::VISIBLE;
let h = Visibility::HIDDEN;

assert!(v.is_visible());
assert!(!h.is_visible());
```

Implements `Pod + Zeroable`.

## Name

A human-readable name for entities (useful for debugging and scene graphs).

```rust
use redlilium_ecs::Name;

let name = Name::new("Player");
assert_eq!(name.as_str(), "Player");
```

## Lights

Three light types for scene illumination:

### DirectionalLight

Infinite-distance light (like the sun).

```rust
use redlilium_ecs::DirectionalLight;

let light = DirectionalLight {
    color: [1.0, 1.0, 0.9],
    intensity: 1.0,
};
```

### PointLight

Omnidirectional light from a point.

```rust
use redlilium_ecs::PointLight;

let light = PointLight {
    color: [1.0, 0.8, 0.6],
    intensity: 10.0,
    range: 50.0,
};
```

### SpotLight

Directional cone light.

```rust
use redlilium_ecs::SpotLight;

let light = SpotLight {
    color: [1.0, 1.0, 1.0],
    intensity: 20.0,
    range: 30.0,
    inner_cone_angle: 0.3,
    outer_cone_angle: 0.5,
};
```

## Parent / Children

Hierarchy components (see [hierarchy.md](hierarchy.md)):

```rust
use redlilium_ecs::{Parent, Children};

// Parent — stored on child entities
let p = Parent(parent_entity);

// Children — stored on parent entities
let c = Children(vec![child_a, child_b]);
assert_eq!(c.len(), 2);
assert!(!c.is_empty());
```

## Registration

All standard components are registered via a single function:

```rust
use redlilium_ecs::register_std_components;

let mut world = World::new();
register_std_components(&mut world);
```

This registers:
- **With inspector + default**: Transform, GlobalTransform, Visibility, Name, DirectionalLight, PointLight, SpotLight
- **With inspector (no default)**: Camera
- **Storage only**: Parent, Children
- **Physics (feature-gated)**: RigidBody3D, Collider3D, RigidBody2D, Collider2D, handles

## Component Summary

| Component | Fields | Pod | Default | Inspector |
|-----------|--------|-----|---------|-----------|
| Transform | translation, rotation, scale | Yes | Yes | Yes |
| GlobalTransform | Mat4 | Yes | Yes | Yes |
| Camera | projection_matrix, view_matrix, etc. | No | No | View only |
| Visibility | flags | Yes | Yes | Yes |
| Name | String | No | Yes | Yes |
| DirectionalLight | color, intensity | Yes | Yes | Yes |
| PointLight | color, intensity, range | Yes | Yes | Yes |
| SpotLight | color, intensity, range, angles | Yes | Yes | Yes |
| Parent | Entity | No | No | No |
| Children | Vec\<Entity\> | No | No | No |
