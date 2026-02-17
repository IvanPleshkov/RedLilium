# Standard Components

RedLilium ships with a set of commonly-needed components. Register them all at once with:

```rust
register_std_components(&mut world);
```

Or register individually as needed.

## Transform

Local position, rotation, and scale of an entity:

```rust
#[derive(Component)]
struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}
```

```rust
// Common constructors
Transform::IDENTITY
Transform::from_translation(Vec3::new(1.0, 2.0, 3.0))
Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::PI))
Transform::new(translation, rotation, scale)

// Convert to matrix
let matrix: Mat4 = transform.to_matrix();
```

## GlobalTransform

World-space 4x4 matrix, computed by `UpdateGlobalTransforms`:

```rust
#[derive(Component)]
struct GlobalTransform(pub Mat4);
```

```rust
GlobalTransform::IDENTITY

let world_pos: Vec3 = global_transform.translation();
let forward: Vec3 = global_transform.forward();  // -Z
let right: Vec3 = global_transform.right();      // +X
let up: Vec3 = global_transform.up();            // +Y
```

You should never set `GlobalTransform` manually. Set `Transform` and let `UpdateGlobalTransforms` propagate it.

## Camera

Requires `Transform`, `GlobalTransform`, and `Visibility`:

```rust
#[derive(Component)]
#[require(Transform, GlobalTransform, Visibility)]
struct Camera {
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
}
```

```rust
// Perspective camera
let camera = Camera::perspective(
    std::f32::consts::FRAC_PI_4, // 45-degree vertical FOV
    16.0 / 9.0,                   // aspect ratio
    0.1,                           // near plane
    1000.0,                        // far plane
);

// Orthographic camera
let camera = Camera::orthographic(10.0, 10.0, 0.1, 100.0);

// Combined matrix
let vp: Mat4 = camera.view_projection();
```

The `UpdateCameraMatrices` system updates `view_matrix` from `GlobalTransform` each frame.

## Lights

All light components require `Transform`, `GlobalTransform`, and `Visibility`.

### DirectionalLight

```rust
#[derive(Component)]
struct DirectionalLight {
    pub color: Vec3,
    pub intensity: f32,
}
```

```rust
let light = DirectionalLight {
    color: Vec3::new(1.0, 1.0, 0.9),
    intensity: 1.0,
};
```

### PointLight

```rust
#[derive(Component)]
struct PointLight {
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
}
```

```rust
let light = PointLight {
    color: Vec3::ONE,
    intensity: 100.0,
    range: 50.0,
};
// Or using builder
let light = PointLight::default().with_range(50.0);
```

### SpotLight

```rust
#[derive(Component)]
struct SpotLight {
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
}
```

## Visibility

Controls whether an entity is visible:

```rust
#[derive(Component)]
struct Visibility(pub bool);
```

```rust
Visibility::VISIBLE   // Visibility(true)
Visibility::HIDDEN    // Visibility(false)

if visibility.is_visible() { /* render */ }
```

## Name

Human-readable label for debugging and editor UI:

```rust
#[derive(Component)]
struct Name(pub String);
```

```rust
let name = Name::new("Player");
println!("{}", name.as_str());
```

## Disabled

Marker component that excludes an entity from all standard queries:

```rust
#[derive(Component)]
struct Disabled;
```

See [Hierarchy - Enable/Disable](./hierarchy.md#enable--disable) for recursive operations.

## Hierarchy Components

`Parent` and `Children` are managed automatically by `set_parent` / `remove_parent`. See [Hierarchy](./hierarchy.md).

```rust
#[derive(Component)]
struct Parent(pub Entity);

#[derive(Component)]
struct Children(pub Vec<Entity>);
```

## Built-in Systems

| System | Schedule | Description |
|--------|----------|-------------|
| `UpdateGlobalTransforms` | PostUpdate | Propagates `Transform` -> `GlobalTransform` through hierarchy |
| `UpdateCameraMatrices` | PostUpdate (after transforms) | Updates `Camera::view_matrix` from `GlobalTransform` |
