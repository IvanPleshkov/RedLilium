# Physics Integration

The ECS provides feature-gated physics integration via the rapier physics engine. Both 2D and 3D physics are supported with configurable precision (f32 or f64).

## Feature Flags

Enable physics via Cargo features:

| Feature | Description |
|---------|-------------|
| `physics-3d` | 3D physics with f64 precision |
| `physics-3d-f32` | 3D physics with f32 precision |
| `physics-2d` | 2D physics with f64 precision |
| `physics-2d-f32` | 2D physics with f32 precision |
| `physics` | Enables all physics features |

```toml
[dependencies]
redlilium-ecs = { path = "../ecs", features = ["physics-3d-f32"] }
```

## Components

### 3D Physics (requires `physics-3d` or `physics-3d-f32`)

```rust
use redlilium_ecs::physics::components3d::{RigidBody3D, Collider3D};

// RigidBody3D — describes the physics body type
let rb = RigidBody3D::default(); // Dynamic body

// Collider3D — describes the collision shape
let col = Collider3D::default();
```

A handle component is also registered for the physics engine's internal reference:

```rust
use redlilium_ecs::physics::physics3d::RigidBody3DHandle;
// Managed by the physics system, not user-created
```

### 2D Physics (requires `physics-2d` or `physics-2d-f32`)

```rust
use redlilium_ecs::physics::components2d::{RigidBody2D, Collider2D};
use redlilium_ecs::physics::physics2d::RigidBody2DHandle;
```

## Registration

Physics components are auto-registered when you call `register_std_components`:

```rust
register_std_components(&mut world);
// If physics-3d feature is enabled:
//   - RigidBody3D (inspector + default)
//   - Collider3D (inspector + default)
//   - RigidBody3DHandle (storage only)
// If physics-2d feature is enabled:
//   - RigidBody2D (inspector + default)
//   - Collider2D (inspector + default)
//   - RigidBody2DHandle (storage only)
```

## Usage Pattern

```rust
let entity = world.spawn();
world.insert(entity, Transform::from_translation(Vec3::new(0.0, 10.0, 0.0))).unwrap();
world.insert(entity, GlobalTransform::default()).unwrap();
world.insert(entity, RigidBody3D::default()).unwrap();
world.insert(entity, Collider3D::default()).unwrap();

// The physics system will:
// 1. Read RigidBody3D + Collider3D to create rapier bodies
// 2. Step the simulation
// 3. Write back to Transform
```

## Module Structure

```
ecs/src/physics/
├── mod.rs            — Feature-gated module entry
├── components3d.rs   — RigidBody3D, Collider3D
├── components2d.rs   — RigidBody2D, Collider2D
├── physics3d.rs      — 3D physics system + RigidBody3DHandle
└── physics2d.rs      — 2D physics system + RigidBody2DHandle
```

## Inspector Support

All descriptor components (RigidBody3D, Collider3D, etc.) are registered with `register_inspector_default`, so they:
- Appear in the inspector UI
- Can be added via the "Add Component" button
- Have editable fields

Handle components are storage-only (not visible in inspector).
