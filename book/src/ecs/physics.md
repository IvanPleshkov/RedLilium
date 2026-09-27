# Physics Integration

RedLilium integrates the [Rapier](https://rapier.rs/) physics engine via feature flags. Both 2D and 3D physics are supported.

## Feature Flags

Enable physics in your `Cargo.toml`:

```toml
[dependencies]
redlilium-ecs = { path = "../ecs", features = ["physics-3d"] }
```

| Feature | Description |
|---------|-------------|
| `physics-3d` | 3D physics with f64 precision (default Rapier) |
| `physics-3d-f32` | 3D physics with f32 precision |
| `physics-2d` | 2D physics with f64 precision |
| `physics-2d-f32` | 2D physics with f32 precision |
| `physics` | Enables all physics features |

## Synchronization lifecycle

For ongoing synchronization, run `SyncPhysicsBodies3D`, then
`SyncPhysicsJoints3D`, then `StepPhysics3D` in `FixedUpdate`, with explicit
ordering edges. The `2D` systems follow the same order. The exclusive sync
systems publish handles immediately. Regular `SyncPhysicsBodiesSystem*` and
`SyncPhysicsJointsSystem*` publish through deferred commands, so joints can
appear on the next schedule invocation after their bodies.

A managed body requires `RigidBody*`, `Collider*` and `Transform`. On the next
body sync, removing any of these components, despawning the entity or excluding
it from game queries removes the Rapier body, its colliders and attached joints.
Their mappings and handle components are cleaned up too. In 3D, body removal
also clears `PhysicsInterpolation`; recreating a body starts a fresh history.
Deferred creation validates the full entity identity and prerequisites again
when publishing handles and removes cancelled Rapier objects.

Both step systems use `Time::fixed_delta()` when `Time` is present. Without it,
they preserve the manually configured `integration_parameters.dt`. The 2D
builder initializes rotation from the transform's Z angle; stepping preserves
the transform's Z translation for draw ordering.

## 3D Physics

### Setup

```rust
use redlilium_ecs::physics::*;

// Insert the physics world resource
world.insert_resource(PhysicsWorld3D::default());

// Or with custom gravity
world.insert_resource(PhysicsWorld3D::with_gravity(vector![0.0, -9.81, 0.0]));
```

### Rigid Bodies

Attach a `RigidBody3D` descriptor to entities:

```rust
let entity = world.spawn_with((
    Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
    GlobalTransform::IDENTITY,
    RigidBody3D::dynamic(),
    Collider3D::ball(0.5),
));
```

Rigid body types:

```rust
RigidBody3D::dynamic()              // affected by forces and gravity
RigidBody3D::fixed()                // immovable (ground, walls)
RigidBody3D::kinematic_position()   // moved by setting position
RigidBody3D::kinematic_velocity()   // moved by setting velocity
```

Configure body properties:

```rust
let body = RigidBody3D {
    body_type: RigidBodyType::Dynamic,
    linear_damping: 0.5,
    angular_damping: 0.1,
    gravity_scale: 1.0,
};
```

### Collider Shapes

```rust
Collider3D::ball(radius)
Collider3D::cuboid(half_x, half_y, half_z)
Collider3D::capsule_y(half_height, radius)
Collider3D::cylinder(half_height, radius)
```

Available shapes via `ColliderShape3D`:

```rust
ColliderShape3D::Ball { radius: 0.5 }
ColliderShape3D::Cuboid { half_extents: [1.0, 0.5, 1.0] }
ColliderShape3D::CapsuleY { half_height: 0.5, radius: 0.25 }
ColliderShape3D::Cylinder { half_height: 1.0, radius: 0.5 }
```

### Building the Physics World

After attaching descriptors, build the Rapier objects:

```rust
build_physics_world_3d(&mut world);
```

This reads all `RigidBody3D` and `Collider3D` components and creates the corresponding Rapier objects in `PhysicsWorld3D`. Handle components (`RigidBody3DHandle`, `Collider3DHandle`) are inserted on the entities.

### Stepping the Simulation

Use the built-in `StepPhysics3D` system:

```rust
schedules.get_mut::<FixedUpdate>().add(StepPhysics3D);
```

This system:
1. Steps the Rapier simulation
2. Syncs resulting positions back to `Transform` and `GlobalTransform`

### Full Example

```rust
fn setup_physics(world: &mut World, schedules: &mut Schedules) {
    world.insert_resource(PhysicsWorld3D::default());

    // Ground (fixed)
    world.spawn_with((
        Transform::IDENTITY,
        GlobalTransform::IDENTITY,
        RigidBody3D::fixed(),
        Collider3D::cuboid(50.0, 0.1, 50.0),
    ));

    // Falling ball (dynamic)
    world.spawn_with((
        Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        GlobalTransform::IDENTITY,
        RigidBody3D::dynamic(),
        Collider3D::ball(0.5),
    ));

    // Build rapier objects from descriptors
    build_physics_world_3d(world);

    // Step physics in FixedUpdate
    schedules.get_mut::<FixedUpdate>().add(StepPhysics3D);
    schedules.get_mut::<PostUpdate>().add(UpdateGlobalTransforms);
}
```

## 2D Physics

The 2D API mirrors 3D with `PhysicsWorld2D`, `RigidBody2D`, `Collider2D`, and `StepPhysics2D`:

```rust
world.insert_resource(PhysicsWorld2D::default());

world.spawn_with((
    Transform::IDENTITY,
    GlobalTransform::IDENTITY,
    RigidBody2D::dynamic(),
    Collider2D::ball(0.5),
));

build_physics_world_2d(&mut world);
schedules.get_mut::<FixedUpdate>().add(StepPhysics2D);
```

## Accessing Rapier Directly

For advanced usage, access the Rapier data structures through the resource:

```rust
ctx.lock::<(ResMut<PhysicsWorld3D>,)>()
    .execute(|(mut physics,)| {
        // Direct Rapier access
        for (handle, body) in physics.bodies.iter() {
            let position = body.translation();
            // ...
        }

        // Apply impulse via handle
        // let body = physics.bodies.get_mut(handle).unwrap();
        // body.apply_impulse(vector![0.0, 100.0, 0.0], true);
    });
```
