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

## Pose ownership and motion

Physics bodies must be root entities (no `Parent`) with unit `Transform.scale`.
Put scaled visuals on child entities and specify collision dimensions in the
collider. Body sync and step return `SystemError::InvalidConfiguration` for an
unsupported hierarchy/scale or invalid pose, before their simulation step.

`Transform` supplies the initial pose. After creation:

| Body kind | Motion input |
|---|---|
| Dynamic | Simulation, forces, impulses and velocity changes through the physics world |
| KinematicPosition | `KinematicTarget3D` / `KinematicTarget2D` |
| KinematicVelocity | `KinematicVelocity3D` / `KinematicVelocity2D` |
| Fixed | `Transform`, read before every step |

Dynamic and kinematic transforms are presentation output. Writing one does not
teleport a body. `PhysicsWorld3D::pose(entity)` (or its 2D equivalent) reads the
current simulation pose. In 3D, `RecordPhysicsPose` reads Rapier directly after
each step; `InterpolatePhysics` writes presentation transforms before global
transform propagation. Fixed bodies are never interpolated.

Call `register_std_components` when preparing the world, including the control
components. A position target is persistent: absent input holds the current
pose. Velocity input is also persistent; absent input means zero velocity. The
2D rotation/angular velocity fields are scalar Z angles in radians and radians
per second; 3D uses a quaternion and angular-velocity vector.

```rust,ignore
use redlilium_ecs::physics::control3d::{KinematicTarget3D, PhysicsPose3D};
use redlilium_ecs::physics::TeleportVelocity;

world.insert(platform, KinematicTarget3D {
    translation: Vec3::new(0.0, 2.0, 0.0),
    rotation: Quat::identity(),
}).unwrap();

// The body must already have been synchronized. Queue before StepPhysics3D.
world.resource_mut::<PhysicsWorld3D>().teleport(
    player,
    PhysicsPose3D { translation: spawn_position, rotation: Quat::identity() },
    TeleportVelocity::Reset, // Or Preserve; choose explicitly.
)?;
```

A teleport is consumed by the next step; the last queued request for a body
wins. It resets interpolation and cannot follow a removed/recreated body.
A present kinematic target is moved to the teleport pose. `Reset` also zeros a
present kinematic velocity input. With `Preserve`, subsequent motion still obeys
the body's control mode (position-based kinematics derives velocity from its target).

## Editable settings

`RigidBody*`, `Collider*` and `ImpulseJoint*` remain editable settings. Run the
corresponding sync after game systems edit settings and before the physics step.
Body and collider changes update the existing body/collider handles, preserving
the body and its connections. Changing shape/density updates mass properties
through Rapier on its next step. Changed settings wake the body; unchanged
settings do not wake it or overwrite its runtime velocity. Body-type changes
also reset interpolation history.

Changing a joint descriptor rebuilds that joint, including when endpoints
change. Other bodies and joints remain intact. If an endpoint is unavailable,
the old joint is removed and creation waits for valid endpoints. Physics-world
caches retain the last applied descriptors to avoid rebuilding unchanged objects.

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

### Synchronizing and stepping

Use sync systems for ongoing creation, removal and settings updates. The
`build_physics_world_*` helpers are deprecated. Register explicit dependencies:

```rust,ignore
let fixed = schedules.get_mut::<FixedUpdate>();
fixed.add_exclusive(SyncPhysicsBodies3D);
fixed.add_exclusive(SyncPhysicsJoints3D);
fixed.add(StepPhysics3D);
fixed.add_edge::<SyncPhysicsBodies3D, SyncPhysicsJoints3D>().unwrap();
fixed.add_edge::<SyncPhysicsJoints3D, StepPhysics3D>().unwrap();
```

The step writes dynamic/kinematic `Transform` values. Run
`UpdateGlobalTransforms` separately afterward (after `InterpolatePhysics` when
using render interpolation).

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

    let fixed = schedules.get_mut::<FixedUpdate>();
    fixed.add_exclusive(SyncPhysicsBodies3D);
    fixed.add(StepPhysics3D);
    fixed.add_edge::<SyncPhysicsBodies3D, StepPhysics3D>().unwrap();
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

let fixed = schedules.get_mut::<FixedUpdate>();
fixed.add_exclusive(SyncPhysicsBodies2D);
fixed.add(StepPhysics2D);
fixed.add_edge::<SyncPhysicsBodies2D, StepPhysics2D>().unwrap();
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
