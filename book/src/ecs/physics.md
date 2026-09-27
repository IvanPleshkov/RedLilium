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

### Validation

Both regular and exclusive sync systems validate descriptors before changing
Rapier objects. Invalid settings return `SystemError::InvalidConfiguration`
with the entity and offending parameter. The failing sync invocation leaves
its previously applied bodies/colliders or joints intact; correct the descriptor
and run sync again. This is not a transaction across the whole schedule.

All numeric settings must be finite. Additional constraints are:

| Setting | Accepted values |
|---|---|
| Linear/angular damping, friction, density | Nonnegative |
| Restitution | From 0 to 1 inclusive |
| Gravity scale | Any finite value, including negative |
| Radius, cuboid half extents, cylinder half height | Positive |
| Capsule half height | Nonnegative (zero gives a sphere/circle) |
| Joint anchors | Finite coordinates |
| Joint axis | Finite, nonzero direction; normalized during conversion |
| Joint endpoints | Different entities; unavailable bodies remain pending |

Constructing, editing or deserializing descriptors does not itself validate
them. Validation happens at sync, before creation or application of edits.
The deprecated `build_physics_world_*` helpers also return a validation `Result`
before replacing the physics resource. The motion API and caller-owned free colliders use Rapier values directly;
descriptor validation does not apply to them.

## Serialization

`RigidBody2D/3D`, `Collider2D/3D` and `ImpulseJoint2D/3D` support the standard
scene/world and prefab serialization paths, including every shape and joint
variant. Call `register_std_components` in the destination world before loading.
Joint endpoint references are remapped to the newly created entities, including
when instantiating the same prefab multiple times.

Rapier handles and the physics world are runtime state and are not serialized.
After loading, run body sync followed by joint sync to rebuild them from the
descriptors and transforms. This saves scene configuration, not a simulation
checkpoint: runtime velocities, contacts and solver state are not preserved.
Bodies loaded as children of a prefab root must be detached before physics sync,
as required by the root-body contract above.

## Collision events and triggers

Opt in per collider with `Option<SensorSettings>` and
`Option<CollisionEventSettings>`. Both settings types currently have no fields;
`None` disables the feature and `Some(Default::default())` enables it. The
options are serialized and editable in the inspector and through body sync.
Enabling sensor behavior does not automatically enable collision events.

```rust,ignore
use redlilium_ecs::physics::{SensorSettings, CollisionEventSettings};
use redlilium_ecs::physics::events3d::CollisionEvent3D;

world.add_event::<CollisionEvent3D>();
let trigger = Collider3D::ball(2.0)
    .with_sensor(Some(SensorSettings::default()))
    .with_collision_events(Some(CollisionEventSettings::default()));
```

At least one participant must opt in; opting in on both does not duplicate the
pair event. Free colliders use Rapier's `ActiveEvents::COLLISION_EVENTS` when
created. Events do not enable otherwise filtered-out collision pairs: Rapier's
collision rules still apply, including its body-type rules.

`StepPhysics3D` publishes to `Events<CollisionEvent3D>` after stepping and writing
transforms. The 2D equivalents are `StepPhysics2D` and
`events2d::CollisionEvent2D`. If tracking is enabled (or tracked pairs await
closure) without the registered queue, Step returns `InvalidConfiguration`
before applying motion or advancing simulation. Pose validation errors likewise
leave pending pair closures for the next successful step.

Each event carries:

- `step`: a per-physics-world counter, starting at 1 for the first simulation step;
- `a`, `b`: participants in stable canonical order, without an initiator role;
- `phase`: `CollisionPhase::Started` or `Stopped(CollisionStopReason)`.

Each participant contains its collider handle, optional body handle, optional
full entity identity and `is_sensor`. These are snapshots captured when tracking
starts. Removed entities remain `Some(original_entity)`; they may no longer be
alive. Free colliders have no body or entity. A recycled index/handle never
retargets an old event. If either participant is a sensor, the event describes a
trigger intersection; otherwise it describes a physical contact.

`Started` means tracking an active pair began. Enabling events inside an existing
contact/intersection also starts tracking on the next successful step. Stop
reasons are:

| Reason | Meaning |
|---|---|
| `Separated` | The reported contact/intersection ended |
| `Removed` | At least one collider was removed, including ECS despawn/exclusion or loss of a required component |
| `TrackingDisabled` | Neither participant requests collision events anymore |
| `Reconfigured` | A participant changed its sensor role |

Sensor-role changes close the old pair using its old snapshots, then start a new
pair if the interaction is still active. Ordinary shape/material changes do not
force an artificial stop/start; actual separation still produces a stop.
Settings are observed at physics-step boundaries. Removal closures are published
by the next successful step, not by sync or `remove_free_collider` itself.
Dropping/replacing the entire physics world ends its stream without mass stop
events; already queued events remain historical snapshots.

Game systems use an independent `EventCursor` each. Put fixed-step consumers
after `StepPhysics*` with an explicit dependency. Update consumers see the events
from all fixed substeps in that frame. No start/stop transitions are coalesced
across those steps, and reads advance the cursor without consuming other
readers' events:

```rust,ignore
ctx.lock::<(Res<Events<CollisionEvent3D>>,)>().execute(|(events,)| {
    for event in events.read(&self.collision_cursor) {
        // Match event.phase and inspect event.a / event.b.
    }
});
```

The standard queue retains events for the current and previous frame;
`Schedules::run_frame` rotates it once per frame, not per fixed step. When
running containers manually, call `World::update_events` once per frame.
Readers that lag beyond retention skip expired events and can inspect
`EventCursor::missed()`. Ordering of independent pairs is not guaranteed.

Tracking keeps snapshots and adjacency only for observed pairs. Configuration
changes inspect the changed colliders' neighborhoods rather than scanning every
contact on every step. Collision force, normal and contact-point events are not
part of this API.

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

## Physics world API

ECS descriptors and sync systems own body, attached collider and joint lifetimes.
Remove the descriptor or despawn the entity, then run sync. `PhysicsWorld` does
not expose public body/joint insertion, deletion or mutable collections.
`bodies()`, `colliders()`, `impulse_joints()` and `narrow_phase()` provide
read-only Rapier access for inspection and contact queries. Use
`body_for_entity`, `entity_for_body` and `joint_for_entity` for handle lookup.

`body_motion(handle)` returns a temporary `BodyMotion3D` / `BodyMotion2D`:
it provides read access to the body and methods for velocities, forces,
impulses and sleeping. It cannot replace the body or change descriptor-owned
settings. Teleports go through `PhysicsWorld::teleport`; kinematic motion uses
the target/velocity components described above. Stepping goes through the ECS
step system so input handling and transform synchronization happen together.

```rust,ignore
ctx.lock::<(ResMut<PhysicsWorld3D>,)>()
    .execute(|(mut physics,)| {
        let Some(handle) = physics.body_for_entity(player) else { return };
        let Some(mut body) = physics.body_motion(handle) else { return };
        body.apply_impulse(Vector::new(0.0, 100.0, 0.0), true);
    });
```

### Standalone colliders

`add_free_collider(rapier_collider)` supports static scenery with custom geometry,
such as polyline or trimesh terrain. These colliders have no parent body or ECS
owner. The caller owns their lifetime and removes them with
`remove_free_collider(handle)` (or drops the physics world). They are not part of
scene serialization and survive ECS body cleanup. Removal returns `false` for
stale handles or any collider attached to a body, so it cannot remove an ECS
body's collider. The physics demo uses this path for its custom terrain.

### Ray queries

`cast_ray(origin, dir, max_toi)` returns `Option<RayHit3D>` / `Option<RayHit2D>`.
A hit contains the collider handle, optional body handle, optional ECS entity,
and `toi`. A collider with no entity still produces a hit. `toi` parameterizes
`origin + dir * toi`; it is a distance only when `dir` has unit length.

`cast_ray_filtered` accepts Rapier's `QueryFilter` for groups, body/collider
exclusions, sensor filtering and custom predicates. Filters run before choosing
the closest hit. For an ECS-only query:

```rust,ignore
let only_ecs = |_: ColliderHandle, collider: &Collider| {
    collider.parent().and_then(|body| physics.entity_for_body(body)).is_some()
};
let hit = physics.cast_ray_filtered(
    origin, dir, max_toi, QueryFilter::default().predicate(&only_ecs),
);
```

Queries use the broad phase from the last physics step. After syncing new
objects or requesting teleports, run `StepPhysics*` before querying their new
positions.
