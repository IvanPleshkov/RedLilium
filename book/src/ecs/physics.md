# Physics Integration

RedLilium integrates the [Rapier](https://rapier.rs/) physics engine via feature flags. Both 2D and 3D physics are supported.

The backend uses Rapier 0.36 and its default integration parameters. In
particular, fast dynamic bodies use CCD against fixed colliders by default;
Rapier's per-body CCD flag additionally enables sweeps against moving bodies.
`integration_parameters.max_ccd_substeps = 0` disables CCD entirely. The ECS
integration currently exposes rigid bodies only; Rapier's required soft-body
set remains internal and empty.

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
ordering edges. The `2D` systems follow the same order. Body sync creates/updates bodies first, then their colliders. Native identity
maps are available when sync finishes. The exclusive systems also publish ECS
handle components immediately; regular `SyncPhysicsBodiesSystem*` and
`SyncPhysicsJointsSystem*` publish them through deferred commands. Order joint
sync after body sync, even when both are regular systems.

A managed body requires `RigidBody*` and `Transform`. On the next body sync,
removing either component, despawning the entity or excluding it from game
queries removes the Rapier body, its colliders and attached joints. Their mappings
and handle components are cleaned up too. Removing a `Collider*` removes only
that collider; the body and its joints survive even with no colliders. In 3D, body removal
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
the body and its connections. Changing mass settings or collider shape/density/local pose updates mass
properties by the end of body sync. Changed settings wake the body; unchanged
settings do not wake it or overwrite its runtime velocity. Body-type changes
also reset interpolation history.

Changing joint endpoints, kind or local frames rebuilds that joint. Limits,
motor mode, targets, gains and effort caps update the existing joint in place. Other bodies and joints remain intact. If an endpoint is unavailable,
the old joint is removed and creation waits for valid endpoints. Physics-world
caches retain the last applied descriptors to avoid rebuilding unchanged objects.

### Joint frames, limits and motors

`ImpulseJoint2D/3D` stores a `JointType2D/3D` kind, two `JointFrame2D/3D` local
frames, and optional `JointLimits` and `JointMotor`. Frames are relative to each
body's origin, independently of its centre of mass. Their matching orientations
define zero angle. Local X is the slider axis and the 3D hinge axis; 2D hinges
rotate around Z. Coordinates and velocities describe frame 2 relative to frame 1.
Frames contain translation and rotation (radians in 2D, a normalized-on-use
quaternion in 3D), without scale. Short anchor/axis constructors build frames;
`with_local_frames` specifies each body's frame independently. `JointFrame*::from_axis`
normalizes finite nonzero directions; invalid directions produce invalid frames
that sync rejects.

Revolute and prismatic joints support `.with_limits(Some(...))` and
`.with_motor(Some(...))`. `None` disables the corresponding feature. Fixed and
spherical joints reject these settings. Spherical cone/twist limits and orientation
motors are not part of this single-axis API.

```rust,ignore
use redlilium_ecs::physics::{JointLimits, JointMotor, JointMotorModel};
use redlilium_ecs::physics::components3d::{ImpulseJoint3D, JointFrame3D};

let mut hinge = ImpulseJoint3D::revolute(body_a, body_b, axis, anchor_a, anchor_b)
    .with_limits(Some(JointLimits { min: -1.0, max: 1.0 }))
    .with_motor(Some(JointMotor::position(0.0, 30.0, 8.0)
        .with_model(JointMotorModel::AccelerationBased)
        .with_max_effort(Some(80.0))));
// Edit the ECS descriptor before SyncPhysicsJoints*, not native joint storage.
hinge.set_motor_position_target(0.7)?;
hinge.set_motor_velocity_target(0.0)?;
```

Limits are inclusive: equal bounds lock the coordinate. Hinge angles and angular
velocities use radians and radians/second; sliders use world length units and
length/second. Hinge limits must fit inside `[-π, π]`, with width less than a full
turn and no interval crossing the branch cut. Position targets use that same angular
range, must lie inside configured limits, and do not count revolutions. A velocity
motor can rotate continuously with limits disabled. Solver constraints allow small
errors depending on timestep, iterations and load.

Angular position motors use Rapier's shortest-rotation error. They do not plan a
path around limits: with limits `[-3, 3]`, a target change from `+2.8` to `-2.8`
pushes against the `+3` stop. To take the longer allowed path, first target `0`,
then `-2.8` after reaching the intermediate target. Merely being inside the limits
does not guarantee that a position target is reachable by the motor's chosen path.

`JointDrive::Velocity { velocity, damping }` requires positive finite damping.
`JointDrive::Position { position, velocity, stiffness, damping }` requires positive
finite stiffness and nonnegative finite damping. The position constructor sets
target velocity to zero. All targets must be finite. Position target setters reject
a velocity drive; both setters reject disabled motors and preserve the descriptor
on error. Direct field edits and builders are validated by sync before any native
mutation in the batch.

`AccelerationBased` is the default and makes gains less dependent on mass/inertia.
`ForceBased` expresses physical stiffness and damping. Both respect `max_effort`:
force for sliders, torque for hinges, `None` for no authored cap, and zero for no
actuation. A position drive provides spring/damper behavior, including suspension.

Configuration and current targets have a single owner: the ECS descriptor. They
are serialized with the scene and exposed in the inspector. Parameter edits wake
connected bodies; unchanged sync preserves sleep. Existing handles and unrelated
constraint impulses survive parameter changes; changing a limit, drive mode,
gains, model or cap resets the affected cached impulse. Target-only edits preserve
it. Structural edits rebuild the joint. Run sync after gameplay target edits and
before stepping physics.

### Collider ownership and local geometry

The same contract applies to 2D and 3D. A `Collider3D` without `ColliderBody3D`
uses the rigid body on its own entity. An explicit `ColliderBody3D { body }`
attaches it to that entity's body. There is no search through `Parent` and no
fallback when an explicit owner is unavailable. One entity carries at most one
collider per dimension; a body can have any number of collider entities.

```rust,ignore
use redlilium_ecs::physics::components3d::{
    Collider3D, ColliderBody3D, ColliderPose3D, ColliderPart3D,
};

// Body and primary collider can still share one entity.
let car = world.spawn_with((
    Transform::IDENTITY,
    RigidBody3D::dynamic(),
    Collider3D::cuboid(Vec3::new(1.0, 0.5, 2.0)),
))?;

// The sensor has its own identity/settings and an explicit physical owner.
let sensor = world.spawn_with((
    ColliderBody3D { body: car },
    Collider3D::ball(2.0)
        .with_sensor(Some(SensorSettings {}))
        .with_local_pose(ColliderPose3D {
            translation: Vec3::new(0.0, 1.0, 0.0),
            ..Default::default()
        }),
))?;

// A compound is ONE collider with shared settings and no part identities.
let furniture = Collider3D::compound(vec![
    ColliderPart3D::cuboid(Vec3::new(1.0, 0.1, 1.0)),
    ColliderPart3D::cuboid(Vec3::new(0.1, 1.0, 0.1))
        .with_local_pose(ColliderPose3D {
            translation: Vec3::new(0.8, -1.0, 0.8),
            ..Default::default()
        }),
]);
```

`ColliderPose*` defaults to identity, has no scale, and is relative to the
body origin, not its centre of mass. In 2D its rotation is a Z angle in radians;
in 3D it is a quaternion. The world pose of a compound part is
`body_pose * collider_local_pose * part_local_pose`. Collider-entity `Transform`
and `Parent` do not supply a physical pose. They may still serve presentation
or prefab grouping; the existing root/unit-scale rule applies to body entities.

Compounds contain a nonempty, flat list of `ColliderPart*` values. Each part has
primitive geometry (`ColliderPrimitive*`) and a local pose. Parts share material,
density, sensor status, groups and event settings. For independently identifiable
or configurable parts, use separate collider entities. Overlapping parts are
not a geometric union for mass calculation; their mass contributions add.
Sensors retain the configured density too; use density zero if they should not
contribute mass. Bodies with no colliders have no collider-derived mass/inertia;
fully explicit body mass properties can supply them.

A missing, disabled or removed body leaves collider descriptors inactive, without
despawning their entities. Restoring the body activates them at the next sync.
The reference includes the entity generation, so slot reuse cannot retarget it.
Removing an explicit `ColliderBody*` switches back to the colocated-body rule.
Changing the physical owner recreates the collider, ending its old observed
pairs with `Removed`; new contacts may start on the next step. Geometry, material
and local-pose edits preserve the collider handle and wake its body. Unchanged
settings preserve sleep; a pose-only edit reuses the existing shared geometry.
Geometry is built on creation/shape changes, not on every sync. Validation checks
the complete sync batch before mutating native objects, including waiting colliders.

`Collider*Handle` belongs to the collider entity; `RigidBody*Handle` belongs to
the body entity. Physics worlds expose `collider_for_entity` and
`entity_for_collider` alongside the existing body lookup methods. Collision and
force participants and query targets carry **both** `collider_entity` and
`body_entity`. They coincide for a colocated collider. Free colliders have neither.
Events remain per collider pair; there is no implicit deduplication by body or
separate event identity for a compound part. Events retain their captured identities
after removal/reparenting. Descriptors, local geometry and explicit owner references
round-trip through scenes/prefabs, with entity references remapped on load.

### Final mass, centre of mass and inertia

`RigidBody2D/3D::with_mass_properties` accepts `Option<MassSettings2D/3D>`.
`None` derives all mass properties from attached colliders and their densities.
`Some(settings)` sets the **final total mass**, not additional mass. The centre
of mass and central angular inertia can each be inferred or overridden:

```rust,ignore
use redlilium_ecs::physics::{MassSettings3D, AngularInertia3D};

let car = RigidBody3D::dynamic().with_mass_properties(Some(
    MassSettings3D::new(1200.0)
        .with_center_of_mass(Some(Vec3::new(0.0, -0.3, 0.0))),
));

// Fully explicit properties also work without any colliders.
let body = RigidBody3D::dynamic().with_mass_properties(Some(
    MassSettings3D::new(10.0)
        .with_center_of_mass(Some(Vec3::zeros()))
        .with_inertia(Some(AngularInertia3D::diagonal(Vec3::new(2.0, 3.0, 4.0)))),
));
```

Settings contain `mass`, `center_of_mass: Option<Vec2/Vec3>` and
`inertia: Option<f32/AngularInertia3D>`. In 3D, `AngularInertia3D` contains
`principal: Vec3` and `rotation: Quat`: moments about the centre of mass and
orientation of their principal axes relative to the body. Identity rotation
aligns these axes with the body's local XYZ axes. The 2D moment is about Z.
Mass is in kg and inertia in kg·m² when the length unit is a metre.

For inferred values, the integration combines collider geometry, local poses
and authored densities, then scales the distribution to the requested total
mass. This preserves the density-weighted centre and scales central inertia
proportionally. Zero-density colliders do not contribute; sensors follow the
same density rule. Compound parts contribute additively.

An overridden centre is relative to the body origin, not world space. Moving it
preserves the inferred **central** inertia: this authors a virtual translation
of the mass distribution relative to collision geometry. It does not move the
body pose, colliders or joint anchors. Explicit moments are already for the
requested final mass and are not automatically scaled.

Sync validates all proposed mass models before mutating any native objects,
including collider removal, owner changes and density/geometry edits. If an
inferred value lacks a positive finite collider mass source, sync returns
`InvalidConfiguration` and retains the previously applied physics state. Fully
explicit settings require no geometry. Each authored mass/moment must be finite
and positive; centre coordinates must be finite; 3D principal moments must satisfy
the triangle inequalities (with rounding tolerance), and the axes quaternion
must be finite and nonzero. Derived properties must also be representable in the
active physics precision. Use `locked_axes` for locked rotation, not zero inertia.

After sync, mass properties are immediately usable by impulses, before stepping.
Edits preserve handles, joints, body-origin pose, linear centre-of-mass velocity
and angular velocity; they do not conserve momentum automatically. Changed mass
models wake their bodies; no-op sync preserves sleep. Fixed and kinematic bodies
retain the settings for a later switch to dynamic.

Only affected bodies are recalculated. Friction, event settings and other edits
unrelated to mass do not rebuild the mass model. In explicit mode the backend
colliders have zero mass contribution; ECS retains authored densities for
inference and for returning to automatic mode. The raw read-only Rapier collider
`density()` therefore reports zero in this mode. Total properties belong to the
body. Returning to `None` restores collider contributions and removes the explicit
override. Ordinary automatic bodies may have zero collider-derived mass.

If earlier deferred commands cancel a newly synchronized collider, publication
recalculates surviving explicit bodies. When cancellation removes a required
mass source, `StepPhysics*` rejects the incomplete model until descriptors are
corrected and body sync succeeds. Settings are serialized with the body and
editable in the inspector.

### Axis locks

`RigidBody2D/3D::with_locked_axes` accepts `Option<LockedAxes2D/3D>`. `None`
leaves all degrees of freedom available. In a settings value, `true` means
locked; `Default::default()` leaves every axis free.

```rust,ignore
use redlilium_ecs::physics::LockedAxes3D;

// Keep a dynamic character upright while allowing yaw around world Y.
let body = RigidBody3D::dynamic().with_locked_axes(Some(LockedAxes3D {
    rotation_x: true,
    rotation_z: true,
    ..Default::default()
}));

// Or prohibit every rotation while leaving translation available.
let body = RigidBody3D::dynamic()
    .with_locked_axes(Some(LockedAxes3D::rotations()));
```

3D settings have `translation_x/y/z` and `rotation_x/y/z`. The 2D settings
have `translation_x`, `translation_y`, and `rotation` (around Z). Both types
provide `all()`, `translations()`, and `rotations()` presets. These are world
axes, independent of body orientation. Rotation locks constrain angular
velocity, not individual Euler angles or an absolute orientation target.

Locks affect **dynamic** simulation, including forces, impulses, gravity and
contacts. Sync updates them on the existing body, preserving its collider and
joints. Enabling a lock discards the corresponding current velocity component;
other components and the pose stay unchanged. `BodyMotion::set_linvel` and
`set_angvel` also discard locked components for dynamic bodies. Removing a
lock does not restore discarded velocity. Persistent forces/torques are kept
and can accelerate the body after unlocking, until explicitly reset.

Fixed transforms, both kinematic control modes and explicit teleports keep
their existing pose ownership. Locks are retained on non-dynamic descriptors
and become effective when the body changes to dynamic; that transition also
discards forbidden velocities. A teleport can move a locked body to a new pose:
locks do not bind it to a previous world position.

Settings are serialized with the body and editable in the inspector. Changed
descriptors wake the body; syncing an unchanged descriptor preserves sleep.
Every boolean combination is valid, without additional numeric validation.

Rapier 0.36's gyroscopic correction can inject angular velocity on a locked axis
of an asymmetric 3D body. The integration therefore disables gyroscopic forces
while any rotation axis is locked and re-enables them when all rotation locks
are removed. Translation-only locks keep gyroscopic forces enabled. This avoids
per-frame correction passes or forbidden rotation inside solver substeps.

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
| Contact-force event threshold | Nonnegative |
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

## Continuous collision detection (CCD)

`RigidBody2D` and `RigidBody3D` expose `ccd: Option<CcdSettings>`:

```rust,ignore
use redlilium_ecs::physics::CcdSettings;

let projectile = RigidBody3D::dynamic()
    .with_ccd(Some(CcdSettings::default()));
```

- `None` keeps Rapier's automatic CCD for fast dynamic bodies against fixed
  colliders. It does **not** disable all continuous collision detection.
- `Some(CcdSettings::default())` enables the extended ("bullet") mode, adding
  sweeps against kinematic and non-bullet dynamic bodies.
- `physics.integration_parameters.max_ccd_substeps = 0` disables CCD globally,
  regardless of the per-body setting.

`CcdSettings` currently has no fields. The option is serialized with the body,
editable in the inspector, and applied by either body-sync system in place.
Changing it preserves handles, joints, pose and velocities and wakes the body;
syncing an unchanged descriptor does not wake it. The setting can be stored on
any body type, but it only acts on dynamic bodies. Switching a configured body
back to dynamic makes it effective again.

Rapier 0.36 does not perform CCD sweeps between two bullet bodies. Enable this
mode selectively for fast objects; enabling it on every body does not give
universal CCD coverage. Collision groups and body-type rules still apply.
CCD limits motion at an impact; it does not guarantee that contact velocities
are fully resolved in that same step.

Sensor crossings detected by CCD can publish `Started` and `Stopped(Separated)`
within the same physics step, even when there is no overlap at either endpoint.
Sensors do not stop the moving body. Register the collision-event queue and opt
in on a participating collider as described below.

## Collision groups

`Collider2D` and `Collider3D` share `Option<CollisionGroups>`. Each setting has
two `u32` masks: `memberships` identifies the groups the collider belongs to,
and `filter` identifies the groups it accepts. The 32 bits are user-defined;
the engine does not reserve any layers.

```rust,ignore
use redlilium_ecs::physics::CollisionGroups;

const PLAYER: u32 = 1 << 0;
const WORLD: u32 = 1 << 1;
const ENEMY: u32 = 1 << 2;

let player = Collider3D::ball(0.5)
    .with_collision_groups(Some(CollisionGroups::new(PLAYER, WORLD | ENEMY)));
let enemy = Collider3D::ball(0.5)
    .with_collision_groups(Some(CollisionGroups::new(ENEMY, PLAYER | WORLD)));
```

Both sides must accept the pair:
`(a.memberships & b.filter) != 0 && (b.memberships & a.filter) != 0`.
`None` and `Some(CollisionGroups::default())` both mean all memberships and
all accepted groups. A zero mask in either field rejects every pair. Masks
apply to solid contacts and sensor intersections alike. They do not override
the body-type rules (for example, fixed–fixed pairs are excluded by default).
This configures collision detection, not just solver forces.

Groups are serialized and editable in the inspector as hexadecimal masks.
Body sync applies edits in place, preserving handles, joints and body motion.
At the next successful step, a newly rejected tracked pair closes with
`Stopped(FilteredOut)`; a newly allowed active pair emits `Started`.
An edit that keeps the pair allowed does not force a stop/start.
If several settings change together, removal takes priority, then disabling
tracking, then a sensor-role change, then group/type filtering.

The shared settings convert into either dimension's Rapier `InteractionGroups`
using `.into()`, including for free-collider builders and scene queries:

```rust,ignore
let filter = QueryFilter::default()
    .groups(CollisionGroups::new(PLAYER, WORLD | ENEMY).into());
let hit = physics.cast_ray(origin, displacement, RayCastOptions::default(), filter);
```

Query masks use the same bilateral rule: the query's memberships must also be
accepted by the collider. `cast_ray` and a default `QueryFilter` do not filter
by groups, so they can still hit colliders whose masks reject all collisions.

## Collision types

`Collider2D` and `Collider3D` expose `collision_types: Option<CollisionTypes>`.
The six boolean fields select unordered body-type pairs. Both kinematic modes
use the same kinematic flags; standalone colliders count as fixed.

| Field | Default |
|---|---|
| `dynamic_dynamic` | true |
| `dynamic_kinematic` | true |
| `dynamic_fixed` | true |
| `kinematic_kinematic` | false |
| `kinematic_fixed` | false |
| `fixed_fixed` | false |

`None` uses these defaults, just like `Some(CollisionTypes::default())`.
`CollisionTypes::all()` enables all six pairs; `CollisionTypes::none()` requests
none. Following Rapier, **either collider** can enable detection for a pair.
Thus `Some(CollisionTypes::none())` on one collider alone does not prevent
collisions: the other collider may still request them. Use group masks to
unconditionally reject a pair from one side; groups must allow both directions.

For example, a fixed trigger can detect a kinematic character:

```rust,ignore
use redlilium_ecs::physics::{CollisionTypes, SensorSettings};

let trigger = Collider3D::ball(2.0)
    .with_sensor(Some(SensorSettings::default()))
    .with_collision_types(Some(CollisionTypes {
        kinematic_fixed: true,
        ..Default::default()
    }));
```

These flags enable contact/intersection detection; they do not make fixed or
kinematic bodies respond to impulses. A kinematic character still needs its
own movement/controller logic to stop at walls. Event tracking remains a
separate collider option, and the event queue must be registered before Step.

Rules are serialized, editable in the inspector, and applied by body sync
without recreating colliders. Changes to either the rules or the body's type
are observed on the next successful Step: rejected tracked pairs emit
`Stopped(FilteredOut)`, and newly allowed active pairs emit `Started`. Pairs
that remain allowed keep their tracking. Unchanged descriptors do not wake
bodies or force pair recomputation.

The shared settings convert to either dimension's Rapier `ActiveCollisionTypes`
via `.into()` for free-collider builders. Ray queries use their own `QueryFilter`;
these body-pair rules do not hide colliders from queries.

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
| `FilteredOut` | The current collision groups or body-type rules reject the pair |
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

## Contact-force events

Force reporting is independent of collision transitions. Enable it per collider
and register its own event queue:

```rust,ignore
use redlilium_ecs::physics::ContactForceSettings;
use redlilium_ecs::physics::events3d::ContactForceEvent3D;

world.add_event::<ContactForceEvent3D>();
let collider = Collider3D::ball(0.5)
    .with_contact_force_events(Some(ContactForceSettings { min_force: 100.0 }));
```

`None` disables reporting; the default settings enable a threshold of zero.
The threshold must be finite and nonnegative. Settings are serialized, exposed
in the inspector, and applied by body sync without replacing handles. Enabled
reporting without `Events<ContactForceEvent3D>` returns `InvalidConfiguration`
before any motion is applied or simulation advances. The 2D equivalent is
`events2d::ContactForceEvent2D`.

Each event describes **one pair over one complete physics step**:

- `step`, `a`, `b`: the same world counter and canonical participant snapshots as
  collision events, including entity generations. Reading them does not require
  the bodies or colliders to remain alive.
- `dt`: the full step duration in seconds.
- `normal_impulse`: the sum of normal impulse magnitudes across all solver
  contacts and all internal CCD substeps. Friction impulses are excluded.
- `average_force()`: `normal_impulse / dt`, compared strictly against the
  thresholds of the participants that enabled reporting. Exceeding either
  threshold is sufficient; enabling both sides does not duplicate the event.
- `strongest_contact`: one representative contact with `point`, `normal`, and
  `normal_impulse`. It is the largest individual point impulse in any CCD
  substep, not the sum for a tracked contact point across substeps. The point is
  the midpoint of the solver's world-space surface points when collected. The
  normal points from canonical participant `a` toward `b`. Ties choose one of
  the equally strong contacts, without promising which one.

Scalar measurements use `f64` with either backend precision; points and normals
use the engine's `Vec2`/`Vec3`. In a scene using meters, kilograms and seconds,
impulse is measured in N·s and force in N. The force is an average over the
whole step, not a peak force or a measure of damage.

An awake supporting contact can report every step above threshold, even without
a new collision. Sensors and sleeping pairs produce no force events. Collision
groups and body-type filtering still apply. Gameplay decides what constitutes
an impact and handles sound, damage and cooldowns. There is no force-specific
`Started`/`Stopped` state.

Events use the usual `Events<T>`/`EventCursor<T>` lifetime, so several fixed steps
in one frame remain individually readable. Pair order within a step is not
specified. Standalone colliders may enable Rapier's
`ActiveEvents::CONTACT_FORCE_EVENTS` and set `contact_force_event_threshold`
before insertion. The world captures that threshold and uses an internal zero
threshold to collect every CCD contribution before applying the full-step test.

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
`body_for_entity`, `entity_for_body`, `collider_for_entity`,
`entity_for_collider` and `joint_for_entity` for handle lookup.

`body_motion(handle)` returns a temporary `BodyMotion3D` / `BodyMotion2D`:
it provides read access to the body and methods for velocities, forces,
impulses and sleeping. It cannot replace the body or change descriptor-owned
settings. Teleports go through `PhysicsWorld::teleport`; kinematic motion uses
the target/velocity components described above. Stepping goes through the ECS
step system so input handling and transform synchronization happen together.

`body_motion(handle).sleep()` zeros velocities and requests sleep. An isolated
body holds its pose across subsequent steps despite gravity, until woken with
`wake_up(true)`, a waking motion input, or an interaction with an awake body.
Connected/contacting bodies follow Rapier's island sleep and wake rules.

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

### Spatial queries

`PhysicsWorld2D/3D` offers three read-only operations with matching semantics:

```rust,ignore
use redlilium_ecs::physics::{RayCastOptions, ShapeCastOptions};
use redlilium_ecs::physics::queries3d::QueryFilter;
use std::ops::ControlFlow;

let ray_hit = physics.cast_ray(
    origin, displacement, RayCastOptions::default(), filter,
)?;
let shape_hit = physics.cast_shape(
    &shape, pose, displacement, ShapeCastOptions::default(), filter,
)?;
let completed = physics.visit_overlaps(&shape, pose, filter, |target| {
    targets.push(target); // Reuse a caller-owned buffer, or break on the first hit.
    ControlFlow::Continue(())
})?;
```

`shape` is a `ColliderShape2D/3D` descriptor and `pose` is a
`PhysicsPose2D/3D`, without visual scale. Query primitives are borrowed from
stack storage without geometry allocations. Compound query descriptors build
temporary shared geometry for the query; they do not create a collider in the
world. A shape query uses the shape and its part poses, not the `Collider`'s
outer local pose; compose that explicitly into the query pose when needed.
They can still hit complex scene geometry, including free terrain colliders.

Casts return `Result<Option<Hit>, PhysicsQueryError>`: `Ok(None)` is a valid
query with no hit, while malformed input is an error. Origins, translations,
rotations and displacements must be finite; displacements must be nonzero and
have finite endpoints. A 3D quaternion must have finite squared norm at least
`1e-12` and is normalized. Shape dimensions follow the same rules as collider
descriptors. `target_distance` must be finite and nonnegative. Validation runs
even in an empty world, before invoking an overlap visitor.

Both casts traverse the finite displacement vector and report a `fraction` in
`[0, 1]`. The reached translation is `origin + displacement * fraction`; the
travelled distance is `displacement.norm() * fraction`. There is no direction
normalization requirement or separate maximum time/distance argument. Shape
casts hold orientation fixed and treat other colliders at their physical poses,
without predicting their motion.

Results live in `physics::queries2d/queries3d` and are also re-exported by the
world modules. `RayHit*` and `ShapeCastHit*` contain a `QueryTarget*` with collider
handle, optional body handle, `collider_entity` and `body_entity`. Free colliders
remain valid hits with neither ECS identity. These are snapshots, not references; handles and
full entity identities may already be dead when read later.

`RayHit*` contains `fraction`, world-space `point`, and an optional unit
`normal`. For closed shapes a surface hit's normal points outward, including
an exit hit; open surfaces use the backend's hit-face orientation. Default
`RayCastOptions { solid: true }` reports a hit at fraction zero when the origin
is inside a collider. Its point is the origin and its normal is `None`, since
there is no unique hit surface. `solid: false` seeks the exit surface instead.
Degenerate normals are also represented as `None`. For a compound, hollow rays
select a boundary of an individual part, not the exit from the union of parts.

`ShapeCastHit*` contains `fraction`, `status`, and optional `geometry` with
world-space `point_on_collider`, `point_on_shape` at impact, and a unit normal
outward from the obstacle. `ShapeCastOptions` defaults to zero clearance and
`stop_at_penetration: true`. Positive `target_distance` reports a hit before
touching, leaving that clearance. `stop_at_penetration: false` permits skipping
initial contacts when moving apart; it does not discard every initial overlap.

`ShapeCastStatus` distinguishes `Converged`, `InitialContact` (including being
within clearance), `OutOfIterations`, and `Failed`. The latter two retain the
backend's conservative hit estimate instead of pretending the way is clear.
Geometry is optional when the backend cannot supply usable points/normals;
a nonfinite or out-of-range fraction returns `PhysicsQueryError::InvalidResult`.
Initial-contact geometry is resolved through a contact query, so the reported
points are on the surfaces rather than penetration-scaled sweep witnesses.

`visit_overlaps` tests actual geometry, not merely intersecting bounding boxes.
Each collider is visited once (including compounds), in unspecified order;
results are not deduplicated by entity. The visitor returns `ControlFlow<()>`,
and the method returns `Result<ControlFlow<()>, PhysicsQueryError>` to distinguish
completion from early exit. It does not allocate a result array or sort results.

Every operation takes Rapier's `QueryFilter`, re-exported from each query module.
It supports groups, body/collider exclusions, sensors, body types and a custom
predicate. Filtering runs **before** selecting the closest hit. For example:

```rust,ignore
let only_ecs = |_: ColliderHandle, collider: &Collider| {
    collider.parent().and_then(|body| physics.entity_for_body(body)).is_some()
};
let hit = physics.cast_ray(
    origin, displacement, RayCastOptions::default(),
    QueryFilter::default().exclude_sensors().predicate(&only_ecs),
)?;
```

Queries use the broad phase maintained by the physics step. After syncing new
objects or requesting teleports, run `StepPhysics*` before querying their new
positions. A query never applies pending commands, updates the search structure,
or advances simulation. It does not promise a coherent old snapshot if the
collider data has been edited since that broad phase was updated.
