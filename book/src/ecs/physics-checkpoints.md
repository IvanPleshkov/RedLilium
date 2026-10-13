# Live physics checkpoints and fixed replay

Physics checkpoints are opaque, reusable **in-memory continuation states** for
one physics resource in one `World`. They are separate from scene serialization.
Each checkpoint is a full copy; history memory grows with retained snapshots and
scene state.
Both dimensions and both scalar precisions are supported. This provides a physics
rollback primitive; games own input history, combat state, network protocol and
replay policy. The ECS does not acquire a dependency on a physics backend.

## Public boundary

Use `PhysicsBindings2D::with` (or `PhysicsBindings3D::with`) between completed
fixed ticks. The callback receives the physics resource and a restricted ECS
context; capture/restore are methods of the resource:

```rust,ignore
use redlilium_ecs::physics::world2d::{PhysicsBindings2D, PhysicsCheckpoint2D};

let physics_checkpoint: PhysicsCheckpoint2D = PhysicsBindings2D::with(&mut world,
    |physics, bindings| physics.capture_checkpoint(bindings))?;
let clock = schedules.capture_fixed_clock(&world)?;
// Save the game's authoritative components/resources and tick-indexed inputs.

// Later, restore the game's data and physical descriptors first.
PhysicsBindings2D::with(&mut world,
    |physics, bindings| physics.restore_checkpoint(&physics_checkpoint, bindings))?;
schedules.restore_fixed_clock(&mut world, &clock)?;
for input in historical_inputs {
    // Install this tick's game input, then run the same FixedUpdate as live play.
    schedules.run_fixed_tick(&mut world, &runner)?;
}
```

No special replay physics loop is needed. Keep the normal ordering: body sync, joint sync, gameplay
motion inputs, step, pose recording and game event consumption, with explicit
system dependency edges. Register standard components before capture. Complete
sync, deferred commands and observers before capturing; do not capture in a
running physics system or while background tasks can mutate simulation data.
The binding constructor reports `Busy` if a required resource is borrowed.

Capture includes native bodies and colliders, joints/motors, islands, contacts,
broad/narrow phase, CCD, applied-descriptor caches, entity/handle maps, physics
configuration, event tracking, pending teleports/forces and pose-reset state.
Rapier's pipeline is scratch workspace; restore rebuilds it and retains the
persistent solver state. Immutable geometry may be shared by native clones;
mutable simulation state is independent. Custom shapes with externally mutable
interior state are outside this continuation guarantee.

The ECS side restores physics handles, interpolation history, body transforms,
kinematic targets and velocities. Handles preserve their identities. Runtime
component storage order is restored as well. Game lifecycle hooks, observers and
required-component insertion are **not** invoked for these restored values.
Normal subsequent system mutations retain their usual lifecycle behavior.
Global transforms and other derived visual caches must be recomputed before
rendering, through the normal presentation schedule.

## Preconditions and structural changes

Before restore, the game must provide:

- The same live `World`, physics resource instance and entity identities. A new
  resource/world, even with identical contents, cannot consume the checkpoint.
- Exactly the saved sets and values of rigid-body, collider, collider-owner and
  impulse-joint descriptors. Restore descriptors removed or edited since capture;
  remove descriptors added since capture. Do not synchronize this reconstruction:
  physics restore installs the saved native topology and handles directly.
- Saved flags and parent relationships for participating entities, the same
  presence of body transforms, and saved transforms for fixed bodies. Dynamic
  and kinematic presentation poses are restored by physics.

An entity needed by the checkpoint must remain alive throughout the rollback
window. Removing a joint *component* is supported: reinsert its saved descriptor
on the same entity, then restore physics. Despawning that entity and reusing its
slot is incompatible. Games can defer final despawn until history expires. This
API does not resurrect entities, remap references or rewind arbitrary ECS state.

Validation and native cloning precede writes. An incompatible restore returns
`PhysicsCheckpointError` without changing live physics, ECS bindings or event
queues. Checkpoints remain reusable after failed and successful restores. The
physics and scheduler operations are independently validated operations, not a
transaction over arbitrary game state; a game must not resume simulation after
only part of its own restore sequence succeeds.

## Simulation clock and scheduler

`FixedTime` contains `tick()`, `elapsed()` and `delta()`. `Schedules::run_frame`
and `run_fixed_tick` use the same fixed execution path. Tick zero is before the
first fixed tick, with zero elapsed/delta; inside tick one the clock is already
`1`, `fixed_timestep`, `fixed_timestep`. A failed tick still counts as an attempted
tick and may have partially changed game state. `run_fixed_tick` returns its
system errors; restore all relevant state before retrying.

`run_fixed_tick` executes the runner, including command/observer flushing. It
sets `Time::delta()` and `Time::fixed_delta()` to the fixed step while systems run
and restores the previous effective delta afterward. It does not advance
`GameTime`, `RealTime`, frame elapsed time, frame queues or reactive trigger
buffers; execute frame/state-transition schedules; or spend accumulator time.
It runs even when `GameTime` is paused. `run_schedule::<FixedUpdate>` remains a
raw schedule invocation and does not provide this clock contract.

`capture_fixed_clock` / `restore_fixed_clock` rewind only the simulation clock.
They require the same `Schedules`, `World` and fixed timestep. Keep schedule
composition, ordering and timestep unchanged throughout a replay window. The
render accumulator and `Time::fixed_alpha()` stay at their current live values:
after replay to the live tick, interpolation uses corrected previous/current
poses at the existing render phase. Do not render intermediate replay ticks.

Use `FixedTime` or fixed delta for replayable gameplay timers. `GameTime` is a
scaled **frame** clock; `RealTime` is suitable for real-duration UI/envelopes.
Gameplay writes to these resources are not suppressed during replay. The game
must explicitly restore/reconcile its slow-motion timeline and other side
effects rather than assuming the scheduler will deduce their ownership.

This is not an ECS checkpoint: system-local mutable state, previous system
results, event cursors, state-machine transitions, triggers and change-detection
ticks are not rewound. `Changed`/`Added` reflect new restore writes on the live
monotonic ECS timeline. Replay-critical decisions must depend on explicit saved
game data and tick inputs, with deterministic iteration/ordering. Restore game
storage order where game logic depends on it. Put presentation-only side effects
outside the replay schedule or explicitly suppress/deduplicate them in the game.

## Tick events

Default `PhysicsEventDelivery::FrameQueues` preserves the existing collision and
contact-force `Events<T>` behavior. Physics restore discards these two legacy
queues, without rewinding event IDs or arbitrary consumers' `EventCursor`s.
Other event queues are untouched.

For replay, set `physics.event_delivery = PhysicsEventDelivery::TickOnly`.
Stock step systems then require no collision/force frame queues and publish
nothing to them. Read `physics.last_step_events()` after the step, using an
explicit ordering edge. `PhysicsStepEvents2D` / `PhysicsStepEvents3D` contain a
physics `step` number, collisions and contact forces. Each step replaces the
batch; consume each tick before the next one. Checkpoints preserve that counter,
the last batch and contact lifecycle tracking. Restoring a persistent contact
therefore does not manufacture a new start solely because of restore. Physics
step numbers rewind and can recur during replay; the game owns external-event
deduplication. The physics step counter is independent of `FixedTime` (one
physics step per fixed tick is an application schedule invariant).

## Verification and determinism scope

The guarantee is continuation from a checkpoint in the same instance, build,
precision and execution environment, with identical subsequent inputs and
ordering. It is not a stable wire format or a cross-platform lockstep guarantee.
The default Rapier feature set is retained. Selecting cross-machine deterministic
simulation requires a separate audit of engine and game ordering, math, compiler
features and supported platforms.

`ecs/tests/physics_checkpoints.rs` covers both dimensions: exact sphere
continuation, loaded motor joint and removed-joint restoration, persistent
contacts, sleep/wake, CCD, pending teleports/forces, kinematic inputs,
interpolation, reusable checkpoints, incompatible identities/descriptors, borrowed
resources and hook-free restore. `ecs/tests/fixed_replay.rs` checks clock errors,
frame isolation, deferred commands and residual accumulator preservation.

The headless probe runs a separate 101-body contact scene, verifies local
continuation, reports checkpoint cost, and fingerprints 64 ticks for comparison.
The WASM build also runs the same physics and scheduler contract cases:

```sh
cargo test -p redlilium-ecs --features physics --offline
cargo test -p redlilium-ecs --features 'physics-2d-f32 physics-3d-f32' --offline
cargo run -p redlilium-ecs --release --example physics_checkpoint_probe \
  --features physics --offline
cargo build -p redlilium-ecs --release --example physics_checkpoint_probe \
  --target wasm32-unknown-unknown --features physics --offline
# Use a wasm-bindgen CLI version matching Cargo.lock.
wasm-bindgen target/wasm32-unknown-unknown/release/examples/physics_checkpoint_probe.wasm \
  --target nodejs --out-dir /tmp/physics-checkpoint-probe
node -e 'console.log(require("/tmp/physics-checkpoint-probe/physics_checkpoint_probe.js").probe())'
```

For f32 measurements use `--features 'physics physics-2d-f32 physics-3d-f32'`.
`retained_bytes` is additional requested heap storage plus the checkpoint's inline
size, excluding shared geometry already alive in the world, allocator overhead
and temporary peak allocations. Capture timings include dropping each captured
value; restore timings include validation and ECS restoration. They are averages
of 100 operations in one warmed scene, including allocation-counter overhead,
not worst-case game budgets. Matching
fingerprints in this probe do not establish general native/WASM determinism.

### Recorded release sample

Measured on Apple M3, `aarch64-apple-darwin`, Rust 1.97.1, Rapier 0.36.0;
WASM used `wasm32-unknown-unknown` through Node 22.16.0 and wasm-bindgen 0.2.108.
Both full native ECS test matrices passed (1,218 tests each, 77 ignored), as did
the focused mixed-precision and physics-free clock tests. WASM ran the 16 physics
and 3 scheduler contract cases for each precision. Browser rendering and full
game rollback were not part of this headless verification.

| Dimension / scalar | Environment | Retained bytes | Capture + drop, µs | Restore, µs |
|---|---|---:|---:|---:|
| 2D f64 | Native | 422,785 | 70.2 | 71.4 |
| 2D f64 | WASM | 389,549 | 75.8 | 95.9 |
| 3D f64 | Native | 462,003 | 69.4 | 74.9 |
| 3D f64 | WASM | 423,507 | 71.3 | 96.9 |
| 2D f32 | Native | 357,027 | 66.8 | 69.7 |
| 2D f32 | WASM | 318,055 | 78.4 | 100.0 |
| 3D f32 | Native | 400,756 | 65.1 | 74.9 |
| 3D f32 | WASM | 357,084 | 72.4 | 95.4 |

All eight runs reproduced their own local continuation exactly. Comparing native
against WASM produced:

| Configuration | Native fingerprint | WASM fingerprint | Equal |
|---|---|---|---|
| 2D f64 | `6b17d60b3615d55c` | `6b17d60b3615d55c` | Yes |
| 3D f64 | `21c38e6dd75442a3` | `21c38e6dd75442a3` | Yes |
| 2D f32 | `19708bf62217d33b` | `19708bf62217d33b` | Yes |
| 3D f32 | `ccc7c31489f8f638` | `1af93463845a079d` | **No** |

The 3D f32 difference is a measured cross-environment divergence in the default
configuration, despite exact local checkpoint continuation on both sides. Do not
select cross-platform lockstep from these results. Matching rows also require
broader validation before any cross-platform determinism guarantee. A network
strategy with authoritative corrections still needs game-owned state transfer;
the opaque checkpoint itself is local history, not a transferable network packet.
