# System Sets and Stages (Partial)

## What Are System Sets and Stages?

Modern ECS frameworks organize system execution into **stages** (fixed execution phases like PreUpdate, Update, PostUpdate) and **system sets** (named groups of systems that share ordering constraints or run conditions). This allows complex scheduling without specifying pairwise edges for every system.

```rust
// Bevy-style stages and sets (not available in RedLilium)
app.add_systems(Update, (
    movement_system,
    collision_system.after(movement_system),
));

app.add_systems(PostUpdate, (
    update_transforms,
    update_camera.after(update_transforms),
));

// System sets group systems with shared constraints
app.configure_sets(Update, PhysicsSet.before(RenderingSet));
```

## What RedLilium Has

RedLilium provides **explicit dependency edges** between individual systems via `SystemsContainer`:

```rust
let mut systems = SystemsContainer::new();

systems.add(MovementSystem);
systems.add(CollisionSystem);
systems.add(UpdateGlobalTransforms);
systems.add(UpdateCameraMatrices);

// Pairwise edges
systems.add_edge::<MovementSystem, UpdateGlobalTransforms>().unwrap();
systems.add_edge::<CollisionSystem, UpdateGlobalTransforms>().unwrap();
systems.add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>().unwrap();
```

The container resolves these into a topological order using Kahn's algorithm with cycle detection.

## Why It's Partial

| Aspect | Full Stage/Set System | RedLilium |
|--------|----------------------|-----------|
| Named stages | PreUpdate, Update, PostUpdate, etc. | No stages — single flat graph |
| System sets | Named groups with shared constraints | Individual systems only |
| Label-based ordering | `.before("physics")`, `.after("rendering")` | TypeId-based edges only |
| Run conditions | Systems run only when condition is true | All systems run every frame |
| Multiple schedules | Startup, Main, FixedUpdate schedules | Single SystemsContainer |
| Stage barriers | Synchronization points between stages | No barriers — pure dependency graph |
| Batch ordering | "All physics before all rendering" | Must add edges for each pair |

The dependency graph is **functionally correct** — it guarantees ordering and detects cycles. However, it lacks **organizational abstraction**:

- Adding 10 gameplay systems that must all run before transforms requires 10 edges, instead of placing them in a "Gameplay" set with a single set-level edge.
- No way to conditionally skip systems (e.g., pause physics when paused).
- No separation between "startup" systems (run once) and "update" systems (run every frame).

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | Full schedule system: `Schedule`, `SystemSet`, run conditions, `Startup`/`Update`/`FixedUpdate` schedules, `.before()`/`.after()` labels |
| **flecs** | Phases (pre-frame, on-update, post-frame), pipelines, system enable/disable |
| **Unity DOTS** | `ComponentSystemGroup` hierarchy, `UpdateBefore`/`UpdateAfter` attributes, `SystemGroup` ordering |
| **Legion** | Stages with manual barrier insertion, `Schedule` builder |
| **Shipyard** | Workloads (named groups of systems with ordering) |
| **EnTT** | No built-in scheduling (manual) |
| **hecs** | No built-in scheduling (manual) |
