# System Sets and Stages (Partial)

## What Are System Sets and Stages?

Modern ECS frameworks organize system execution into **stages** (fixed execution phases like PreUpdate, Update, PostUpdate) and **system sets** (named groups of systems that share ordering constraints or run conditions). This allows complex scheduling without specifying pairwise edges for every system.

## System Sets (Implemented)

RedLilium supports **system sets** — named groups that allow batch ordering and shared run conditions.

### Defining a Set

```ignore
struct PhysicsSet;
impl SystemSet for PhysicsSet {}

struct RenderSet;
impl SystemSet for RenderSet {}
```

### Adding Systems to Sets

```ignore
let mut systems = SystemsContainer::new();

systems.add(MovementSystem);
systems.add(CollisionSystem);
systems.add(UpdateGlobalTransforms);

// Group systems into sets
systems.add_to_set::<MovementSystem, PhysicsSet>().unwrap();
systems.add_to_set::<CollisionSystem, PhysicsSet>().unwrap();
systems.add_to_set::<UpdateGlobalTransforms, RenderSet>().unwrap();

// One edge orders all physics before all rendering
systems.add_set_edge::<PhysicsSet, RenderSet>().unwrap();
```

### Set-Level Run Conditions

```ignore
systems.add_condition(GameRunningCondition);
systems.add_set_condition::<PhysicsSet, GameRunningCondition>().unwrap();
// All physics systems are skipped when the condition is false.
```

### How It Works

Each set creates two invisible "barrier" nodes in the dependency graph:
- **SetEnter** — runs before all member systems
- **SetExit** — runs after all member systems

`add_to_set::<Sys, MySet>()` creates edges: `SetEnter → Sys → SetExit`

`add_set_edge::<SetA, SetB>()` creates edge: `SetA_Exit → SetB_Enter`

This gives O(N+M) edges instead of O(N×M) and integrates with the existing topological sort and cycle detection.

## What's Still Partial

| Aspect | Full Stage/Set System | RedLilium |
|--------|----------------------|-----------|
| Named stages | PreUpdate, Update, PostUpdate, etc. | No stages — single flat graph |
| System sets | Named groups with shared constraints | **Implemented** via `SystemSet` trait |
| Label-based ordering | `.before("physics")`, `.after("rendering")` | TypeId-based edges and set edges |
| Run conditions | Per-system and per-set conditions | **Implemented** via `Condition<T>` and `add_set_condition` |
| Multiple schedules | Startup, Main, FixedUpdate schedules | Single SystemsContainer |
| Stage barriers | Synchronization points between stages | No barriers — pure dependency graph |
| Batch ordering | "All physics before all rendering" | **Implemented** via `add_set_edge` |

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
