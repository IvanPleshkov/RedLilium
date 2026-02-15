# RedLilium ECS — Design Document

## Why a Custom ECS?

Existing ECS solutions treat async compute as an afterthought — something bolted on through external task pools. In a real game engine, CPU cores sit idle while the slowest ECS system in a dependency stage finishes. Background work (navmesh rebuilds, pathfinding, LOD calculations, asset processing) has no way to fill those gaps.

RedLilium ECS is designed from the ground up with a **unified scheduling model**: ECS systems and background compute tasks share the same thread pool. When a core finishes its system and no other systems are ready, it automatically picks up async compute work. No idle cores.

```
Traditional ECS (wasted CPU):

Core 1: [physics ████████████████████]
Core 2: [AI ████] [idle ░░░░░░░░░░░░░]  ← wasted
Core 3: [anim ██] [idle ░░░░░░░░░░░░░░]  ← wasted
Core 4: [cull █]  [idle ░░░░░░░░░░░░░░░]  ← wasted

RedLilium ECS (full utilization):

Core 1: [physics ████████████████████]
Core 2: [AI ████] [navmesh ▒▒▒▒▒▒▒▒▒]  ← async compute fills gap
Core 3: [anim ██] [LOD ▒▒▒▒] [path ▒▒▒]  ← async compute fills gap
Core 4: [cull █]  [navmesh ▒▒▒] [LOD ▒▒]  ← async compute fills gap
```

## Goals

1. **Unified scheduling** — ECS systems and compute tasks share one work-stealing thread pool. Idle cores automatically pick up background work.

2. **Priority-based execution** — Critical systems (physics, rendering) always run first. Background tasks (navmesh, pathfinding) fill gaps without affecting frame time.

3. **Multiple worlds** — First-class support for multiple independent ECS worlds. Use cases: game world + editor world + preview world, server-side simulation, parallel scene loading.

4. **Simplicity over cleverness** — Sparse set storage, runtime borrow checking, no compile-time query magic. Easy to understand, debug, and extend.

5. **Cross-platform** — Works on native (multi-threaded) and web (single-threaded). Same API, different scheduling backends.

## Non-Goals

- Competing with archetype-based ECS storage performance for millions of entities
- Plugin ecosystem or scripting integration (can be added later)
- Editor/inspector reflection system (can be added later)

## Architecture Overview

### Sync Systems with Lock-Execute Pattern

The key architectural decision: **ECS systems are synchronous functions that access the World through a lock-execute pattern.** Component locks are confined to closures and automatically dropped when the closure returns, preventing deadlocks in multi-threaded execution.

- **Sync systems** access the World through `ctx.lock::<A>().execute(|items| {...})`. All systems complete within a single `runner.run()` call.
- **Compute tasks** receive owned data (copies/clones extracted from execute closures). They run on the shared pool and may span multiple frames. Systems can wait for results via `compute.block_on()` or fire-and-forget.

```
┌─────────────────────────────────────────────────────────┐
│ system with compute (completes within one frame)         │
│                                                          │
│  let data = ctx.lock::<(Read<NavMesh>,)>()               │
│      .execute(|(nav,)| nav.clone());  ← locks released   │
│                                                          │
│  let mut handle = ctx.compute().spawn(Priority::High,    │
│      |ctx| async { heavy_pathfinding(data) });           │
│  let result = ctx.compute().block_on(&mut handle);       │
│                      ← ticks pool until task completes   │
│                                                          │
│  if let Some(paths) = result {                           │
│      ctx.commands(move |world| { apply(world, paths); });│
│  }                                                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ fire-and-forget compute tasks (may span multiple frames) │
│                                                          │
│  let geometry = ctx.lock::<(Read<Geometry>,)>()          │
│      .execute(|(geo,)| extract_geometry(&geo));          │
│                                                          │
│  compute.spawn(Priority::Low, |ctx| async move {        │
│      rebuild_navmesh(geometry)  ← runs across frames    │
│  });                                                     │
│  // system returns, task continues in background         │
│  // next frame: try_recv() to check for results          │
└─────────────────────────────────────────────────────────┘
```

### Unified Thread Pool

One pool, two priority levels, work-stealing:

```
┌──────────────────────────────────────────────────┐
│            Shared Work-Stealing Pool              │
│                                                   │
│  ┌───────────────────┐  ┌─────────────────────┐  │
│  │ Systems           │  │ Compute Tasks       │  │
│  │ (high priority)   │  │ (fill gaps)         │  │
│  │                   │  │                     │  │
│  │ physics           │  │ navmesh             │  │
│  │ AI                │  │ pathfind            │  │
│  │ animation         │  │ LOD                 │  │
│  │ culling           │  │ terrain             │  │
│  └───────────────────┘  └─────────────────────┘  │
│                                                   │
│  Worker threads: run assigned systems,            │
│                  tick compute tasks when idle      │
└──────────────────────────────────────────────────┘
```

Each worker thread loop:
1. Run assigned system to completion
2. No systems ready? Tick compute tasks
3. Nothing at all? Park until woken

### Priority Levels

| Priority | Use Case | Behavior |
|----------|----------|----------|
| **Critical** | ECS systems, physics, render prep | Must complete this frame |
| **High** | AI decisions, animation blending | Should complete this frame |
| **Low** | Navmesh rebuild, LOD, asset processing | Fills gaps, may span multiple frames |

### Multiple Worlds

Each World is independent — its own entities, components, resources, and system schedule. Worlds share the thread pool but not data.

Use cases:
- **Game + Editor**: Separate simulation from editor state
- **Game + Preview**: Material/asset preview without affecting game
- **Server simulation**: Headless world running game logic
- **Parallel loading**: Load a new level in a separate world, swap when ready
- **Testing**: Isolated worlds for deterministic unit tests

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  World A    │  │  World B    │  │  World C    │
│  (Game)     │  │  (Editor)   │  │  (Loading)  │
│             │  │             │  │             │
│ entities    │  │ entities    │  │ entities    │
│ components  │  │ components  │  │ components  │
│ resources   │  │ resources   │  │ resources   │
│ schedule    │  │ schedule    │  │ schedule    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌───────────────────┐
              │   Shared Pool     │
              │   (N threads)     │
              └───────────────────┘
```

Worlds can communicate through channels or shared resources (Arc-wrapped, external to any world).

## Component Storage: Sparse Sets

We use sparse sets instead of archetypes. The tradeoffs:

| | Sparse Sets | Archetypes |
|---|---|---|
| **Iteration speed** | Good (dense array, but indirect) | Excellent (contiguous memory) |
| **Add/remove component** | O(1), no data movement | O(N), moves entity to new archetype |
| **Memory overhead** | Higher (sparse array per type) | Lower (packed tables) |
| **Implementation** | ~200 lines | ~1000+ lines |
| **Cache behavior** | Good for single-component, scattered for multi | Excellent for multi-component |

For our use case (thousands, not millions of entities), sparse sets are fast enough and dramatically simpler. If iteration performance becomes a bottleneck, we can add archetype storage later without changing the query API.

## Query System: Runtime Borrow Checking

Queries borrow component storages at runtime using `RefCell`-like tracking. This is simpler than compile-time checking and catches bugs immediately with clear error messages.

```rust
// Read-only access — multiple systems can read simultaneously
let positions = world.read::<Position>();
let velocities = world.read::<Velocity>();

// Mutable access — exclusive
let mut transforms = world.write::<Transform>();

// Panics at runtime if another system already has &mut Transform
```

## System Scheduling

Systems implement the `System` trait: a synchronous `run` method that receives a `SystemContext`. The scheduler resolves dependencies and runs non-conflicting systems in parallel:

```rust
struct PhysicsSystem;

impl System for PhysicsSystem {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Write<Transform>, Read<RigidBody>)>()
            .execute(|(mut transforms, bodies)| {
                // ... physics step
            });
    }
}

// Registration with ordering constraints:
let mut container = SystemsContainer::new();
container.add(PhysicsSystem);
container.add(AnimationSystem);
container.add_edge::<PhysicsSystem, AnimationSystem>().unwrap();

runner.run(&mut world, &container); // all systems complete within this call
```

## Compute Integration

Systems can spawn compute tasks and wait for results within the same frame using `block_on`:

```rust
impl System for PathfindSystem {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        // Phase 1: extract data (locks released when execute returns)
        let graph = ctx.lock::<(Read<NavMesh>,)>()
            .execute(|(nav,)| {
                nav.iter().next().map(|(_, n)| n.clone())
            });

        // Phase 2: heavy compute (no locks held)
        if let Some(graph) = graph {
            let mut handle = ctx.compute().spawn(Priority::High, |_ctx| async move {
                compute_paths(graph)
            });
            let paths = ctx.compute().block_on(&mut handle);

            // Phase 3: apply results via deferred command
            if let Some(paths) = paths {
                ctx.commands(move |world| {
                    // apply paths to agents
                });
            }
        }
    }
}
```

Fire-and-forget tasks for background work that spans multiple frames:

```rust
// Spawn from a system — task continues after the system returns
let handle = compute.spawn(Priority::Low, |ctx| async move {
    let mut mesh = NavMesh::new();
    for chunk in geometry.chunks(256) {
        mesh.process(chunk);
        ctx.yield_now().await;  // cooperative yielding
    }
    mesh
});

// Next frame: check for results
if let Some(mesh) = handle.try_recv() {
    // apply mesh
}
```

## Frame Flow

All systems complete within a single `schedule.run()` call. There is no cross-frame system state — if work is too heavy for one frame, spawn it as a compute task.

```
1. world.advance_tick();               ← start new frame

2. runner.run(&mut world, &systems);   ← systems execute by dependency order
   // Stage 1: [physics, AI, animation] ← parallel, non-conflicting
   //   idle threads tick compute pool
   // Stage 2: [transform_propagation]  ← depends on physics
   // Stage 3: [camera_update, culling] ← depends on transforms

3. world.apply_commands();             ← deferred spawn/despawn/insert

4. render(&world);                     ← render submission
```

## Platform Differences

| | Native | Web (WASM) |
|---|---|---|
| **Thread pool** | N worker threads | Single-threaded |
| **Systems** | Parallel via pool (per stage) | Sequential on main thread |
| **Async compute** | Multi-core, work-stealing | Cooperative on main thread |
| **IO** | tokio (separate thread) | wasm-bindgen-futures / fetch API |
| **API** | Same | Same |

On web, `pool.scope()` runs tasks sequentially, and async compute tasks tick cooperatively between frames. The API is identical — only the scheduling backend changes.

## Implementation Plan

### Phase 1: Foundation
- Entity storage (generational IDs)
- Component storage (sparse sets, type-erased)
- Basic queries (iteration, With/Without filters)
- Resources (typed singletons)
- World struct

### Phase 2: Scheduling
- Thread pool with sync scope + async executor
- `yield_now()` and priority levels
- System registration with access declarations
- Dependency resolution and parallel execution
- Channel-based result bridge

### Phase 3: Features
- Change detection (Changed<T>, Added<T>)
- Commands (deferred spawn/despawn/insert)
- Events (typed channels between systems)
- Parent-child hierarchy with cascading delete

### Phase 4: Polish
- Run conditions
- App states and transitions
- On-add / on-remove hooks
- Profiling integration (Tracy)
- Multiple world management
