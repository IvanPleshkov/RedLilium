# RedLilium ECS — Design Document

## Why a Custom ECS?

Existing ECS solutions treat async compute as an afterthought — something bolted on through external task pools. In a real game engine, CPU cores sit idle while the slowest ECS system in a dependency stage finishes. Background work (navmesh rebuilds, pathfinding, LOD calculations, asset processing) has no way to fill those gaps.

RedLilium ECS combines synchronous systems with cooperatively polled async compute. Native systems can run concurrently, and parallel entity queries use reusable workers owned by their world. A unified work-stealing scheduler is a longer-term goal; it is not the current execution model.

## Goals

1. **Unified scheduling (planned)** — Share execution capacity between ECS systems and compute tasks so idle cores can pick up background work.

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
- **Compute tasks** receive owned data (copies/clones extracted from execute closures). They are polled by the compute executor and may span multiple frames. Systems can wait for results via `compute.block_on()` or fire-and-forget.

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

### Execution and parallel queries

The current executors have separate responsibilities:

- `EcsRunnerMultiThread` starts scoped OS threads for ready systems, bounded by the runner's `num_threads`.
- `ComputePool` stores futures and polls them through `tick` / `block_on`; it does not own persistent worker threads.
- Each `World` owns a `ParallelExecutor` for synchronous `par_for_each` calls. Its capacity defaults to `available_parallelism`, including the calling thread. Native helpers start lazily and are reused across calls and systems in that world.

`QueryGuard::par_for_each`, `LockRequest::par_for_each`, and `par_for_each(...)` systems all use the world's executor. Low-level `ForEachAccess::run_par_for_each` / `run_par_for_each_with` take an explicit `&ParallelExecutor`. A standalone executor can be created with `ParallelExecutor::new(num_threads)`.

`ParConfig::num_threads` limits participants **per call**, including its caller, capped by the executor's capacity. `None` uses that capacity; zero means one. `min_batch_size` defaults to 64, and zero means one. Queries with fewer than 128 candidate entities, or insufficient batches for two participants, run sequentially. Other calls distribute disjoint batches through an atomic counter, with no scheduling lock per entity.

Only idle workers receive jobs. Busy workers are skipped and the caller always participates, so nested or concurrent calls can run with fewer participants. A callback must not wait for sibling callbacks to run concurrently. The executor does not poll unrelated compute tasks or steal systems while component locks are held. Nested queries still have to obey component/resource borrow rules.

All submitted borrowed jobs finish before the call returns, including when a callback panics. A panic is propagated on the caller after draining those jobs; workers remain available. Writes performed before a panic are not rolled back. Worker startup failure reduces parallelism instead of losing work.

A world creates at most `capacity - 1` helper threads. Calling system threads are additional participants: this is **not a shared CPU budget** across the runner, compute, and multiple worlds. Integrating those executors is separate scheduler work.

For hot reload, `World::purge_source` stops and joins all query workers before removing source registrations; dropping the world also joins them. This runs worker TLS destructors while game code is still mapped. Queries after a purge lazily start fresh workers. Lazy startup uses the spawn entry point captured when the executor was constructed, so a guest query cannot relocate the worker loop into its own image. An executor must be dropped before the image that constructed it is unloaded; standalone executors running guest callbacks must also be dropped before unloading that guest.

On WASM, the same query API runs sequentially and creates no workers.

### Deferred command failures

Commands from `SystemContext` are collected by the runner and applied before exclusive systems and at the end of a run. Each flush attempts every command in queue order. A panicking command produces a `CommandError` containing its message and enqueue location (`file`, `line`, `column`); later commands and systems continue. Changes made before a panic remain in the world. Command application is not a transaction, and subsequent code sees that partial state.

The runners report one `SystemError::DeferredCommandsFailed { errors }` per failed flush through `run` / `RunResult.errors`. `run_system_once` also collects all command errors, flushes observers, and returns this variant instead of the system result. Errors from system execution remain separate. Observer callbacks invoked later by `flush_observers` have their own execution path and are outside this command boundary.

`World::apply_commands()` applies the separate `CommandBuffer` resource and returns `Vec<CommandError>`. `CommandBuffer::apply` and `CommandCollector::apply` provide the same checked batch application for standalone queues. Commands queued during application wait for the next flush; they do not extend the current batch. `drain()` returns `DeferredCommand` values, each consumed with `apply(&mut world) -> Result<(), CommandError>`.

The panic boundary is inside the generic wrapper created when a command is queued, before erasing its closure type. This keeps guest-command panic capture inside the originating image, following the system panic boundary used for hot reload. Reports own their strings and can outlive that image; pending commands must still be applied or dropped before unloading their module. Queue-location tracking adds a captured location pointer; string copies and error-vector allocation occur only when a command fails.

### Priority Levels

| Priority | Use Case | Behavior |
|----------|----------|----------|
| **Critical** | ECS systems, physics, render prep | Must complete this frame |
| **High** | AI decisions, animation blending | Should complete this frame |
| **Low** | Navmesh rebuild, LOD, asset processing | Fills gaps, may span multiple frames |

### Multiple Worlds

Each World is independent — its own entities, components, resources, and system schedule. Each world owns its query executor; a runner can drive multiple worlds.

Use cases:
- **Game + Editor**: Separate simulation from editor state
- **Game + Preview**: Material/asset preview without affecting game
- **Server simulation**: Headless world running game logic
- **Parallel loading**: Load a new level in a separate world, swap when ready
- **Testing**: Isolated worlds for deterministic unit tests

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

### Low-level access safety

`AccessSet` and `AccessElement` are unsafe implementation contracts. Their
metadata must include every accessed storage, the correct read/write mode and
any main-thread requirement. `Or` and `Any` propagate these requirements from
their nested filters.

Direct `fetch` and `fetch_unlocked` calls are unsafe. Even the locking `fetch`
cannot protect an existing unlocked `World::get` reference obtained through
`&World`; filters and main-thread resources also do not acquire their own locks.
The caller must uphold the documented aliasing, lifetime and thread guarantees.
Use `ctx.query`, `ctx.lock().execute`, or `World::query` in application code.

The unsafe contracts add no locks, allocations or per-entity checks.
Custom access-trait implementations require `unsafe impl`.

### QueryGuard storage views

`QueryGuard::items()` returns a tuple of read-only views by value;
`items_mut()` returns views borrowed exclusively from the guard. Component
views preserve the existing `Ref`/`RefMut` methods, exclusion masks and write
ticks. Bind writable component views with `mut`:

```rust,ignore
let mut q = ctx.query::<(Write<Position>, Read<Velocity>)>();
let (mut positions, velocities) = q.items_mut();
```

Resource views are ordinary `&T` / `&mut T`, and filters are borrowed read-only.
Options keep their shape: taking an `OptionalWrite` view does not remove the
stored item from the query. The view's lifetime keeps the guard borrowed, so
neither `take`, `replace`, nor `swap` can leave access alive after its locks are
released. Creating these views adds no locks, allocations or Arc operations.
Custom fetched item types use `unsafe impl QueryBorrow` to expose their own
views; its contract forbids exposing the original, longer storage lifetime.

`ForEachAccess::run_*` helpers require mutable access to the fetched tuple and
callbacks whose references cannot escape. To collect component references
within a guard borrow, use `QueryGuard::iter_mut`.

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
| **Parallel queries** | Reusable world-owned workers + caller | Sequential |
| **Systems** | Scoped threads, bounded by the runner | Sequential on main thread |
| **Async compute** | Cooperatively polled futures | Cooperative on main thread |
| **IO** | tokio (separate thread) | wasm-bindgen-futures / fetch API |
| **API** | Same | Same |

On web, parallel queries run sequentially, and async compute tasks tick cooperatively. The query API is identical across platforms.

## Implementation Plan

### Entity Flags

Each entity carries a `u32` flags field in its 128-bit identifier. Flags control query visibility without requiring component insertion/removal:

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | `DISABLED` | Manually disabled by user/system |
| 1 | `INHERITED_DISABLED` | Disabled because a parent was disabled |
| 2 | `STATIC` | Manually marked static (rarely-changing) |
| 3 | `INHERITED_STATIC` | Static because a parent was marked static |

**Filtering semantics**:
- `Read<T>` / `Write<T>` — exclude entities with `DISABLED` or `STATIC` set
- `ReadAll<T>` — exclude `DISABLED` only, include static entities
- `World::get()` / `World::get_mut()` — no filtering (exclusive system access)
- `_unfiltered` methods on `Ref`/`RefMut` — bypass all flag checks

Both disabled and static flags propagate through the parent-child hierarchy. The `INHERITED_*` variants distinguish manual vs propagated state so that `enable`/`unmark_static` preserve manually-flagged children.

### Phase 1: Foundation
- Entity storage (128-bit IDs with spawn_tick and flag bits)
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
