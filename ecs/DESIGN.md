# RedLilium ECS — Design Document

## Why a Custom ECS?

Existing ECS solutions treat async compute as an afterthought — something bolted on through external task pools. In a real game engine, CPU cores sit idle while the slowest ECS system in a dependency stage finishes. Background work (navmesh rebuilds, pathfinding, LOD calculations, asset processing) has no way to fill those gaps.

RedLilium ECS is designed from the ground up with a **unified scheduling model**: synchronous ECS systems and asynchronous compute tasks share the same thread pool. When a core finishes its ECS system and no other systems are ready, it automatically picks up async compute work. No idle cores.

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

1. **Async-native scheduling** — ECS systems (sync) and compute tasks (async) share one work-stealing thread pool. Idle cores automatically pick up background work.

2. **Priority-based execution** — Critical systems (physics, rendering) always run first. Background tasks (navmesh, pathfinding) fill gaps without affecting frame time.

3. **Multiple worlds** — First-class support for multiple independent ECS worlds. Use cases: game world + editor world + preview world, server-side simulation, parallel scene loading.

4. **Simplicity over cleverness** — Sparse set storage, runtime borrow checking, no compile-time query magic. Easy to understand, debug, and extend.

5. **Cross-platform** — Works on native (multi-threaded) and web (single-threaded). Same API, different scheduling backends.

## Non-Goals

- Competing with archetype-based ECS storage performance for millions of entities
- Plugin ecosystem or scripting integration (can be added later)
- Editor/inspector reflection system (can be added later)

## Architecture Overview

### Sync Systems + Async Compute Separation

The key architectural decision: **ECS systems are synchronous, compute tasks are asynchronous.** This eliminates the hard problem of borrowing World data across await points.

- **Sync systems** borrow the World normally (Rust's standard borrow rules). They run to completion within a frame.
- **Async compute tasks** receive copies/clones of data extracted from the World. They can yield at explicit `.await` points, spreading work across multiple frames. Results are sent back via channels.

```
World (owned by sync systems)          Async Compute Tasks
  │                                         ▲
  │ extract data (copy)                     │ results (channel)
  ▼                                         │
┌──────────────┐    spawn     ┌─────────────────────────┐
│ sync system  │ ───────────► │ async task              │
│              │   data copy  │                         │
│ borrows      │              │ process chunk           │
│ &World /     │              │ yield_now().await        │
│ &mut World   │              │ process chunk           │
│              │              │ yield_now().await        │
│ runs to      │              │ ...                     │
│ completion   │              │ done → send(result)     │
└──────────────┘              └─────────────────────────┘
      │
      ▼
┌──────────────┐
│ sync system  │ ◄── channel.try_recv()
│ applies      │
│ results      │
└──────────────┘
```

### Unified Thread Pool

One pool, two priority levels, work-stealing:

```
┌──────────────────────────────────────────────────┐
│            Shared Work-Stealing Pool              │
│                                                   │
│  ┌───────────────────┐  ┌─────────────────────┐  │
│  │ Sync System Tasks │  │ Async Compute Tasks │  │
│  │ (high priority)   │  │ (fill gaps)         │  │
│  │                   │  │                     │  │
│  │ physics           │  │ navmesh.await       │  │
│  │ AI                │  │ pathfind.await      │  │
│  │ animation         │  │ LOD.await           │  │
│  │ culling           │  │ terrain.await       │  │
│  └───────────────────┘  └─────────────────────┘  │
│                                                   │
│  Worker threads: steal sync first,                │
│                  tick async when idle              │
└──────────────────────────────────────────────────┘
```

Each worker thread loop:
1. Try to steal a sync task → run it
2. No sync work? Tick an async compute future
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

Systems declare their component access. The scheduler resolves dependencies and runs non-conflicting systems in parallel:

```rust
scheduler.add_system("physics")
    .writes::<Transform>()
    .reads::<RigidBody>()
    .reads::<Collider>()
    .run(physics_system);

scheduler.add_system("animation")
    .writes::<Transform>()
    .reads::<AnimationState>()
    .after("physics")
    .run(animation_system);

// physics and animation both write Transform,
// so animation runs after physics (explicit ordering + conflict detection)
```

## Async Compute Integration

Async tasks are spawned from sync systems and live on the shared thread pool:

```rust
fn request_navmesh_rebuild(world: &World, pool: &ComputePool) {
    let geometry = extract_geometry(world);  // copy data out

    pool.spawn(Priority::Low, async move {
        let mut mesh = NavMesh::new();
        for chunk in geometry.chunks(256) {
            mesh.process(chunk);
            yield_now().await;  // can be paused here
        }
        mesh  // result available via Task handle
    });
}
```

## Frame Flow

```
1. pool.scope(|s| {                     ← parallel sync systems
       s.spawn(|| physics(world));
       s.spawn(|| ai(world));
       s.spawn(|| animation(world));
       // idle threads tick async tasks
   });

2. drain_results(&mut world);           ← apply completed compute results

3. pool.scope(|s| {                     ← dependent sync systems
       s.spawn(|| transform_propagation(world));
       s.spawn(|| frustum_culling(world));
   });

4. render(&world);                      ← render submission
```

## Platform Differences

| | Native | Web (WASM) |
|---|---|---|
| **Thread pool** | N worker threads | Single-threaded |
| **Sync systems** | Parallel via pool | Sequential on main thread |
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
