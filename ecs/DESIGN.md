# RedLilium ECS — Design Document

## Why a Custom ECS?

Existing ECS solutions treat async compute as an afterthought — something bolted on through external task pools. In a real game engine, CPU cores sit idle while the slowest ECS system in a dependency stage finishes. Background work (navmesh rebuilds, pathfinding, LOD calculations, asset processing) has no way to fill those gaps.

RedLilium ECS combines synchronous systems with wake-driven async compute. Native systems, parallel entity queries, and background compute share reusable workers through an explicitly owned executor. Background tasks continue between frames; no process-global scheduler or separate compute thread pool is required.

## Goals

1. **Unified scheduling** — Share execution capacity between ECS systems and compute tasks so idle cores can pick up background work.

2. **Priority-based execution** — Ready systems take precedence over queued compute. Background tasks fill idle capacity and yield cooperatively; a long non-yielding poll can delay a system and affect frame time.

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

Native systems and parallel queries share a `ParallelExecutor`. It owns a fixed maximum number of persistent workers, started lazily; clones share the same pool. There is no process-global executor. `ParallelExecutor::new(worker_threads)` counts background CPU workers, **excluding** coordinating/calling threads and IO threads; zero means one. The default reserves one available CPU for the caller, with at least one worker.

```rust
let executor = ParallelExecutor::new(4);
let runner = EcsRunner::multi_thread_with_executor(executor.clone());
let mut world = World::with_parallel_executor(executor.clone());
```

`EcsRunner::multi_thread(n)` creates its own pool with at most `n` workers. When run, a runner attaches its executor to the world, so all query entry points use the same workers, including queries in exclusive systems. Attaching a different executor joins the previous pool first and fails if that pool has active scopes. `World::new()` remains a standalone world with a lazy default executor. Sharing handles explicitly avoids pool switches when several runners drive the same worlds.

The coordinator maintains a ready queue and submits systems only to available workers. Completion releases the worker before notifying the coordinator and unlocks DAG dependents. A ready exclusive system stops new admissions; already running systems finish before the exclusive system receives `&mut World`. Conditions, virtual nodes, result reuse, change ticks, command flushes, and diagnostics retain their roles. Workers and coordinator cannot recursively invoke a runner on the same executor: that reports `SystemError::ExecutorUnavailable` instead of waiting for themselves. Systems must express scheduling dependencies through the DAG, not synchronously wait for another queued system to start.

`QueryGuard::par_for_each`, `LockRequest::par_for_each`, and parallel function systems use the world's executor. Low-level `ForEachAccess::run_par_for_each*` take an explicit `&ParallelExecutor`. Queries claim disjoint batches through an atomic counter, with no scheduling lock per entity. They submit helpers only to idle workers and always execute work on the caller; saturated nested queries therefore finish inline. Waiting queries never steal unrelated systems or compute jobs while holding component locks. Query callbacks must not wait for sibling callbacks to run concurrently.

`ParConfig::num_threads` limits participants per query call, including the caller. An external caller can use up to `worker_threads + 1` participants; a caller already running on a worker can use only the remaining idle workers as helpers. `None` uses executor capacity; zero means one. `min_batch_size` defaults to 64 and normalizes zero to one. Fewer than 128 candidate entities, or insufficient batches for two participants, use sequential iteration. All submitted work drains before return or panic propagation.

The coordinating thread services main-thread resource requests throughout a parallel phase. On coordinator unwind, its request receiver is dropped before the borrowed task scope drains, releasing waiting workers' result channels. System-job completion is published even if system setup or result reuse panics. Persistent threads do not imply persistent borrows: every phase drains before world mutation resumes. Per-system dispatch allocates one job; query helpers borrow one shared batch closure without allocating a job per entity or per helper.

Both runners attach their compute queue to the same executor used by their systems and queries. Native workers poll ready compute tasks whenever no system is waiting for worker capacity, and continue doing so between `runner.run()` calls. Sources sharing the pool get round-robin service; each compute queue preserves priority ordering with fairness rounds. Even `Priority::Critical` compute yields admission to a ready system. Already-running polls are not preempted: expensive tasks must use checkpoints.

The coordinator sleeps on system-completion, main-thread-request, and worker-availability events; it does not periodically poll compute or delay dispatch until a timer expires. A single-thread runner still executes systems on its caller, while its parallel queries and compute share background workers. This pool bounds background workers across sharing worlds and runners; explicit callers of `block_on` or query iteration are additional participants.

Before hot reload, stop guest producers and release guest worlds and externally retained results, then call `runner.prepare_reload(timeout)`. It drops cached system results, drains compute and IO, and joins CPU/Tokio workers (including TLS destruction) before guest code is unmapped. The timeout is shared across the drain stages; CPU polls and destructors remain cooperative. Work spawned by task or worker TLS destructors is checked again before success. Any error forbids unload. Standalone/shared pools outside that runner must also be stopped with `shutdown_workers()`; all runners retaining guest results need preparation. Shutdown rejects active scopes and live registered compute tasks with `ExecutorBusy`, and admissions are rejected while workers are joining. Later execution can restart workers lazily. The editor reload path aborts the swap if worker shutdown fails.

Dropping a world stops workers only when it owns the last executor handle; a pool shared with a runner survives that world. `World::purge_source` joins the world's executor before removing registrations and rejects an active shared executor. The constructor's image owns the stored worker-start function pointer and must outlive the pool. Pending guest code and TLS must be drained before unloading that guest.

On WASM, queries run sequentially and create no workers; the single-thread runner drives ready compute work cooperatively between systems and at frame end. IO callbacks mark tasks ready for subsequent ticks. Blocking waits cannot drive the browser event loop; await handles from async code when progress depends on IO.

### Wake-driven compute

`ComputePool::new(io)` is a standalone manually driven queue. `ComputePool::with_executor(io, executor)` registers it for autonomous native execution; runner constructors use this form. `tick`, `tick_all`, and `tick_with_budget` execute only ready tasks, and return zero when all tasks are asleep. A task is initially ready, then after `Pending` waits until its waker fires. Repeated wakes coalesce; a wake during polling schedules a subsequent poll without permitting concurrent polling of the same future.

`TaskHandle` and `IoHandle` are notifying futures. Result publication and sender destruction wake their awaiters. `IoHandle::channel()` returns an `IoSender` and receiver together so a sender cannot forget the notification. Cancellation through any `CancellationToken` clone wakes the task for its final cleanup poll. Cancellation is a request: awaiting the handle waits for a result or sender destruction, and the final poll may still produce a result. `ComputeMutex` and `ComputeRwLock` register waiters and wake them on unlock; dropped waits unregister themselves.

`block_on` helps ready compute work on its caller and parks when there is nothing ready, waking on new work or the awaited result. Release all ECS borrows before calling it; this method can execute other compute tasks. It never steals unrelated systems or query jobs. One-worker configurations therefore make progress even when a system synchronously waits for compute using IO.

The ready queue stores task IDs and reuses each task's waker state; worker polling does not allocate a fresh job for every poll. Task creation allocates the boxed future and shared notification/cancellation state. No entity-query helper allocations are added by compute integration. Idle workers sleep on a condition variable, rather than polling waiting IO.

Create shared compute pools in the host image. Waker constructors are captured there, so a late external wake after guest task destruction only sees a completed task; it cannot call a guest vtable. The constructor's image must outlive retained wakers, even after pool destruction. Pools unregister their background source when dropped, discard queued futures, and let in-flight polls finish cleanup. Before guest unload, quiesce explicitly and join workers instead of relying on dropping one shared owner. Runner reload preparation also drains its IO domain and joins its workers; dropping an IO result handle does not cancel the IO operation.

### Compute task lifetime

`ComputePool::active_count()` includes tasks being constructed, queued, polled by another thread, or destroyed. Admission acquires a lifetime counter before calling the task factory, and a task releases it only after its future and metadata have been destroyed. `pending_count()` remains the queue length; an empty queue does not establish quiescence. Runner `graceful_shutdown` uses the full active count and drains work without implicitly requesting cancellation.

`quiesce(timeout)` returns `Result<(), QuiesceTimeout>`, with the remaining active count on timeout. Callers must stop external producers before waiting and keep them stopped through reload. Tasks spawned by existing tasks, including from their destructors, participate in the same accounting. A timeout leaves the pool usable but forbids unloading the guest. The time budget is cooperative: a blocking poll or destructor cannot be preempted.

Polling and future destruction have generic panic boundaries inside the originating image. Panic payloads are converted into owned strings before returning to the host, and failures remain available through `TaskHandle::panic_message()`. `is_done()` becomes true after future destruction, including cancellation; receiving a result may happen earlier. Quiescence does not destroy results retained by handles or guest thread-local state: those owners must also be released, and guest execution threads joined, before module unload.

Lifetime tracking shares the pool progress signal and updates its active counter at task admission and final destruction. The future remains a single boxed allocation. Shutdown uses the same progress notifications as blocking waits, and does not poll sleeping tasks repeatedly.

### IO task lifetime and reload

An `IoRuntime` and all its clones form one lifetime domain. Each runner owns its own domain; independent host IO that must survive guest reload uses a separate runtime. Native workers start lazily and drive tasks between frames. No per-task module IDs or global task registry are required.

`IoRuntime::active_count()` covers submission, polling and destruction of the future and its result sender. Dropping an `IoHandle` does not cancel or remove the operation from this accounting. `quiesce(timeout)` waits for these lifetimes without stopping workers; `graceful_shutdown` on a runner drains both compute and IO, including tasks spawned from one domain into the other. Neither operation implicitly cancels IO. Stop external producers before draining and keep them stopped through reload.

`IoRuntime::shutdown(timeout)` first drains tasks, then joins the native runtime, including its blocking operations and thread-local destructors. A task timeout leaves the runtime running. A worker timeout leaves it shutting down on a host helper thread; new submissions then return disconnected handles until a successful shutdown retry. The old module must remain mapped throughout this interval. After success, later submissions start fresh workers. Dropping the last runtime clone is not an unload barrier.

Create runtimes in the host image. Their submission function is captured there and schedules an erased task, keeping Tokio task/waker vtables in the host even for futures submitted by a guest. Late retained wakers can be invoked and dropped after guest unload. The generic poll/destruction boundary enters the host runtime handle inside the guest image, so its own Tokio context TLS can resolve timers and IO. Polling and future destruction have separate panic boundaries inside the originating image; panics are logged and the result channel disconnects if no result was published. A result may arrive before the future is destroyed; only quiescence establishes the lifetime barrier. Retained results with guest drop glue must be released separately before unloading.

Guest async work must enter through `IoRunner::run`; spawning raw Tokio tasks bypasses the tracked lifetime and panic boundaries. Shutdown also waits for blocking work on this runtime, but it cannot account for work moved to other runtimes or external threads. On WASM, synchronous IO quiescence only checks the active count: callers must yield to the browser event loop before retrying if operations remain.

The native cross-image regression probe runs with `python3 scripts/check-io-reload.py` (macOS/Linux, no GPU). It builds host and guest from identical Cargo artifacts and exercises timers, poll/destructor panics, TLS teardown and late wakers over three unload/reload cycles.

### Deferred command failures

Commands from `SystemContext` are collected by the runner and applied before exclusive systems and at the end of a run. Each flush attempts every command in queue order. A panicking command produces a `CommandError` containing its message and enqueue location (`file`, `line`, `column`); later commands and systems continue. Changes made before a panic remain in the world. Command application is not a transaction, and subsequent code sees that partial state.

The runners report one `SystemError::DeferredEffectsFailed { commands, observers }` per failed flush through `run` / `RunResult.errors`. `run_system_once` also collects all command errors, flushes observers, and returns this variant instead of the system result. Errors from system execution remain separate. At the end of the run, both command and observer errors are retained in the report; pre-exclusive command flushes have an empty observer list.

`World::apply_commands()` applies the separate `CommandBuffer` resource and returns `Vec<CommandError>`. `CommandBuffer::apply` and `CommandCollector::apply` provide the same checked batch application for standalone queues. Commands queued during application wait for the next flush; they do not extend the current batch. `drain()` returns `DeferredCommand` values, each consumed with `apply(&mut world) -> Result<(), CommandError>`.

The panic boundary is inside the generic wrapper created when a command is queued, before erasing its closure type. This keeps guest-command panic capture inside the originating image, following the system panic boundary used for hot reload. Reports own their strings and can outlive that image; pending commands must still be applied or dropped before unloading their module. Queue-location tracking adds a captured location pointer; string copies and error-vector allocation occur only when a command fails.

### Deferred observer failures

Each deferred observer invocation has an in-image panic boundary installed by `observe_add` / `observe_insert` / `observe_remove`. `ObserverError::Panicked` identifies the registration source, trigger type, entity, panic message, and registration location. Its strings are owned, so reports may outlive a game module. Other handlers and triggers continue after a callback failure, and partial mutations remain.

During a flush, a guard owns the detached handler map and restores it on every exit, including unwinding. New registrations join the original handlers at the end of each wave, preserving registration order; they start receiving events in the next wave. A nested flush leaves pending work to the active outer flush, which owns the cascade budget and reports its errors.

One flush processes at most 100 waves. If work remains afterward, it returns `ObserverError::CascadeLimitExceeded { iterations, discarded_triggers }` and discards the remaining queued triggers. Completing exactly on wave 100 succeeds. Registrations survive the limit, and later newly generated triggers work normally; a stopped cascade is not replayed automatically next frame.

Runners and both `run_system_once` helpers report observer failures through `SystemError::DeferredEffectsFailed`. `run_system_once` retains command errors and observer errors together. Immediate component lifecycle hooks (`on_add`, `on_insert`, `on_replace`, `on_remove`) remain a separate mechanism from deferred observers.

Callbacks execute under their registration source, so observers/resources registered from a guest callback are attributed to that guest even if it subsequently panics. The previous source is restored after the call. `purge_source` is rejected before mutation while a flush is active; source unloading must wait until callbacks finish. Outside a flush, purging removes the source's handlers and pending triggers whose keys have no surviving handlers, preventing delivery of old triggers to replacement registrations.

The hot path keeps one boxed callback and adds a registration-location pointer plus a panic boundary per invocation. Restoring the handler map uses safe disjoint borrows, without cloning all callbacks or locking each entity. Error text and report storage are allocated only on failure.

### Priority Levels

| Priority | Use Case | Behavior |
|----------|----------|----------|
| **Critical** | ECS systems, physics, render prep | Must complete this frame |
| **High** | AI decisions, animation blending | Should complete this frame |
| **Low** | Navmesh rebuild, LOD, asset processing | Fills gaps, may span multiple frames |

### Multiple Worlds

Each World is independent — its own entities, components, resources, and system schedule. Worlds and runners can share an explicitly owned worker pool.

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
   //   shared workers execute ready compute; block_on callers can help
   // Stage 2: [transform_propagation]  ← depends on physics
   // Stage 3: [camera_update, culling] ← depends on transforms

3. world.apply_commands();             ← deferred spawn/despawn/insert

4. render(&world);                     ← render submission
```

## Platform Differences

| | Native | Web (WASM) |
|---|---|---|
| **Parallel queries** | Shared workers + caller | Sequential |
| **Systems** | Persistent shared workers with a fixed limit | Sequential on main thread |
| **Async compute** | Wake-driven shared workers, continuing between frames | Wake-driven cooperative ticks |
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
