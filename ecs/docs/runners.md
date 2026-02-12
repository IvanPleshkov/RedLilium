# System Runners

Runners are the execution engines that drive ECS systems each frame. The `EcsRunner` enum dispatches to either a single-threaded or multi-threaded implementation.

## EcsRunner Enum

```rust
use redlilium_ecs::EcsRunner;

// Single-threaded (works everywhere, including WASM)
let runner = EcsRunner::single_thread();

// Multi-threaded with specific thread count (native only)
let runner = EcsRunner::multi_thread(4);

// Multi-threaded using available CPU cores (native only)
let runner = EcsRunner::multi_thread_default();
```

## Running Systems

```rust
let mut world = World::new();
// ... register components, spawn entities ...

let mut container = SystemsContainer::new();
container.add(MovementSystem);
container.add(PhysicsSystem);
container.add_edge::<MovementSystem, PhysicsSystem>().unwrap();

let runner = EcsRunner::single_thread();

// Run all systems (one frame)
runner.run(&mut world, &container);
```

Each call to `run()`:
1. Creates a `SystemContext` from the world, compute pool, and IO runtime.
2. Executes all systems respecting dependency ordering.
3. Applies deferred commands to `&mut World`.
4. Drains remaining compute tasks.

## Single-Threaded Runner

**`EcsRunnerSingleThread`** runs systems sequentially in pre-computed topological order.

```rust
let runner = EcsRunnerSingleThread::new();
runner.run(&mut world, &container);
```

Execution model:
```
System A → poll to completion
System B → poll to completion
System C → poll to completion
Apply commands
Drain compute tasks
```

Between each system poll, the compute pool is ticked so spawned tasks make progress.

Properties:
- Zero locking overhead (no contention on a single thread)
- Deterministic execution order
- Works on all platforms including WASM
- Simple to reason about

## Multi-Threaded Runner

**`EcsRunnerMultiThread`** runs independent systems in parallel using `std::thread::scope`.

```rust
let runner = EcsRunnerMultiThread::new(4); // 4 worker threads
// or
let runner = EcsRunnerMultiThread::with_default_threads(); // auto-detect
runner.run(&mut world, &container);
```

Execution model:
```
Thread 1: System A (no deps) ──────────→ done → Signal
Thread 2: System B (no deps) ──→ done → Signal
Main:     Tick compute pool, coordinate
Thread 1: System C (depends on A,B) ──→ done → Signal
Apply commands
Drain compute (parallel across threads)
```

How it works:
1. Initialize atomic dependency counters from `in_degrees()`.
2. Start all systems with zero dependencies on worker threads.
3. Main thread coordination loop:
   - Wait for system completion signals (1ms timeout).
   - Decrement dependents' counters.
   - Start newly-ready systems on available threads.
   - Tick the compute pool during idle time.
4. After all systems complete, apply commands sequentially.
5. Drain remaining compute tasks across all threads using `tick_extract()`.

Thread safety is ensured by:
- Per-component `RwLock` synchronization.
- TypeId-sorted lock acquisition in `execute()` (prevents deadlocks).
- Atomic dependency counters for progress tracking.

## Graceful Shutdown

Both runners support graceful shutdown with a time budget:

```rust
use std::time::Duration;

match runner.graceful_shutdown(Duration::from_secs(5)) {
    Ok(()) => println!("All compute tasks completed"),
    Err(ShutdownError::Timeout { remaining_tasks }) => {
        println!("{} tasks still running after timeout", remaining_tasks);
    }
}
```

This continuously ticks the compute pool until all pending tasks complete or the budget expires.

## Accessing Runner Resources

```rust
// Get the compute pool (for spawning tasks outside systems)
let compute = runner.compute();
assert_eq!(compute.pending_count(), 0);

// Get the IO runtime
let io = runner.io();
```

## Typical Game Loop

```rust
let runner = EcsRunner::multi_thread_default();
let mut world = World::new();
register_std_components(&mut world);

let mut systems = SystemsContainer::new();
systems.add(InputSystem);
systems.add(MovementSystem);
systems.add(UpdateGlobalTransforms);
systems.add(UpdateCameraMatrices);
systems.add_edges(&[
    Edge::new::<InputSystem, MovementSystem>(),
    Edge::new::<MovementSystem, UpdateGlobalTransforms>(),
    Edge::new::<UpdateGlobalTransforms, UpdateCameraMatrices>(),
]).unwrap();

loop {
    world.advance_tick();
    runner.run(&mut world, &systems);

    // Render, handle window events, etc.
    if should_exit { break; }
}

runner.graceful_shutdown(Duration::from_secs(5)).ok();
```

## Public API

| Method | Description |
|--------|-------------|
| `EcsRunner::single_thread()` | Create single-threaded runner |
| `EcsRunner::multi_thread(n)` | Create multi-threaded with n threads |
| `EcsRunner::multi_thread_default()` | Create with auto-detected thread count |
| `runner.run(&mut world, &systems)` | Execute all systems for one frame |
| `runner.compute()` | Access the compute pool |
| `runner.io()` | Access the IO runtime |
| `runner.graceful_shutdown(budget)` | Drain compute tasks within time budget |
