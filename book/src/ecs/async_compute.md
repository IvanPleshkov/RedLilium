# Async Compute

A key design goal of RedLilium's ECS is filling idle CPU cores with background work. The async compute pool shares a thread pool with the ECS runner -- when sync systems leave cores idle, async tasks are automatically picked up via work-stealing.

## Spawning Tasks

From within a system, use `ctx.compute()`:

```rust
impl System for PathfindingSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        let mut handle = ctx.compute().spawn(Priority::High, |ctx| async move {
            // Heavy computation runs on the thread pool
            let path = compute_path(start, goal);
            path
        });

        // Block until the result is ready
        if let Some(path) = ctx.compute().block_on(&mut handle) {
            // Use the path
        }

        Ok(())
    }
}
```

## Priority Levels

```rust
pub enum Priority {
    Critical,  // ECS systems, physics -- highest priority
    High,      // AI, animation, pathfinding
    Low,       // Navmesh rebuilds, LOD computation, background tasks
}
```

Higher-priority tasks are polled first. ECS sync systems run at `Critical` priority, so they always take precedence.

## Fire-and-Forget Tasks

For background work where you don't need the result immediately:

```rust
ctx.compute().spawn(Priority::Low, |ctx| async move {
    // Rebuild navmesh in the background
    rebuild_navmesh(&mesh_data);
});
// Handle is dropped -- task continues running
```

## TaskHandle API

```rust
let mut handle: TaskHandle<PathResult> = ctx.compute().spawn(Priority::High, |ctx| async {
    compute_path(start, goal)
});

// Block until done (drives the pool internally)
let result: Option<PathResult> = ctx.compute().block_on(&mut handle);

// Non-blocking check
if handle.is_done() {
    let result = handle.try_recv();
}

// Timeout
let result = handle.recv_timeout(Duration::from_millis(16));

// Cancel
handle.cancel();
assert!(handle.is_cancelled());
```

## Cooperative Yielding

Async tasks should yield periodically to let higher-priority work execute:

```rust
ctx.compute().spawn(Priority::Low, |ctx| async move {
    for chunk in data.chunks(1000) {
        process_chunk(chunk);
        ctx.yield_now().await;  // give other tasks a chance to run
    }
});
```

## Using the Pool Directly

Outside of systems, create and drive a `ComputePool` manually:

```rust
let pool = ComputePool::new(io_runtime);

let mut handle = pool.spawn(Priority::High, |ctx| async { 42 });

// Drive the pool
pool.tick();            // poll one highest-priority task
pool.tick_all();        // poll all tasks once
pool.tick_with_budget(Duration::from_millis(8)); // poll until budget exceeded

let result = pool.block_on(&mut handle);
println!("pending: {}", pool.pending_count());
```

## Cross-Platform Behavior

| Platform | Behavior |
|----------|----------|
| Native | Multi-core work-stealing across all pool threads |
| WASM | Single-threaded cooperative scheduling on main thread |

No API changes needed -- the same code works on both platforms.
