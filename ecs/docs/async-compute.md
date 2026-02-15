# Async Compute

The async compute system is a core innovation of this ECS. It lets systems offload heavy computation to background tasks that fill idle CPU cores during ECS execution, without blocking system progress.

## Architecture

- **ComputePool**: A manually-polled task executor. Tasks are stored internally and polled via `tick()` / `tick_all()`.
- **TaskHandle<T>**: A handle returned when spawning a task. Implements `Future` for `.await` integration.
- **Priority**: Tasks have priority levels (Critical, High, Low) that determine polling order.
- **EcsComputeContext**: Passed to each task, providing cooperative yielding and IO access.

## Spawning Tasks

From within a system:

```rust
struct HeavySystem;

impl System for HeavySystem {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        // Spawn a background compute task
        let mut handle = ctx.compute().spawn(Priority::Low, |_cctx| async {
            // This runs during idle time on the compute pool
            expensive_calculation()
        });

        // Wait for the result (ticks compute pool until task completes)
        let result = ctx.compute().block_on(&mut handle);

        if let Some(value) = result {
            ctx.commands(move |world| {
                world.insert_resource(value);
            });
        }
    }
}
```

## Priority Levels

| Priority | Use Case |
|----------|----------|
| `Priority::Critical` | Must complete this frame (physics results, input processing) |
| `Priority::High` | Should complete soon (AI decisions, pathfinding) |
| `Priority::Low` | Background work (LOD generation, asset loading prep) |

Higher priority tasks are polled first. Within the same priority, earlier-spawned tasks are preferred (FIFO).

## TaskHandle API

```rust
let handle = pool.spawn(Priority::Low, |_ctx| async { 42u32 });

// Non-blocking check
if handle.is_done() {
    let value = handle.try_recv(); // Some(42) or None
}

// Blocking wait (avoid in frame loops)
let value = handle.recv(); // blocks until done

// Wait with timeout
let value = handle.recv_timeout(Duration::from_millis(100));

// Cancel a running task
handle.cancel();
assert!(handle.is_cancelled());
```

### TaskHandle as Future

`TaskHandle<T>` implements `Future<Output = Option<T>>`, so you can `.await` it:

```rust
let mut handle = pool.spawn(Priority::Low, |_ctx| async { 42u32 });

// Use &mut handle to await without consuming it
let result: Option<u32> = (&mut handle).await;
assert_eq!(result, Some(42));
```

Returns `None` if the task was cancelled or the sender was dropped.

## Cooperative Yielding

Long-running tasks should yield periodically to prevent starving higher-priority work:

```rust
use redlilium_ecs::{yield_now, set_yield_interval};

// Set yield interval globally (default is 1ms)
set_yield_interval(Duration::from_micros(500));

pool.spawn(Priority::Low, |cctx| async move {
    for chunk in large_dataset.chunks(1000) {
        process_chunk(chunk);
        // Yield control to the scheduler
        cctx.yield_now().await;
    }
});
```

The `yield_now()` function uses a timer-based approach: it only actually suspends if the configured interval has elapsed since the last yield. This avoids excessive context switches for tasks that yield frequently.

## Driving the Pool

The compute pool doesn't drive itself — it must be ticked by the runner:

```rust
let io = IoRuntime::new();
let pool = ComputePool::new(io);

let handle = pool.spawn(Priority::Low, |_ctx| async { 42u32 });

// Manual ticking (the runner does this automatically)
while pool.pending_count() > 0 {
    pool.tick();     // Poll one highest-priority task
    // or:
    pool.tick_all(); // Poll all tasks once each
}

assert_eq!(handle.try_recv(), Some(42));
```

### Tick Methods

| Method | Behavior | Lock held during poll? |
|--------|----------|----------------------|
| `tick()` | Poll one highest-priority task | Yes (holds mutex) |
| `tick_all()` | Poll all tasks once each | Yes (holds mutex) |
| `tick_extract()` | Extract one task, poll outside lock, return if pending | No (mutex-free polling) |

`tick_extract()` is designed for parallel draining — multiple threads can call it concurrently without contention.

## Multi-Threaded Compute Draining

The multi-threaded runner uses `tick_extract()` after all systems complete to drain remaining compute tasks across all worker threads:

```rust
// This happens automatically in EcsRunnerMultiThread::run()
std::thread::scope(|scope| {
    for _ in 0..num_threads {
        scope.spawn(|| {
            while pool.tick_extract() > 0 {}
        });
    }
});
```

## Integration with Systems

Systems can interact with the compute pool in two ways:

1. **Fire-and-forget**: Spawn a task and let it run in the background. Check results later via `try_recv()`.
2. **Block on result**: Use `ctx.compute().block_on(&mut handle)` to tick the pool until the task completes, then use the result immediately.

The runner also ticks the compute pool between systems and after all systems complete, ensuring background tasks make progress.

## EcsComputeContext

Each spawned task receives an `EcsComputeContext` that implements the `ComputeContext` trait:

```rust
pool.spawn(Priority::Low, |cctx: EcsComputeContext| async move {
    // Cooperative yielding
    cctx.yield_now().await;

    // IO access (file reads, network, etc.)
    let data = cctx.io().run(async {
        tokio::fs::read_to_string("data.json").await.unwrap()
    }).await;

    process(data)
});
```
