# Parallel Iteration

## What Is It?

Parallel iteration (par_iter) splits component iteration across multiple threads **within a single system**. Instead of one thread processing all entities sequentially, the entity range is divided into chunks and processed concurrently using a thread pool (typically via rayon).

```rust
// Bevy-style parallel iteration (not available in RedLilium)
fn movement_system(mut query: Query<(&mut Transform, &Velocity)>) {
    query.par_iter_mut().for_each(|(mut transform, velocity)| {
        transform.translation += velocity.0 * DELTA_TIME;
    });
}
```

### How It Works

1. The component storage is divided into equal-sized batches.
2. Each batch is submitted as a task to a thread pool.
3. Threads process their batches concurrently.
4. System completes when all batches finish.

```
Thread 0: [Entity 0..999]      ← processes first 1000
Thread 1: [Entity 1000..1999]  ← processes next 1000
Thread 2: [Entity 2000..2999]  ← processes next 1000
Thread 3: [Entity 3000..3999]  ← processes last 1000
```

### When It Helps

| Scenario | Sequential | Parallel |
|----------|-----------|----------|
| 100 entities, simple math | ~1μs | ~5μs (overhead > work) |
| 10,000 entities, simple math | ~50μs | ~15μs |
| 100,000 entities, complex computation | ~5ms | ~1.5ms |
| 1M entities, physics update | ~50ms | ~15ms |

Parallel iteration is most beneficial for:
- **High entity counts** (10,000+) where per-entity work is non-trivial.
- **CPU-bound computations** (physics, AI, pathfinding) where each entity's work is independent.
- **Embarrassingly parallel** problems where entities don't interact.

### Comparison With System-Level Parallelism

| Approach | Granularity | When It Helps |
|----------|-------------|---------------|
| System-level parallelism | Different systems on different threads | Many independent systems |
| Within-system parallelism | Same system, entities split across threads | Few systems with many entities |

These are complementary — a fully parallel ECS uses both.

## Current Approach in RedLilium

RedLilium parallelizes at the **system level** (multi-threaded runner runs independent systems concurrently) but iterates **sequentially** within each system:

```rust
ctx.lock::<(Write<Transform>, Read<Velocity>)>()
    .execute(|world, (mut transforms, velocities)| {
        // Sequential iteration — single thread
        for (entity, transform) in transforms.iter_mut() {
            if let Some(velocity) = velocities.get(entity) {
                transform.translation += velocity.0;
            }
        }
    })
    .await;
```

The async compute pool can be used as a manual workaround for heavy per-entity work, but it requires extracting data out of the ECS, processing it, and writing results back.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `query.par_iter()` / `query.par_iter_mut()` via `ComputeTaskPool`, configurable batch size |
| **flecs** | `ecs_iter_t` with multi-threaded pipeline, system-level parallelism + stage-level worker distribution |
| **Unity DOTS** | `IJobEntityBatch`, `Entities.ForEach` with `ScheduleParallel()`, Burst compiler for SIMD |
| **EnTT** | No built-in (manual rayon integration possible via view iterators) |
| **hecs** | No built-in parallel iteration |
| **Legion** | `par_for_each()` on query results via rayon |
| **Shipyard** | `par_iter()` on views via rayon |
