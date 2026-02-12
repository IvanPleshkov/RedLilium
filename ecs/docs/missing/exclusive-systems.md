# Exclusive Systems

## What Are They?

Exclusive systems are special systems that get **mutable access to the entire World** — not just specific components. They block all other systems from running concurrently, giving them full authority to perform arbitrary structural changes, run sub-schedules, or access any combination of components and resources.

```rust
// Bevy-style exclusive system (not available in RedLilium)
fn apply_deferred_physics(world: &mut World) {
    // Full mutable access — can do anything
    let physics_world = world.resource::<PhysicsWorld>();

    for (entity, rb, transform) in world.query::<(Entity, &RigidBody, &mut Transform)>().iter(world) {
        // Read physics results, write to ECS
        let position = physics_world.body_position(rb.handle);
        transform.translation = position;
    }

    // Can also spawn/despawn, add/remove components, etc.
    world.spawn((Transform::default(), Velocity::default()));
}
```

### Use Cases

- **Physics synchronization**: Read from external physics engine, write back to ECS.
- **Applying deferred operations**: Flush command buffers at a specific point.
- **Sub-schedule execution**: Run a nested schedule inside a system.
- **World migration**: Move entities between worlds.
- **Complex structural changes**: Spawn/despawn patterns that depend on multiple queries.

### Why Exclusive?

Normal systems in a parallel ECS can only access components they declare upfront. This enables the scheduler to run non-conflicting systems in parallel. Exclusive systems sacrifice parallelism for flexibility — they're a synchronization barrier.

## Current Approach in RedLilium

RedLilium systems receive `&SystemContext<'_>` (an immutable reference to the world), not `&mut World`. All mutations happen through:

1. **Lock-execute**: Borrow specific components, mutate within closure.
2. **Commands**: Deferred structural changes (spawn, despawn, insert, remove).
3. **Resources**: Shared state via `Res`/`ResMut`.

```rust
// No way to get &mut World inside a system
impl System for MySystem {
    async fn run(&self, ctx: &SystemContext<'_>) {
        // ctx.world() returns &World (immutable)
        // Must use lock-execute for mutations
        ctx.lock::<(Write<Transform>,)>()
            .execute(|world, (mut transforms,)| {
                // Can only access Transform here
            })
            .await;
    }
}
```

For operations needing full world access, code must run **outside** the runner:

```rust
// Between frames — full access
runner.run(&mut world, &systems);  // systems run with &SystemContext
world.apply_commands();            // full &mut World access here
do_exclusive_work(&mut world);     // your code with &mut World
```

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `fn my_system(world: &mut World)` — exclusive system parameter, runs as a scheduling barrier |
| **flecs** | `ecs_run()` with `ecs_world_t*` pointer gives full access; systems can be marked single-threaded |
| **Unity DOTS** | `SystemBase.EntityManager` provides exclusive entity operations; `[UpdateInGroup]` for ordering |
| **Legion** | `Schedule::execute()` provides `&mut World` between stages |
| **EnTT** | No system framework — user code always has `entt::registry&` |
| **hecs** | No system framework — user code always has `&mut World` |
| **Shipyard** | `AllStoragesViewMut` for exclusive world access in workloads |
