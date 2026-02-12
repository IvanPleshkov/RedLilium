# Commands (Deferred Operations)

Commands are deferred world mutations that are queued during system execution and applied after all systems complete. They exist because systems receive `&World` (immutable), so structural changes like spawning entities require deferral.

## Why Commands?

Systems access the world through a shared reference (`&World`) with per-component RwLock synchronization. This allows multiple systems to run in parallel. But operations like `spawn()`, `despawn()`, and `insert()` need `&mut World`. Commands bridge this gap.

## Two Command Mechanisms

### 1. CommandCollector (System-side)

Systems push commands through `SystemContext::commands()`:

```rust
struct SpawnerSystem;

impl System for SpawnerSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        // Read some data
        let should_spawn = ctx.lock::<(Read<SpawnRequest>,)>()
            .execute(|(requests,)| !requests.is_empty())
            .await;

        if should_spawn {
            ctx.commands(|world| {
                let entity = world.spawn();
                world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
                world.insert(entity, Velocity { x: 1.0, y: 0.0 }).unwrap();
            });
        }
    }
}
```

The `CommandCollector` is internal to the runner — it collects commands from all systems and applies them after the frame.

### 2. CommandBuffer (Resource-based)

The `CommandBuffer` is a World resource that provides a richer API:

```rust
// Initialize during setup
world.init_commands();

// Systems can access it as a resource
struct DespawnSystem;

impl System for DespawnSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Read<Health>, Res<CommandBuffer>)>()
            .execute(|(healths, commands)| {
                for (idx, health) in healths.iter() {
                    if health.value <= 0.0 {
                        let entity = /* resolve entity from idx */;
                        commands.despawn(entity);
                    }
                }
            }).await;
    }
}

// After systems run:
world.apply_commands();
```

## CommandBuffer API

### Basic Commands

```rust
let commands = CommandBuffer::new();

// Queue a raw closure
commands.push(|world: &mut World| {
    world.insert_resource(42u32);
});

// Queue entity despawn
commands.despawn(entity);

// Queue component insertion
commands.insert(entity, Position { x: 1.0, y: 2.0 });

// Queue component removal
commands.remove::<Health>(entity);
```

### SpawnBuilder

For spawning entities with multiple components:

```rust
commands.spawn_entity()
    .with(Position { x: 0.0, y: 0.0 })
    .with(Velocity { x: 1.0, y: 0.0 })
    .with(Visibility::VISIBLE)
    .with(Name::new("Bullet"))
    .build();
```

The builder collects component insertions and executes them in a single command. The entity is spawned and all components are inserted atomically.

### Batch Spawning

```rust
for i in 0..100 {
    commands.spawn_entity()
        .with(Position { x: i as f32, y: 0.0 })
        .with(Transform::IDENTITY)
        .build();
}
```

## Thread Safety

`CommandBuffer` uses an internal `Mutex`, so multiple parallel systems can push commands concurrently:

```rust
// Thread 1 (System A): commands.push(|w| { ... });
// Thread 2 (System B): commands.push(|w| { ... });
// Both safe — Mutex protects the internal Vec
```

## Execution Order

Commands execute in the order they were pushed:

```rust
let commands = CommandBuffer::new();

commands.insert(entity, Health(100));
commands.push(move |world| {
    let h = world.get_mut::<Health>(entity).unwrap();
    h.0 += 50; // Health is now 150
});

// After apply_commands(): Health is 150
```

## Lifecycle in a Frame

```
1. world.advance_tick()
2. runner.run(&mut world, &systems)
   ├── Systems execute, push commands via ctx.commands(...)
   ├── All systems complete
   ├── Commands from CommandCollector applied to &mut World
   └── Compute tasks drained
3. (Optional) world.apply_commands() for CommandBuffer resource
```

Note: The runner automatically applies commands from `CommandCollector`. The `CommandBuffer` resource requires a separate `world.apply_commands()` call if you use both mechanisms.

## Public API

### CommandBuffer

| Method | Description |
|--------|-------------|
| `CommandBuffer::new()` | Create empty buffer |
| `push(closure)` | Queue a raw `FnOnce(&mut World)` |
| `despawn(entity)` | Queue entity despawn |
| `insert(entity, component)` | Queue component insertion |
| `remove::<T>(entity)` | Queue component removal |
| `spawn_entity()` | Start building a spawn command |
| `drain()` | Extract all commands (empties buffer) |
| `len()` / `is_empty()` | Query buffer state |

### SpawnBuilder

| Method | Description |
|--------|-------------|
| `.with(component)` | Add a component to the spawn |
| `.build()` | Finalize and queue the command |

### SystemContext

| Method | Description |
|--------|-------------|
| `ctx.commands(closure)` | Push a deferred command (via CommandCollector) |
