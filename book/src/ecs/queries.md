# Queries and Filters

Queries let you access component data inside systems. The access pattern is declared as a tuple of access types, and the ECS validates at runtime that no conflicting borrows exist.

## Locking Components

The primary way to query data is through `ctx.lock()`:

```rust
impl System for MovementSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        ctx.lock::<(Write<Position>, Read<Velocity>)>()
            .for_each(|(pos, vel): (&mut Position, &Velocity)| {
                pos.x += vel.x;
                pos.y += vel.y;
            });
        Ok(())
    }
}
```

### Access Types

| Type | Provides | Notes |
|------|----------|-------|
| `Read<T>` | `Ref<'_, T>` (shared) | Skips disabled and static entities |
| `ReadAll<T>` | `Ref<'_, T>` (shared) | Skips disabled only, includes static |
| `Write<T>` | `RefMut<'_, T>` (exclusive) | Skips disabled and static entities |
| `OptionalRead<T>` | `Option<Ref<'_, T>>` | Returns `None` if not registered |
| `OptionalWrite<T>` | `Option<RefMut<'_, T>>` | Returns `None` if not registered |
| `Res<T>` | `ResourceRef<'_, T>` | Shared resource access |
| `ResMut<T>` | `ResourceRefMut<'_, T>` | Exclusive resource access |

Access tuples of up to 8 elements are supported.

## Iteration Styles

### for_each -- Per-Entity

Iterates over all entities that have **all** the required components:

```rust
ctx.lock::<(Write<Position>, Read<Velocity>)>()
    .for_each(|(pos, vel): (&mut Position, &Velocity)| {
        pos.x += vel.x;
        pos.y += vel.y;
    });
```

Only entities that have both `Position` **and** `Velocity` are visited.

### par_for_each -- Parallel Per-Entity

Same as `for_each`, but distributes work across threads:

```rust
ctx.lock::<(Write<Position>, Read<Velocity>)>()
    .par_for_each(|(pos, vel): (&mut Position, &Velocity)| {
        pos.x += vel.x;
        pos.y += vel.y;
    });
```

Falls back to sequential on WASM automatically.

### execute -- Full Storage Access

When you need more control than per-entity iteration:

```rust
ctx.lock::<(Write<Position>, Read<Velocity>)>()
    .execute(|(mut positions, velocities)| {
        // Access the full sparse set storages
        for (entity, pos) in positions.iter_mut() {
            if let Some(vel) = velocities.get(entity) {
                pos.x += vel.x;
                pos.y += vel.y;
            }
        }
    });
```

### QueryGuard -- Hold Locks

If you need to hold locks across multiple operations:

```rust
let mut query = ctx.query::<(Write<Position>, Read<Velocity>)>();
let (positions, velocities) = &mut query.items;
// locks held until `query` is dropped
```

## Filters

Filters refine which entities are visited during iteration.

### With / Without

Filter by component presence without reading the data:

```rust
ctx.lock::<(Write<Position>, Read<Velocity>, With<Player>, Without<Dead>)>()
    .for_each(|(pos, vel, _, _): (&mut Position, &Velocity, _, _)| {
        // Only entities that have Player and don't have Dead
        pos.x += vel.x;
    });
```

### Added / Removed

React to structural changes since a given tick:

```rust
ctx.lock::<(Read<Health>, Added<Health>)>()
    .for_each(|(health, added): (&Health, _)| {
        for &entity in added.iter() {
            println!("New health component on {:?}", entity);
        }
    });

ctx.lock::<(Removed<Enemy>,)>()
    .execute(|(removed,)| {
        for &entity in removed.iter() {
            println!("Enemy component removed from {:?}", entity);
        }
    });
```

`MaybeAdded<T>` and `MaybeRemoved<T>` are non-panicking variants that match nothing if the component isn't registered.

### Or / Any -- Combining Filters

Combine filters with logical OR:

```rust
// Entities that have either Player OR Enemy
ctx.lock::<(Read<Position>, Or<With<Player>, With<Enemy>>)>()
    .for_each(|(pos, _): (&Position, _)| {
        // ...
    });

// Any of multiple filters (2-8 supported)
ctx.lock::<(Read<Position>, Any<(With<Player>, With<Enemy>, With<NPC>)>)>()
    .for_each(|(pos, _): (&Position, _)| {
        // ...
    });
```

## Mixing Resources and Components

You can access both resources and components in the same lock:

```rust
ctx.lock::<(Write<Position>, Read<Velocity>, Res<Time>)>()
    .for_each(|(pos, vel, time): (&mut Position, &Velocity, &Time)| {
        let dt = time.delta_f32();
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;
    });
```

## Main Thread Resources

For non-`Send` resources (e.g. windowing handles), use `MainThreadRes` / `MainThreadResMut`. The ECS dispatches access to the main thread automatically:

```rust
ctx.lock::<(MainThreadRes<Window>,)>()
    .execute(|(window,)| {
        let size = window.inner_size();
        // ...
    });
```
