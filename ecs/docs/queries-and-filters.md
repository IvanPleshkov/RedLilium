# Queries and Filters

The query system provides ways to iterate and filter entities based on their component composition. Unlike archetype-based ECS libraries, queries in this sparse-set ECS work by iterating one component storage and cross-referencing others.

## Basic Query Pattern

Pick a "driving" component to iterate, then look up related components:

```rust
let positions = world.read::<Position>().unwrap();
let velocities = world.read::<Velocity>().unwrap();

for (entity_idx, pos) in positions.iter() {
    if let Some(vel) = velocities.get(entity_idx) {
        println!("Entity {} at ({}, {}) moving ({}, {})",
            entity_idx, pos.x, pos.y, vel.x, vel.y);
    }
}
```

## Filters

Filters check component presence/absence **without borrowing the component data**. They use the sparse array directly for O(1) checks.

### With<T> — Presence Filter

```rust
let positions = world.read::<Position>().unwrap();
let has_health = world.with::<Health>();

for (idx, pos) in positions.iter() {
    if has_health.matches(idx) {
        // Entity has both Position and Health
    }
}
```

### Without<T> — Absence Filter

```rust
let positions = world.write::<Position>().unwrap();
let not_frozen = world.without::<Frozen>();

for (idx, pos) in positions.iter_mut() {
    if not_frozen.matches(idx) {
        // Entity has Position but NOT Frozen
        pos.x += 1.0;
    }
}
```

### Changed<T> — Change Detection Filter

Only match entities whose component was modified since a given tick:

```rust
let transforms = world.read::<Transform>().unwrap();
let changed = world.changed::<Transform>(last_tick);

for (idx, transform) in transforms.iter() {
    if changed.matches(idx) {
        // This transform was modified since last_tick
        recompute_bounding_box(idx, transform);
    }
}
```

### Added<T> — Addition Detection Filter

Only match entities that received the component since a given tick:

```rust
let meshes = world.read::<MeshHandle>().unwrap();
let newly_added = world.added::<MeshHandle>(last_tick);

for (idx, mesh) in meshes.iter() {
    if newly_added.matches(idx) {
        // This mesh was just added — initialize GPU resources
        upload_to_gpu(mesh);
    }
}
```

## Combining Filters

Filters compose naturally with `&&`:

```rust
let positions = world.write::<Position>().unwrap();
let has_velocity = world.with::<Velocity>();
let not_frozen = world.without::<Frozen>();
let recently_changed = world.changed::<Position>(last_tick);

for (idx, pos) in positions.iter_mut() {
    if has_velocity.matches(idx) && not_frozen.matches(idx) {
        // Has velocity and not frozen — apply movement
    }
    if recently_changed.matches(idx) {
        // Position was modified — update spatial index
    }
}
```

## In Systems (Lock-Execute Pattern)

Within systems, use the access type system for locking, then filters for iteration:

```rust
struct MovementSystem;

impl System for MovementSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Write<Position>, Read<Velocity>)>()
            .execute(|(mut positions, velocities)| {
                for (idx, pos) in positions.iter_mut() {
                    if let Some(vel) = velocities.get(idx) {
                        pos.x += vel.x;
                        pos.y += vel.y;
                    }
                }
            }).await;
    }
}
```

Filters in systems (combining world filters with component access):

```rust
struct SelectiveUpdateSystem;

impl System for SelectiveUpdateSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Write<GlobalTransform>, Read<Transform>)>()
            .execute(|(mut globals, locals)| {
                let world = /* accessed via the lock pattern */;
                // In practice, you'd pass the world reference or
                // pre-compute filters outside the lock
                for (idx, local) in locals.iter() {
                    if let Some(global) = globals.get_mut(idx) {
                        *global = GlobalTransform(local.to_matrix());
                    }
                }
            }).await;
    }
}
```

## Single Entity Lookup

For random access by entity:

```rust
let positions = world.read::<Position>().unwrap();

// O(1) lookup by entity index
if let Some(pos) = positions.get(entity.index()) {
    println!("Entity at ({}, {})", pos.x, pos.y);
}
```

## Filter Edge Cases

| Scenario | `with::<T>()` | `without::<T>()` |
|----------|---------------|-------------------|
| `T` registered, entity has it | `true` | `false` |
| `T` registered, entity lacks it | `false` | `true` |
| `T` never registered | `false` (matches nothing) | `true` (matches everything) |

## ContainsChecker

The `ContainsChecker` struct is the runtime filter object:

```rust
pub struct ContainsChecker<'a> {
    storage: Option<&'a ComponentStorage>,
    inverted: bool,
}
```

- Created by `world.with::<T>()` or `world.without::<T>()`.
- Borrows only the component storage metadata (sparse array), not the component data.
- `matches(entity_index)` is O(1) — just a bounds check and Option check in the sparse array.

## Public API

| Method | Returns | Description |
|--------|---------|-------------|
| `world.with::<T>()` | `ContainsChecker` | Filter: entity has component T |
| `world.without::<T>()` | `ContainsChecker` | Filter: entity lacks component T |
| `world.changed::<T>(tick)` | `ChangedFilter` | Filter: T changed since tick |
| `world.added::<T>(tick)` | `AddedFilter` | Filter: T added since tick |
| `filter.matches(entity_idx)` | `bool` | Check if entity passes filter |
