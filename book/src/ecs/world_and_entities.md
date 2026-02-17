# World and Entities

## World

`World` is the central container that owns all entities, component storages, and resources. You can create multiple independent worlds.

```rust
let mut world = World::new();
```

## Entities

An `Entity` is a lightweight identifier composed of an index and a generation counter. The generation prevents stale references -- when an entity is despawned and its index reused, the generation increments.

```rust
let entity = world.spawn();
println!("{}", entity); // "Entity(0:0)"

assert!(world.is_alive(entity));
```

### Spawning

```rust
// Spawn a single empty entity
let e = world.spawn();

// Spawn with a bundle of components
let e = world.spawn_with((
    Position { x: 0.0, y: 0.0 },
    Velocity { x: 1.0, y: 0.0 },
));

// Spawn many at once
let entities = world.spawn_batch(100);

// Spawn many with the same components
let entities = world.spawn_batch_with(100, (
    Position { x: 0.0, y: 0.0 },
    Velocity { x: 0.0, y: 0.0 },
));

// Spawn many with per-entity initialization
let entities = world.spawn_batch_with_fn(100, |i| (
    Position { x: i as f32, y: 0.0 },
    Name::new(format!("entity_{}", i)),
));
```

### Despawning

```rust
// Despawn a single entity (returns false if already dead)
let was_alive = world.despawn(entity);

// Despawn many at once
world.despawn_batch(&[e1, e2, e3]);
```

### Querying Entities

```rust
// Check if alive
if world.is_alive(entity) {
    // ...
}

// Total live entity count
let count = world.entity_count();

// Iterate all live entities
for entity in world.iter_entities() {
    println!("{}", entity);
}
```

## Component Management

Before using a component type, register it with the world:

```rust
world.register_component::<Position>();
world.register_component::<Velocity>();
```

### Insert and Remove

```rust
// Insert a single component
world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();

// Insert a bundle (tuple of components)
world.insert_bundle(entity, (
    Position { x: 0.0, y: 0.0 },
    Velocity { x: 1.0, y: 0.0 },
)).unwrap();

// Insert the same component for many entities at once
world.insert_batch(&entities, vec![
    Position { x: 0.0, y: 0.0 }; entities.len()
]).unwrap();

// Remove a component (returns the removed value)
let old_pos: Option<Position> = world.remove::<Position>(entity);
```

### Direct Access

For one-off reads outside of systems, you can access components directly:

```rust
// Read a single entity's component
if let Some(pos) = world.get::<Position>(entity) {
    println!("x={}, y={}", pos.x, pos.y);
}

// Mutate a single entity's component
if let Some(pos) = world.get_mut::<Position>(entity) {
    pos.x += 10.0;
}
```

### Storage Borrows

For bulk access, borrow the entire component storage:

```rust
// Shared borrow of all Position components
let positions = world.read::<Position>().unwrap();
if let Some(pos) = positions.get(entity) {
    println!("{:?}", pos);
}

// Exclusive borrow of all Position components
let mut positions = world.write::<Position>().unwrap();
for (entity, pos) in positions.iter_mut() {
    pos.x += 1.0;
}
```

These borrows are runtime-checked (like `RefCell`). Attempting a write while a read is held will panic.

## Change Tracking

The world maintains a tick counter for change detection:

```rust
let tick = world.current_tick();
world.advance_tick();

// Insert with change tracking
world.insert_tracked(entity, Position { x: 0.0, y: 0.0 }).unwrap();
```

Change detection is used by filters like `Added` and `Changed` in queries (see [Queries and Filters](./queries.md)).
