# Entities

Entities are lightweight identifiers that represent objects in the ECS world. An entity on its own holds no data — it's just an ID. Components are attached to entities to give them behavior and state.

## Generational IDs

Each entity is a 64-bit value composed of two parts:

- **Index** (lower 32 bits): The slot in the internal storage arrays.
- **Generation** (upper 32 bits): A counter that increments each time a slot is recycled.

This prevents the **ABA problem**: if you hold a reference to entity `(index=5, gen=0)` and that entity is despawned and the slot reused, the new entity will be `(index=5, gen=1)`. Your old reference will correctly report as "not alive."

## Creating and Destroying Entities

```rust
let mut world = World::new();

// Spawn a new entity
let entity = world.spawn();
assert!(world.is_alive(entity));

// Check total alive count
assert_eq!(world.entity_count(), 1);

// Despawn — removes the entity and all its components
world.despawn(entity);
assert!(!world.is_alive(entity));
assert_eq!(world.entity_count(), 0);
```

## Entity Recycling

When an entity is despawned, its index slot is added to a free list. The next `spawn()` call reuses the slot with an incremented generation:

```rust
let mut world = World::new();

let old = world.spawn();
assert_eq!(old.index(), 0);
assert_eq!(old.generation(), 0);

world.despawn(old);

let new = world.spawn();
assert_eq!(new.index(), 0);       // Same slot reused
assert_eq!(new.generation(), 1);  // New generation

// Old reference is stale — no longer alive
assert!(!world.is_alive(old));
assert!(world.is_alive(new));
```

## Iterating Over Entities

You can iterate all alive entities in the world:

```rust
let mut world = World::new();
let e1 = world.spawn();
let e2 = world.spawn();
let e3 = world.spawn();
world.despawn(e2);

let alive: Vec<Entity> = world.iter_entities().collect();
assert_eq!(alive.len(), 2);
assert!(alive.contains(&e1));
assert!(alive.contains(&e3));
```

## Key Properties

| Property | Value |
|----------|-------|
| Size | 8 bytes (`u64`) |
| Copy | Yes (`Clone + Copy`) |
| Hash | Yes (can be used as HashMap key) |
| Equality | Compares both index and generation |
| Display | `Entity(index:generation)` format |

## Public API

| Method | Description |
|--------|-------------|
| `entity.index()` | Returns the 32-bit index portion |
| `entity.generation()` | Returns the 32-bit generation portion |
| `world.spawn()` | Allocates a new entity |
| `world.despawn(entity)` | Destroys entity and removes all components |
| `world.is_alive(entity)` | Checks if entity is alive with matching generation |
| `world.entity_count()` | Returns count of alive entities |
| `world.iter_entities()` | Iterator over all alive entity IDs |
