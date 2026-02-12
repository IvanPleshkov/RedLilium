# World

The `World` is the central container of the ECS. It owns all entities, component storages, and resources. Each World is fully self-contained — multiple worlds can coexist in the same process.

## Creating a World

```rust
use redlilium_ecs::World;

let mut world = World::new();
```

## Registering Components

Before inserting a component on an entity, the type must be registered:

```rust
struct Position { x: f32, y: f32 }
struct Velocity { x: f32, y: f32 }

world.register_component::<Position>();
world.register_component::<Velocity>();
```

There are three registration levels:

| Method | Storage | Inspector UI | "Add Component" button |
|--------|---------|-------------|----------------------|
| `register_component::<T>()` | Yes | No | No |
| `register_inspector::<T>()` | Yes | Yes | No |
| `register_inspector_default::<T>()` | Yes | Yes | Yes (uses `Default`) |

## Entity-Component Operations

```rust
let entity = world.spawn();

// Insert a component
world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();

// Get a reference
let pos = world.get::<Position>(entity);
assert_eq!(pos, Some(&Position { x: 1.0, y: 2.0 }));

// Get a mutable reference
let pos = world.get_mut::<Position>(entity).unwrap();
pos.x = 5.0;

// Remove a component (returns the removed value)
let removed = world.remove::<Position>(entity);
assert!(removed.is_some());

// Despawn removes entity + all its components
world.despawn(entity);
```

## Query Access (Runtime Borrow-Checked)

The World provides runtime borrow-checked access to entire component storages:

```rust
// Shared read access (multiple readers OK)
let positions = world.read::<Position>().unwrap();
let velocities = world.read::<Velocity>().unwrap();

// Iterate and cross-reference
for (idx, pos) in positions.iter() {
    if let Some(vel) = velocities.get(idx) {
        println!("pos ({}, {}) vel ({}, {})", pos.x, pos.y, vel.x, vel.y);
    }
}
drop(positions);
drop(velocities);

// Exclusive write access
let mut positions = world.write::<Position>().unwrap();
for (_, pos) in positions.iter_mut() {
    pos.x += 1.0;
}
```

### Optional Access

Use `try_read` / `try_write` when the component type might not be registered:

```rust
// Returns None instead of Err if unregistered
if let Some(positions) = world.try_read::<Position>() {
    println!("Found {} positions", positions.len());
}
```

## Change Detection

The World maintains a global tick counter for change tracking:

```rust
// Advance tick at the start of each frame
world.advance_tick();

// Insert with tick tracking
world.insert_tracked(entity, Position { x: 0.0, y: 0.0 }).unwrap();

// Query what changed since a specific tick
let current_tick = world.current_tick();
```

## Resource Management

Resources are global singletons stored by type:

```rust
// Insert a resource
world.insert_resource(42u32);
world.insert_resource("delta_time".to_string());

// Check existence
assert!(world.has_resource::<u32>());

// Borrow immutably
let val = world.resource::<u32>();
assert_eq!(*val, 42);

// Borrow mutably
let mut val = world.resource_mut::<u32>();
*val = 99;

// Remove
let removed = world.remove_resource::<u32>();
```

## Filters

Create lightweight filters that check component presence without borrowing data:

```rust
let positions = world.read::<Position>().unwrap();
let has_health = world.with::<Health>();
let not_frozen = world.without::<Frozen>();
let recently_changed = world.changed::<Position>(last_tick);
let recently_added = world.added::<Position>(last_tick);

for (idx, pos) in positions.iter() {
    if has_health.matches(idx) && not_frozen.matches(idx) {
        // Process entities with Health but without Frozen
    }
    if recently_changed.matches(idx) {
        // Only process entities whose Position changed
    }
}
```

## Commands

For deferred structural changes from within systems:

```rust
// Initialize the command buffer resource
world.init_commands();

// Systems push commands during execution...
// (typically via SystemContext::commands())

// Apply all queued commands
world.apply_commands();
```

## Events

Register typed event channels:

```rust
world.add_event::<CollisionEvent>();
// Creates an Events<CollisionEvent> resource
```

## Standard Components Registration

Register all built-in component types with a single call:

```rust
use redlilium_ecs::register_std_components;
register_std_components(&mut world);
// Registers: Transform, GlobalTransform, Camera, Visibility, Name,
// DirectionalLight, PointLight, SpotLight, Parent, Children,
// and physics components (if feature-gated)
```

## Public API Summary

| Category | Methods |
|----------|---------|
| **Entity** | `spawn`, `despawn`, `is_alive`, `entity_count`, `iter_entities` |
| **Component** | `register_component`, `insert`, `insert_tracked`, `get`, `get_mut`, `remove` |
| **Query** | `read`, `write`, `try_read`, `try_write` |
| **Filters** | `with`, `without`, `changed`, `added` |
| **Resource** | `insert_resource`, `remove_resource`, `has_resource`, `resource`, `resource_mut` |
| **Change** | `current_tick`, `advance_tick` |
| **Commands** | `init_commands`, `apply_commands` |
| **Events** | `add_event` |
| **Inspector** | `register_inspector`, `register_inspector_default`, `inspectable_components_of`, `addable_components_of`, `inspect_by_name`, `remove_by_name`, `insert_default_by_name` |
