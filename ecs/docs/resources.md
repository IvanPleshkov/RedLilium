# Resources

Resources are **global singleton values** stored in the World by type. Unlike components (which are per-entity), each resource type has at most one instance in the world.

Resources are used for data that doesn't belong to any specific entity: time, configuration, input state, shared caches, etc.

## Basic Usage

```rust
let mut world = World::new();

// Insert a resource
world.insert_resource(0.016f32); // delta time

// Check existence
assert!(world.has_resource::<f32>());

// Borrow immutably (returns ResourceRef<T>)
let dt = world.resource::<f32>();
assert_eq!(*dt, 0.016);
drop(dt);

// Borrow mutably (returns ResourceRefMut<T>)
let mut dt = world.resource_mut::<f32>();
*dt = 0.032;
drop(dt);

// Remove a resource
let removed = world.remove_resource::<f32>();
assert_eq!(removed, Some(0.032));
```

## Thread Safety

Resources use per-resource `RwLock` synchronization:

- Multiple shared borrows (`resource::<T>()`) can coexist.
- An exclusive borrow (`resource_mut::<T>()`) requires no other borrows.
- Conflicts panic immediately (instant detection, not deadlock).

```rust
world.insert_resource(42u32);

// Two shared borrows — OK
let _a = world.resource::<u32>();
let _b = world.resource::<u32>();
```

```rust
// Shared + exclusive — PANICS
let _a = world.resource::<u32>();
// let _b = world.resource_mut::<u32>(); // panics!
```

## In Systems (via Access Types)

Systems access resources through the lock-execute pattern using `Res<T>` and `ResMut<T>`:

```rust
struct GameConfig {
    gravity: f32,
    time_scale: f32,
}

struct PhysicsSystem;

impl System for PhysicsSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Read<Velocity>, Res<GameConfig>)>()
            .execute(|(velocities, config)| {
                println!("Gravity: {}", config.gravity);
                // use config.gravity with velocities...
            }).await;
    }
}
```

To modify a resource from a system:

```rust
ctx.lock::<(ResMut<GameConfig>,)>()
    .execute(|(mut config,)| {
        config.time_scale = 0.5;
    }).await;
```

## ResourceRef / ResourceRefMut

These are RAII guard types that auto-release locks on drop:

| Type | Access | Implements |
|------|--------|-----------|
| `ResourceRef<'a, T>` | Shared read | `Deref<Target = T>` |
| `ResourceRefMut<'a, T>` | Exclusive write | `Deref + DerefMut<Target = T>` |

Both implement `Send + Sync` for use in multi-threaded contexts.

## Common Resource Patterns

### Configuration

```rust
struct AppConfig {
    window_width: u32,
    window_height: u32,
    vsync: bool,
}

world.insert_resource(AppConfig {
    window_width: 1920,
    window_height: 1080,
    vsync: true,
});
```

### Frame Timing

```rust
struct DeltaTime(pub f32);

world.insert_resource(DeltaTime(0.016));

// Update each frame:
let mut dt = world.resource_mut::<DeltaTime>();
dt.0 = measured_delta;
```

### Shared Caches

```rust
struct AssetCache {
    textures: HashMap<String, TextureHandle>,
}

world.insert_resource(AssetCache {
    textures: HashMap::new(),
});
```

## Public API

| Method | Description |
|--------|-------------|
| `world.insert_resource::<T>(value)` | Insert or replace a resource |
| `world.remove_resource::<T>()` | Remove and return resource |
| `world.has_resource::<T>()` | Check if resource exists |
| `world.resource::<T>()` | Shared borrow (panics if missing or exclusively borrowed) |
| `world.resource_mut::<T>()` | Exclusive borrow (panics if missing or any borrow active) |
