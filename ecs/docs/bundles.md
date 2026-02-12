# Component Bundles

Bundles let you insert multiple components on an entity in a single call using tuples. This avoids forgetting a component and reduces boilerplate.

## Bundle Trait

The `Bundle` trait is implemented for tuples of 1-8 components:

```rust
pub trait Bundle: Send + 'static {
    fn insert_into(self, world: &mut World, entity: Entity)
        -> Result<(), ComponentNotRegistered>;
}

// Implemented for (A,), (A, B), (A, B, C), ... up to 8 elements
```

Each element must be `Send + Sync + 'static` (the standard component bounds).

## Derive Macro

Use `#[derive(Bundle)]` to implement the trait for a struct. Each field becomes a component inserted individually:

```rust
#[derive(Bundle)]
struct PlayerBundle {
    transform: Transform,
    global_transform: GlobalTransform,
    visibility: Visibility,
    name: Name,
}

let entity = world.spawn_with(PlayerBundle {
    transform: Transform::IDENTITY,
    global_transform: GlobalTransform::default(),
    visibility: Visibility::VISIBLE,
    name: Name::new("Player"),
});
```

### Nested Bundles

Use `#[bundle]` on a field to treat it as a nested bundle rather than a component:

```rust
#[derive(Bundle)]
struct SpatialBundle {
    transform: Transform,
    global_transform: GlobalTransform,
    visibility: Visibility,
}

#[derive(Bundle)]
struct EnemyBundle {
    health: Health,
    #[bundle]
    spatial: SpatialBundle,
}
```

## World API

### `insert_bundle` — insert a tuple of components on an existing entity

```rust
let entity = world.spawn();
world.insert_bundle(entity, (
    Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    GlobalTransform::default(),
    Visibility::VISIBLE,
    Name::new("Player"),
)).unwrap();
```

### `spawn_with` — spawn a new entity with a bundle

```rust
let entity = world.spawn_with((
    Transform::IDENTITY,
    GlobalTransform::default(),
    Visibility::VISIBLE,
));
```

## Deferred Commands (from systems)

### Via SystemContext

```rust
impl System for SpawnSystem {
    async fn run(&self, ctx: &SystemContext<'_>) {
        // Spawn with bundle
        ctx.spawn_with((
            Transform::IDENTITY,
            GlobalTransform::default(),
            Visibility::VISIBLE,
        ));

        // Insert bundle on existing entity
        ctx.insert_bundle(entity, (Health(100), Armor(50)));
    }
}
```

### Via SpawnBuilder

```rust
ctx.spawn_entity()
    .with_bundle((Transform::IDENTITY, GlobalTransform::default()))
    .with(Name::new("Extra"))
    .build();
```

The builder's `.with()` and `.with_bundle()` can be mixed freely.

### Via CommandCollector directly

```rust
let commands = CommandCollector::new();
commands.spawn_with((Transform::IDENTITY, Visibility::VISIBLE));
commands.insert_bundle(entity, (Health(100), Armor(50)));
```

## Error Handling

`World::insert_bundle` returns `Result` — if any component type is unregistered, it returns `ComponentNotRegistered`. The deferred variants (`ctx.spawn_with`, `ctx.insert_bundle`) panic when applied if a component is unregistered, matching the behavior of `ctx.insert`.

## Public API

| Method | On | Description |
|--------|----|-------------|
| `World::insert_bundle(entity, bundle)` | `&mut World` | Insert tuple of components |
| `World::spawn_with(bundle)` | `&mut World` | Spawn entity + insert bundle |
| `SystemContext::insert_bundle(entity, bundle)` | `&SystemContext` | Deferred insert bundle |
| `SystemContext::spawn_with(bundle)` | `&SystemContext` | Deferred spawn + bundle |
| `CommandCollector::insert_bundle(entity, bundle)` | `&CommandCollector` | Deferred insert bundle |
| `CommandCollector::spawn_with(bundle)` | `&CommandCollector` | Deferred spawn + bundle |
| `SpawnBuilder::with_bundle(bundle)` | `SpawnBuilder` | Add bundle to builder |
| `#[derive(Bundle)]` | struct | Auto-implement `Bundle` for a struct |
| `#[bundle]` field attr | field | Treat field as nested bundle |
