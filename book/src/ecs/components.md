# Components

Components are plain data types attached to entities. Any type that implements `Component` (which requires `Send + Sync + 'static`) can be stored in the world.

## The Component Trait

```rust
pub trait Component: Send + Sync + 'static {
    const NAME: &'static str;

    fn component_name(&self) -> &'static str { Self::NAME }
    fn inspect_ui(&mut self, ui: &mut egui::Ui);
    fn collect_entities(&self, collector: &mut Vec<Entity>) {}
    fn remap_entities(&mut self, map: &mut dyn FnMut(Entity) -> Entity) {}
    fn register_required(_world: &mut World) {}
}
```

## Derive Macro

In practice, you'll almost always use the derive macro:

```rust
#[derive(Component)]
struct Health {
    current: f32,
    max: f32,
}

#[derive(Component)]
struct Player;  // marker component (no data)
```

The derive macro generates the `NAME` constant from the struct name and provides default implementations for all methods.

## Required Components

When a component logically depends on other components, use `#[require]` to automatically insert them:

```rust
#[derive(Component)]
#[require(Transform, GlobalTransform, Visibility)]
struct Camera {
    pub view_matrix: Mat4,
    pub projection_matrix: Mat4,
}
```

When `Camera` is registered with `register_inspector_default`, inserting a `Camera` on an entity will automatically insert `Transform`, `GlobalTransform`, and `Visibility` if they aren't already present. The required components must implement `Default`.

## Registration

Components must be registered before use:

```rust
let mut world = World::new();

// Basic registration
world.register_component::<Health>();

// Registration with inspector support (feature: inspector)
world.register_inspector::<Health>();           // viewable in inspector UI
world.register_inspector_default::<Health>();   // viewable + "Add" button in inspector

// Enable cloning (needed for prefabs)
world.enable_clone::<Health>();
```

### Required Component Registration

You can also declare that inserting one component should automatically insert another:

```rust
world.register_required::<Camera, Transform>();
// Now inserting Camera on an entity will also insert Transform::default()
// if the entity doesn't already have one
```

## Bundles

Bundles are tuples of components that can be inserted together. Tuples of 1-8 `Send + Sync + 'static` types automatically implement the `Bundle` trait:

```rust
// Insert a bundle
world.insert_bundle(entity, (
    Health { current: 100.0, max: 100.0 },
    Position { x: 0.0, y: 0.0 },
    Player,
)).unwrap();

// Spawn with a bundle
let entity = world.spawn_with((
    Health { current: 100.0, max: 100.0 },
    Position { x: 0.0, y: 0.0 },
    Player,
));
```

## Entity References in Components

If your component stores `Entity` references (e.g. a target to follow), implement `collect_entities` and `remap_entities` so the prefab system can correctly remap IDs when cloning:

```rust
#[derive(Component)]
struct FollowTarget {
    target: Entity,
}

// The derive macro handles this automatically if you annotate:
// #[derive(Component)]
// struct FollowTarget {
//     #[entity]
//     target: Entity,
// }
```

## Entity Flags

Entities carry per-entity flag bits that control query visibility. Flags are set and cleared via `World::set_entity_flags` / `World::clear_entity_flags`, or through the hierarchy functions.

### Disabled

Disabled entities are excluded from all `Read<T>` and `Write<T>` queries:

```rust
// Disable an entity and all its descendants
disable(&mut world, entity);

// Re-enable
enable(&mut world, entity);
```

### Static

Static entities represent rarely-changing data (terrain, baked lighting, static geometry). They are excluded from both `Read<T>` and `Write<T>` queries — regular systems cannot see or mutate them:

```rust
// Mark an entity and all descendants as static
mark_static(&mut world, entity);

// Unmark
unmark_static(&mut world, entity);
```

To **read** static entity data, use `ReadAll<T>` instead of `Read<T>`:

```rust
ctx.lock::<(ReadAll<Transform>,)>()
    .execute(|(transforms,)| {
        // Sees ALL active entities, including static ones
    });
```

To **mutate** static entity components, use exclusive system access (`&mut World`).

Both disabled and static flags propagate through the hierarchy. See [Hierarchy](./hierarchy.md) for details.
