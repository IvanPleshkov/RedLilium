# Bundles (Partial)

## What Are Bundles?

Bundles are predefined groups of components that are commonly inserted together as a unit. Instead of adding components one-by-one, a bundle lets you insert a logically related set in a single call — with compile-time guarantees that required components aren't forgotten.

```rust
// Bevy-style bundle (not available in RedLilium)
#[derive(Bundle)]
struct PlayerBundle {
    transform: Transform,
    global_transform: GlobalTransform,
    visibility: Visibility,
    health: Health,
    name: Name,
}

world.spawn(PlayerBundle {
    transform: Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    global_transform: GlobalTransform::default(),
    visibility: Visibility::VISIBLE,
    health: Health { current: 100.0, max: 100.0 },
    name: Name::new("Player"),
});
```

## What RedLilium Has

RedLilium provides a **builder pattern** via `SpawnBuilder` for inserting multiple components, but this is not a formal bundle system:

```rust
// Current RedLilium approach
commands.spawn_entity()
    .with(Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)))
    .with(GlobalTransform::default())
    .with(Visibility::VISIBLE)
    .with(Name::new("Player"))
    .build();

// Or direct insertion
let entity = world.spawn();
world.insert(entity, Transform::IDENTITY).unwrap();
world.insert(entity, GlobalTransform::default()).unwrap();
world.insert(entity, Visibility::VISIBLE).unwrap();
```

The `spawn_scene()` function also inserts multiple components per node (Transform, GlobalTransform, Visibility, Name, Camera), acting as a hard-coded "scene node bundle."

## Why It's Partial

| Aspect | Full Bundle System | RedLilium |
|--------|-------------------|-----------|
| Compile-time completeness | Bundle struct ensures all fields present | Builder can forget components |
| Type-safe group | Single type represents the group | No group type — just chained calls |
| Archetype optimization | Bundle maps to a known archetype | N/A (sparse sets, not archetypes) |
| Nested bundles | Bundles can contain other bundles | Not applicable |
| Required components | Framework enforces dependencies | Manual responsibility |
| Ergonomic spawning | `world.spawn(MyBundle { ... })` | `spawn_entity().with(A).with(B).build()` |

The builder pattern achieves the **ergonomic** goal (less boilerplate than individual inserts), but lacks the **safety** guarantee — nothing prevents you from spawning an entity with `Transform` but forgetting `GlobalTransform`.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `#[derive(Bundle)]` trait with nested bundles, required components (0.15+) |
| **hecs** | Tuples as bundles: `world.spawn((Position(...), Velocity(...)))` |
| **Legion** | Tuples as bundles: `world.push((pos, vel))` |
| **Shipyard** | Tuples for batch insertion |
| **flecs** | Prefab entities as templates (different approach) |
| **Unity DOTS** | `IComponentData` archetypes, `EntityManager.AddComponentData` with typed groups |
| **EnTT** | No formal bundles (manual per-component) |
