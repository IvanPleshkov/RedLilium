# Prefabs and Blueprints

## What Are They?

Prefabs (or blueprints) are reusable entity templates that define a set of components with default values. You can instantiate a prefab to create entities with a known configuration, and optionally override specific values. They serve as the ECS equivalent of object-oriented class instantiation.

```rust
// flecs-style prefab (not available in RedLilium)
let spaceship = world.prefab()
    .set(Transform::default())
    .set(GlobalTransform::default())
    .set(Visibility::VISIBLE)
    .set(Health { current: 100.0, max: 100.0 })
    .set(MeshRef("spaceship.glb"));

// Instantiate with overrides
let player_ship = world.entity()
    .is_a(spaceship)
    .set(Name::new("PlayerShip"))
    .set(Transform::from_translation(Vec3::new(0.0, 0.0, 5.0)));

let enemy_ship = world.entity()
    .is_a(spaceship)
    .set(Name::new("EnemyShip"))
    .set(Health { current: 50.0, max: 50.0 });
```

### Key Properties

- **Reusable templates**: Define once, instantiate many times.
- **Overrides**: Instances can override specific component values.
- **Inheritance**: Prefabs can inherit from other prefabs (prefab hierarchy).
- **Live updates**: In some systems, changing a prefab updates all instances that haven't overridden that component.
- **Editor integration**: Prefabs are the foundation of visual scene editors.

### Use Cases

- **Game objects**: Enemy types, weapon types, particle configurations.
- **UI widgets**: Button templates, panel layouts.
- **Level design**: Place prefab instances in a scene editor.
- **Networking**: Spawn replicated entities from a shared template ID.

## Current Workarounds in RedLilium

RedLilium provides `spawn_scene()` for glTF-sourced entity creation, but no general-purpose prefab system:

```rust
// Workaround 1: Factory functions
fn spawn_spaceship(world: &mut World, position: Vec3) -> Entity {
    let entity = world.spawn();
    world.insert(entity, Transform::from_translation(position)).unwrap();
    world.insert(entity, GlobalTransform::default()).unwrap();
    world.insert(entity, Visibility::VISIBLE).unwrap();
    world.insert(entity, Health { current: 100.0, max: 100.0 }).unwrap();
    entity
}

// Workaround 2: Scene-based templates
let scene = load_gltf(include_bytes!("spaceship.glb"), &layouts)?;
let roots = spawn_scene(&mut world, &scene.scenes[0]);
```

Limitations of workarounds:
- Factory functions are code-only — not data-driven or editor-friendly.
- Scene spawning is tied to glTF format and file structure.
- No override mechanism — must modify components after spawning.
- No inheritance between templates.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **flecs** | First-class prefabs with `is_a()` relationship, component inheritance, override tracking, nested prefabs |
| **Bevy** | `SceneBundle` for scene instancing; `DynamicScene` for runtime serialization; no native prefab-with-overrides |
| **Unity DOTS** | `Prefab` → `Entity` conversion via `SubScene`, `EntityManager.Instantiate()`, baking pipeline |
| **EnTT** | `entt::prototype` for entity prototyping (user library, not core) |
| **hecs** | No built-in prefabs |
| **Legion** | No built-in prefabs |
| **Shipyard** | No built-in prefabs |
