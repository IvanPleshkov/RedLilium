# Prefabs and Scene Spawning

## Prefabs

A `Prefab` is a portable snapshot of an entity tree that can be instantiated into any world, any number of times. Entity references within the prefab are automatically remapped on instantiation.

### Enabling Cloning

Before extracting prefabs, enable cloning for every component type that should be included:

```rust
world.enable_clone::<Transform>();
world.enable_clone::<GlobalTransform>();
world.enable_clone::<Visibility>();
world.enable_clone::<Name>();
world.enable_clone::<Health>();
world.enable_clone::<Weapon>();
```

Components without cloning enabled are silently skipped during extraction.

### Extracting a Prefab

```rust
// Build an entity tree
let root = world.spawn_with((
    Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    GlobalTransform::IDENTITY,
    Name::new("Soldier"),
));

let weapon = world.spawn_with((
    Transform::IDENTITY,
    GlobalTransform::IDENTITY,
    Name::new("Rifle"),
    Weapon { damage: 25.0 },
));

set_parent(&mut world, weapon, root);

// Extract the tree as a prefab
let prefab: Prefab = world.extract_prefab(root);
```

### Instantiating

```rust
// Spawn a copy of the prefab
let new_entities: Vec<Entity> = prefab.instantiate(&mut world);

// The first entity is the root
let new_root = new_entities[0];

// Spawn more copies
let another: Vec<Entity> = prefab.instantiate(&mut world);
let yet_another: Vec<Entity> = prefab.instantiate(&mut world);
```

Each instantiation creates fresh entities with new IDs. `Parent`/`Children` references and any `Entity` fields in components are remapped to point to the new copies.

### Storing Prefabs

Prefabs are `Clone`, so you can store them as resources for reuse:

```rust
struct PrefabLibrary {
    soldier: Prefab,
    vehicle: Prefab,
}

world.insert_resource(PrefabLibrary {
    soldier: world.extract_prefab(soldier_template),
    vehicle: world.extract_prefab(vehicle_template),
});

// Later, in a system:
ctx.commands(|world| {
    let lib = world.resource::<PrefabLibrary>();
    let soldier = lib.soldier.clone();
    let entities = soldier.instantiate(world);
});
```

### Entity Remapping

If your components store `Entity` references, implement `collect_entities` and `remap_entities` on the `Component` trait (or use the derive macro with `#[entity]` attribute) so the prefab system can correctly remap them:

```rust
#[derive(Component)]
struct FollowTarget {
    target: Entity,
}
```

## Scene Spawning

`spawn_scene` converts a `redlilium_core::scene::Scene` (loaded from glTF or built programmatically) into live entities:

```rust
use redlilium_ecs::spawn_scene;

let roots: Vec<Entity> = spawn_scene(&mut world, &scene);
```

For each `SceneNode`, the spawner creates an entity with:
- `Transform` (from node's local transform)
- `GlobalTransform::IDENTITY` (computed later by `UpdateGlobalTransforms`)
- `Visibility::VISIBLE`
- `Name` (if the node has a name)
- `Camera` (if the node references a camera)
- `Parent` / `Children` (hierarchy from the scene graph)

The returned vector contains the root-level entities.

### Combining with Prefabs

A common pattern is to load a scene, then extract prefabs from it for runtime instantiation:

```rust
// Load a glTF file
let doc = load_gltf(&bytes, &layouts)?;
let scene = &doc.scenes[0];

// Spawn once to create template entities
let roots = spawn_scene(&mut world, scene);

// Extract prefab from the first root
let prefab = world.extract_prefab(roots[0]);

// Despawn the template
despawn_recursive(&mut world, roots[0]);

// Now instantiate the prefab as many times as needed
for i in 0..10 {
    let entities = prefab.instantiate(&mut world);
    let root = entities[0];
    if let Some(t) = world.get_mut::<Transform>(root) {
        t.translation.x = i as f32 * 2.0;
    }
}
```

## Cloning Entity Trees

For one-off copies without creating a prefab:

```rust
// Clone a single entity (no children)
world.clone_entity(source, destination);

// Clone an entire tree (creates new entities for all descendants)
let new_entities: Vec<Entity> = world.clone_entity_tree(root);
```
