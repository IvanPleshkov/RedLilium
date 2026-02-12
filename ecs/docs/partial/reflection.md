# Reflection and Type Registry (Partial)

## What Is a Type Registry?

A type registry is a runtime catalog of types that enables dynamic operations: serialization, deserialization, cloning, comparison, and UI generation — all without knowing concrete types at compile time. It bridges the gap between Rust's static type system and the dynamic needs of editors, save systems, and networking.

```rust
// Bevy-style type registry (not available in RedLilium)
let mut registry = TypeRegistry::new();
registry.register::<Transform>();
registry.register::<Health>();

// Dynamic operations
let type_info = registry.get(TypeId::of::<Transform>()).unwrap();
let serializer = type_info.get::<ReflectSerialize>().unwrap();
let json = serializer.serialize(&transform);

// Clone via reflection
let cloned = type_info.get::<ReflectClone>().unwrap().clone_value(&transform);
```

## What RedLilium Has

RedLilium provides **component-level reflection** via the `Component` trait and inspector metadata:

```rust
use redlilium_ecs::Component;

#[derive(Component)]
struct Health {
    current: f32,
    max: f32,
}
// Generates: NAME = "Health", inspect_ui() for egui rendering

// Registration levels
world.register_component::<InternalFlag>();           // Storage only
world.register_inspector::<Camera>();                  // Inspector visible
world.register_inspector_default::<Transform>();       // Inspector + add via UI

// Type-erased inspector operations
let components = world.inspectable_components_of(entity);  // -> ["Transform", "Camera"]
let addable = world.addable_components_of(entity);         // -> ["PointLight", "Visibility"]
world.inspect_by_name(entity, "Transform", &mut ui);       // Render UI by name
world.remove_by_name(entity, "Health");                     // Remove by name
world.insert_default_by_name(entity, "Visibility");         // Add default by name
```

The `InspectorEntry` stores type-erased function pointers:

```rust
struct InspectorEntry {
    has_fn: fn(&World, Entity) -> bool,
    inspect_fn: fn(&mut World, Entity, &mut Ui) -> bool,
    remove_fn: fn(&mut World, Entity) -> bool,
    insert_default_fn: Option<fn(&mut World, Entity)>,
}
```

## Why It's Partial

| Aspect | Full Type Registry | RedLilium |
|--------|-------------------|-----------|
| Type catalog | All registered types with metadata | Only inspector-registered components |
| Serialization | `ReflectSerialize` / `ReflectDeserialize` | No serialization support |
| Dynamic cloning | `ReflectClone` / `ReflectFromReflect` | No dynamic cloning |
| Field access | Named field get/set by string | No field-level access |
| Enum reflection | Variant names, values, iteration | No enum reflection |
| Type path | Fully qualified type path string | Only `Component::NAME` (short name) |
| Cross-crate | Registry shared across plugins | Inspector entries per-world |
| Trait registration | Register any trait impl dynamically | Only Component + Default |
| Nested reflection | Recursive field inspection | Only top-level `inspect_ui()` |
| Format-agnostic | JSON, RON, binary, etc. | egui UI only |

The current reflection covers the **inspector use case** well — you can browse entities, view/edit component fields via egui, and add/remove components at runtime. However, it does not support:

- **Serialization**: Cannot save/load entity data to any format.
- **Networking**: Cannot replicate component data without knowing concrete types.
- **Undo/redo**: Cannot snapshot and restore component values generically.
- **Scripting**: Cannot access component fields from a scripting language.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `bevy_reflect` crate: `#[derive(Reflect)]`, `TypeRegistry`, `ReflectSerialize`, `ReflectDefault`, `ReflectComponent`, field-level access, enum reflection, path-based access (`"field.nested.value"`) |
| **flecs** | Built-in reflection with `ecs_struct`, `ecs_member`, JSON serialization, REST API for live inspection |
| **Unity DOTS** | Full C# reflection, `[GenerateAuthoringComponent]`, serialized fields |
| **EnTT** | `entt::meta` runtime reflection, custom meta factories |
| **Legion** | `World::pack`/`unpack` with serde-based serialization |
| **hecs** | Minimal — `column` access for raw bytes, no built-in reflection |
| **Shipyard** | No built-in reflection |
