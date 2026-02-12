# Dynamic Components

## What Are They?

Dynamic components are component types defined at **runtime** rather than compile time. Instead of requiring a Rust struct known at compile time, a dynamic component is described by a schema (name, field types, size, alignment) and stored as raw bytes. This enables scripting languages, editors, and data-driven workflows to create new component types without recompiling.

```rust
// Bevy-style dynamic component (not available in RedLilium)
let mut builder = TypeBuilder::new("EnemyConfig");
builder.add_field::<f32>("speed");
builder.add_field::<i32>("health");
builder.add_field::<bool>("is_boss");

let component_id = world.register_dynamic_component(builder.build());

// Insert dynamic component via reflection
let mut data = DynamicStruct::new();
data.insert("speed", 5.0f32);
data.insert("health", 100i32);
data.insert("is_boss", false);

world.entity_mut(entity).insert_by_id(component_id, data);
```

### Use Cases

- **Scripting integration**: Lua/Python scripts define custom components without Rust compilation.
- **Editor workflows**: Artists add custom data fields to entities in a visual editor.
- **Hot reloading**: Change component schemas without restarting the application.
- **Data-driven design**: Component definitions loaded from configuration files.
- **Modding**: Game mods add new component types without engine recompilation.
- **Network replication**: Deserialize component types defined by a remote schema.

### Challenges

| Challenge | Description |
|-----------|-------------|
| **Type safety** | No compile-time guarantees — schema mismatches caught at runtime |
| **Performance** | Raw byte access, no monomorphization, potential layout overhead |
| **Interop** | Bridging dynamic components with typed Rust queries |
| **Storage** | Must handle arbitrary sizes and alignments |

## Current Approach in RedLilium

All components must be known at compile time as Rust types implementing `Component + Send + Sync + 'static`:

```rust
// All component operations require a concrete type T
world.register_component::<MyComponent>();
world.insert(entity, MyComponent { ... }).unwrap();
let comp = world.get::<MyComponent>(entity).unwrap();
```

The inspector provides name-based operations (`insert_default_by_name`, `remove_by_name`), but these only work with **pre-registered** Rust types — they don't support defining new types at runtime.

For scripting-like use cases, a workaround is a generic "data bag" component:

```rust
// Workaround: generic key-value component
#[derive(Component)]
struct ScriptData {
    values: HashMap<String, ScriptValue>,
}

enum ScriptValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    String(String),
}
```

This loses per-field type safety and ECS query efficiency.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `DynamicComponent`, `ComponentDescriptor`, `insert_by_id()`, `bevy_reflect` dynamic structs |
| **flecs** | Full runtime component registration: `ecs_component_init()` with name/size/alignment, runtime struct definition via `ecs_struct()` + `ecs_member()` |
| **Unity DOTS** | `DynamicComponentTypeHandle`, `EntityManager.AddComponent(entity, componentType)` with runtime `ComponentType` |
| **EnTT** | `entt::meta` for runtime type information, but storage is still typed |
| **hecs** | Column-level raw byte access for custom serialization, no dynamic types |
| **Legion** | Dynamic component registration via `ComponentMeta` and layout descriptions |
| **Shipyard** | No dynamic components |
