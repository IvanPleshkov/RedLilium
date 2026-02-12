# Serialization and Snapshots

## What Is It?

World serialization is the ability to save and restore the entire ECS state (or subsets of it) to/from a persistent format — JSON, binary, RON, etc. Snapshots capture entity-component data for save/load, undo/redo, networking, or hot-reloading.

```rust
// Bevy-style scene serialization (not available in RedLilium)
let scene = DynamicScene::from_world(&world);
let json = scene.serialize(&type_registry)?;

// Later: deserialize and spawn
let scene = DynamicScene::deserialize(&json, &type_registry)?;
scene.write_to_world(&mut world, &mut entity_map)?;
```

### Key Capabilities

| Capability | Description |
|-----------|-------------|
| **World save/load** | Persist entire world state to disk |
| **Entity snapshots** | Capture and restore individual entity state |
| **Scene export** | Export a subtree of entities as a reusable scene file |
| **Undo/redo** | Snapshot before each edit, restore on undo |
| **Networking** | Serialize delta changes for replication |
| **Hot reload** | Save state → recompile → restore state |

### Requirements

A serialization system typically needs:
1. **Type registry**: Know which types exist and how to serialize them.
2. **Component registry**: Map component TypeId to serializer/deserializer.
3. **Entity mapping**: Remap entity references (Parent, targets) after deserialization.
4. **Selective serialization**: Exclude transient components (GPU handles, caches).
5. **Format flexibility**: Support multiple output formats (JSON for debugging, binary for production).

## Current State in RedLilium

RedLilium has **POD byte serialization** via `bytemuck` for components implementing `Pod + Zeroable`:

```rust
// POD components can be serialized to raw bytes
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Transform { /* ... */ }

let bytes = bytemuck::bytes_of(&transform);  // &[u8]
let restored: Transform = bytemuck::from_bytes(bytes);
```

Limitations:
- Only works for `#[repr(C)]` POD types (no `String`, `Vec`, `Entity` references, `Option`).
- No entity reference remapping.
- No world-level serialization — must manually iterate and serialize each component.
- No format choice (raw bytes only).
- Components like `Name(String)`, `Children(Vec<Entity>)`, `Parent(Entity)` cannot be serialized.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `DynamicScene` + `bevy_reflect` + serde: serialize any reflected component to RON/JSON/binary, entity mapping, filtered scenes |
| **flecs** | Built-in JSON serialization via reflection, REST API for live state inspection, `ecs_to_json()` / `ecs_from_json()` |
| **Unity DOTS** | `SubScene` serialization, entity remapping, `BlobAssetStore` for binary data, baking pipeline |
| **Legion** | `World::pack()` / `World::unpack()` with serde support, custom serialization registry |
| **EnTT** | `entt::snapshot` / `entt::snapshot_loader` with customizable serialization backend |
| **hecs** | `hecs::serialize::column` module for raw column serialization, user-provided serializers |
| **Shipyard** | No built-in serialization |
