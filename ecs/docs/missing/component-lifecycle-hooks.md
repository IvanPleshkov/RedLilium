# Component Lifecycle Hooks

## What Are They?

Component lifecycle hooks are callbacks registered on a **component type** that fire automatically whenever that component is added to, removed from, or modified on any entity. Unlike observers (which are user-defined reactive systems), lifecycle hooks are intrinsic to the component type itself.

```rust
// flecs-style lifecycle hooks (not available in RedLilium)
world.component::<Transform>()
    .on_add(|entity, transform| {
        println!("Transform added to {:?}", entity);
    })
    .on_remove(|entity, transform| {
        println!("Transform removed from {:?}", entity);
    })
    .on_set(|entity, transform| {
        // Mark spatial index as dirty
        spatial_index.invalidate(entity);
    });
```

### Hook Types

| Hook | Trigger | Typical Use |
|------|---------|-------------|
| `on_add` | Component inserted (first time) | Initialize derived state |
| `on_set` | Component value written | Sync external systems |
| `on_remove` | Component about to be removed | Cleanup resources |
| `on_move` | Entity moved between archetypes | Update archetype caches |

### Difference From Observers

| Aspect | Lifecycle Hooks | Observers |
|--------|----------------|-----------|
| Registration | On the component type | On the world/app |
| Scope | All entities with this component | Can be filtered |
| Execution | Immediate (during insert/remove) | May be deferred |
| Purpose | Type-intrinsic behavior | User-defined reactions |
| Multiple | Usually one per type | Many per event type |

### Use Cases

- **Resource management**: Release GPU handles when a mesh component is removed.
- **Index maintenance**: Update spatial hash when transform changes.
- **Validation**: Clamp health to [0, max] whenever it's set.
- **Parent-child consistency**: When Parent is removed, update the old parent's Children list.
- **Physics sync**: Create/destroy physics bodies when RigidBody is added/removed.

## Current Approach in RedLilium

No lifecycle hooks exist. Changes are detected via polling with change tracking:

```rust
// Poll for newly added components each frame
for entity in world.added::<RigidBody3D>(last_tick).iter(world) {
    // Set up physics body
}

// Poll for removed components — not directly supported
// Must track "seen" set and diff against current entities
```

The hierarchy system (`set_parent`, `remove_parent`) manually maintains Parent/Children consistency in imperative code rather than through hooks.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **flecs** | `on_add`, `on_set`, `on_remove` per-component hooks, executed synchronously during structural changes |
| **EnTT** | `on_construct<T>`, `on_update<T>`, `on_destroy<T>` signals via `registry.on_construct<T>().connect(callback)` |
| **Bevy** | Component hooks via `Component` trait: `on_add`, `on_insert`, `on_replace`, `on_remove` (added in 0.14+) |
| **Unity DOTS** | `ISystemStateComponentData` for cleanup tracking; no per-component hooks |
| **hecs** | No lifecycle hooks |
| **Legion** | No lifecycle hooks |
| **Shipyard** | No lifecycle hooks |
