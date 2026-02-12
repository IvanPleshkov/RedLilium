# Observers and Hooks

## What Are They?

Observers (also called hooks or triggers) are reactive callbacks that fire automatically when specific ECS events occur — a component is added, removed, or modified on an entity. Instead of polling for changes every frame, observers let you respond to structural changes immediately.

```rust
// Bevy-style observer (not available in RedLilium)
app.add_observer(on_add::<RigidBody3D>, |trigger: Trigger<OnAdd, RigidBody3D>, mut commands: Commands| {
    // Automatically create physics body when RigidBody3D is added
    let entity = trigger.entity();
    commands.entity(entity).insert(PhysicsHandle::new());
});

// flecs-style hook
world.component::<Position>()
    .on_set(|entity, pos| {
        println!("Position changed on {:?}: {:?}", entity, pos);
    });
```

### Common Event Types

| Event | When It Fires |
|-------|--------------|
| `OnAdd` | Component is inserted on an entity for the first time |
| `OnInsert` | Component value is set (includes re-insertion) |
| `OnRemove` | Component is about to be removed from an entity |
| `OnReplace` | Component value is replaced with a new value |

### Use Cases

- **Automatic setup**: When `RigidBody3D` is added, create the physics engine handle.
- **Cleanup**: When `AudioSource` is removed, stop the playing sound.
- **Synchronization**: When `Transform` changes, mark spatial index as dirty.
- **Cascading updates**: When `Parent` is removed, also remove `Children` entry from old parent.
- **Validation**: When `Health` is set, clamp it to `[0, max]`.

## Current Workarounds in RedLilium

Without observers, you must poll for changes each frame:

```rust
// Poll-based approach (current)
impl System for PhysicsSetupSystem {
    async fn run(&self, ctx: &SystemContext<'_>) {
        ctx.lock::<(Read<RigidBody3D>, Write<PhysicsHandle>)>()
            .execute(|world, (bodies, mut handles)| {
                // Check every entity with RigidBody3D
                for entity in world.added::<RigidBody3D>(self.last_tick).iter(world) {
                    // Create handle for newly added bodies
                    if !handles.contains(entity) {
                        world.insert(entity, PhysicsHandle::new()).unwrap();
                    }
                }
            })
            .await;
    }
}
```

This works but has drawbacks:
- Must run every frame even when no entities changed
- Change detection has one-frame delay
- Structural changes (insert/remove) within the polling system require deferred commands

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `Observer` system type, `Trigger<Event, Component>`, `OnAdd`/`OnInsert`/`OnRemove` events, custom trigger events |
| **flecs** | `on_add`, `on_set`, `on_remove` component hooks, observers with filter queries, monitor queries |
| **Unity DOTS** | `ISystemStateComponentData` for cleanup tracking, `EntityCommandBuffer` playback triggers |
| **EnTT** | Signals: `on_construct`, `on_update`, `on_destroy` per component type via `registry.on_construct<T>().connect(...)` |
| **hecs** | No built-in hooks |
| **Legion** | No built-in hooks |
| **Shipyard** | No built-in hooks |
