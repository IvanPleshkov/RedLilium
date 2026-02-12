# One-Shot Systems

## What Are They?

One-shot systems are systems that run **on demand** rather than every frame. They are registered but not scheduled — instead, they're invoked explicitly by other systems or game logic when needed. This is useful for event-driven logic, input handling, or infrequent operations that don't belong in the frame loop.

```rust
// Bevy-style one-shot system (not available in RedLilium)
let system_id = app.register_system(|mut commands: Commands, assets: Res<AssetServer>| {
    commands.spawn(SceneBundle {
        scene: assets.load("level_02.glb"),
        ..default()
    });
});

// Later, invoke it from another system
fn on_level_complete(mut commands: Commands) {
    commands.run_system(system_id);
}
```

### Use Cases

- **Level loading**: Trigger scene load when player reaches a checkpoint.
- **Input actions**: Run a system when a button is pressed, not every frame.
- **Editor commands**: Execute operations from UI buttons (undo, redo, save).
- **State transitions**: Run setup/teardown logic when entering/leaving a game state.
- **Callbacks**: Pass a system as a callback to be invoked later.

### Benefits Over Regular Systems

| Aspect | Regular System | One-Shot System |
|--------|---------------|----------------|
| Execution | Every frame | Only when invoked |
| CPU cost when idle | Still scheduled/checked | Zero (not in schedule) |
| Trigger | Automatic by scheduler | Explicit by caller |
| Parameters | Same every frame | Can vary per invocation |

## Current Approach in RedLilium

RedLilium has `run_system_blocking()` for running a single system outside the scheduler, but this is a low-level utility, not a one-shot system framework:

```rust
// Current approach — manual invocation outside runner
let system = MySetupSystem;
let resources = Resources::new();
run_system_blocking(&world, &resources, &system);

// Or: use events to trigger logic within the frame loop
// System checks for event, only does work when event exists
impl System for LevelLoadSystem {
    async fn run(&self, ctx: &SystemContext<'_>) {
        ctx.lock::<(Res<Events<LoadLevel>>,)>()
            .execute(|_world, (events,)| {
                for event in events.current() {
                    // Load the level
                }
            })
            .await;
    }
}
```

Limitations:
- `run_system_blocking()` doesn't integrate with the scheduler or commands.
- Event-based approach requires the system to run every frame and check for events.
- No way to dynamically register and invoke systems by ID from within another system.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `app.register_system()` → `SystemId`, `commands.run_system(id)`, `commands.run_system_with_input(id, data)` |
| **flecs** | Any system can be run manually with `ecs_run(world, system, delta_time)` |
| **Unity DOTS** | `World.GetExistingSystem<T>().Update()` for manual system invocation |
| **EnTT** | No system framework — functions called manually |
| **hecs** | No system framework — functions called manually |
| **Legion** | No built-in one-shot systems |
| **Shipyard** | No built-in one-shot systems |
