# Resources and Events

## Resources

Resources are singleton values stored in the world, accessible from any system. Use them for global state like time, configuration, or shared data structures.

### Inserting Resources

```rust
struct GameConfig {
    gravity: f32,
    max_enemies: u32,
}

world.insert_resource(GameConfig {
    gravity: -9.81,
    max_enemies: 100,
});
```

### Accessing Resources

```rust
// Direct access (outside systems)
let config = world.resource::<GameConfig>();
println!("gravity: {}", config.gravity);

let mut config = world.resource_mut::<GameConfig>();
config.max_enemies = 200;

// Check existence
if world.has_resource::<GameConfig>() { /* ... */ }

// Non-panicking variants
if let Some(config) = world.try_resource::<GameConfig>() { /* ... */ }
```

### In Systems

Use `Res<T>` for shared access and `ResMut<T>` for exclusive access:

```rust
impl System for GravitySystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        ctx.lock::<(Write<Velocity>, Res<GameConfig>)>()
            .for_each(|(vel, config): (&mut Velocity, &GameConfig)| {
                vel.y += config.gravity;
            });
        Ok(())
    }
}
```

### Removing Resources

```rust
let old_config: Option<GameConfig> = world.remove_resource::<GameConfig>();
```

### Main Thread Resources

For non-`Send` types (window handles, GPU contexts), use main-thread resources:

```rust
world.insert_main_thread_resource(window);

// Access (must be on main thread, or dispatched via MainThreadRes in systems)
unsafe {
    let window = world.main_thread_resource::<Window>();
}
```

## Time

The `Time` resource is automatically inserted and updated by the schedule runner:

```rust
ctx.lock::<(Res<Time>,)>()
    .execute(|(time,)| {
        let dt: f32 = time.delta_f32();       // frame delta as f32
        let dt: f64 = time.delta();            // frame delta as f64
        let elapsed: f64 = time.elapsed();     // total elapsed time
        let fixed_dt: f64 = time.fixed_delta(); // fixed timestep value
    });
```

## Events

Events provide a double-buffered communication channel between systems. Events from the current frame are available until the next frame's update.

### Setup

```rust
world.add_event::<DamageEvent>();
```

This inserts an `Events<DamageEvent>` resource and registers an `EventUpdateSystem` that swaps buffers each frame.

### Sending Events

```rust
#[derive(Clone)]
struct DamageEvent {
    target: Entity,
    amount: f32,
}

impl System for CombatSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        ctx.lock::<(ResMut<Events<DamageEvent>>,)>()
            .execute(|(mut events,)| {
                events.send(DamageEvent {
                    target: enemy,
                    amount: 25.0,
                });
            });
        Ok(())
    }
}
```

### Reading Events

```rust
impl System for HealthSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        ctx.lock::<(Res<Events<DamageEvent>>, Write<Health>)>()
            .execute(|(events, mut health)| {
                // Iterate events from both current and previous frame
                for event in events.iter() {
                    if let Some(hp) = health.get_mut(event.target) {
                        hp.current -= event.amount;
                    }
                }

                // Or only current frame events
                for event in events.iter_current() {
                    // ...
                }
            });
        Ok(())
    }
}
```

Events are automatically double-buffered: `iter()` yields events from both the current and previous frame, while `iter_current()` yields only the current frame's events. The buffers are swapped at the beginning of each frame by the built-in `EventUpdateSystem`.
