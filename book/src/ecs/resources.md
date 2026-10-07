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
        let dt: f32 = time.delta_f32();        // fixed game step in FixedUpdate,
        let dt: f64 = time.delta();            // real frame delta otherwise
        let elapsed: f64 = time.elapsed();     // total real elapsed time
        let fixed_dt: f64 = time.fixed_delta(); // constant step in game seconds
    });
```

### Gameplay time scale

`GameTime::set_scale` controls both the game clock and the stock `FixedUpdate`
schedule. The runtime inserts `GameTime` and `RealTime` at boot. A headless or
custom host opts in by inserting them before calling `Schedules::run_frame`:

```rust
world.insert_resource(GameTime::default());
world.insert_resource(RealTime::default());
world.resource_mut::<GameTime>().set_scale(0.25);
```

The fixed accumulator receives scaled game time, while the fixed timestep stays
constant in game seconds. With a 1/128-second fixed step and 64 render frames per
real second, scales 1, 0.25, and 0 execute 128, 32, and 0 steps respectively.
`StepPhysics2D` / `StepPhysics3D` in `FixedUpdate` and their stock interpolation
systems follow this clock automatically. `Time::fixed_alpha()` uses the same
scaled remainder; at scale zero it holds the current interpolated pose.
Restoring normal speed preserves that remainder and never catches up the real
time omitted by slow-motion or pause. The existing `max_fixed_steps` limit still
drops excess simulation debt during a long stall or fast-forward; `GameTime`
itself reports the full scaled frame duration, including dropped debt.

Choose the clock by the work being timed:

| Work | Clock |
|------|-------|
| Fixed thrust, physics, bot timers, hitstun, contact rearming | `Time::delta()` or `Time::fixed_delta()` inside `FixedUpdate` |
| Variable-step gameplay in `Update` | `GameTime::delta()` |
| Slow-motion duration, real-time UI | `RealTime::delta()` / `RealTime::elapsed()` in `Update` |
| Frame timing and input collection | Unscaled `Time::frame_delta()` |

`GameTime::delta()` is always the scaled **frame** delta, including inside fixed
systems. Do not apply it once per fixed step or multiply the fixed delta by the
scale again. `Time::delta()` outside `FixedUpdate`, `Time::elapsed()`, and
`Time::smoothed_frame_delta()` retain their unscaled behavior. `PreUpdate`,
`Update`, and `PostUpdate` keep running at scale zero; the host can keep rendering
and collecting input. Worlds without `GameTime` retain normal-speed simulation.
Calling a fixed schedule manually does not perform clock accumulation.

The scale is sampled **after `PreUpdate` and state transitions**, when both
clocks advance. Changes in those phases affect this frame. A hit accepted in
`FixedUpdate` can call `set_scale`, but the change takes effect next frame; all
remaining fixed steps this frame use the already sampled delta. Changes in
`Update` and `PostUpdate` likewise affect the next frame. `RealTime` in
`PreUpdate` still describes the previous frame; use `Time::frame_delta()` there
if a timer must update the scale before this frame's sample.

For a combat slow-motion envelope, record accepted hits in fixed gameplay and
advance the envelope in `Update` using `RealTime`. For example, hold 0.25× for
0.30 real seconds and smoothly return to 1× over 0.15 seconds. Define retriggers
in game code (restart or extend the envelope), and on reset clear the envelope
and call `set_scale(1.0)`. `GameTime::reset()` only zeroes its elapsed/delta clock;
it intentionally preserves the scale and does not reset the fixed accumulator.
Scales must be finite and non-negative; reverse simulation is unsupported.

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
