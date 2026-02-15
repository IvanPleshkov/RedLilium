# System Piping

## What Is It?

System piping connects the output of one system to the input of another, creating a data-flow chain. Instead of communicating through shared resources or events, piped systems pass data directly — the return value of one system becomes the parameter of the next.

```rust
// Bevy-style system piping (not available in RedLilium)
fn parse_config(input: In<String>) -> Result<Config, ConfigError> {
    toml::from_str(&input.0)
}

fn apply_config(In(config): In<Result<Config, ConfigError>>, mut settings: ResMut<Settings>) {
    match config {
        Ok(c) => settings.apply(c),
        Err(e) => eprintln!("Config error: {}", e),
    }
}

// Chain them
app.add_systems(Startup, parse_config.pipe(apply_config));
```

### Key Properties

- **Type-safe**: The output type of system A must match the input type of system B.
- **Composable**: Pipes can be chained: `a.pipe(b).pipe(c)`.
- **Error propagation**: `Result` types can be piped through error handlers.
- **Adapter patterns**: Common adapters like `.pipe(system_adapter::ignore)` to discard output.

### Use Cases

- **Error handling**: `fallible_system.pipe(error_handler)`.
- **Data transformation**: `load_raw_data.pipe(parse_data).pipe(validate_data)`.
- **Conditional execution**: `check_condition.pipe(run_if_true)`.
- **Logging/debugging**: `any_system.pipe(log_output)`.

### Comparison With Other Communication Methods

| Method | Coupling | Timing | Type Safety |
|--------|---------|--------|-------------|
| System piping | Direct (A → B) | Same frame, sequential | Compile-time checked |
| Events | Indirect (broadcast) | Same or next frame | Runtime type match |
| Resources | Indirect (shared state) | Any order, race possible | Runtime borrow checks |

## Current Approach in RedLilium

Systems communicate through resources and events — no direct piping:

```rust
// Communication via resource
impl System for ConfigLoader {
    async fn run(&self, ctx: &SystemContext<'_>) {
        ctx.lock::<(ResMut<ConfigState>,)>()
            .execute(|_world, (mut state,)| {
                state.config = load_config();
            })
            .await;
    }
}

impl System for ConfigApplier {
    async fn run(&self, ctx: &SystemContext<'_>) {
        ctx.lock::<(Res<ConfigState>, ResMut<Settings>)>()
            .execute(|_world, (state, mut settings)| {
                settings.apply(&state.config);
            })
            .await;
    }
}
```

This works but requires an intermediate resource type and explicit ordering edges.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `system_a.pipe(system_b)`, `In<T>` input parameter, built-in adapters (`ignore`, `unwrap`, `dbg`) |
| **flecs** | No direct piping (systems communicate via components) |
| **Unity DOTS** | No direct piping (systems communicate via components/buffers) |
| **EnTT** | No system framework |
| **hecs** | No system framework |
| **Legion** | No built-in piping |
| **Shipyard** | No built-in piping |

Note: System piping is primarily a Bevy feature. Most ECS libraries use shared components, resources, or events for inter-system communication.
