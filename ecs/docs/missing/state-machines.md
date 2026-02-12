# State Machines

## What Are They?

A built-in state machine system manages application-level or game-level states (menu, playing, paused, loading) within the ECS. Systems can be configured to run only in specific states, and transitions between states trigger setup/teardown logic automatically.

```rust
// Bevy-style states (not available in RedLilium)
#[derive(States, Debug, Clone, PartialEq, Eq, Hash, Default)]
enum GameState {
    #[default]
    Menu,
    Loading,
    Playing,
    Paused,
}

app.init_state::<GameState>()
    .add_systems(OnEnter(GameState::Playing), setup_game)
    .add_systems(OnExit(GameState::Playing), cleanup_game)
    .add_systems(Update, player_movement.run_if(in_state(GameState::Playing)))
    .add_systems(Update, menu_ui.run_if(in_state(GameState::Menu)));

// Transition
fn start_game(mut next_state: ResMut<NextState<GameState>>) {
    next_state.set(GameState::Playing);
}
```

### Key Properties

- **Conditional system execution**: Systems only run in their associated state.
- **Enter/exit hooks**: Run setup on state entry, cleanup on state exit.
- **Transition events**: Other systems can observe state changes.
- **Nested states**: Sub-states within a parent state (e.g., Playing → Combat, Playing → Exploration).
- **Computed states**: Derived states that automatically update based on other states.

### Use Cases

- **Game flow**: Menu → Loading → Playing → Paused → GameOver.
- **Level management**: Setup entities on level enter, despawn on level exit.
- **UI screens**: Show/hide UI panels based on current state.
- **Network**: Connected, Disconnected, Reconnecting states.
- **Editor modes**: Select, Move, Rotate, Scale tool modes.

## Current Approach in RedLilium

Without built-in states, you can use a resource + manual checks:

```rust
#[derive(Debug, Clone, PartialEq)]
enum GameState { Menu, Playing, Paused }

// Store as resource
world.insert_resource(GameState::Menu);

// Systems check state manually
impl System for PlayerMovementSystem {
    async fn run(&self, ctx: &SystemContext<'_>) {
        ctx.lock::<(Res<GameState>, Write<Transform>)>()
            .execute(|world, (state, mut transforms)| {
                if *state != GameState::Playing {
                    return; // Skip when not playing
                }
                // movement logic...
            })
            .await;
    }
}
```

Limitations:
- Every state-dependent system needs an `if` check — boilerplate.
- No automatic enter/exit hooks — must track previous state manually.
- Systems still get scheduled even when their state isn't active (wasted scheduling overhead).
- No framework support for transitions, sub-states, or computed states.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `States` derive macro, `OnEnter`/`OnExit`/`OnTransition` schedules, `run_if(in_state())`, `NextState<S>` resource, sub-states, computed states |
| **flecs** | Pipeline phases + system enable/disable; no formal state machine but systems can be toggled dynamically |
| **Unity DOTS** | Custom `SystemGroup` enable/disable, `World.GetExistingSystem<T>().Enabled`, no built-in state enum |
| **EnTT** | No built-in states |
| **hecs** | No built-in states |
| **Legion** | No built-in states |
| **Shipyard** | No built-in states |
