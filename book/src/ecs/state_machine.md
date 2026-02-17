# State Machine

RedLilium includes a built-in application state machine for managing game states like menus, gameplay, and pause screens.

## Defining States

Implement the `States` trait on an enum:

```rust
#[derive(Clone, PartialEq, Eq, Hash)]
enum GameState {
    Menu,
    Playing,
    Paused,
    GameOver,
}

impl States for GameState {}
```

## Initialization

Register the state with your schedules:

```rust
let mut schedules = Schedules::new();
schedules.init_state(&mut world, GameState::Menu);
```

This inserts:
- `State<GameState>` resource (current state)
- `NextState<GameState>` resource (pending transition)
- `ApplyStateTransition<GameState>` system in PreUpdate

## Transitioning States

From any system or exclusive system, set the next state:

```rust
impl System for MenuSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        ctx.lock::<(ResMut<NextState<GameState>>,)>()
            .execute(|(mut next_state,)| {
                // Queue a transition to Playing
                next_state.set(GameState::Playing);
            });
        Ok(())
    }
}
```

The transition is applied during the next PreUpdate phase.

## State-Driven Schedules

Run systems when entering or exiting a specific state:

```rust
schedules.on_enter(GameState::Playing).add(SetupLevelSystem);
schedules.on_enter(GameState::Playing).add(SpawnPlayerSystem);

schedules.on_exit(GameState::Playing).add(CleanupLevelSystem);
schedules.on_exit(GameState::Playing).add(DespawnEnemiesSystem);
```

These run once during the frame when the transition occurs.

## State Conditions

Gate systems to only run while in a specific state using condition macros:

```rust
// Define conditions
state_condition!(WhenPlaying, GameState, GameState::Playing);
state_condition!(WhenPaused, GameState, GameState::Paused);
state_enter_condition!(OnEnterMenu, GameState, GameState::Menu);
state_exit_condition!(OnExitMenu, GameState, GameState::Menu);
```

Use them in a schedule:

```rust
let update = schedules.get_mut::<Update>();

update.add_condition(WhenPlaying);
update.add(PlayerMovement);
update.add(EnemyAI);
update.add_edge::<WhenPlaying, PlayerMovement>().unwrap();
update.add_edge::<WhenPlaying, EnemyAI>().unwrap();
// PlayerMovement and EnemyAI only run while in GameState::Playing
```

## Reading Current State

```rust
ctx.lock::<(Res<State<GameState>>,)>()
    .execute(|(state,)| {
        match **state {
            GameState::Menu => { /* ... */ }
            GameState::Playing => { /* ... */ }
            _ => {}
        }
    });
```

## Complete Example

```rust
#[derive(Clone, PartialEq, Eq, Hash)]
enum AppState { Loading, Menu, InGame }
impl States for AppState {}

state_condition!(WhenInGame, AppState, AppState::InGame);

fn setup(world: &mut World, schedules: &mut Schedules) {
    schedules.init_state(world, AppState::Loading);

    // Startup: load assets
    schedules.get_mut::<Startup>().add(LoadAssetsSystem);

    // On entering Menu: show UI
    schedules.on_enter(AppState::Menu).add(ShowMenuSystem);

    // On entering InGame: spawn world
    schedules.on_enter(AppState::InGame).add(SpawnWorldSystem);
    schedules.on_exit(AppState::InGame).add(DespawnWorldSystem);

    // Update: game logic only runs in InGame
    let update = schedules.get_mut::<Update>();
    update.add_condition(WhenInGame);
    update.add(PlayerSystem);
    update.add_edge::<WhenInGame, PlayerSystem>().unwrap();
}
```
