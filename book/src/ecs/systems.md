# Systems and Scheduling

Systems are the logic that operates on components each frame. RedLilium provides several ways to define systems, from trait implementations to lightweight function syntax.

## The System Trait

```rust
pub trait System: Send + Sync + 'static {
    type Result: Send + Sync + 'static;

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError>;
}
```

A system receives a `SystemContext` that provides access to component storages, resources, and the compute pool.

### Example: Trait-Based System

```rust
struct MovementSystem;

impl System for MovementSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        ctx.lock::<(Write<Position>, Read<Velocity>)>()
            .for_each(|(pos, vel): (&mut Position, &Velocity)| {
                pos.x += vel.x;
                pos.y += vel.y;
            });
        Ok(())
    }
}
```

## Function Systems

For simple per-entity logic, use function-based systems to avoid boilerplate:

```rust
// Per-entity function system
fn movement((pos, vel): (&mut Position, &Velocity)) {
    pos.x += vel.x;
    pos.y += vel.y;
}

// Register with a container
container.add_fn::<(Write<Position>, Read<Velocity>), _>(movement);
```

### Parallel Function Systems

Run the per-entity function across multiple threads:

```rust
container.add_par_fn::<(Write<Position>, Read<Velocity>), _>(movement);
```

On WASM, `par_fn` automatically falls back to single-threaded execution.

### Raw Function Systems

When you need access to entire storages (not per-entity), use raw function systems:

```rust
fn apply_gravity((mut velocities,): (RefMut<Velocity>,)) {
    for (_, vel) in velocities.iter_mut() {
        vel.y -= 9.81;
    }
}

container.add_fn_raw::<(Write<Velocity>,), _>(apply_gravity);
```

## Exclusive Systems

Exclusive systems receive `&mut World` directly, acting as a scheduling barrier. No other systems run concurrently with an exclusive system:

```rust
struct SpawnWaveSystem {
    wave: u32,
}

impl ExclusiveSystem for SpawnWaveSystem {
    fn run(&mut self, world: &mut World, commands: &CommandCollector) {
        for _ in 0..self.wave {
            world.spawn_with((
                Position { x: 0.0, y: 0.0 },
                Enemy,
            ));
        }
        self.wave += 1;
    }
}

container.add_exclusive(SpawnWaveSystem { wave: 1 });
```

## Deferred Commands

Systems that don't have exclusive access can still perform structural changes (spawn, despawn, insert, remove) through deferred commands:

```rust
impl System for CleanupSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        let mut to_despawn = Vec::new();

        ctx.lock::<(Read<Health>,)>()
            .for_each(|(health,): (&Health,)| {
                // Collect entities to remove (can't mutate world here)
            });

        // Defer the mutations to after all systems complete
        ctx.commands(|world| {
            for entity in to_despawn {
                world.despawn(entity);
            }
        });

        Ok(())
    }
}
```

## SystemsContainer

Systems are grouped into a `SystemsContainer` that manages their execution order:

```rust
let mut container = SystemsContainer::new();

container.add(MovementSystem);
container.add(GravitySystem);
container.add_exclusive(SpawnWaveSystem { wave: 1 });
```

### Ordering

By default, systems with non-conflicting data access can run in parallel. Use edges to enforce ordering:

```rust
// GravitySystem runs before MovementSystem
container.add_edge::<GravitySystem, MovementSystem>().unwrap();

// Multiple edges at once
container.add_edges(&[
    Edge::new::<InputSystem, MovementSystem>(),
    Edge::new::<MovementSystem, CollisionSystem>(),
]).unwrap();
```

### System Sets

Group related systems and order them as a unit:

```rust
struct PhysicsSet;
impl SystemSet for PhysicsSet {}

struct RenderSet;
impl SystemSet for RenderSet {}

container.add_to_set::<GravitySystem, PhysicsSet>().unwrap();
container.add_to_set::<CollisionSystem, PhysicsSet>().unwrap();

// All physics systems run before all render systems
container.add_set_edge::<PhysicsSet, RenderSet>().unwrap();
```

### Conditions

Condition systems gate the execution of dependent systems:

```rust
struct IsPlayingCondition;

impl System for IsPlayingCondition {
    type Result = Condition;

    fn run(&self, ctx: &SystemContext) -> Result<Condition, SystemError> {
        ctx.lock::<(Res<GameState>,)>()
            .execute(|(state,)| {
                if *state == GameState::Playing {
                    Ok(Condition::True(()))
                } else {
                    Ok(Condition::False)
                }
            })
    }
}

container.add_condition(IsPlayingCondition);
container.add(PlayerMovement);
container.add_edge::<IsPlayingCondition, PlayerMovement>().unwrap();
// PlayerMovement only runs when IsPlayingCondition returns True
```

## Schedules

Schedules organize system containers into named phases that run in a fixed order each frame:

```rust
let mut schedules = Schedules::new();

// Built-in phases (in execution order):
// Startup     -- runs once at initialization
// PreUpdate   -- input handling, state transitions
// FixedUpdate -- fixed timestep (physics)
// Update      -- main game logic
// PostUpdate  -- cleanup, transform propagation

schedules.get_mut::<Update>().add(MovementSystem);
schedules.get_mut::<PostUpdate>().add(UpdateGlobalTransforms);
```

### Running

```rust
let runner = EcsRunner::single_thread();

// Run startup systems once
schedules.run_startup(&mut world, &runner);

// Run one frame (call this in your game loop)
let delta_time = 1.0 / 60.0;
schedules.run_frame(&mut world, &runner, delta_time);
```

`run_frame` executes: PreUpdate -> (state transitions if any) -> FixedUpdate (0..N times) -> Update -> PostUpdate.

### Custom Schedule Labels

```rust
struct MyCustomPhase;
impl ScheduleLabel for MyCustomPhase {}

schedules.get_mut::<MyCustomPhase>().add(MySystem);
schedules.run_schedule::<MyCustomPhase>(&mut world, &runner);
```

## Runners

```rust
// Single-threaded (works on all platforms including WASM)
let runner = EcsRunner::single_thread();

// Multi-threaded (native only)
#[cfg(not(target_arch = "wasm32"))]
let runner = EcsRunner::multi_thread(4); // 4 worker threads
```

The runner handles system parallelism, command application, and observer flushing automatically.

## System Results

Systems can produce results that other systems read:

```rust
struct RaycastSystem;

impl System for RaycastSystem {
    type Result = Vec<HitResult>;

    fn run(&self, ctx: &SystemContext) -> Result<Vec<HitResult>, SystemError> {
        let hits = vec![]; // ... perform raycasts
        Ok(hits)
    }
}

struct DamageSystem;

impl System for DamageSystem {
    type Result = ();

    fn run(&self, ctx: &SystemContext) -> Result<(), SystemError> {
        if let Some(hits) = ctx.system_result::<RaycastSystem>() {
            for hit in hits {
                // apply damage
            }
        }
        Ok(())
    }
}

container.add(RaycastSystem);
container.add(DamageSystem);
container.add_edge::<RaycastSystem, DamageSystem>().unwrap();
```

## One-Shot Execution

Run a system outside of the schedule for testing or initialization:

```rust
let result = run_system_once(&world, &compute_pool, &io_runtime, &MySystem);
run_exclusive_system_once(&mut world, &mut MyExclusiveSystem);
```
