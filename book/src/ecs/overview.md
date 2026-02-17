# ECS Overview

The `redlilium-ecs` crate provides a custom Entity Component System designed around a unified scheduling model where ECS systems and async compute tasks share a single work-stealing thread pool.

## Key Ideas

- **Entities** are lightweight IDs (index + generation) that act as keys into component storage
- **Components** are plain Rust structs attached to entities
- **Systems** are functions that operate on components each frame
- **Resources** are singleton values shared across systems (e.g. time, input state, physics world)
- **Schedules** group systems into phases that run in a defined order each frame

## Architecture

```text
  ┌─────────────────────────────────────────────┐
  │                  Schedules                   │
  │  Startup → PreUpdate → FixedUpdate →        │
  │            Update → PostUpdate              │
  └──────────────────┬──────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   SystemsContainer  │
          │  (dependency graph) │
          └──────────┬──────────┘
                     │
         ┌───────────▼───────────┐
         │      EcsRunner        │
         │  single / multi-thread│
         └───────────┬───────────┘
                     │
     ┌───────────────▼───────────────┐
     │     Unified Thread Pool       │
     │  sync systems (high priority) │
     │  async compute (fills gaps)   │
     └──────────────────────────────┘
```

Sync systems borrow the `World` and run in parallel when their data access doesn't conflict. Async compute tasks own their data and yield at `.await` points, letting idle cores pick up background work (pathfinding, navmesh rebuilds, LOD computation) automatically.

## Cross-Platform

The API is identical on native and WebAssembly. On native, systems run in parallel across multiple threads. On WASM, everything runs single-threaded with cooperative scheduling -- no code changes required.

## Minimal Example

```rust
use redlilium_ecs::*;

// Define components
#[derive(Component)]
struct Position { x: f32, y: f32 }

#[derive(Component)]
struct Velocity { x: f32, y: f32 }

// Define a system
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

fn main() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Velocity>();

    // Spawn entities
    let entity = world.spawn();
    world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
    world.insert(entity, Velocity { x: 1.0, y: 0.5 }).unwrap();

    // Build schedule
    let mut schedules = Schedules::new();
    schedules.get_mut::<Update>().add(MovementSystem);

    // Run
    let runner = EcsRunner::single_thread();
    schedules.run_frame(&mut world, &runner, 1.0 / 60.0);
}
```

The following chapters cover each feature in depth.
