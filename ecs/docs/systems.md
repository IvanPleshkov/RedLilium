# Systems

Systems are the logic processors of the ECS. Each system is a synchronous function that reads and writes components/resources through a context object.

## The System Trait

All systems implement the `System` trait:

```rust
pub trait System: Send + Sync + 'static {
    type Result: Send + Sync + 'static;
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Self::Result;
}
```

Systems are:
- **Synchronous**: Run to completion, using the lock-execute pattern for component access.
- **Send + Sync**: Can run on any thread.
- **Stateless or self-contained**: Receive data through `SystemContext`, not constructor args (unless stored as struct fields).

## Basic System

```rust
struct MovementSystem;

impl System for MovementSystem {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Write<Position>, Read<Velocity>)>()
            .execute(|(mut positions, velocities)| {
                for (idx, pos) in positions.iter_mut() {
                    if let Some(vel) = velocities.get(idx) {
                        pos.x += vel.x;
                        pos.y += vel.y;
                    }
                }
            });
    }
}
```

## SystemContext

The `SystemContext` provides everything a system needs:

| Method | Purpose |
|--------|---------|
| `ctx.lock::<A>()` | Request component/resource access (returns `LockRequest`) |
| `ctx.compute()` | Access the compute pool for background tasks |
| `ctx.io()` | Access the IO runtime for async IO |
| `ctx.commands(closure)` | Queue a deferred world mutation |

## Lock-Execute Pattern

The core access pattern: component access is done through a **lock-execute** pattern that ensures locks are always released deterministically when the closure returns.

```rust
// The execute closure is FnOnce (synchronous) — locks are held only inside
ctx.lock::<(Write<Position>, Read<Velocity>)>()
    .execute(|(mut positions, velocities)| {
        // Locks are held here
        for (idx, pos) in positions.iter_mut() {
            if let Some(vel) = velocities.get(idx) {
                pos.x += vel.x;
            }
        }
        // Locks released when closure returns
    });
```

## Systems with Compute Tasks

Extract data in one phase, offload computation, then apply results:

```rust
struct PathfindSystem;

impl System for PathfindSystem {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        // Phase 1: Extract data (locks acquired and released)
        let nav_data = ctx.lock::<(Read<NavMesh>,)>()
            .execute(|(nav,)| {
                nav.iter().next().map(|(_, n)| n.clone())
            });

        // Phase 2: Offload heavy computation (no locks held)
        if let Some(data) = nav_data {
            let mut handle = ctx.compute().spawn(Priority::Low, |_cctx| async move {
                compute_paths(data)
            });
            let paths = ctx.compute().block_on(&mut handle);

            // Phase 3: Apply results via deferred command
            if let Some(paths) = paths {
                ctx.commands(move |world| {
                    // apply paths to world
                });
            }
        }
    }
}
```

## System with State

Systems can hold internal state as struct fields:

```rust
struct PeriodicSpawner {
    interval: f32,
    timer: std::sync::Mutex<f32>,
}

impl System for PeriodicSpawner {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        let dt = ctx.lock::<(Res<DeltaTime>,)>()
            .execute(|(dt,)| dt.0);

        let mut timer = self.timer.lock().unwrap();
        *timer += dt;

        if *timer >= self.interval {
            *timer -= self.interval;
            ctx.commands(|world| {
                let e = world.spawn();
                world.insert(e, Position { x: 0.0, y: 0.0 }).unwrap();
            });
        }
    }
}
```

## SystemResult — Inter-System Communication

Systems can return typed results that downstream systems can read via dependency edges:

```rust
struct ProducerSystem;

impl System for ProducerSystem {
    type Result = u32;
    fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> u32 {
        42
    }
}

struct ConsumerSystem;
impl System for ConsumerSystem {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        let value = *ctx.system_result::<ProducerSystem>();
        println!("Got: {}", value);
    }
}

// Registration with dependency edge:
let mut container = SystemsContainer::new();
container.add(ProducerSystem);
container.add(ConsumerSystem);
container.add_edge::<ProducerSystem, ConsumerSystem>().unwrap();
```

## Running Systems Outside a Runner

For testing or one-off execution:

```rust
use redlilium_ecs::{run_system_blocking, ComputePool, IoRuntime};

let world = World::new();
let compute = ComputePool::new(IoRuntime::new());
let io = IoRuntime::new();

run_system_blocking(&MovementSystem, &world, &compute, &io);
```

## DynSystem (Internal)

`DynSystem` is the object-safe wrapper used internally for type erasure. The blanket implementation converts any `System` into a `DynSystem` by boxing the result. You don't need to interact with this directly.
