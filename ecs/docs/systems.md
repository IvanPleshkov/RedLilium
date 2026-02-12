# Systems

Systems are the logic processors of the ECS. Each system is an async function that reads and writes components/resources through a context object.

## The System Trait

All systems implement the `System` trait:

```rust
pub trait System: Send + Sync + 'static {
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> impl Future<Output = ()> + Send + 'a;
}
```

Systems are:
- **Async**: Return a future, enabling `.await` for compute tasks and lock-execute.
- **Send + Sync**: Can run on any thread.
- **Stateless or self-contained**: Receive data through `SystemContext`, not constructor args (unless stored as struct fields).

## Basic System

```rust
struct MovementSystem;

impl System for MovementSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Write<Position>, Read<Velocity>)>()
            .execute(|(mut positions, velocities)| {
                for (idx, pos) in positions.iter_mut() {
                    if let Some(vel) = velocities.get(idx) {
                        pos.x += vel.x;
                        pos.y += vel.y;
                    }
                }
            }).await;
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

The core innovation of this ECS: component access is done through a **lock-execute** pattern that prevents holding locks across `.await` points at compile time.

```rust
// The execute closure is FnOnce (synchronous) — you can't .await inside it
ctx.lock::<(Write<Position>, Read<Velocity>)>()
    .execute(|(mut positions, velocities)| {
        // Locks are held here
        for (idx, pos) in positions.iter_mut() {
            if let Some(vel) = velocities.get(idx) {
                pos.x += vel.x;
            }
        }
        // Locks released when closure returns
    }).await;

// Safe to .await here — no locks held
```

## Two-Phase Async System

Extract data in one phase, process asynchronously, then apply results:

```rust
struct PathfindSystem;

impl System for PathfindSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        // Phase 1: Extract data (locks acquired and released)
        let nav_data = ctx.lock::<(Read<NavMesh>,)>()
            .execute(|(nav,)| {
                nav.iter().next().map(|(_, n)| n.clone())
            }).await;

        // Phase 2: Offload heavy computation (no locks held)
        if let Some(data) = nav_data {
            let mut handle = ctx.compute().spawn(Priority::Low, |_cctx| async move {
                compute_paths(data)
            });
            let paths = (&mut handle).await;

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
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        let dt = ctx.lock::<(Res<DeltaTime>,)>()
            .execute(|(dt,)| dt.0).await;

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

Systems can publish typed results as resources for other systems to read:

```rust
struct PhysicsResult {
    collision_count: usize,
}

struct PhysicsSystem;

impl System for PhysicsSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        let result = PhysicsResult { collision_count: 42 };
        ctx.commands(move |world| {
            world.insert_resource(SystemResult::<PhysicsSystem, PhysicsResult>::new(result));
        });
    }
}

// Consumer system reads the result
struct DebugSystem;
impl System for DebugSystem {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(Res<SystemResult<PhysicsSystem, PhysicsResult>>,)>()
            .execute(|(result,)| {
                println!("Collisions: {}", result.value.collision_count);
            }).await;
    }
}
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

`DynSystem` is the object-safe wrapper used internally for type erasure. The blanket implementation converts any `System` into a `DynSystem` by boxing the future. You don't need to interact with this directly.
