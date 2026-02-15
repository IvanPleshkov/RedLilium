# Access Control

The access control system provides compile-time safe, deadlock-free component and resource access in systems. It uses **marker types** to declare data dependencies and a **lock-execute pattern** to ensure deterministic lock release.

## Access Marker Types

| Marker | Closure Receives | Panics if |
|--------|-----------------|-----------|
| `Read<T>` | `Ref<'_, T>` | Component `T` not registered |
| `Write<T>` | `RefMut<'_, T>` | Component `T` not registered |
| `OptionalRead<T>` | `Option<Ref<'_, T>>` | Never (returns `None`) |
| `OptionalWrite<T>` | `Option<RefMut<'_, T>>` | Never (returns `None`) |
| `Res<T>` | `ResourceRef<'_, T>` | Resource `T` doesn't exist |
| `ResMut<T>` | `ResourceRefMut<'_, T>` | Resource `T` doesn't exist |

## Basic Usage

```rust
use redlilium_ecs::{Read, Write, Res, ResMut, OptionalRead};

struct MySystem;

impl System for MySystem {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        // Read two components and one resource
        ctx.lock::<(Write<Position>, Read<Velocity>, Res<DeltaTime>)>()
            .execute(|(mut positions, velocities, dt)| {
                for (idx, pos) in positions.iter_mut() {
                    if let Some(vel) = velocities.get(idx) {
                        pos.x += vel.x * dt.0;
                        pos.y += vel.y * dt.0;
                    }
                }
            });
    }
}
```

## Optional Access

Use `OptionalRead<T>` or `OptionalWrite<T>` when a component might not be registered:

```rust
ctx.lock::<(Write<Position>, OptionalRead<Gravity>)>()
    .execute(|(mut positions, gravity)| {
        let g = gravity.as_ref().map(|g| g.iter().next())
            .flatten()
            .map(|(_, g)| g.value)
            .unwrap_or(9.81);

        for (_, pos) in positions.iter_mut() {
            pos.y -= g;
        }
    });
```

## How Lock Ordering Works

When `execute()` is called:

1. **Collect** `AccessInfo` from each element (TypeId + read/write flag).
2. **Sort** by TypeId — this prevents deadlocks when multiple systems acquire locks.
3. **Deduplicate** — if the same TypeId appears twice, the write flag wins.
4. **Acquire** all locks in sorted order via `World::acquire_sorted()`.
5. **Fetch** typed data without per-fetch locking (locks already held).
6. **Call** the user's closure.
7. **Drop** all lock guards when the closure returns.

## Tuple Support

Access sets support tuples up to 8 elements:

```rust
// All of these work:
ctx.lock::<(Read<A>,)>()
ctx.lock::<(Read<A>, Write<B>)>()
ctx.lock::<(Read<A>, Write<B>, Res<C>)>()
// ... up to 8 elements
ctx.lock::<(Read<A>, Write<B>, Read<C>, Res<D>, ResMut<E>, OptionalRead<F>, OptionalWrite<G>, Read<H>)>()
```

The empty tuple is also valid (no-op):

```rust
ctx.lock::<()>().execute(|()| {
    // no data access needed
});
```

## Why Lock-Execute?

The execute closure is `FnOnce` (synchronous). This is a deliberate design choice:

```rust
ctx.lock::<(Write<Position>,)>()
    .execute(|(mut positions)| {
        for (_, pos) in positions.iter_mut() {
            pos.x += 1.0;
        }
    });
```

This ensures locks are always released deterministically when the closure returns. In a multi-threaded executor, this prevents deadlocks by guaranteeing that locks cannot be held indefinitely.

## Return Values from Execute

The closure can return a value:

```rust
let total_health = ctx.lock::<(Read<Health>,)>()
    .execute(|(healths,)| {
        healths.iter().map(|(_, h)| h.value).sum::<f32>()
    });

println!("Total health: {}", total_health);
```

## AccessSet Trait

The `AccessSet` trait is implemented for tuples of `AccessElement` types. Each `AccessElement` provides:

- `access_info()` — Returns `AccessInfo { type_id, is_write }`.
- `fetch(world)` — Fetches data with lock acquisition.
- `fetch_unlocked(world)` — Fetches data assuming locks are already held.

You don't need to implement these traits yourself — they work automatically with the provided marker types.
