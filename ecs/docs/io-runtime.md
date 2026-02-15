# IO Runtime

The IO runtime bridges the ECS's custom manual-polling executor with a real async runtime capable of driving IO futures (file reads, network requests, timers).

## The Problem

Compute tasks run on a manually-polled executor — they don't use a real async runtime. This means standard `tokio` or browser APIs won't work directly because no one drives their wakers.

The `IoRuntime` solves this by spawning IO work on a real runtime and communicating results back via channels.

## Platform Implementations

| Platform | Runtime | Worker Threads |
|----------|---------|---------------|
| Native (x86_64, ARM) | tokio multi-thread | 1 dedicated IO thread |
| WASM (wasm32) | `wasm_bindgen_futures::spawn_local` | Browser event loop |

## Creating an IO Runtime

```rust
use redlilium_ecs::IoRuntime;

let io = IoRuntime::new();

// Clone is cheap (Arc-wrapped internally)
let io_clone = io.clone();
```

## Running IO Tasks

```rust
use redlilium_ecs::IoRuntime;

let io = IoRuntime::new();

// Spawn an async IO operation
let handle = io.run(async {
    tokio::fs::read_to_string("config.json").await.unwrap()
});

// Block until ready (for testing)
let config = handle.recv();
```

### IoHandle as Future

The returned `IoHandle<T>` implements `Future`, so you can `.await` it from within compute tasks:

```rust
pool.spawn(Priority::Low, |cctx| async move {
    let mut handle = cctx.io().run(async {
        reqwest::get("https://api.example.com/data").await.unwrap()
    });

    // Poll-based await — works with noop waker
    let response = (&mut handle).await;
    process(response)
});
```

### Non-Blocking Check

```rust
let handle = io.run(async { 42u32 });

// Non-blocking check
match handle.try_recv() {
    Some(val) => println!("Got: {}", val),
    None => println!("Not ready yet"),
}

// Blocking receive
let val = handle.recv(); // Some(42)
```

## In Systems

Access the IO runtime through `SystemContext`:

```rust
struct LoaderSystem;

impl System for LoaderSystem {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        // Spawn IO from within a compute task, then block_on for the result
        let mut task = ctx.compute().spawn(Priority::High, |cctx| async move {
            let data = cctx.io().run(async {
                tokio::fs::read("asset.bin").await.unwrap()
            }).await;
            parse_asset(data)
        });

        let asset = ctx.compute().block_on(&mut task);
    }
}
```

## Cross-Platform Usage

The `IoRuntime` implements the `IoRunner` trait from `redlilium-core`, so libraries can be generic:

```rust
// In redlilium-core
pub trait IoRunner: Clone + Send + Sync + 'static {
    fn run<T, F>(&self, future: F) -> IoHandle<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static;
}
```

Code using `impl IoRunner` works on both native and WASM without modification.

## Ownership

The IO runtime is owned by the ECS runner and shared via `Arc`:

```
EcsRunner
  └── IoRuntime (Arc)
       ├── ComputePool (has a clone)
       ├── EcsComputeContext (has a clone per task)
       └── SystemContext (has a reference)
```

## Public API

| Method | Description |
|--------|-------------|
| `IoRuntime::new()` | Create a new IO runtime |
| `io.run(future)` | Spawn an IO future, returns `IoHandle<T>` |
| `io.clone()` | Cheap clone (Arc) |
| `handle.try_recv()` | Non-blocking result check |
| `handle.recv()` | Blocking receive |
| `handle.await` | Poll-based await (implements Future) |
