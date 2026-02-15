# Cross-Platform Support

The ECS is designed to run on both native desktop platforms and WebAssembly (WASM), with the same public API but different internal implementations.

## Platform Matrix

| Feature | Native (x86_64, ARM) | WASM (wasm32) |
|---------|----------------------|---------------|
| Single-threaded runner | Yes | Yes (sequential) |
| Multi-threaded runner | Yes | No |
| Compute pool | Full (tick, tick_all, tick_extract) | Full |
| IO runtime | tokio (1 worker thread) | wasm_bindgen_futures |
| Thread scope | `std::thread::scope` | N/A |
| Profiling | puffin integration | N/A |

## Conditional Compilation

The ECS uses `#[cfg(target_arch = "wasm32")]` and `#[cfg(not(target_arch = "wasm32"))]` to select platform-specific implementations:

```rust
// Multi-threaded runner only available on native
#[cfg(not(target_arch = "wasm32"))]
pub use runner::EcsRunnerMultiThread;

// IoRuntime uses tokio on native, wasm_bindgen on web
#[cfg(not(target_arch = "wasm32"))]
impl IoRuntime {
    pub fn new() -> Self {
        // tokio multi-thread runtime with 1 worker
    }
}

#[cfg(target_arch = "wasm32")]
impl IoRuntime {
    pub fn new() -> Self {
        // lightweight handle (spawn_local is global)
    }
}
```

## Writing Cross-Platform Code

Use `EcsRunner` enum (not the concrete runner types) for portable code:

```rust
// This works on all platforms
let runner = EcsRunner::single_thread();
runner.run(&mut world, &systems);

// This only compiles on native
#[cfg(not(target_arch = "wasm32"))]
let runner = EcsRunner::multi_thread_default();
```

### Recommended Pattern

```rust
fn create_runner() -> EcsRunner {
    #[cfg(not(target_arch = "wasm32"))]
    {
        EcsRunner::multi_thread_default()
    }
    #[cfg(target_arch = "wasm32")]
    {
        EcsRunner::single_thread()
    }
}
```

## IO Differences

### Native IO

Runs on a dedicated tokio thread. Results arrive synchronously within the same frame poll cycle:

```rust
let io = IoRuntime::new();
let handle = io.run(async {
    tokio::fs::read_to_string("config.json").await.unwrap()
});
// Result available after tokio thread processes it
```

### WASM IO

Runs on the browser event loop via `wasm_bindgen_futures::spawn_local`. Results arrive after control returns to the browser (typically next frame):

```rust
let io = IoRuntime::new();
let handle = io.run(async {
    // fetch API or other browser async operations
    web_sys::window().unwrap().fetch_with_str("data.json")
});
// Result arrives when browser event loop processes the future
```

## Compute Pool on WASM

The compute pool works identically on both platforms — it's a manual polling system that doesn't depend on OS threads. On WASM:

- Tasks are polled during `tick()` / `tick_all()` calls.
- The single-threaded runner drives the pool between system polls.
- `yield_now()` works the same way (timer-based cooperative yielding).
- No `tick_extract()` parallelism (single thread).

## Graceful Shutdown on WASM

```rust
// On WASM, Instant::now() may not be available
// The shutdown implementation handles this:
#[cfg(not(target_arch = "wasm32"))]
let start = std::time::Instant::now();

while compute.pending_count() > 0 {
    #[cfg(not(target_arch = "wasm32"))]
    if start.elapsed() >= time_budget {
        return Err(ShutdownError::Timeout { ... });
    }
    compute.tick_all();
}
```

## Platform-Specific Dependencies

```toml
# In ecs/Cargo.toml

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "time", "fs", "io-util"] }

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen-futures = "0.4"
```

## Building for WASM

```bash
# Build the demos for web
wasm-pack build demos --target web --out-dir web/pkg

# Or check compilation without building
cargo check --target wasm32-unknown-unknown -p redlilium-ecs
```
