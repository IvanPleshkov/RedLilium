# Thread-Local Resources

## What Are They?

Thread-local (or non-Send) resources are global state objects that are **not thread-safe** — they cannot be accessed from arbitrary threads. The ECS stores them on the main thread and only allows access from systems that run on the main thread (exclusive systems or main-thread-only systems).

```rust
// Bevy-style non-Send resource (not available in RedLilium)
// A window handle that's only valid on the main thread
#[derive(Resource)]
struct WindowHandle(*mut RawWindow);

// NOT Send + Sync — can only be accessed on main thread
app.insert_non_send_resource(WindowHandle(raw_ptr));

// Access in exclusive system (runs on main thread)
fn render_system(window: NonSend<WindowHandle>) {
    unsafe { draw_to_window(window.0); }
}
```

### Why They Exist

Some platform APIs require main-thread access:
- **Window/UI handles**: OS window pointers (Win32 HWND, Cocoa NSWindow).
- **OpenGL contexts**: GL contexts are thread-bound.
- **Platform APIs**: macOS Cocoa, Android JNI, iOS UIKit — all main-thread-only.
- **C library state**: Thread-unsafe FFI libraries (e.g., some audio APIs).

### Design Considerations

| Aspect | Send + Sync Resources | Non-Send Resources |
|--------|----------------------|--------------------|
| Access from any thread | Yes | No — main thread only |
| Parallel system access | Yes (with RwLock) | No — serialized |
| Storage location | Thread pool shared | Main thread pinned |
| Use case | Game state, configs | Platform handles |

## Current Approach in RedLilium

All resources in RedLilium must be `Send + Sync + 'static`:

```rust
// From ecs/src/resource.rs
impl Resources {
    pub fn insert<T: Send + Sync + 'static>(&mut self, resource: T) { ... }
}
```

For platform handles that are not thread-safe, workarounds include:

```rust
// Workaround 1: Wrap in Arc<Mutex> (adds overhead)
world.insert_resource(Arc::new(Mutex::new(unsafe_handle)));

// Workaround 2: Keep outside the ECS
struct App {
    window: WindowHandle,  // Not in ECS
    world: World,
    runner: EcsRunner,
}

// Access between frames
fn game_loop(app: &mut App) {
    app.runner.run(&mut app.world, &systems);
    // Access window handle here — we're on the main thread
    render(&app.window, &app.world);
}
```

The second approach (keeping non-Send state outside the ECS) is idiomatic for RedLilium given the runner executes on the calling thread.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `NonSend<T>` / `NonSendMut<T>` system parameters, `insert_non_send_resource()`, main-thread scheduling |
| **flecs** | No explicit non-Send concept (single-threaded access available via system ordering) |
| **Unity DOTS** | `[MainThread]` attribute for systems, presentation system group runs on main thread |
| **EnTT** | No concept — all registry access is user-managed |
| **hecs** | No concept — user manages thread safety |
| **Legion** | No explicit non-Send resources |
| **Shipyard** | `NonSend` / `NonSync` / `NonSendSync` storage markers |
