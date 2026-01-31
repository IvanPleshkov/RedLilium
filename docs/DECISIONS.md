# Architecture Decision Records

This document tracks significant architectural decisions for RedLilium Engine.

## ADR Format

Each decision follows this format:
- **Status**: Proposed | Accepted | Deprecated | Superseded
- **Context**: What is the issue?
- **Decision**: What was decided?
- **Consequences**: What are the trade-offs?

---

## ADR-001: Rust as Primary Language

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need to choose a language for building a custom game engine.

### Decision
Use Rust as the primary language.

### Consequences
- ✅ Memory safety without garbage collection
- ✅ Excellent WebAssembly support
- ✅ Strong type system catches bugs at compile time
- ✅ Modern package management with Cargo
- ⚠️ Steeper learning curve
- ⚠️ Longer compile times

---

## ADR-002: Workspace with Multiple Crates

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need to organize code for a game engine with multiple subsystems.

### Decision
Use a Cargo workspace with separate crates: `core`, `graphics`, `demos`.

### Consequences
- ✅ Clear separation of concerns
- ✅ Parallel compilation of independent crates
- ✅ Can publish crates independently
- ✅ Enforces API boundaries between modules
- ⚠️ More complex project structure
- ⚠️ Need to manage inter-crate dependencies

---

## ADR-003: winit for Window Management

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need a cross-platform window management library that supports both native and web.

### Decision
Use `winit` version 0.30.12.

### Consequences
- ✅ Cross-platform (Windows, Linux, macOS, Web)
- ✅ Well-maintained with active community
- ✅ Integrates well with wgpu
- ✅ Supports WebAssembly
- ⚠️ API changes between versions require updates

---

## ADR-004: Web Support via WebAssembly

**Date**: 2025-01-30
**Status**: Accepted

### Context
Want to support running demos in web browsers.

### Decision
Use wasm-pack to compile to WebAssembly with wasm-bindgen.

### Consequences
- ✅ Demos run in browsers without plugins
- ✅ Easy sharing via URLs
- ✅ Same codebase for native and web
- ⚠️ Some features unavailable on web (file system, threading)
- ⚠️ Performance may differ from native

---

## ADR-005: Documentation Strategy

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need documentation that stays in sync with code and is useful for both humans and AI assistants.

### Decision
Use a layered documentation approach:
1. Rust doc comments for API documentation
2. `docs/` folder for architecture and decisions
3. Per-crate READMEs for module-specific info

### Consequences
- ✅ Doc comments are checked by compiler
- ✅ Examples in docs are tested via `cargo test --doc`
- ✅ AI can read markdown files for context
- ✅ Architecture docs separate from API docs
- ⚠️ Requires discipline to keep docs updated

---

## ADR-006: Render Graph Architecture

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need a flexible and efficient way to describe rendering operations that:
- Works across multiple graphics backends
- Handles synchronization automatically
- Supports both simple and complex rendering pipelines
- Allows optimization at the graph level

### Decision
Implement an abstract render graph system where:
1. Users declare passes and their resource dependencies
2. The graph compiler determines optimal execution order
3. The executor handles synchronization and resource management
4. Backend implementations translate to native API calls

### Consequences
- ✅ Declarative API is easier to use than manual barriers
- ✅ Graph-level optimizations (memory aliasing, barrier batching)
- ✅ Backend-agnostic rendering code
- ✅ Automatic resource lifetime management
- ⚠️ Initial overhead for graph compilation
- ⚠️ Less control over low-level details
- ⚠️ Additional abstraction layer complexity

---

## ADR-007: Triple Backend Strategy (Vulkan, wgpu, Dummy)

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need to support multiple platforms with different graphics capabilities:
- Desktop platforms with Vulkan support need maximum performance
- Web and platforms without Vulkan need cross-platform support
- Testing requires graphics-free execution

### Decision
Implement three backends behind a common trait:

1. **Vulkan Backend** (via `ash` crate)
   - Direct Vulkan API access for maximum performance
   - Full access to Vulkan extensions (ray tracing, mesh shaders)
   - Explicit memory management with `gpu-allocator`
   - Target: Windows, Linux desktop

2. **wgpu Backend** (version 28.0.0)
   - Cross-platform via WebGPU abstraction
   - Automatic fallback to Vulkan/Metal/DX12
   - WebAssembly support for browsers
   - Target: All platforms including Web

3. **Dummy Backend**
   - No-op implementation for testing
   - Validates API usage without GPU
   - Enables CI testing without graphics hardware

### Consequences
- ✅ Maximum performance on desktop via Vulkan
- ✅ Web support via wgpu
- ✅ Testability without GPU hardware
- ✅ Future extensibility (can add Metal backend, etc.)
- ⚠️ Three implementations to maintain
- ⚠️ Need to ensure feature parity across backends
- ⚠️ wgpu limits available features to common denominator

---

## ADR-008: Multithreaded Render Graph Execution

**Date**: 2025-01-30
**Status**: Accepted

### Context
Modern games need to utilize multiple CPU cores efficiently. The render graph should support parallel command recording to maximize performance.

### Decision
Design the render graph for thread-safety:

1. **Construction Phase** (Single-threaded)
   - Graph building is single-threaded for determinism
   - Clear ownership during setup

2. **Execution Phase** (Multi-threaded)
   - Command buffers recorded in parallel per pass
   - Each thread gets its own command buffer pool
   - Graph data is immutable during execution

3. **Synchronization Primitives**
   - Use `Arc` for shared resource references
   - Use `parking_lot` for fast mutexes where needed
   - Lock-free handle allocation via atomics

### Consequences
- ✅ Scales with CPU core count
- ✅ Reduced frame latency via parallel recording
- ✅ Clear threading model (build single, execute parallel)
- ⚠️ Requires careful API design to prevent data races
- ⚠️ Per-thread resource pools increase memory usage
- ⚠️ Debugging parallel execution is harder

---

## ADR-009: Multiple Render Graphs per Backend

**Date**: 2025-01-31
**Status**: Accepted

### Context
A process may contain multiple ECS worlds (e.g., game world, editor world, preview worlds). Each world may require different rendering pipelines or render graphs. The rendering backend (GPU device, command queues) is expensive to create and should be shared.

### Decision
Support a one-to-many relationship between backends and render graphs:

1. **Single Backend Instance**
   - One backend per process manages GPU resources
   - All render graphs share the same GPU device and memory pools
   - Synchronization primitives are shared across graphs

2. **Multiple Render Graphs**
   - Each ECS world can own one or more render graphs
   - Render graphs are independent and can have different pass configurations
   - No direct communication between render graphs (isolation)

3. **ECS World Independence**
   - Multiple ECS worlds can coexist in a process
   - Each world extracts render data independently
   - Worlds can target different render graphs

4. **Resource Sharing**
   - GPU resources (buffers, textures) can be shared via handles
   - Backend manages resource lifetimes across all graphs
   - Synchronization patterns (fences, semaphores) are shared

### Consequences
- ✅ Efficient GPU resource utilization across worlds
- ✅ Flexible multi-world architecture (game + editor)
- ✅ Render graphs can be created/destroyed independently
- ✅ Supports split-screen, picture-in-picture, previews
- ⚠️ Need careful resource lifetime management
- ⚠️ Backend complexity increases with shared state
- ⚠️ Cross-graph synchronization requires explicit barriers

---

## ADR-010: Streaming Graph Submission

**Date**: 2025-01-31
**Status**: Accepted

### Context
Traditional render systems batch all work and submit at frame end. This leaves the GPU idle while the CPU builds subsequent passes. We wanted to maximize CPU-GPU parallelism.

### Decision
Implement streaming submission via `FrameSchedule`:

1. Each `submit()` call immediately sends work to the GPU
2. GPU semaphores synchronize dependencies between graphs
3. CPU continues building while GPU executes

```rust
let shadows = schedule.submit("shadows", shadow_graph, &[]);     // GPU starts now
let main = schedule.submit("main", main_graph, &[shadows]);      // Waits on shadow
```

### Consequences
- ✅ GPU starts earlier, reducing frame latency
- ✅ Better CPU-GPU parallelism
- ✅ Natural expression of rendering dependencies
- ✅ Semaphores handle GPU-side ordering
- ⚠️ More complex than batch submission
- ⚠️ Dependency graph must be acyclic
- ⚠️ Submitted graphs cannot be modified

---

## ADR-011: Frame Pipelining with Fences

**Date**: 2025-01-31
**Status**: Accepted

### Context
With streaming submission, we don't want the CPU to wait for the GPU after each frame. Multiple frames should be "in flight" simultaneously for maximum throughput.

### Decision
Implement `FramePipeline` to manage N frames in flight:

1. Each frame slot has a fence for CPU-GPU synchronization
2. `begin_frame()` waits only if reusing a slot still in use, returns `FrameSchedule`
3. `end_frame(schedule)` takes ownership of schedule, extracts fence, advances slot
4. `wait_idle()` ensures graceful shutdown

```rust
let mut pipeline = device.create_pipeline(2);  // Device creates pipeline

while running {
    let mut schedule = pipeline.begin_frame();  // Waits + returns schedule
    // ... submit graphs to schedule ...
    schedule.present("present", graph, &[deps]);
    pipeline.end_frame(schedule);               // Takes ownership
}
pipeline.wait_idle();                           // Graceful shutdown
```

### Consequences
- ✅ CPU can work on frame N+1 while GPU renders frame N
- ✅ Higher throughput (better GPU utilization)
- ✅ Clean separation from scheduling logic
- ✅ Graceful shutdown prevents resource destruction races
- ⚠️ Higher input latency (frames queued ahead)
- ⚠️ Each frame slot needs its own resources (uniform buffers, etc.)
- ⚠️ 2-3 frames typical; more increases memory usage

---

## ADR-013: Hierarchical API for Pipeline and Schedule

**Date**: 2025-01-31
**Status**: Accepted

### Context
The initial API allowed creating `FramePipeline` and `FrameSchedule` independently:

```rust
let pipeline = FramePipeline::new(2);
let mut schedule = FrameSchedule::new();
// ... submit graphs ...
let fence = schedule.submit_and_present(...);
pipeline.end_frame(fence);
```

This had issues:
- No clear ownership hierarchy
- Users could accidentally use a schedule from a different frame
- Fence extraction was manual and error-prone
- Pipeline and Schedule lifetimes weren't enforced

### Decision
Establish a clear creation hierarchy:

1. **Device creates Pipeline**: `device.create_pipeline(frames_in_flight)`
2. **Pipeline creates Schedule**: `pipeline.begin_frame()` returns `FrameSchedule`
3. **Schedule consumed by Pipeline**: `pipeline.end_frame(schedule)`

```rust
let mut pipeline = device.create_pipeline(2);

while running {
    let mut schedule = pipeline.begin_frame();  // Returns schedule
    let main = schedule.submit("main", graph, &[]);
    schedule.present("present", post, &[main]);   // Marks complete
    pipeline.end_frame(schedule);                 // Takes ownership
}
```

The `present()` method replaces `submit_and_present()` and doesn't return a fence.
Instead, `end_frame()` extracts the fence internally.

### Consequences
- ✅ Clear ownership: Device → Pipeline → Schedule
- ✅ Prevents misuse (can't mix schedules between pipelines)
- ✅ Cleaner API (no manual fence handling)
- ✅ `present()` must be called before `end_frame()` (enforced with panic)
- ✅ `FrameSchedule::new()` is `pub(crate)` - can't create directly
- ⚠️ Slightly more opinionated API
- ⚠️ Must call `present()` even for off-screen rendering (may revisit)

---

## ADR-014: Debounced Window Resize with Strategies

**Date**: 2025-01-31
**Status**: Accepted

### Context
Window resize is problematic for real-time rendering:

1. OS sends many resize events during drag (30+ per second)
2. Each resize requires swapchain recreation
3. Swapchain recreation requires GPU synchronization
4. Naive approach: recreate on every event → severe stuttering

Professional engines need smooth resize without visible hitches.

### Decision
Implement `ResizeManager` with three components:

**1. Debouncing**: Buffer resize events, only apply after quiet period (50-100ms)

```rust
let mut manager = ResizeManager::new((1920, 1080), 50, strategy);

// Events buffered
manager.on_resize_event(800, 600);
manager.on_resize_event(900, 700);
manager.on_resize_event(1000, 800);

// Only applied after 50ms quiet
if let Some(event) = manager.update() {
    // Single resize to 1000x800
}
```

**2. Per-Slot Waiting**: `wait_current_slot()` instead of `wait_idle()`

- `wait_idle()`: waits for ALL frames (2-3 frame times)
- `wait_current_slot()`: waits for ONE frame
- Result: 2-3x faster resize

**3. Render Strategies**: Configurable behavior during resize

| Strategy | Description |
|----------|-------------|
| `Stretch` | Render at old size, OS stretches |
| `IntermediateTarget` | Fixed-size render target |
| `DynamicResolution` | Reduced resolution during resize |

### Consequences
- ✅ Smooth resize without stuttering
- ✅ Single swapchain recreation per resize gesture
- ✅ Configurable quality/performance tradeoff
- ✅ `wait_current_slot()` minimizes GPU stall
- ✅ Works with any windowing library
- ⚠️ 50-100ms delay before resize takes effect
- ⚠️ `DynamicResolution` requires upscaling support
- ⚠️ Application must integrate with event loop
