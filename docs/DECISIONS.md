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
2. `begin_frame()` waits only if reusing a slot still in use
3. `end_frame()` stores the fence and advances to next slot
4. `wait_idle()` ensures graceful shutdown

```rust
let mut pipeline = FramePipeline::new(2);  // 2 frames in flight

while running {
    pipeline.begin_frame();   // Waits if slot still in use
    // ... render ...
    pipeline.end_frame(fence);
}
pipeline.wait_idle();         // Graceful shutdown
```

### Consequences
- ✅ CPU can work on frame N+1 while GPU renders frame N
- ✅ Higher throughput (better GPU utilization)
- ✅ Clean separation from scheduling logic
- ✅ Graceful shutdown prevents resource destruction races
- ⚠️ Higher input latency (frames queued ahead)
- ⚠️ Each frame slot needs its own resources (uniform buffers, etc.)
- ⚠️ 2-3 frames typical; more increases memory usage
