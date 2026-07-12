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

---

## ADR-015: D3D/wgpu-Style Coordinate System with [0, 1] Depth Range

**Date**: 2025-02-01
**Status**: Accepted

### Context

Different graphics APIs use different coordinate system conventions:

| API | NDC Depth Range | Y-Axis Direction |
|-----|-----------------|------------------|
| OpenGL | [-1, 1] | +Y up |
| Vulkan | [0, 1] | +Y down |
| D3D/Metal | [0, 1] | +Y down |
| wgpu | [0, 1] | +Y down |

We need to choose a consistent coordinate system convention that:
- Works efficiently with our Vulkan backend
- Matches our wgpu backend for compatibility
- Allows straightforward porting of shaders and content

### Decision

Adopt the **D3D/wgpu coordinate system convention**:

1. **Depth Range**: `[0, 1]` (near = 0, far = 1)
2. **Y-Axis**: +Y points down in normalized device coordinates (NDC)
3. **Origin**: Top-left corner in screen space

**Implementation Details:**

1. **Vulkan Backend**: Set viewport `minDepth = 0.0`, `maxDepth = 1.0`
   - This is Vulkan's native convention, so no transformation needed
   - Clear depth values use 1.0 for far plane

2. **wgpu Backend**: Uses `[0, 1]` depth range natively
   - wgpu handles this automatically across all backend APIs

3. **Projection Matrices**: Must be built for `[0, 1]` depth
   - Use `glam::Mat4::perspective_rh()` (right-handed, zero-to-one depth)
   - Or use libraries' `_zo` (zero-to-one) projection variants

**Why not use VK_EXT_depth_clip_control?**

The `VK_EXT_depth_clip_control` extension allows using OpenGL's `[-1, 1]` convention on Vulkan. We don't need it because:
- Vulkan natively uses `[0, 1]` which matches our target
- The extension is designed for OpenGL-over-Vulkan layering
- Not using the extension means broader hardware compatibility

**Shader Implications:**

Shaders receive depth in `[0, 1]` range after projection. No shader-side transformation like `gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0` is needed.

### Consequences

- ✅ Consistent behavior across Vulkan and wgpu backends
- ✅ Native Vulkan convention (no extension required)
- ✅ Better depth precision than OpenGL's `[-1, 1]` mapped to `[0, 1]`
- ✅ Compatible with reverse-Z for improved precision (minDepth=1, maxDepth=0)
- ✅ Matches industry-standard D3D/Metal/wgpu convention
- ⚠️ OpenGL shaders/content may need projection matrix adjustment
- ⚠️ Users of glm/nalgebra must use depth-zero-to-one projection functions

---

## ADR-016: Deferred GPU Resource Destruction for Vulkan

**Date**: 2025-02-02
**Status**: Accepted

### Context

GPU commands execute asynchronously - when work is submitted, the CPU continues while the GPU processes commands 1-3 frames behind. This creates a critical problem: if a resource (buffer, texture, etc.) is destroyed while the GPU is still using it, the result is:

- Vulkan validation errors
- Undefined behavior (corrupted rendering, crashes, GPU hangs)
- Hard-to-debug intermittent failures

The **wgpu backend doesn't have this problem** because wgpu handles deferred destruction internally. When you drop a `wgpu::Buffer`, wgpu tracks resource usage and automatically defers destruction until the GPU is done. This safety is built into wgpu's design.

Our **Vulkan backend uses raw `ash`** (direct Vulkan bindings), which provides maximum performance but no automatic safety. When `vkDestroyBuffer()` is called, destruction is immediate. We needed to implement equivalent protection.

**The Problem Illustrated:**

```
CPU Frame 0: Submit commands using Buffer A → GPU starts
CPU Frame 1: Submit commands using Buffer B → Continue
CPU Frame 2: User drops Arc<Buffer> for A (refcount = 0)
             → Immediate vkDestroyBuffer() ← WRONG!
GPU Frame 0: Still reading from Buffer A! ← CRASH/CORRUPTION
```

### Decision

Implement a deferred destruction system for the Vulkan backend:

**1. Resource Queuing**

When a Vulkan resource's `Arc` is dropped, instead of immediately calling `vkDestroy*`, the resource handle is queued:

```rust
impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let GpuBuffer::Vulkan { device, buffer, allocation, deferred, .. } = self {
            deferred.queue(DeferredResource::Buffer {
                device: device.clone(),
                buffer: *buffer,
                allocation: allocation.lock().take(),
            });
        }
    }
}
```

**2. Frame-Indexed Queues**

The `DeferredDestructor` maintains `MAX_FRAMES_IN_FLIGHT` (3) queues, one per frame slot:

```
Frame Queues:
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Frame 0  │  │ Frame 1  │  │ Frame 2  │
│ [buf, tex]│  │ [sampler]│  │ []       │
└──────────┘  └──────────┘  └──────────┘
```

**3. Frame Boundary Processing**

When `FramePipeline::begin_frame()` waits on a fence, it means the GPU has finished with an old frame. After the fence signals, we advance the destructor:

```rust
// In FramePipeline::begin_frame()
if let Some(fence) = &self.frame_fences[self.current_slot] {
    fence.wait();  // GPU done with old frame
}
device.advance_deferred_destruction();  // Safe to destroy old resources
```

**4. Resource Types Covered**

All Vulkan resource types use deferred destruction:
- `GpuBuffer` - vertex/index/uniform buffers
- `GpuTexture` - images and image views
- `GpuSampler` - texture samplers
- `GpuFence` - CPU-GPU synchronization
- `GpuSemaphore` - GPU-GPU synchronization

**5. Allocator Integration**

Memory allocations (via `gpu-allocator`) are freed along with their resources through a weak reference to the allocator. If the allocator is already dropped (during shutdown), resources are destroyed without freeing allocations (the allocator cleanup handles this).

### Alternatives Considered

**1. Manual Lifetime Management**

Require users to track resource lifetimes and call explicit destruction methods.

- ❌ Error-prone and tedious
- ❌ Doesn't match Rust's RAII patterns
- ❌ Poor developer experience

**2. Reference Counting with Frame Tracking**

Track which frames use which resources via reference counting.

- ❌ Complex bookkeeping
- ❌ Every resource access needs tracking
- ❌ Performance overhead per draw call

**3. Global Device Wait on Every Drop**

Call `vkDeviceWaitIdle()` in every resource destructor.

- ❌ Massive performance impact
- ❌ Defeats the purpose of async GPU execution
- ❌ Completely impractical

### Consequences

- ✅ **Safe**: Resources destroyed only after GPU is done with them
- ✅ **Transparent**: Users use `Arc<Buffer>` normally; destruction is automatic
- ✅ **Zero-cost for wgpu**: wgpu handles this internally; no double-deferral
- ✅ **Integrates with FramePipeline**: Cleanup happens at natural frame boundaries
- ✅ **Graceful shutdown**: `flush_all()` destroys all pending resources when device waits idle
- ⚠️ **Memory overhead**: Resources held slightly longer than strictly necessary
- ⚠️ **Vulkan-specific**: Only the Vulkan backend needs this complexity
- ⚠️ **Frame timing dependent**: Resources destroyed at frame boundaries, not immediately

---

## ADR-017: Automatic Texture Layout Tracking and Barrier Placement

**Date**: 2026-02-02
**Status**: Accepted

### Context

Vulkan requires explicit image layout transitions via pipeline barriers. Each texture has a current "layout" (e.g., `COLOR_ATTACHMENT_OPTIMAL`, `SHADER_READ_ONLY_OPTIMAL`) that must match what the GPU expects. Transitioning between layouts requires:

1. Knowing the current layout of each texture
2. Knowing the required layout for the upcoming operation
3. Issuing a `vkCmdPipelineBarrier` with appropriate stage and access masks

Manual barrier management is error-prone:
- Easy to forget transitions
- Easy to use wrong old/new layouts
- Leads to Vulkan validation errors or undefined behavior
- Each pass must track what layouts textures were left in

The original implementation always used `VK_IMAGE_LAYOUT_UNDEFINED` as the old layout, which:
- Works but may discard texture contents
- Prevents multi-pass workflows where a texture is rendered then sampled
- No optimization for consecutive passes using the same layout

### Decision

Implement automatic texture layout tracking for the Vulkan backend:

**1. Per-Frame Layout State**

Track texture layouts per frame-in-flight since the GPU may be processing old frames while the CPU records new ones:

```rust
pub struct TextureLayoutTracker {
    frame_states: Vec<FrameLayoutState>,  // One per frame in flight
    current_frame: usize,
}

pub struct FrameLayoutState {
    layouts: HashMap<TextureId, TextureLayout>,
}
```

**2. Usage Inference from Pass Configuration**

Instead of requiring explicit layout declarations, infer texture usage from pass configuration:

```rust
impl GraphicsPass {
    pub fn infer_resource_usage(&self) -> PassResourceUsage {
        let mut usage = PassResourceUsage::new();
        // Color attachments → RenderTargetWrite
        // Depth attachments → DepthStencilWrite or DepthStencilReadOnly
        // Material textures → ShaderRead
        usage
    }
}
```

**3. Barrier Generation at Encode Time**

Before encoding each pass, generate barriers for all textures that need transitions:

```rust
fn execute_graph(&self, compiled: &CompiledGraph) {
    for pass_handle in compiled.pass_order() {
        let pass = &passes[pass_handle.index()];
        let usage = pass.infer_resource_usage();

        // Generate and submit barriers
        let barriers = self.layout_tracker.generate_barriers(&usage);
        barriers.submit(cmd);

        // Encode the pass
        self.encode_pass(cmd, pass)?;
    }
}
```

**4. Batched Barrier Submission**

Collect all barriers for a pass and submit them in a single `vkCmdPipelineBarrier` call with combined stage masks for efficiency.

**5. Access Mode to Layout Mapping**

```rust
pub enum TextureAccessMode {
    RenderTargetWrite    → ColorAttachment
    DepthStencilWrite    → DepthStencilAttachment
    DepthStencilReadOnly → DepthStencilReadOnly
    ShaderRead           → ShaderReadOnly
    StorageReadWrite     → General
    TransferRead         → TransferSrc
    TransferWrite        → TransferDst
}
```

### Alternatives Considered

**1. Explicit Layout Annotations**

Require users to declare layouts for each texture in each pass.

- ❌ Verbose and error-prone
- ❌ Duplicates information already present in pass configuration
- ❌ Easy to get out of sync with actual usage

**2. Layout Tracking at Texture Level**

Store current layout in each `Texture` object.

- ❌ Doesn't work with frames in flight (GPU may be using old layout)
- ❌ Race conditions between CPU recording and GPU execution
- ❌ Complex synchronization needed

**3. Always Use UNDEFINED → Optimal**

Keep the simple approach of always transitioning from UNDEFINED.

- ❌ Discards texture contents (can't sample a texture after rendering to it)
- ❌ Misses optimization opportunities
- ❌ Only works for single-pass scenarios

### Consequences

- ✅ **Zero user burden**: No manual barrier management required
- ✅ **Correct by construction**: Layouts always match actual usage
- ✅ **Multi-pass workflows**: Textures can be rendered then sampled
- ✅ **Optimized barriers**: Skip transitions when layout already correct
- ✅ **Batched submission**: Single barrier call per pass
- ✅ **wgpu compatible**: wgpu handles this internally; system is Vulkan-only
- ⚠️ **Vulkan-specific complexity**: Adds code only used by Vulkan backend
- ⚠️ **Per-frame memory**: Layout maps consume memory per frame in flight
- ⚠️ **Inference limitations**: Some edge cases may need manual hints (future work)

---

## ADR-018: Automatic Buffer Barrier Placement

**Date**: 2026-02-02
**Status**: Accepted

### Context

While ADR-017 addressed automatic texture layout tracking, buffers also require memory barriers for correct synchronization in Vulkan. Unlike textures, buffers don't have "layouts" but still need barriers for:

1. **Write-After-Read (WAR)**: Ensure reads complete before writes begin
2. **Read-After-Write (RAW)**: Ensure writes are visible before reads begin
3. **Write-After-Write (WAW)**: Ensure sequential writes don't overlap

Common scenarios requiring buffer barriers:
- Compute shader writes storage buffer → Graphics pass reads as vertex buffer
- Transfer operation copies to buffer → Shader reads buffer data
- Indirect draw arguments written by compute → Draw indirect reads arguments
- Multi-pass scenarios where the same buffer is used differently

Without proper barriers, the GPU may read stale data or writes may occur out of order.

### Decision

Extend the automatic barrier system to cover buffers:

**1. Buffer Access Mode Enumeration**

Define access modes that map to Vulkan access flags and pipeline stages:

```rust
pub enum BufferAccessMode {
    VertexBuffer,       // Vertex input stage, vertex attribute read
    IndexBuffer,        // Vertex input stage, index read
    UniformRead,        // Vertex/fragment shader, uniform read
    StorageRead,        // Shader stages, shader read
    StorageWrite,       // Shader stages, shader write
    StorageReadWrite,   // Shader stages, shader read | write
    IndirectRead,       // Draw indirect stage, indirect command read
    TransferRead,       // Transfer stage, transfer read
    TransferWrite,      // Transfer stage, transfer write
}
```

**2. Buffer Usage Declaration**

Pass configurations declare buffer usage via `BufferUsageDecl`:

```rust
pub struct BufferUsageDecl {
    pub buffer: Arc<Buffer>,
    pub access: BufferAccessMode,
    pub offset: u64,
    pub size: u64,
}
```

**3. Usage Inference from Pass Configuration**

Buffer usages are automatically inferred from pass configuration:

- **GraphicsPass**: Indirect draw buffers → `IndirectRead`
- **TransferPass**:
  - `BufferToBuffer` src → `TransferRead`, dst → `TransferWrite`
  - `BufferToTexture` src → `TransferRead`
  - `TextureToBuffer` dst → `TransferWrite`

**4. Batched Barrier Submission**

Buffer barriers are collected alongside image barriers and submitted in a single `vkCmdPipelineBarrier` call:

```rust
unsafe {
    device.cmd_pipeline_barrier(
        cmd,
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[],                // memory barriers (global)
        &buffer_barriers,   // buffer memory barriers
        &image_barriers,    // image memory barriers
    );
}
```

### Alternatives Considered

**1. Per-Buffer State Tracking (like textures)**

Track previous access mode per buffer across passes.

- ✅ Optimal barriers (only insert when state changes)
- ❌ Significant memory overhead for many buffers
- ❌ Complex state management
- ❌ Buffers change usage more frequently than textures

**2. Manual Buffer Barriers**

Require users to explicitly declare buffer barriers.

- ❌ Error-prone and tedious
- ❌ Inconsistent with automatic texture barriers
- ❌ Easy to forget and cause subtle bugs

**3. No Buffer Barriers (rely on implicit guarantees)**

Rely on command buffer ordering for synchronization.

- ❌ Incorrect for cross-pass dependencies
- ❌ Would cause undefined behavior in multi-pass scenarios
- ❌ Different behavior between backends

### Consequences

- ✅ **Consistent with texture system**: Same inference pattern
- ✅ **Zero user burden**: Barriers automatically inferred from pass config
- ✅ **Correct synchronization**: Proper RAW/WAR/WAW handling
- ✅ **Batched submission**: Efficient single barrier call per pass
- ✅ **wgpu compatible**: wgpu handles this internally; Vulkan-only
- ⚠️ **Conservative barriers**: May insert barriers where not strictly needed
- ⚠️ **No per-buffer state tracking**: Future optimization opportunity
- ⚠️ **Vulkan-specific**: Only affects Vulkan backend complexity

## ADR-019: Sequential Vertex Attribute Locations in Layout Declaration Order

**Date**: 2026-07-04
**Status**: Accepted

### Context

The two backends assigned vertex shader locations differently (audit finding
XB-C1, `docs/audits/2026-07-graphics-backends/`):

- **wgpu** numbered attributes sequentially (0, 1, 2, ...) in `VertexLayout`
  declaration order.
- **Vulkan** used the fixed semantic-index table (`Position=0` ... `Weights=7`,
  so e.g. `TexCoord0=3`).

The same `VertexLayout` therefore produced different locations per backend, and
shader annotations in the tree were split between the two conventions: shaders
annotated with semantic indices (`egui.slang`, `debug_draw.slang`) worked on
Vulkan only by construction, while sequentially-annotated ones
(`opaque_textured.slang`, `entity_index.slang`) had broken UV input on Vulkan.

A semantic-index convention cannot be implemented on the wgpu path at all:
Slang's WGSL emission ignores `[[vk::location(N)]]` on vertex inputs and always
numbers them sequentially in struct declaration order.

### Decision

Shader locations are **sequential (0, 1, 2, ...) in `VertexLayout` attribute
declaration order** on both backends.

Contract for shader authors:

1. Declare vertex shader inputs in the same order as the mesh layout's
   attributes.
2. A shader may consume a **prefix** of the layout's attributes (e.g. use only
   position + normal of a position/normal/uv layout), but not an arbitrary
   subset — skipping an attribute shifts every later location on the WGSL path.
3. `[[vk::location(N)]]` annotations must match the declaration index; they are
   kept for documentation and SPIR-V explicitness, but WGSL output ignores them.

`VertexAttributeSemantic::index()` remains for semantic-set comparison and is
**not** a shader location.

### Alternatives Considered

**1. Semantic-index locations everywhere**

- ❌ Impossible on wgpu: Slang WGSL output cannot produce location gaps
- ✅ Would allow arbitrary attribute subsets in shaders

**2. Reflection-based matching (match shader inputs to attributes by semantic)**

- ✅ Most robust
- ❌ SPIR-V does not reliably carry HLSL semantics; significant complexity

### Consequences

- ✅ Identical attribute binding on both backends
- ✅ Fixes UV input for sequentially-annotated shaders on Vulkan
- ⚠️ Shaders consuming a non-prefix attribute subset are not expressible;
  use a dedicated `VertexLayout` for such passes

---

## ADR-020: Game Code Authoring — Rust Plugins over a Shared Engine Dylib

**Date**: 2026-07-05
**Status**: Accepted

### Context

Today a "game" is a hand-written `AppHandler` binary that owns its `World`,
`SystemsContainer` and render graph; the editor hosts only engine systems and
cannot run user gameplay code. Issue #4 asks: how does a user (or an AI agent
driving the editor remotely) write game logic **without recompiling the
editor**? Candidates: a C API, scripting/WASM, or hot-reloadable Rust dylibs.

Additional inputs:

- Rust has no stable ABI, so any native dylib boundary requires either a
  C-shaped stable layer or a same-toolchain contract.
- A domain scripting language already exists: **tyroxine**
  (`github/procedural`) — a pure, deterministic DSL for procedural mesh
  generation, designed to be embedded in this engine for procedural assets.
  It is explicitly not a general-purpose gameplay language, and its embedding
  `Context` is expensive to build (~1s analyze) and not `Send`.
- True live hot reload (swapping code under a running world) is notoriously
  fragile: stale function pointers in hooks/systems/resources, `TypeId`
  churn, closures capturing old code.

### Decision

Game logic is written in **Rust as plugins**, hosted either by the editor
(dev) or by a thin runtime binary (shipping):

1. **Plugin contract.** Game code implements `trait Plugin` over an `App`
   builder wrapping `World` + `Schedules` (+ runner/window config), with two
   methods: `build(&self, &mut App)` registers components/resources/events and
   adds systems to named schedules (including `Render`); `spawn_scene(&self,
   &mut App)` (default no-op) populates the initial scene. The split is
   load-bearing for reload (point 5): `build` re-runs against every fresh world
   generation, while `spawn_scene` runs only on first boot — a reload restores
   the scene from a snapshot instead, so scene population must never live in
   `build`. The host owns the frame loop and the Render bracket; game code
   never owns `main()`'s event loop internals.

2. **New thin crate `redlilium-runtime`** glues app+ecs+graphics+assets:
   `App` builder, `redlilium_runtime::run(MyGamePlugin)` for shipped
   binaries (native and wasm). The `app` crate stays a pure window/GPU layer
   with no ecs dependency.

3. **Hosting model.** The editor is an engine **binary** that loads the game
   as a **cdylib** exporting exactly two symbols: an entry
   `redlilium_game_module() -> Box<dyn Plugin>` and an ABI fingerprint
   (rustc version + engine build id). Fingerprint mismatch → refuse to load
   with a clear error. The shipped game is a separate fully static binary
   reusing the same `Plugin`.

4. **Dev linking: one shared engine dylib.** In dev profiles the engine is
   built as a single dynamic library (`redlilium-dylib` re-export crate,
   `crate-type = ["dylib"]`, à la Bevy `dynamic_linking`), linked by **both**
   the editor binary and the game cdylib. One copy of engine code, statics
   and `TypeId`s per process makes passing `&mut World` across the boundary
   sound under the same-toolchain contract. Release builds are fully static.

5. **Reload = warm restart, not live hot reload.** The editor process keeps a
   persistent host-owned `EngineContext` — window, `GraphicsInstance` /
   `Device` / `Surface`, egui, asset DB + processor + VFS mounts + watcher,
   GPU caches (`TextureManager`, `MeshManager`, `PipelineCache`,
   `ShadingRegistry`), tyroxine `Context` — injected into each new world via
   `insert_resource_shared`. On reload: serialize scene (name-based) → drop
   worlds/schedules → unload dylib → load new dylib → fingerprint check →
   register std + plugin (`build`, **not** `spawn_scene`) → deserialize scene.
   Undo history and selection are deliberately reset. This is the **same
   snapshot machinery as Play mode (#1) and scene save/load (#2)** — one
   mechanism, three features. A remote channel `reload` command makes this
   drivable by an AI agent.

   *Status (#45, slice A):* the world-lifecycle half landed first, decoupled
   from dylib linking. `App::capture` + `App::reload` (`runtime/src/app.rs`)
   do build-new-world → re-run `build` → `deserialize_world_into`, skipping
   `spawn_scene`/`run_startup`, and preserve the `EngineContext` managers as
   the same shared `Arc`s across the swap (headless-tested). This runs against
   a **statically linked** plugin under `SourceId::HOST`; the dylib
   load/unload and real per-reload `SourceId` generations (slices B/C) build on
   this seam.

6. **Scripting stays domain-specific.** tyroxine is the embedded DSL for
   procedural assets / contextual generation, living host-side in the
   persistent `EngineContext` (surviving game reloads). Gameplay logic is
   Rust; no general-purpose scripting layer.

### Alternatives Considered

**1. C API boundary**

- ✅ Editor and game could use different compiler versions
- ❌ The boundary would have to cover the *entire* World/graphics/assets API
  as a stable, type-erased surface — enormous cost, and it discards the
  typed ECS API (GAT queries, derives, typed events) just built
- Rejected in favor of a same-rustc fingerprint check.

**2. Scripting/WASM for gameplay**

- ✅ Sandbox, stable boundary, compiler-independence
- ❌ Per-call boundary overhead on queries, a component-access ABI to design,
  wasm-inside-wasm on the web target; no external users to justify it
- Deferred indefinitely; tyroxine covers the scripting niche that matters.

**3. Editor as a library in the game project (game owns both binaries)**

- ✅ No ABI boundary at all
- ❌ Every game change relinks the editor binary and restarts the process,
  losing warm GPU/asset state — incompatible with a long-lived editor driven
  over the remote channel
- Rejected.

**4. True live hot reload (state migration under a running world)**

- ❌ Stale function pointers in hooks/systems/resources, `TypeId` churn,
  closures capturing unloaded code
- Rejected: warm restart gives most of the value with none of the UB surface.

### Consequences

- ✅ Game code uses the full typed engine API directly — the narrow boundary
  is one entry symbol, not a wrapper layer
- ✅ Sub-second reload of a running editor: GPU resources and asset DB stay
  resident, only worlds are rebuilt
- ✅ One snapshot mechanism serves Play mode, scene save/load and reload
- ✅ The same `Plugin` runs statically in shipped binaries, including wasm
- ✅ Demos and editor converge on one bootstrap model (`Plugin` + `Schedules`)
- ⚠️ Dylib mode requires same rustc + same engine build (fingerprint-enforced)
- ⚠️ Dev builds move to a dylib profile; release stays static (two link modes
  to keep healthy in CI)
- ⚠️ GPU caches / asset managers must migrate from world resources to the
  host-owned `EngineContext` (`Arc` + `insert_resource_shared`)
- ⚠️ Undo history and selection reset on reload (accepted as expected behavior)

### Amendment (2026-07-06): Type identity is `QualifiedTypeId`, not bare `TypeId`

**Status**: Accepted (design). The `QualifiedTypeId` / `SourceId` primitive is
introduced as a `HOST`-only, behavior-preserving plinth in #49; #45 wires it to
real per-reload generations.

The original decision (points 3–4) relied on "one copy of `TypeId`s per
process" as the thing that makes `&mut World` sound across the boundary. That
is the *goal*, but bare `TypeId` is the wrong primitive to build on, for two
reasons the original text glossed over:

- **`TypeId` is not guaranteed stable across separately-compiled artifacts.**
  It is derived from the defining crate's `StableCrateId` = hash(crate name +
  `-C metadata` + rustc version). Engine types (`Transform`, defined in the
  shared engine dylib, one compiled copy) *do* unify. But a game compiled as a
  separate `cargo` invocation against `redlilium-*` as external deps gets a
  different `-C metadata` → different `StableCrateId` → **different `TypeId`**,
  and the failure is *silent*: `insert::<Transform>` from the game lands under a
  key the editor's `query::<Transform>` never looks up. Components vanish with
  no link error.
- **Game-defined types get a fresh `TypeId` on every reload** (the game crate's
  metadata changes per build), so a component's identity is not even stable
  across the reload it must survive conceptually.

**Decision.** Type identity in the engine is
`QualifiedTypeId { type_id: TypeId, source: SourceId }`, where
`SourceId(u32)` names the *origin of the type's definition* — **not** the
calling library — with `SourceId::HOST = 0` for the host/shared-dylib and a
monotonic generation index (incremented per dylib reload) for game-defined
types. The source is assigned **at registration time** (engine
`register_std_components` stamps `HOST`; the plugin-load path stamps the
current generation), never inferred from `TypeId::of` in generic code — a
generic call site cannot know where `T` was defined. A `TypeId → SourceId`
map maintained at registration resolves a live `TypeId::of::<T>()` back to its
qualified key.

The source binding **must** be by defining origin (variant B). Binding it to
the *calling* library (variant A, a lib-local static) would tag the shared
`Transform` as `(V, HOST)` when the editor inserts it but `(V, gen_N)` when the
game inserts it — splitting one engine type into two keys. That is exactly the
breakage this amendment prevents.

**Where the tag earns its keep.** Because worlds are dropped on reload and
exactly one game generation is live at a time, a bare `TypeId` is already
unique *within* a live `World` — so `source` is **not** added to the hot query
key. It pays off at the **boundaries**:

1. **`Any::downcast` UB-guard.** Downcasting data produced by generation *N*
   with generation *N+1*'s vtable is UB even if the `TypeId` coincidentally
   matches. `source` lets retained/serialized data from an unloaded generation
   be *refused* rather than reinterpreted. This is the real prize and it
   protects snapshot blobs, host-retained resources, and anything surviving a
   reload.
2. **Load-time fail-fast.** A plugin registering a type whose `TypeId`
   collides with an engine type from a different `source` is an explicit error
   instead of silent aliasing. This subsumes the earlier "ABI probe" idea —
   the qualified id *is* the probe, generalized to a multi-origin world.

**What this does not change.** The shared-engine-dylib discipline (point 4)
stands: engine types must still unify to `SourceId::HOST` on both sides, which
requires the game to link the *same prebuilt* `libredlilium.dylib`
(`-C prefer-dynamic`, not a recompiled copy), same rustc, same engine version,
**same feature set** of the shared crates (feature unification changes
metadata → changes `TypeId`). `QualifiedTypeId` does not remove that
requirement — it makes violations of it *loud* instead of silently dropping
components.

**Not affected.** The world-snapshot core (#48) is single-process and carries
no dylib boundary. Notably the serialization layer is already **name-keyed**
(`SerializedComponent.type_name`, `name_index`, `SnapshotResource::NAME`), so
the save/reload path is naturally resilient to a `TypeId` shift — which is part
of why "reload = snapshot" holds. Only the *live in-process* path (dylib
systems mutating the host's `World`) depends on `TypeId` identity, and that is
what `QualifiedTypeId` guards.

**Known boundaries (to close in #45).** The #49 plinth records a source for
components and resources registered through `register_component` /
`insert_resource*`, and guards the resource downcast on both the convenience
(`resource()`) and production (`Res<T>::fetch_unlocked`) paths. Deliberately
still outside its scope, because they need real generations to matter:

- **Main-thread resources** (`insert_main_thread_resource`) carry no recorded
  source and are not guarded on downcast — the same future risk class as
  regular resources.
- **No unload API yet.** `remove_resource` does not clear `type_sources`, so a
  `TypeId` stays bound to its first source for the world's life. Fine while a
  world is dropped whole on reload; #45 introduces per-generation registration
  and teardown, at which point the guard becomes reachable in production (today
  it fires only via the test-only `force_type_source_for_test` seam).

---

## ADR-021: Device-Local Buffers with a Pooled Staging Belt (Vulkan)

**Date**: 2026-07-05
**Status**: Accepted

### Context

Every Vulkan buffer with `COPY_DST` usage was allocated host-visible
(`CpuToGpu`), because the old heuristic assumed `COPY_DST` implies direct CPU
writes. All mesh buffers carry `VERTEX|COPY_DST`/`INDEX|COPY_DST`, so all
geometry — including write-once static meshes — lived in host-visible memory:
on discrete GPUs every draw fetches vertices over PCIe instead of VRAM
(invisible on unified-memory dev machines, costly on the discrete-GPU
targets). Meanwhile every `TransferOperation::WriteBuffer` allocated and
destroyed a dedicated transient staging buffer per operation per frame.

The upload architecture itself was already correct: production code uploads
through the frame graph (`create_mesh_deferred`, `upload_texture_data`,
material managers' `pending_uploads`), and transient staging retires per frame
slot after the slot's fence.

### Decision

1. **Memory location follows mappability, not copyability.** `MAP_READ` →
   `GpuToCpu`, `MAP_WRITE` → `CpuToGpu`, everything else — including
   `COPY_DST` — → `GpuOnly`. GPU-side copies do not need host-visible
   destinations; buffers the CPU writes directly must say so with `MAP_WRITE`.
2. **The `RING` usage flag marks host-visible intent** — a ring buffer is by
   definition CPU-written every frame, so the Vulkan backend allocates
   `RING` buffers host-visible. The flag stays engine-side: adding wgpu's
   `MAP_WRITE` instead would be invalid there (it only combines with
   `COPY_SRC`), and wgpu rings are written via `Queue::write_buffer`.
3. **`GpuBackend::write_buffer` becomes dual-path**: mapped write for
   host-visible buffers (unchanged); for device-local buffers it is a
   *blocking convenience path* (staging + one-shot submit + wait), mirroring
   the documented `write_texture` contract. Production uploads go through
   `TransferOperation`s in the frame graph.
4. **Staging belt**: transient staging for graph uploads is sub-allocated from
   pooled chunks (bump allocator) instead of one `vkCreateBuffer` +
   allocation per operation. Chunks retire per frame slot (same fence
   guarantee as the old per-op buffers) and return to a free pool;
   oversized allocations get a dedicated chunk destroyed on retirement.

### Consequences

- ✅ Static geometry and uniform buffers reside in VRAM on discrete GPUs
- ✅ Zero per-operation buffer/allocation churn for graph uploads
- ✅ No API changes for production code (deferred paths already used)
- ⚠️ Direct `Buffer::write` on a non-`MAP_WRITE` buffer now takes the slow
  blocking path — fine for tools/tests, wrong for per-frame data (use a
  `RingBuffer` or `MAP_WRITE`)
- ⚠️ Belt chunks hold memory at the high-water mark of a frame's uploads;
  oversized chunks are destroyed on retirement to bound growth

---

## ADR-025: Editor/Game Schedule Separation via Run Conditions

**Date**: 2026-07-10
**Status**: Accepted
**Resolves**: #67, prerequisite for #45

### Context

Editor and game systems previously shared the same schedule objects (Update, FixedUpdate, PostUpdate, Render). This caused:

1. Editor systems (DrawGrid, UpdateFreeFlyCamera) to conflict with game systems during Play
2. Change-detection reset on Play/Pause to affect both editor and game systems indiscriminately, breaking undo/redo boundaries
3. When game plugins load (#45), they'll add systems to shared schedules, creating unmaintainable mixing
4. Pause would freeze infrastructure systems (transforms, asset loading), breaking hot-reload during pause edits

### Decision

Implement single-flag activation model with run conditions to gate systems:

1. **Single `game_active` flag on Schedules** (not per-container)
   - Default: `true` (ensures standalone runtime works without PlayControl)
   - Set to `true` on Play, `false` on Stop
   - Pause freezes game systems (ticks reset) but keeps infrastructure active

2. **FixedUpdate skipped conditionally**
   - Only execute accumulator loop when `game_active` is true
   - Prevents catch-up burst after long pauses via `reset_fixed_accumulator()`

3. **Run conditions gate individual systems**
   - `GameActiveCondition`: system runs only during Play/Pause (game active)
   - `NotGameActiveCondition`: system runs only during Stopped (editor active)
   - Applied to editor-only systems: DrawGrid, DrawSelectionAabb, UpdateFreeFlyCamera
   - Infrastructure (transforms, asset pipeline, render) always active

4. **State transition semantics**
   - **Play** (Stopped → Playing): `set_game_active(true)`, reset game ticks to 0
   - **Pause** (Playing → Paused): reset game ticks to current_tick (freeze game, let infrastructure run)
   - **Resume** (Paused → Playing): no extra reset (game ticks still frozen, resume continues)
   - **Stop** (Playing/Paused → Stopped): `set_game_active(false)`, `reset_fixed_accumulator()`

### Consequences

- ✅ Editor systems cleanly separated from game (no conflicts during Play)
- ✅ Change-detection reset only affects game schedules (undo/redo boundaries fixed)
- ✅ Infrastructure (transforms, assets) continues during Pause (hot-reload works, edits propagate)
- ✅ Foundation for game plugin loading (#45) — plugins add to shared schedules, conditions gate them
- ✅ No split of Render or PreUpdate (edges/system_result reads are container-scoped; avoiding complexity)
- ✅ Backward compatible: standalone runtime defaults `game_active=true`, needs no changes
- ⚠️ Run conditions must be applied correctly; systems added without conditions during Pause may freeze
- ⚠️ TypeId-based schedule identity doesn't cross dylib boundary — must be resolved before #45 ships plugins

---

## ADR-026: Default Query Masks and Entity Visibility

**Date**: 2026-07-10
**Status**: Accepted

### Context

Entity visibility (enabled/disabled, editor-only, hidden during play) must be correctly enforced across:
- Query iteration (Ref<T>, Read<T>, Write<T>)
- Point lookups (get(), contains())
- Side-table invalidation (physics bodies, asset residents)
- Editor inspection during different play states

Without systematic enforcement, queries and side-table checks diverge: a system might miss an entity that should be excluded, leading to stale references or incorrect state.

### Decision

Use flag-based exclusion masks in the query guard layer:

**Default exclude masks:**
- `DEFAULT_GAME_QUERY_EXCLUDE_MASK = DISABLED | STATIC | EDITOR | HIDDEN_IN_PLAY`
  - Used by Ref, RefMut, Read, Write for normal system queries
  - Excludes entities not relevant to game logic
  
- `INFRASTRUCTURE_QUERY_EXCLUDE_MASK = DISABLED | HIDDEN_IN_PLAY`
  - Used by ReadAll, WriteAll for engine-wide readers (asset systems, transforms, rendering)
  - Allows static and editor entities (but not hidden-in-play) to remain visible

**Query filtering:**
- Point lookups (get, contains) respect the mask of their accessor type
- Iteration automatically filters via mask in query guards
- Mask is checked per entity with `entity_flags & mask`

**Side-table synchronization:**
- Systems using side-tables (physics bodies, asset residents) must validate entries against the same mask as the query that populated them
- Added `World::is_excluded_from_game(entity)` helper to check the default game mask
- Invalidation predicates use this helper: `.filter(|e| !world.is_excluded_from_game(*e))`

**Entity visibility matrix during different play states:**

| State | Disabled | Static | Editor | HiddenInPlay | Visible |
|-------|----------|--------|--------|--------------|---------|
| Stopped | ✗ | ✓ | ✓ | ✗ | Normal query sees: enabled, static, editor, not-hidden |
| Playing | ✗ | ✗ | ✗ | ✗ | Normal query sees: game entities only; ReadAll sees: static, editor |
| Paused | ✗ | ✗ | ✗ | ✓ | Normal query sees: game entities; editor entities shown via flag clear |

Note: At Play transition, editor entities receive HIDDEN_IN_PLAY flag (set); at Pause, flag clears (show).

### Consequences

- ✅ Query visibility is systematic and enforced at guard layer
- ✅ Side-table invalidation stays in sync with queries (physics bodies don't outlive hidden entities)
- ✅ Mask-based filtering crosses dylib boundary (u32 flags are ABI-stable, unlike TypeId)
- ✅ Editor UI remains responsive: during Play, hidden entities aren't visible (correct); during Pause, editor entities are shown again
- ✅ Infrastructure systems (transforms, assets) continue during Pause, independent of game visibility
- ✅ Game plugins automatically inherit mask filtering (no special handling needed)
- ⚠️ Mask is unconditional: never add `game_active` conditional inside query iteration (factoring is: flags are rewritten by state transitions, masks are constant)
- ⚠️ Future expansions: entity bits 0–15 reserved for engine; 16+ available for plugins
- ⚠️ Editor UI requiring direct inspection of hidden entities would need `_unfiltered` query accessors (not yet exposed publicly; future enhancement)

### Related Issues

- #67: Schedule separation (complements this decision)
- #63: Entity classification and default filters (depends on this)
- #45: Plugin loading (masks enable safe ABI-stable visibility)

---

## ADR-024: Resource Lifecycle Management — Hybrid Hook + Event Model

**Date**: 2026-07-12
**Status**: Accepted
**Relates to**: #65, #45

### Context

Game resources (Score, RNG, Audio, Physics) need coordinated state changes across Play/Pause/Resume/Stop transitions. Two competing requirements:

1. **Engine-internal needs**: Deterministic, transactional ordering; ability to veto transitions (e.g., snapshot capture failure).
2. **Game-facing needs**: Decoupled from registry; safe for plugins; no function pointers across dylib boundary.

Previous attempts failed:
- Pure hooks: game systems couldn't react (schedule gating during Stop blocks all listeners).
- Pure events: no veto path (failed snapshot commit = partial transition).

### Decision

Implement **hybrid hook + event model**:

**PlayModeAwareRegistry** (hooks, engine-internal):
- Deterministic dispatch order (registration order)
- Called before event emission; can veto transition
- Snapshot capture/restore happen at hook layer
- Used for engine infrastructure (PhysicsWorld pause flag, asset manager cleanup)

**PlayModeTransition** events (game-facing):
- Emitted after hooks complete + snapshot captured + state committed
- Double-buffered; observable by systems in PostUpdate and later schedules
- Game systems subscribe via `EventCursor<PlayModeTransition>`
- Safe for plugins: no registry access needed; events are serialized data, not function pointers

### Execution Model

```
ManagePlayModeTransitions (exclusive system, PreUpdate):
  1. Validation phase (snapshot capture dry-run, hierarchy checks)
  2. Commitment phase (write state, emit event, dispatch hooks)
  3. Hide/show entities, despawn game entities
  
Same frame (PostUpdate+): game systems read events via EventCursor
```

### API Contracts

**SnapshotResource** + **PlayModeAware** interop:
- `on_play_start()`: called before snapshot capture. Resets state to seed (e.g., RNG).
- `on_pause()` / `on_resume()`: called before hooks complete. Freeze/thaw non-snapshotted state (e.g., Physics).
- `on_stop()`: called before snapshot resources are restored. Best left empty; snapshot restore overwrites all state.
- **Rule**: For resources that are both `SnapshotResource + PlayModeAware`, the snapshot restore is the reset; don't also mutate in `on_stop()`.

**Pause-mode edits**:
- Edits during Pause are inspection-only; Stop restores exact pre-play state.
- Rationale: snapshot restore is destructive (overwrites all components + SnapshotResources). Preserving edits requires differential snapshots (complex, error-prone).

**Stop event visibility**:
- Game systems gate off on `GameActive=false` during Stop.
- Stop transition emits event in PreUpdate, but gated-off systems don't see it (they don't run that frame).
- **Contract**: Stop cleanup is engine-mediated only. Game systems react to Play/Pause/Resume, not Stop.
- Corollary: plugins cannot observe Stop events. They must unload cleanly via explicit engine sequence (not event-driven).

### Consequences

- ✅ Transactional safety: validation before any mutation; failed snapshot veto doesn't leave world in half-state.
- ✅ Deterministic: hook dispatch order is registration order; snapshot capture/restore happen together.
- ✅ Plugin-safe: game systems can't accidentally register function pointers; events are data.
- ✅ Pause-mode editing works as expected: Stop reverts to known good state.
- ✅ Deferred work for plugins (#45): plugins will implement `on_play_start()` for initialization; Stop cleanup delegated to engine (warm-reload sequence).
- ⚠️ Stop event not observable by game systems — requires discipline in plugin architecture (Stop = engine-mediated, not event-driven).
- ⚠️ Snapshot restore is all-or-nothing: if snapshot capture fails, Stop aborts and world reverts (safety + visibility trade-off).
- ⚠️ Pause edits discarded: design assumption that Pause is inspection-only. Future work (differential snapshots) could change this.

### Related Examples

- `GameScore` (SnapshotResource, no hooks) — pure state capture/restore.
- `GameRNG` (SnapshotResource + PlayModeAware) — deterministic seeding on Play.
- `PhysicsWorld3D/2D` (PlayModeAware, no SnapshotResource by default) — pause flag toggled by hooks.
- `GameObserverSystem` (event listener) — game reaction pattern, safe for extension.

### Migration Path for #45 (Plugin Loading)

When game plugins load:
1. Plugin calls `world.insert_play_mode_aware::<MyResource>()` to register hooks.
2. On Play: hooks fire; plugin initializes (e.g., RNG seed, asset load).
3. On Pause/Resume: plugin can react if needed (e.g., pause audio).
4. On Stop: engine unloads plugin dylib cleanly; plugin is not observed.

Plugins must NOT:
- Call `PlayModeAwareRegistry::register()` directly (dylib function pointer hazard).
- Rely on observing Stop events (gated-off systems).
- Store game state across unload (rely on SnapshotResource for persistence).

### Design Rationale

**Why hooks fire before event emission?**
Allows veto path. If snapshot capture fails, hooks haven't fired yet; no need to revert. If event fires first, hook failure leaves event-driven state stale.

**Why Stop events are gated-off?**
Simplifies plugin unload: engine stops all game systems (natural gate-off), unloads dylib, cleans up resources. If plugins could observe Stop, they'd need async cleanup (complex). "Stop = engine-mediated only" is simpler and safer.

**Why Pause edits are inspection-only?**
Preserving edits requires tracking which components changed during Pause vs pre-play, then merging them into the restored snapshot. This is complex (differential snapshots) and error-prone. "Pause is inspection; Stop reverts" is simpler and matches user expectation (undo button).

### Related Issues

- #67: Schedule separation (Stop gate-off depends on game_active flag)
- #45: Plugin loading (hooks + events replace hand-rolled plugin initialization)
- #70: Snapshot reliability (restore tests cover round-trip fidelity)


---

## ADR-027: Device Capability Model — Tier Ladder for Renderer Architecture, Capabilities for Orthogonal Axes

**Date**: 2026-07-12
**Status**: Accepted

### Context

Device and instance creation hardcoded desktop-dev-machine assumptions (#38):
features and extensions enabled without querying support, unclamped
anisotropy/sample counts, an engine-level `DeviceCapabilities` that was made up
rather than queried, no headless/Wayland path, no API-version negotiation.
Beyond fixing the hardcodes, the engine needs a durable answer to "how does
renderer code decide what to run on this GPU?" — advanced features (bindless,
ray tracing, mesh shaders) are planned, and a flat pile of boolean feature
flags combinatorially explodes the paths that must be authored and tested.

### Decision

A two-axis model, mirroring what D3D12 (feature levels + `CheckFeatureSupport`),
Metal (GPU families + queries), Unreal (SM tiers + `GRHISupports*`), and
Vulkan Profiles all converged on:

1. **`DeviceTier` — an ordered ladder naming renderer architectures.** Each
   tier includes everything below it. A feature belongs in a tier **only if
   the renderer has a distinct code path built on it** (bindless material
   binding, GPU-driven submission, ray-traced lighting). The device's tier is
   detected once at device creation; devices below `Baseline` fail creation
   with a named-feature error. A new tier may be added **only together with
   the render path that stands on it** — a tier without a consumer is
   premature hardware classification.

2. **`DeviceCapabilities` — orthogonal queried facts.** Everything whose
   consequence is a clamp or a local fallback of a single pipeline/asset
   choice: max anisotropy, supported sample counts, size limits, wireframe
   support, async compute availability. Filled honestly from adapter queries
   by every backend (including Dummy, which reports what it pretends to
   support); read-only outside the backend. Use-sites clamp/validate against
   it — no downstream hardcodes.

**Litmus test**: "different render path" → tier; "clamp or local fallback" →
capability. Texture compression formats are neither — they are an asset-bake
axis (BC on desktop, ASTC/ETC2 on mobile/web); runtime only answers "is this
format supported".

There is deliberately **no user-facing feature-request API** (wgpu-style
requested features): all rendering flows through the render graph, the engine
knows its own requirements, and capabilities are exposed read-only — the same
philosophy as #47's "no manual sync mode".

### Consequences

- ✅ Bounded test matrix: N tiers of render paths instead of 2^k flag combos
- ✅ Missing optional features degrade (wireframe off, anisotropy clamped)
  instead of failing `vkCreateDevice`
- ✅ Engine-level validation agrees with what the backend actually granted
- ✅ Ladder is ready for bindless/RT tiers without restructuring
- ⚠️ `Baseline` is the only tier until a second render path exists; the enum
  carries one variant for now
- ⚠️ Tier detection must stay in sync with each tier's actual feature set —
  centralized per backend in one place

### Related Issues

- #38: device/instance hardcodes (implementation)
- #47: async compute queue availability is a capability, not a tier

---

## ADR-028: Adapter Selection — Declared Policy, Stable Identity, Restart Semantics

**Date**: 2026-07-12
**Status**: Accepted

### Context

The backend commits to one physical device at instance creation, but the
selection logic was policy-free: hardcoded "discrete beats integrated"
scoring, no user override, `enumerate_adapters` a stub, and
`create_device_with_adapter(index)` inviting selection by enumeration index —
which changes with driver updates, eGPU hotplug, and enumeration order.
Multi-GPU machines (hybrid laptops, workstations with compute accelerators)
need a deliberate answer.

### Decision

Adapter choice is a **declared policy, resolved by the backend** — following
DXGI's `EnumAdapterByGpuPreference` and WebGPU's `request_adapter` rather than
Vulkan's raw "app enumerates and picks":

- `AdapterPreference { Auto, HighPerformance, LowPower, Explicit(AdapterId) }`
  on `InstanceParameters`. **`Auto` = `HighPerformance`, always** — no
  context-dependent defaults (an editor silently choosing a different GPU
  than the game would make capabilities differ between them). `LowPower` is
  a deliberate opt-in at graphics initialization.
- **Selection happens once, at instance creation; changing it requires a
  restart.** Device handles pervade every resource; live GPU migration is a
  browser-tier problem we refuse.
- **Explicit selection is by stable identity** (`AdapterId`: PCI
  `vendor:device` or name substring), never by index. Settings persist the
  id; `AdapterInfo::id()` provides it. An explicitly requested adapter that
  is absent fails loudly — never a silent fallback.
- **`REDLILIUM_ADAPTER` env var** overrides everything (CI, multi-GPU dev
  machines, support).
- Filtering precedes scoring (ADR-027): baseline tier, then presentability —
  display-less compute accelerators are excluded via the surfaceless
  platform queries (`vkGetPhysicalDeviceWin32PresentationSupportKHR`; other
  platforms have no surfaceless query and presume presentable). Scoring:
  preferred device type always wins, device-local memory as tiebreaker,
  software renderers never beat hardware.
- `enumerate_adapters` exists for UI listing (editor settings page), not for
  selection. On WebGPU there is no enumeration — `Explicit` degrades to a
  power hint with a warning.

### Consequences

- ✅ Deterministic default: same GPU for editor and game
- ✅ Persisted choices survive driver updates and hotplug (stable ids)
- ✅ Compute-only accelerators can no longer win selection and die at the
  first window (Windows; other platforms rely on compositor routing)
- ✅ One env var turns any machine into a targeted test box
- ⚠️ No live adapter switching; settings UI must say "requires restart"
- ⚠️ No multi-adapter rendering — by the ADR-027 litmus, no mechanism
  without a consuming render path

### Related Issues

- #38: device/instance hardcodes (XB-L3: adapter enumeration stub)
