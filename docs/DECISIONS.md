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
**Status**: Accepted; hosting model inverted by ADR-037 (2026-07-16) — the
game project owns the editor binary (this ADR's Alternative 3, re-evaluated
under the ADR-036 two-world model), and the cdylib narrows to play-world
behavior reload. The plugin contract, `redlilium-runtime`, and the ABI
fingerprint machinery stand.

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
**Status**: Accepted (decision 3's blocking device-local path removed by ADR-031)

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
**Status**: Superseded by ADR-036 (2026-07-16) — the one-world gating model
(`game_active`, `GameActiveCondition`/`NotGameActiveCondition`) is deleted;
editor Play runs the game in a separate world built by the standalone
composition.
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
**Status**: Accepted; amended by ADR-036 (2026-07-16) — the
`HIDDEN_IN_PLAY`/`INHERITED_HIDDEN_IN_PLAY` flags and every play-state
visibility rule below are deleted (editor Play no longer shares a world with
the game). The masks are now `DISABLED | STATIC | EDITOR` (game queries) and
`DISABLED` (infrastructure queries); the rest of this ADR stands.

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
**Status**: Superseded by ADR-036 (2026-07-16) — `PlayModeAware`, the
registry, `PlayModeTransition` events, and `ManagePlayModeTransitions` are
deleted; a play session's lifecycle is now the play *world's* lifecycle
(created on Play, dropped on Stop), so there are no in-world transitions for
resources to coordinate around. `SnapshotResource` survives (warm reload).
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

### Amendment (2026-07-14, #94): synchronization2 is a baseline requirement

The Vulkan backend records every barrier via `vkCmdPipelineBarrier2` and
submits via `vkQueueSubmit2` — there is no sync1 path. `VK_KHR_synchronization2`
(extension + `synchronization2` feature bit) therefore joins the baseline-tier
requirement list alongside dynamic rendering, timeline semaphores, and
shaderDrawParameters: a device lacking it fails the baseline filter with the
named gap `"synchronization2 feature"` and falls back to the wgpu backend (or,
for explicit `BackendType::Vulkan`, fails loudly). Rationale: sync2 is core in
Vulkan 1.3 and, on 1.2, shipped in the same driver wave as dynamic rendering
(which baseline already requires), so the set of devices passing today's filter
but lacking sync2 is effectively empty. Enforced in
`backend/vulkan/device.rs::{baseline_gaps, create_logical_device}`.

### Amendment (2026-07-14): baseline is Vulkan 1.3 core (sync2 + dynamic rendering)

The Vulkan baseline tier is raised from 1.2 to **1.3**. Both
`synchronization2` and `dynamicRendering` are core *and mandatory* in 1.3, so
the backend stops enabling the `VK_KHR_synchronization2` / `VK_KHR_dynamic_rendering`
extensions and calls the core entry points (`vkCmdPipelineBarrier2`,
`vkQueueSubmit2`, `vkCmdWriteTimestamp2`, `vkCmd{Begin,End}Rendering`) directly
on `ash::Device` — no `khr::*::Device` loaders. Consequences:

- The instance and per-device version gates require 1.3
  (`instance.rs::MINIMUM_API_VERSION`, `device.rs::baseline_gaps`); a 1.2-only
  loader/device fails the filter and Auto falls back to wgpu.
- `baseline_gaps` no longer probes the two extension *names* (a conformant 1.3
  driver may drop the promoted names) — it verifies the feature bits via
  `PhysicalDeviceVulkan13Features` instead, and enables them the same way at
  device creation.
- Rationale: MoltenVK (the macOS path, now the default backend everywhere)
  reports 1.4 and satisfies 1.3 core, so nothing is lost on Apple; requiring
  1.3 removes the extension-advertisement dependency and the dual sync1/sync2
  bookkeeping the KHR loaders implied.

### Amendment (2026-07-14, #95): `gpu_timestamps` is a queried capability

Per-pass GPU timing is exposed as `DeviceCapabilities.gpu_timestamps: bool` —
a capability (an orthogonal axis), never a tier gate. It is queried, never
fabricated (the ADR-027 contract): the Vulkan backend reports `true` only when
the graphics queue family exposes a non-zero `timestampValidBits` and a
non-zero `timestampPeriod`; wgpu and dummy report `false`. When false,
`GraphicsDevice::latest_gpu_timings()` returns an empty `FrameGpuTimings` and
the editor stats window degrades to "unavailable" — no feature is denied, the
data axis is simply absent. Collection lives entirely in the Vulkan backend
(`backend/vulkan/timestamps.rs`): one `vkQueryPool` per (queue, frame-slot),
`vkCmdWriteTimestamp2` (sync2, #94) around each pass and submit, read back
without a WAIT bit when the slot retires in `advance_frame` (the slot fence
already guarantees completion — the staging-belt retirement argument).
Durations only: timestamps from different queues are not compared (that needs
`VK_EXT_calibrated_timestamps`, out of scope).

### Amendment (2026-07-14, #97): GPU crash breadcrumbs are instance/debug tooling, NOT a capability

Per-pass GPU crash breadcrumbs (post-mortem for `VK_ERROR_DEVICE_LOST`:
"which pass did the GPU die in") are deliberately **not** a
`DeviceCapabilities` field. They are debug tooling, gated by an
`InstanceParameters` option (`BreadcrumbsMode`, default on when validation is
enabled, `REDLILIUM_BREADCRUMBS` env override — the ADR-028 override
precedent), not a queried device property the renderer clamps against.
"Breadcrumbs active: <mechanism>" is a single startup info line instead. The
mechanism is chosen at backend creation by extension availability —
`VK_NV_device_diagnostic_checkpoints` → `VK_AMD_buffer_marker` → a portable
`vkCmdFillBuffer` fallback (always available, coarser) — all behind one
interface encoding the same per-pass marker code, so one pure diagnosis
function serves every mechanism (`backend/vulkan/breadcrumbs.rs`).
`VK_EXT_device_fault` is orthogonal: its structured report is appended where
present. The happy path with breadcrumbs off adds zero encode work (the guard
is at the call site, not inside the marker fn). Granularity is per-pass, never
per-draw (per-draw is where breadcrumbs start costing real perf).

### Related Issues

- #38: device/instance hardcodes (implementation)
- #47: async compute queue availability is a capability, not a tier
- #94: synchronization2 baseline requirement (sync1 path deleted)
- #95: per-pass GPU timestamps surfaced as the `gpu_timestamps` capability
- #97: GPU crash breadcrumbs are instance/debug tooling, not a capability

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

---

## ADR-029: Camera Graphics Stack — Serializable Spec, Derived Target, Virtual Texture Assets

**Date**: 2026-07-12
**Status**: Accepted

### Context

Render-to-texture with a per-camera setup was impossible: the camera's GPU
target (`CameraTarget`, holding `Arc<Texture>`) was created by host code
(duplicated between the runtime and the editor), only the first camera
rendered, and nothing could reference a camera's output. The naive fix — a
component owning the GPU target — dead-ends on serialization: GPU resources
cannot live in scene assets.

### Decision

Split the camera's graphics stack the way Unity (`RenderTexture` asset),
Unreal (`TextureRenderTarget2D`), and Bevy (`RenderTarget::Image(Handle)`)
all do — **serialize the intent, derive the resource, reference by stable
identity**:

1. **`CameraOutput` — the serializable spec** (plain data): `Screen` (main
   viewport, composited by the host) or `Offscreen { size, output }` with
   `SizePolicy::{Viewport, ViewportScale, Fixed}`, plus the clear color.
   Formats are engine-standard (`Rgba8Unorm` + `Depth32Float`) until a
   consumer needs per-camera formats.
2. **`CameraTarget` — the runtime-only derived cache** (unchanged type,
   `#[skip_serialization]`), now owned by the **`EnsureCameraTargets`**
   system (Render schedule, before `ForwardRender`): it re-derives targets
   whenever they disagree with the spec. Host-side `ensure_camera_target`
   code is deleted; the host only publishes `MainViewport` and (runtime)
   defaults the first camera to `CameraOutput::screen()`. Cameras **without**
   a spec stay host-managed — the editor's scene view keeps its own path
   until migrated.
3. **Virtual texture assets** — `TextureSource::Virtual(Guid)` +
   `TextureManager::publish_virtual`: an asset whose GPU texture is
   *published* by an engine system instead of loaded. An offscreen camera
   with `output: Some(guid)` publishes its color target under that identity;
   materials sample it like any texture (mirrors, minimaps, portals).
   Precedent: `TextureSource::Solid`, `MeshSource::Generated` (non-file
   sources are an established concept). Republish on resize bumps the
   resident generation, so `AssetRef` holders re-resolve automatically.
4. **`ForwardRender` iterates all cameras** — one pass per
   `Camera + CameraTarget` entity, offscreen passes emitted first, the
   primary (Screen or host-managed) camera last. **Cross-camera ordering is
   never declared**: a camera sampling another's output is ordered by the
   render graph's automatic resource-dependency derivation (#47) — no
   Unity-style camera depth/priority knobs.

### Consequences

- ✅ Render-to-texture is authorable scene data (component + GUID reference)
- ✅ One implementation of target sizing instead of two host copies
- ✅ Serialization dead-end resolved without GPU handles in assets
- ✅ Camera ordering falls out of the frame graph for free
- ⚠️ Editor scene view still host-managed (follow-up migration)
- ⚠️ Per-camera render *paths* (quality variants via #6 system axes,
  deferred tiers per ADR-027) are declared but not yet implemented —
  `CameraOutput` is the natural home for that knob when its consumer exists

### Amendment (2026-07-16, ADR-035): the render-path knob is a separate component

The consumer now exists (ADR-035), and the knob lands as a **separate
serializable `RenderPath` component**, not a `CameraOutput` field as this
ADR anticipated. Rationale: `CameraOutput` is the contract with *consumers*
of the camera's image (where it goes, its size and format); the render path
is the camera's internal production recipe, and its settings payload grows
independently (shadow parameters, quality tiers, HDR chains). Everything
else in this ADR — serialize the intent, derive the resource, graph-derived
ordering — carries over unchanged to the pipeline layer.

### Related Issues

- #74: framework hoist reframed (the frame driver shrinks; camera stack ADR)
- #47: automatic graph ordering makes multi-camera safe
- #6: per-camera quality = system variant axes (future)
- #2: scene save/load consumes the serializable spec
- #128 / ADR-035: per-camera render pipelines (the declared path knob lands)

## ADR-030: Per-Resource Sharing Mode for Cross-Queue Textures

**Date**: 2026-07-13
**Status**: Accepted

### Context

Phase 4 of #47 created **every** buffer and image `VK_SHARING_MODE_CONCURRENT`
whenever the device exposes a distinct async compute family, so cross-queue
access needs no queue-family ownership transfers (QFOT). CONCURRENT images can
cost framebuffer compression (DCC) on AMD hardware — the canonical guidance
(GPUOpen's DOOM article, GCN era) reports a ~3% whole-frame win from EXCLUSIVE
sharing. Research for #88 found: the ecosystem avoids full QFOT (Granite, AnKi,
vkd3d/D3D12 — D3D12 has no QFOT concept at all); `VK_KHR_maintenance9` (2025)
makes QFOT optional where the driver reports it unnecessary; but on Windows the
RDNA1/2 driver branch no longer receives new Vulkan extensions, so ~6% of the
Steam fleet (frozen GCN + Windows RDNA1/2) will never have maintenance9.

### Decision

1. **Textures are EXCLUSIVE by default.** `TextureDescriptor::cross_queue`
   (builder: `with_cross_queue`) declares a texture as accessible by both the
   graphics and the async compute queue; only declared textures are created
   CONCURRENT (and only when a distinct compute family exists). This recovers
   DCC for the overwhelming majority of images on all hardware generations.
2. **Buffers stay CONCURRENT** whenever the async queue exists: compression
   does not apply to buffers, and engine-internal staging (belt chunks) is
   read by async-routed transfer graphs.
3. **Routing safety**: `execute_graph` honors the async hint only when every
   texture the graph touches is declared cross-queue. Accessing an EXCLUSIVE
   image from another queue family leaves contents undefined per spec, so an
   undeclared texture silently falls the graph back to the graphics queue —
   consistent with the hint's existing "honored only when safe" semantics.
4. **maintenance9 fast path** (implemented): when the extension is available
   (hand-rolled FFI in `backend/vulkan/maintenance9.rs` until ash catches up)
   and the graphics/compute families may *mutually* implicitly acquire each
   other's optimal images (`VkQueueFamilyOwnershipTransferPropertiesKHR`),
   declared cross-queue textures stay EXCLUSIVE too — no transfers, no
   compression loss. Skipped when validating under a layer older than the
   extension (spec < 1.4.316), which would emit false "unknown
   VkStructureType" errors and disable its handling of the extension.
5. **Full QFOT (paired release/acquire) is not implemented.** The streaming
   submit model records command buffers before future consumers are known,
   making the release side structurally awkward, and the measured driver
   reality (below) shows the payoff is a handful of images on one vendor.

### Measured driver reality (RX 9070 XT + RX 6400 + RTX 3070, 2026-07)

The AMD Windows driver reports `optimalImageTransferToQueueFamilies == 0`
for **every** queue family: even with maintenance9 enabled, AMD requires
real ownership transfers for optimal-tiling images (consistent with DCC
metadata needing a handoff). On AMD the fast path therefore stays off and
declared textures use the CONCURRENT fallback — which further validates not
building full QFOT. **On NVIDIA the fast path engages** (RTX 3070, driver
573-era: every family reports `0b111111`, so declared textures stay
EXCLUSIVE with no transfers).

**The CONCURRENT fallback measures as free on both AMD generations tested.**
Methodology (`async_overlap_demo`): solid-color fullscreen fills into two
8192² RGBA8 targets — 256 MiB each, larger than any Infinity Cache so fills
are bandwidth-bound; solid color is the best case for compression, so losing
it would show directly as fill time.

- **RDNA4 (RX 9070 XT)**: EXCLUSIVE 335.6 µs vs CONCURRENT 300.0 µs; an
  uncompressed 256 MiB write at ~640 GB/s cannot beat ~440 µs, and both fills
  do. (RGP's Compression column reports N/A on RDNA4 — timing only.)
- **RDNA2 (RX 6400)**: EXCLUSIVE 928 µs vs CONCURRENT 950 µs, both far under
  the ~2.1 ms uncompressed floor at ~128 GB/s — and here **RGP reports
  `Compression: ON` for the CONCURRENT target directly**, driver-confirmed,
  not merely inferred.

The GCN-era "CONCURRENT loses DCC" penalty does not reproduce on RDNA2 or
RDNA4. It remains unmeasured on GCN proper (Polaris/Vega); `async_overlap_demo`
is the harness if such hardware appears.

### Consequences

- ✅ DCC retained for all undeclared images (render targets, sampled assets)
- ✅ No new synchronization semantics; #82-validated CONCURRENT path unchanged
  for declared resources
- ✅ Async hint stays a hint: correctness never depends on the declaration
- ⚠️ Declared textures fall to CONCURRENT on hardware without maintenance9
  (~6% of fleet), but measured DCC loss there is nil on RDNA2/RDNA4 — the
  concern is theoretical for the AMD generations tested, real only on
  unmeasured GCN, and a frame shares only a handful of such images anyway
- ⚠️ Forgetting the declaration serializes an intended-async graph onto the
  graphics queue; each offending texture logs one warning (a hard error would
  break the hint semantics — the same graph legitimately runs single-queue on
  devices without an async queue)

### Related Issues

- #47: multi-queue design (CONCURRENT-everything superseded by this ADR)
- #82: hardware validation of the CONCURRENT path
- #88: research, staged plan, maintenance9 follow-up

## ADR-031: Dedicated Transfer Queue for Asset Streaming

**Date**: 2026-07-13
**Status**: Accepted

### Context

Discrete GPUs expose transfer-only queue families backed by DMA engines (AMD
SDMA, NVIDIA copy engines) that move data across PCIe in parallel with both
graphics and compute at zero CU/SM cost. RedLilium never created one:
`plan_queues` looked for graphics + dedicated compute only, and every asset
upload rode the main frame graph on the graphics queue (`flush_gpu` injected
its `TransferPass` there), serializing streaming with rendering. Vendor
guidance (AMD & NVIDIA) splits transfer work in two: **host→device streaming
belongs on the transfer queue**; **small in-frame dependent copies belong on
the consumer's queue** (DMA engines have weaker device-local bandwidth, and a
semaphore round-trip adds latency the frame immediately pays).

### Decision

1. **Plan a third queue** from a transfer-only family (`TRANSFER` without
   graphics/compute/video/optical-flow bits). A coarse
   `minImageTransferGranularity` is **not** a disqualifier (#92): AMD SDMA
   reports 16×16×8, but buffer copies are granularity-exempt and
   whole-subresource image copies (offset 0, full extent — what asset
   streaming emits) are legal at any granularity. The family is planned with
   its granularity recorded, and the routing layer gates per op: on a coarse
   family a graph with a partial/non-base image copy falls down the ladder
   instead. A 1×1×1 family (NVIDIA copy engines) accepts every copy. No
   same-family fallback: without DMA engines a "transfer queue" buys nothing.
2. **`QueuePreference { Graphics, AsyncCompute, Transfer }`** replaces the
   `prefer_async_compute` bool on `RenderGraph`. Placement stays an explicit
   hint with a fallback ladder — `Transfer` requires a transfer-only graph
   (a transfer family cannot execute compute) and falls back to async
   compute, then graphics; every rung is first-class. Ordering remains fully
   automatic: the #47 trackers generalize to `QUEUE_COUNT = 3` and emit
   timeline waits for cross-queue hazards exactly as before.
3. **Asset streaming is the first consumer**: `AssetProcessor::flush_gpu`
   returns a transfer-only graph flagged `Transfer` instead of injecting
   into the frame graph; hosts submit it before the main graph. In-frame
   copies (material props, ring-buffer data, readbacks) stay on the
   consumer's queue by design.
4. **Sharing generalizes to three families**: staging-belt chunks and
   buffers go CONCURRENT across every distinct family; asset textures are
   declared `cross_queue` (ADR-030 semantics; EXCLUSIVE under the
   maintenance9 implicit fast path, which now requires pairwise support
   across all planned families).
5. **Both blocking convenience paths are deleted.** `write_texture` is gone
   on every backend (it had no callers), and `write_buffer` on a
   device-local buffer is now an error instead of a one-shot staging copy +
   synchronous wait (superseding ADR-021 decision 3) — with the transfer
   queue in place there must be no bypass around the frame graph.
   `write_buffer` remains the mapped-write primitive for host-visible
   (RING/MAP_WRITE) buffers only.

### Consequences

- ✅ On hardware with DMA engines, asset uploads overlap rendering and
  compute; visible as SDMA/copy-engine occupancy in RGP/Nsight
- ✅ Single-queue devices (MoltenVK default, wgpu, web) are untouched — the
  ladder lands on graphics, one extra submit per uploading frame
- ✅ No new synchronization API: routing is the only knob, ordering is
  derived (#47 invariant preserved)
- ⚠️ QFOT for the streaming handoff (upload once → sample forever) remains
  deliberately unimplemented pending #88 measurements on RDNA1/2 + NVIDIA;
  maintenance9 covers the fast path where drivers grant it
- ⚠️ A `Transfer`-flagged graph touching an undeclared texture silently
  falls back to graphics (one warning per texture) — same hint semantics as
  ADR-030

### Amendment (2026-07-14, #96): requires-graphics ops split off the transfer graph

GPU mip-chain generation (`TransferOperation::GenerateMipmaps`, a
`vkCmdBlitImage` chain) is legal **only on a graphics-capable queue** —
neither the transfer family nor a dedicated compute family can blit. So the
per-op routing gate this ADR introduced grows a *requires-graphics* rung:
`RenderGraph::requires_graphics_queue()` makes a graph containing such an op
ineligible for both the Transfer and AsyncCompute routes (alongside the
existing `has_graphics_passes` check). `AssetProcessor::flush_gpu` now returns
**two** graphs per flush: the mip-0 upload stays on the transfer-routed graph;
the mip-gen blit chain goes into a second, graphics-routed graph pushed after
it. Cross-queue ordering (upload on transfer → blit on graphics → sample in
the frame) is derived automatically by the trackers — no manual sync. The op
is gated by `DeviceCapabilities.mip_generation` (Vulkan only; the loader falls
back to a single mip on wgpu/dummy and on blit-ineligible formats). See #96.

### Related Issues

- #89: implementation
- #47: multi-queue machinery the routing generalizes
- #88: sharing-mode measurements gating a QFOT revisit
- #96: GPU mip generation — a requires-graphics transfer op routed off the
  transfer queue

## ADR-032: Inline Ray Tracing via VK_KHR_ray_query

**Date**: 2026-07-15
**Status**: Accepted

### Context

The engine reached the point where an advanced GPU feature is the right next
step (#110). Two candidates: mesh shaders (meshlets) and hardware ray
tracing. The raw-ash Vulkan backend was chosen in ADR-007 precisely to keep
such extensions reachable; the ADR-027 capability model reserved slots for
them. Mesh shaders need new `ShaderStage::{Task, Mesh}` variants, a SPIR-V
bake path (Slang's WGSL output cannot express them), and meshlet
preprocessing (meshoptimizer) from scratch. Inline ray tracing via
`VK_KHR_ray_query` needs none of that: ray queries run inside existing
fragment/compute shaders, and naga — the engine's WGSL→SPIR-V path on the
Vulkan backend — already supports WGSL ray queries, emitting
`SPV_KHR_ray_query`. Both dev GPUs (RTX 3070, RX 6400 / RDNA2) support the
extension bundle, enabling two-vendor validation of the sync code.

### Decision

1. **Capability, not tier** (per the ADR-027 litmus): `ray_query` lands as a
   `DeviceCapabilities` field — no renderer code path stands on it yet. When
   ray-traced lighting becomes a real render path, *that* change introduces
   the tier rung.
2. **All-or-nothing bundle**: `VK_KHR_acceleration_structure` +
   `VK_KHR_ray_query` + `VK_KHR_deferred_host_operations` extensions plus the
   `accelerationStructure`, `rayQuery`, and Vulkan 1.2 `bufferDeviceAddress`
   feature bits. Partial support reads as unsupported. The gpu-allocator is
   created with `buffer_device_address` exactly when the bundle is enabled.
3. **AS storage and scratch are ordinary engine `Buffer`s**, so the existing
   buffer-access tracker covers builds and traversals with no new tracking
   machinery. Four new `BufferAccessMode`s lower to the
   `ACCELERATION_STRUCTURE_BUILD` stage / `ACCELERATION_STRUCTURE_{READ,WRITE}`
   access masks (build input, TLAS-build BLAS read, build write, ray-query
   shader read).
4. **Builds go through the frame graph** (the Phase-0 red line): a fourth
   pass kind, `AccelerationStructureBuildPass`, encodes
   `vkCmdBuildAccelerationStructuresKHR`. Builds within one pass are
   independent by contract; BLAS→TLAS ordering is expressed as pass
   dependencies, and hazards derive from the declared access modes like every
   other pass. The pass is legal on any compute-capable queue (never the
   dedicated transfer queue).
5. **TLAS instance streaming is ring-buffered**: a `Tlas` owns one
   host-visible instance buffer per frame slot; `write_instances` rotates so
   the CPU never overwrites data an in-flight frame's build reads. The TLAS
   snapshots which BLASes its instances reference — that snapshot feeds both
   barrier inference and keep-alive.
6. **Binding model**: `BindingType::AccelerationStructure` →
   `VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR`,
   `BoundResource::AccelerationStructure(Arc<Tlas>)`. Material resource
   extraction declares the TLAS **and** its referenced BLAS backings as
   shader reads (traversal touches both).
7. **Vulkan-only**: wgpu/dummy report `ray_query: false`; wgpu rejects AS
   buffers/bindings/passes with a clear error; the dummy backend accepts
   creation (placeholder handles) so graph-level tests run headlessly.

### Consequences

- ✅ `ray_query_demo`: fullscreen fragment-shader ray tracing (primary rays +
  hard shadows) over GPU-built BLAS/TLAS, TLAS rebuilt per frame; validated
  on RTX 3070 and RX 6400 with sync validation clean
- ✅ WGSL ray-query shaders ride the existing naga path — no new bake
  machinery; guarded by a unit test compiling one to SPIR-V
- ✅ Barrier correctness is machine-checked: the new access modes flow
  through the same tracker the sync validation layer (#99) already audits
- ⚠️ BLAS geometry buffers must be created with
  `BufferUsage::ACCELERATION_STRUCTURE_INPUT` — meshes loaded through the
  standard path don't carry it yet (opt-in mesh AS input is #110 phase 2)
- ✅ BLAS compaction landed (#110 phase 3): `BlasDescriptor::with_compaction()`
  swaps a built structure for a smaller compacted copy transparently a few
  frames later (RTX 3070: 59–70% smaller, sync validation clean)
- ⚠️ Slang cannot target WGSL ray queries, so ray-query shaders are authored
  in WGSL directly for now; a Slang→SPIR-V bake variant is the escape hatch
  when they need to join the baked-material world

### Amendment (2026-07-15): phase decisions

Three follow-up forks were decided with the project owner:

1. **RT shadows integrate as a separate WGSL shadow-mask pass** — a
   fullscreen ray-query pass reconstructs positions from depth and writes a
   shadow-mask texture; the Slang resolve shader samples it as a plain
   texture. The bake pipeline stays untouched; the Slang→SPIR-V bake variant
   is deferred until ray queries need to live inside baked materials.
2. **BLAS compaction swaps transparently in place**: `Blas` gains interior
   mutability for its (handle, backing) pair; the old pair goes into deferred
   retirement until the frame fence. Callers are unaffected —
   `write_instances` picks up the new device address on the next write, while
   in-flight builds keep the old AS alive through their Arc snapshots.
3. **wgpu ray query is dropped** (was phase 4): browser WebGPU has no RT and
   desktop wgpu is shadowed by the native Vulkan backend — dead weight in the
   backlog.

### Update (phase 3 implemented)

Both remaining phases landed together:

- **Async-compute AS builds** ride the existing per-graph queue routing — an
  `AccelerationStructureBuildPass`-only graph with
  `QueuePreference::AsyncCompute` runs on the async compute queue, its
  cross-queue hazards resolved as timeline waits by the trackers (no new sync
  code). `ray_query_demo` builds its BLASes there.
- **Transparent compaction** works as decided: a per-frame-slot
  `ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR` query (per-query reset, so it is
  safe on a single per-slot pool even when the build ran on async compute) reads
  the compacted size back a few frames after the build; a graphics-side driver
  then encodes `vkCmdCopyAccelerationStructureKHR(COMPACT)` through the frame
  graph via `RenderGraph::flush_acceleration_structure_compaction` and swaps the
  `Blas`'s interior-mutable (backing, handle) pair once the copy is guaranteed
  complete, holding the original one further frames-in-flight window (in-flight
  TLAS instance buffers may still carry its device address). The swap/retire
  clock is a monotonic frame counter advanced in `advance_frame`, so it is
  correct regardless of how often the flush runs.

### Related Issues

- #110: feature issue (phase 2: mesh AS input + shadow-mask pass in
  pbr_ibl; phase 3: transparent compaction + async-compute builds)
- #99: sync validation that machine-checks the new barrier paths

## ADR-033: Meshlet Rendering via VK_EXT_mesh_shader

**Date:** 2026-07-15
**Status:** Accepted
**Issue:** #111

### Context

The second "advanced GPU" pillar after inline ray tracing (ADR-032):
meshlet-based rendering with GPU-side per-meshlet culling. Unlike ray query
— where the WGSL/naga path carried the whole feature — **naga has no
mesh-shading support at all**: there is no WGSL syntax for task or mesh
shaders, so the entire shader toolchain question had to be answered first.

### Decision

**Task + mesh pipelines via `VK_EXT_mesh_shader`, authored in Slang, baked
as SPIR-V.** Vulkan-only by construction (WebGPU has no mesh shaders).

1. **Capability, not tier** (ADR-027 litmus): `DeviceCapabilities::
   mesh_shading` — `VK_EXT_mesh_shader` with both `taskShader` and
   `meshShader` feature bits, all-or-nothing. No renderer path stands on it
   yet. wgpu/dummy report `false`; wgpu additionally rejects mesh materials
   and mesh-tasks draws with `FeatureNotSupported`.
2. **Stages are first-class:** `ShaderStage::Task`/`Mesh`,
   `ShaderStageFlags::TASK`/`MESH`, descriptor visibility, baked-key stage
   tags, SPIR-V entry-point probing (TaskEXT/MeshEXT execution models).
   Stage-combination rules live in
   `MaterialDescriptor::validate_stage_combination`: vertex XOR mesh, task
   implies mesh, mesh materials carry no vertex layout.
3. **Second baked artifact kind — SPIR-V.** The default build (Slang off)
   serves Slang sources from the offline bake, but there is no WGSL form for
   mesh stages, so `xtask bake-shaders` emits a `BAKED_SPIRV` table (bytes =
   Slang's SPIR-V verbatim, same `shader_key` keyspace). **A spec containing
   any task/mesh entry bakes ALL its entries to SPIR-V** — Slang's WGSL
   target refuses to even load a module with mesh-shading entry points, so
   the set's fragment stage cannot bake as WGSL either. The runtime tries
   the WGSL table first, then SPIR-V, and misses loudly. `bake-shaders
   --check` covers the new table unchanged.
4. **Pipelines:** `create_graphics_pipeline` takes an ordered stage list and
   an `Option<(&VertexLayout, PrimitiveTopology)>`; mesh pipelines omit the
   vertex-input and input-assembly states entirely (spec-ignored anyway).
5. **Draws without a Mesh:** `MeshTasksDrawCommand { material, group_count }`
   + `GraphicsPass::add_draw_mesh_tasks` → `vkCmdDrawMeshTasksEXT`. Geometry
   lives in storage buffers bound through the material instance, so resource
   inference is `extract_material_resources` alone and the existing
   `BufferAccessTracker` provides the barriers.
6. **Barrier stage unions gain TASK/MESH bits** — but only when the
   extension is enabled (stage flags from a disabled extension are invalid
   API use). The augmentation lives in `BufferAccessTracker` so the recorded
   source scopes and the emitted destination scopes cannot disagree.

### Consequences

- Mesh-shading materials are Slang-only; the WGSL path fails with the
  engine-level cause. Web never sees the feature.
- The `meshlet_demo` partitions a UV sphere into 32 meshlets at generation
  time (grid patches, deliberately no meshoptimizer), instances it 64×, and
  culls per meshlet in the task stage (bounding sphere vs. frustum +
  meshoptimizer-convention backface cone) against a **freezable cull
  camera** — SPACE freezes it, so orbiting on shows the holes where culled
  meshlets were. Validated on RTX 3070 and RX 6400, 300 frames each under
  `REDLILIUM_SYNC_VALIDATION=1`, zero hazards.
- **Known limitation:** texture layout transitions do not yet augment their
  stage masks with TASK/MESH — a texture sampled from a task/mesh stage
  would get a too-narrow barrier scope. No current shader does this; sync
  validation (#99) is the backstop, and the fix mirrors the buffer-side
  augmentation.
- Windows bake gotcha: the Vulkan SDK's `slang.dll` shadows the pinned one
  on PATH — prepend `$SLANG_DIR/bin` before `bake-shaders` or the
  `BAKED_SLANG_TAG` drifts (same class of issue commit ca76afa fixed for
  `test-all.sh`).

### Related Issues

- #111: feature issue
- #110 / ADR-032: the ray-tracing sibling — shared capability-vs-tier
  reasoning, shared "advanced GPU" demo conventions
- #99: sync validation that machine-checks the new barrier paths

## ADR-034: Bindless Texture Heap via Descriptor Indexing

**Date:** 2026-07-15
**Status:** Accepted
**Issue:** #117

### Context

The third "advanced GPU" pillar after ray query (ADR-032) and mesh shading
(ADR-033): runtime-sized, update-after-bind descriptor arrays indexed
non-uniformly from shaders — the prerequisite for GPU-driven rendering,
per-instance materials in ray-query shaders, and textured meshlet materials.
Three forks were settled with the owner up front: **phase-1 scope is sampled
2D textures + samplers** (buffers already reach shaders via
`bufferDeviceAddress`), **registration is an explicit opt-in** (render
targets never burn slots), and **the heap is an explicit material group**,
not a reserved set 0 (no global set-numbering migration).

### Decision

1. **Capability** (ADR-027): `DeviceCapabilities::bindless` — five Vulkan
   1.2 core feature bits (`runtimeDescriptorArray`,
   `descriptorBindingSampledImageUpdateAfterBind` — which also covers
   SAMPLER bindings per the binding-flags VUIDs,
   `descriptorBindingPartiallyBound`,
   `descriptorBindingUpdateUnusedWhilePending`,
   `shaderSampledImageArrayNonUniformIndexing`), all-or-nothing, queried at
   selection. No extension: the engine baseline is Vulkan 1.3. wgpu/dummy:
   false.
2. **One device-owned heap**: a dedicated `UPDATE_AFTER_BIND` pool holding a
   single persistent set — binding 0 = sampled-texture array, binding 1 =
   sampler array, both `PARTIALLY_BOUND | UPDATE_AFTER_BIND |
   UPDATE_UNUSED_WHILE_PENDING`. Capacities are engine caps (16384 / 256)
   clamped by the update-after-bind device limits. The heap's set layout is
   created through the same `ds_layout_cache` material pipelines use, so
   set compatibility holds by construction.
3. **Registration writes descriptors immediately**
   (`GraphicsDevice::bindless_register_texture/_sampler` → slot `u32`);
   legal while the heap is bound in flight because a fresh slot is never
   dynamically used. **Slot recycling is fence-deferred**: an unregistered
   slot (and its keep-alive `Arc`) waits `MAX_FRAMES_IN_FLIGHT` frame
   advances in `BindlessSlots` before reuse.
4. **Barriers stay automatic**: the heap group's entries carry
   `BoundResource::BindlessHeap(Arc<BindlessSlots>)`; pass resource
   inference declares every live (and retirement-pending) texture as
   `ShaderRead`, so a freshly uploaded texture transitions
   TransferDst → ShaderReadOnly before its first bindless draw.
5. **Shaders are Slang with explicit binding layouts**: reflection cannot
   express runtime arrays, so bindless materials pass
   `GraphicsDevice::bindless_heap_layout()` (the shared `Arc`) plus their
   data layouts explicitly. The bake registry gains per-spec `force_spirv`
   (Slang's WGSL target cannot load unbounded arrays — generalizing
   ADR-033's task/mesh auto-rule) and `skip_reflection`.

### Consequences

- One draw call can address every registered texture by integer — the
  `bindless_demo` renders a 48-texture quad grid in a single instanced draw
  and churns one texture every 90 frames (register new → repoint instance
  data through the frame graph → unregister old), exercising deferred
  recycling live. Validated on RTX 3070 and RX 6400, 400 frames each under
  `REDLILIUM_SYNC_VALIDATION=1`, zero hazards.
- Heap visibility is VERTEX|FRAGMENT|COMPUTE for now; task/mesh visibility
  joins when a mesh material first consumes the heap (stage flags require
  the extension).
- Registered textures live as long as their slot: forgetting to unregister
  pins the texture for the device's lifetime (documented; the slot table is
  inspectable via `BindlessSlots::live_counts`).
- Depth-format textures are excluded from the heap in phase 1 (the heap
  records `SHADER_READ_ONLY_OPTIMAL`; sampled-depth layout conventions come
  with a consumer).
- Follow-ups when consumers appear: cube/3D/array texture heaps (same
  mechanism, new binding), reflection support for runtime arrays, TASK/MESH
  heap visibility.

### Related Issues

- #117: feature issue
- #110 / ADR-032, #111 / ADR-033: the sibling pillars
- #114–#116: meshlet follow-ups (barrier scopes, indirect, limits)

## ADR-035: Per-Camera Render Pipelines — Serializable Path, Registered Pipeline, Derived Views

**Date:** 2026-07-16
**Status:** Accepted
**Issues:** #128–#131 (milestone "Per-camera render pipelines")

### Context

ADR-029 gave every camera a serializable output spec (`CameraOutput`) and a
derived GPU target (`CameraTarget`), but the *production* side stayed
hardcoded: `ForwardRender` walks all cameras and emits one fixed
unlit forward pass each. Per-camera render paths were explicitly declared
and deferred. The first concrete consumer has arrived — shadow mapping
needs a camera whose frame is produced differently (a depth-only auxiliary
view feeding the main pass) — and game code will eventually want paths the
engine does not ship.

### Decision

Fill the "how to render" axis with four pieces, reusing ADR-029's
discipline (serialize the intent, derive the resource, let the graph order
passes):

1. **`RenderPath` — the serializable choice** (component):
   `{ pipeline: <stable name>, settings }`. A camera without one defaults
   to `"forward"`. It deliberately lives *next to* `CameraOutput`, not
   inside it (see the ADR-029 amendment): `CameraOutput` is the contract
   with consumers of the image, `RenderPath` is the camera's internal
   production recipe with its own growing settings payload.

2. **`CameraRenderPipeline` — the registered implementation** (trait +
   `PipelineRegistry` resource): `ensure_targets()` derives auxiliary
   resources beyond color+depth (shadow maps, HDR intermediates) into the
   runtime-only `PipelineTargets` component (`#[skip_serialization]`,
   name → texture), with the same re-derive-on-disagreement discipline as
   `EnsureCameraTargets`; `record()` writes the camera's passes into the
   frame graph and returns the main pass handle (the `ScenePass`/overlay
   contract is unchanged). The engine registers `"forward"` (today's
   `ForwardRender` body, extracted); game plugins register their own —
   trait objects cross the plugin dylib boundary, so the ADR-020 /
   DESIGN_45 ABI contract applies. Scene files reference pipelines by
   name only — the registry is the GPU-code analogue of virtual texture
   publication: stable identity in assets, code-owned resource behind it.

3. **`SceneDrawer` — the shared scene walk**: the mesh gathering, ring
   uniform pushes, `PipelineCache` specialization, and binding-group
   assembly currently inlined in `ForwardRender`, extracted and
   parameterized by (view-projection, `RenderPhase`, material override,
   viewport). The visible set is gathered once per frame; each recorded
   view replays the prepared list — auxiliary views must not re-walk the
   World. `RenderPhase` starts as `Opaque` | `ShadowCaster` (with a
   `casts_shadows` flag on `MeshRenderer`); a depth-only pass renders with
   a shared vertex-only override shader and an empty color-format set.

4. **Views are derived, never authored**: a pipeline may produce any
   number of auxiliary views (shadow cascades, cube faces) as passes in
   the same graph. They are frame-internal data — **not** camera entities;
   spawning entities for shadow views would mix authored scene data with
   per-frame derivations and leak into the editor's undo model.
   Cross-pass ordering is never declared: the shadow pass's depth write
   and the main pass's `ShaderRead` of the map order themselves through
   the graph's resource-dependency derivation (#47), exactly like
   cross-camera ordering in ADR-029.

Choosing a pipeline is per-camera *authoring*, orthogonal to ADR-027
tiers: a tier gates which pipelines a device can offer; `RenderPath`
selects among the offered ones. Per-camera *quality* within one pipeline
remains #6 variant-axis territory.

### Consequences

- ✅ Stage 1 (#128) is a pure refactor — the forward path moves, frame
  output is bit-identical; the editor camera takes the default path
- ✅ Shadow mapping (#129/#130) becomes a pipeline, not a special case:
  depth-only phase + `"forward-shadows"` registration; the graphics crate
  already carries the needed vocabulary (`BindingType::DepthTexture`,
  `ComparisonSampler`, depth-attachment co-use from #40/#60)
- ✅ Game-defined paths (portals, stylized renderers) register without
  engine changes
- ⚠️ Zero-color-attachment passes must be validated on all three backends
  (`MaterialDescriptor` allows it; the encoders have never exercised it)
- ⚠️ #130 waits on #125: reversed-Z must settle the depth convention
  before a second depth consumer lands
- ⚠️ Lights are inert until #130 ships the first lit shader — the
  vertical slice is directional light + lambert + one un-cascaded map;
  cascades, spot/point, and a camera-shared shadow cache are #131
- ⚠️ A registry name with no registered pipeline degrades to `"forward"`
  with a warning (scene files stay loadable when a plugin is absent)

### Related Issues

- #128: framework (RenderPath, registry, SceneDrawer, dispatcher)
- #129: depth-only phase groundwork
- #130: forward-shadows vertical slice (blocked by #125, #129)
- #131: follow-ups (cascades, spot/point, shared shadow cache)
- #6, #47, #74, #125: neighboring axes referenced above

---

## ADR-036: Editor Play Runs a Separate Game World Built by the One Composition

**Date**: 2026-07-16
**Status**: Accepted
**Supersedes**: ADR-025; ADR-024 (play-mode role); amends ADR-026
**Relates to**: #67, #65, #58, #45

### Context

Play in the editor showed nothing while the standalone build worked. Root
cause: one world and one set of schedules served two regimes with opposite
invariants, reconciled by a spread of flags and guards (`read_only`
containers, `GameActive` conditions, `is_read_only()` branches inside game
code, whole-world snapshots and entity hiding on transitions). The question
"which systems run in Play, in what order" had no answer that could not
drift from the build.

### Decision

Two worlds, one composition:

- **The game-world composition is a single function** — `App::new` (+ the
  `App::boot` bootstrap). The standalone build and the editor's Play both
  build the game through it. Parity with the build is by construction, not
  by discipline.
- **Play** boots a play world from the hosted module (`PlaySession`,
  `App::boot_scoped` under the module's type generation), seeded with the
  scene open for editing. **Pause** = the world is not ticked. **Stop** =
  the world is dropped whole. The editing world is never touched: nothing
  to snapshot, restore, or hide.
- **Host parameters are the only prod/editor differences**: start scene,
  render destination (window vs. offscreen camera target shown in the scene
  view), input source (window vs. scene view).
- **Plugin contract v2**: `register_types` (every hosting world — the
  editing world must inspect/serialize game components) is split from
  `build`/`spawn_scene` (game worlds only). The editing world hosts
  registrations and nothing else.

Deleted with the old model: `PlayControl`/`PlaySnapshot`/`PlayStartTick`/
`ManagePlayModeTransitions`, `GameActive` + both run conditions +
`Schedules::set_game_active`, `PlayModeAware` + registry + transition
events, `HIDDEN_IN_PLAY` entity flags, `Plugin::on_stop`, and every
host-sniffing branch in game code.

### Consequences

- ✅ Play in the editor shows exactly what the build shows (verified
  end-to-end over the remote channel: `play`/`pause`/`resume`/`stop`)
- ✅ Game code composes unconditionally; no host-policy branches
- ✅ Editing world stays clean during Play — undo history, selection, and
  scene state survive by construction
- ✅ Engine-wide managers (`EngineContext`) are shared `Arc`s, so a play
  world starting/dying does not reload GPU assets
- ⚠️ Two worlds are resident during Play (entities + component storages;
  GPU caches shared)
- ⚠️ Pause-edit-restore (editing the running game's state with restore on
  Stop) is gone by design; play-world inspection is a future remote-channel
  concern
- ⚠️ Game egui (menus) does not yet render inside the editor's play view
  (the play world gets no `GameUi`); gameplay HUD/menu flows need the
  standalone build until that lands

## ADR-037: The Game Owns the Editor Binary; the Dylib Narrows to Play Behavior

**Date**: 2026-07-16
**Status**: Accepted
**Supersedes**: ADR-020 hosting model (point 3) — reverses its rejection of
Alternative 3 under the ADR-036 two-world model
**Relates to**: ADR-036, #45

### Context

The generic-editor-hosts-a-dylib model accumulated friction that is
structural, not incidental:

- **The build recipe lives in the operator's head.** Host and cdylib must
  come from one cargo invocation (feature unification changes `-C metadata`
  → every engine `TypeId`). The fingerprint + TypeId probe *gate* the
  mismatch; nothing *prevents* it. In practice every editor-side commit
  invalidated the running game module.
- **Open soundness debt.** Two engine images mean duplicated statics;
  `runtime/src/abi.rs` documents hazards that "must be resolved before"
  game code runs under the multi-threaded runner alongside host code.
- **No typed extension surface.** Game-specific editor tools (inspectors,
  gizmos, panels) across a dylib boundary would need an ABI-stable editor
  plugin interface. Rust has no stable ABI, so a shipped editor binary
  loading arbitrary user games is not a sound long-term product shape
  anyway — the user compiles either way.

ADR-020 rejected "editor as a library in the game project" because *every*
game change would restart the editor process. ADR-036 changed the calculus:
the editing world consumes only `register_types` (game *data* definitions,
which change rarely); game *behavior* runs only in play worlds, which are
cold-booted on every Play by construction. What must be hot is behavior,
and behavior is exactly what the dylib can carry without touching the
editing world.

### Decision

1. **Editor becomes a library.** `redlilium-editor` exposes
   `run(plugin, …)` (windowed + headless). The game project owns a small
   editor binary (`car-game-editor`) that statically links the game plugin
   and the editor. Editing-world registrations come from the static image:
   authoring (inspector, scene save/load, undo) never depends on a dylib
   and cannot be invalidated by one. Game code injects its own editor
   tools through the same typed API — "editor plugins" are ordinary cargo
   dependencies. The engine repo keeps a plain `redlilium-editor` binary
   for engine development (no game, or `REDLILIUM_GAME` override).

2. **Warm reload is tiered by what changed.**
   - **Tier 1 — behavior (the hot loop).** Play worlds boot from a cdylib
     of the same game crate. The editor *owns the rebuild*: it invokes
     cargo itself with a fixed invocation that includes its own package
     (unification therefore matches the running binary by construction);
     the fingerprint + probe gates remain as the backstop. After loading,
     the editor diffs component schemas (dylib `register_types` into a
     scratch world vs. the static image): equal → reload valid, play uses
     new code with the editing world untouched; different → soft
     degradation — play still runs the new code, authoring of the new
     fields needs a Tier-2 restart.
   - **Tier 2 — data / editor tools / engine.** Exec-restart with session
     carry: the editor persists session state (open scene, camera pose,
     selection, optionally a play-world snapshot), rebuilds, and execs the
     new binary. Undo history dies per restart — accepted. Remote agents
     reconnect via `.redlilium/editor.port`.
   - **Horizon (non-binding):** play world in a child game process
     (crash isolation; our protocols are already data-keyed), and
     function-level patching as a sub-second tier. Nothing in this design
     may preclude the child-process shape.

3. **Source watching marks, never acts.** A file watcher sets a "game
   module stale" indicator (UI badge + remote `state` field). Rebuild and
   restart are explicit commands; module swap happens only on a green
   build (a red build leaves the old module running, compile errors go to
   the log/remote channel). Auto-rebuild is per-session opt-in, never
   project-wide config — a code-editing agent must not be able to yank a
   parallel editor session it does not own. Agents editing game code work
   in git worktrees.

4. **`project.toml` stays project config** (mounts, start scene). It does
   not name a game: the game is known statically. `REDLILIUM_GAME` remains
   a dev override for hosting a foreign dylib in the engine-repo editor.

### Alternatives Considered

- **Editor keeps owning the build of a configured game** (project.toml
  names the crate): codifies the invocation but keeps authoring hostage to
  the dylib and offers no typed tool surface. Rejected — the inversion
  subsumes it (the editor still owns the *cdylib* rebuild in Tier 1).
- **Static-only with exec-restart as the only reload**: simplest, deletes
  all ABI machinery, but puts a full editor relink + process boot in the
  hottest loop (behavior tuning). Rejected while iteration speed is the
  point of the editor; it survives as Tier 2.
- **Shared engine dylib** (ADR-020 point 4): still the only fix for
  duplicated statics if cross-image execution ever widens; deferred, the
  hazards stay scoped to play sessions.

### Consequences

- ✅ Authoring is never blocked or invalidated by module drift; worst case
  is a stale inspector with a clear "restart to author new fields" state
- ✅ Behavior iteration keeps the seconds-scale dylib loop; the editing
  world is not even warm-restarted (registrations are static)
- ✅ Typed, in-process API for game-specific editor tooling
- ✅ The invocation-discipline failure class disappears from the user's
  hands (the editor is the only party that builds the cdylib)
- ⚠️ Static and dylib copies of the game coexist — with **identical**
  `TypeId`s: the cdylib and the rlib linked into the editor come from one
  rustc invocation of the game crate, so both images agree on every game
  `TypeId` (empirically confirmed; a fresh-generation attempt aborts on the
  registry's cross-generation conflict check). Behavior modules therefore
  register under the authoring generation (idempotent), and dylib-image
  liveness is enforced structurally (the play session stops before any
  swap), not by the generation registry. Editing world = static, play
  worlds = dylib; name-keyed serialization bridges the schema-diverged case
- ⚠️ Cross-image hazards (duplicated statics, in-image panic shields)
  remain for play sessions — unchanged from today, now explicitly scoped
- ⚠️ Editor-tool and engine changes cost a process restart (Tier 2);
  undo history does not survive it
