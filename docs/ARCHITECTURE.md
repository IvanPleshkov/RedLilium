# RedLilium Engine Architecture

This document describes the high-level architecture of RedLilium Engine.

## Overview

RedLilium Engine is structured as a Cargo workspace with four main crates:

```
┌─────────────────────────────────────────────────────────┐
│                   redlilium-demos                       │
│              (Demo scenes and examples)                 │
├──────────────────────────┬──────────────────────────────┤
│     redlilium-ecs        │     redlilium-graphics       │
│  (ECS components/systems)│  (Rendering, Shaders, GPU)   │
├──────────────────────────┴──────────────────────────────┤
│                    redlilium-core                       │
│               (Tools and common utilities)              │
└─────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Separation of Concerns

- **Core** handles tools and common parts with no graphics or ECS dependencies
- **ECS** handles entity-component-system architecture with no rendering logic
- **Graphics** handles all rendering with no game logic
- **Demos** combines all crates to create runnable applications

### 2. Platform Abstraction

All crates support both native and web targets:
- Native: Windows, Linux, macOS
- Web: WebAssembly with WebGL2 and WebGPU

### 3. Data-Driven Design

- Configuration over code where possible
- Render graph for flexible pipeline configuration
- ECS for data-oriented entity management

## Module Overview

### redlilium-core

Core utilities and common functionality shared across all crates.

### redlilium-ecs

Entity-Component-System module built on [bevy_ecs](https://docs.rs/bevy_ecs):

```
ecs/
├── components/         # ECS components
│   ├── transform.rs    # Transform, GlobalTransform
│   ├── hierarchy.rs    # ChildOf, Children, HierarchyDepth
│   ├── material.rs     # Material, TextureHandle, AlphaMode
│   ├── render_mesh.rs  # RenderMesh, MeshHandle, Aabb, RenderLayers
│   └── collision.rs    # Collider, RigidBody, CollisionLayer
└── systems/            # ECS systems
    └── transform_propagation.rs  # Transform hierarchy propagation
```

Key types:
- **Transform/GlobalTransform**: Local and world-space transforms
- **RenderMesh/Material**: Rendering components
- **ChildOf/Children**: Hierarchy relationship components

### redlilium-graphics

Custom rendering engine built around an abstract render graph:

```
graphics/
├── backend/            # Graphics backend implementations
│   ├── mod.rs         # Backend trait definition
│   ├── dummy.rs       # No-op testing backend
│   └── error.rs       # Backend error types
├── graph/             # Render graph infrastructure
│   ├── mod.rs         # RenderGraph, CompiledGraph
│   ├── pass.rs        # RenderPass, PassHandle
│   └── resource.rs    # ResourceHandle, TextureHandle, BufferHandle
└── types/             # GPU resource types
    ├── texture.rs     # TextureDescriptor, TextureFormat
    ├── buffer.rs      # BufferDescriptor, BufferUsage
    └── sampler.rs     # SamplerDescriptor
```

Key abstractions:
- **RenderGraph**: Declarative render pipeline description
- **Backend trait**: Graphics API abstraction (Vulkan, wgpu, Dummy)
- **SceneRenderer**: Connects ECS world to render graph

## Rendering Pipeline Architecture

The rendering system is organized in four layers, from low-level to high-level:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FramePipeline                                  │
│  Manages multiple frames in flight. Handles CPU-GPU synchronization     │
│  via fences. Enables frame overlap for maximum throughput.              │
│                                                                         │
│  Responsibilities:                                                      │
│  - Track fences for N frames in flight                                  │
│  - Wait for frame slot availability (begin_frame)                       │
│  - Graceful shutdown (wait_idle)                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                          FrameSchedule                                  │
│  Orchestrates multiple render graphs within ONE frame. Enables          │
│  streaming submission (submit graphs as they're ready).                 │
│                                                                         │
│  Responsibilities:                                                      │
│  - Accept compiled graphs and submit immediately to GPU                 │
│  - Track dependencies between graphs via semaphores                     │
│  - Return fence for frame completion                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                           RenderGraph                                   │
│  Describes a set of passes and their dependencies. Represents one       │
│  logical rendering task (e.g., "shadow rendering", "main scene").       │
│                                                                         │
│  Responsibilities:                                                      │
│  - Store passes (graphics, transfer, compute)                           │
│  - Track pass-to-pass dependencies                                      │
│  - Compile to execution order                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                              Pass                                       │
│  A single unit of GPU work (draw calls, copies, dispatches).            │
│                                                                         │
│  Types:                                                                 │
│  - GraphicsPass: vertex/fragment shaders, rasterization                 │
│  - TransferPass: buffer/texture copies                                  │
│  - ComputePass: compute shaders                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Synchronization Model

Different synchronization primitives are used at different levels:

| Level | Primitive | Purpose |
|-------|-----------|---------|
| Pass → Pass | Barriers | Resource state transitions within a graph |
| Graph → Graph | Semaphores | GPU-GPU sync within a frame |
| Frame → Frame | Fences | CPU-GPU sync across frames |

### Automatic Texture Layout Tracking (Vulkan Backend)

The Vulkan backend automatically tracks texture layouts and generates memory barriers,
eliminating the need for manual layout management. This system ensures correct
synchronization while optimizing barrier placement.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TextureLayoutTracker                          │
│  Per-frame tracking of texture layouts (one state per frame     │
│  in flight to handle async GPU execution)                        │
├─────────────────────────────────────────────────────────────────┤
│                     BarrierBatch                                 │
│  Collects all barriers (image + buffer) for a pass, submits     │
│  them once with optimal pipeline stage masks                     │
└─────────────────────────────────────────────────────────────────┘
```

**How it works:**

1. **Pass Configuration**: Each pass declares its resource usage via render targets,
   material bindings, and transfer operations.

2. **Usage Inference**: At encode time, `infer_resource_usage()` extracts texture
   and buffer usages from the pass configuration:
   - Color attachments → `RenderTargetWrite`
   - Depth attachments → `DepthStencilWrite` or `DepthStencilReadOnly`
   - Material textures → `ShaderRead`
   - Transfer sources → `TransferRead`
   - Transfer destinations → `TransferWrite`
   - Indirect draw buffers → `IndirectRead`
   - Buffer copy sources → `TransferRead`
   - Buffer copy destinations → `TransferWrite`

3. **Barrier Generation**: Before encoding each pass, the system:
   - Queries current layout from the tracker (for textures)
   - Determines required layout/access from usage
   - Generates barriers if transitions are needed
   - Updates tracked layout (for textures)

4. **Batched Submission**: All barriers for a pass are collected and submitted
   in a single `vkCmdPipelineBarrier` call with combined stage masks.

**Example transition sequence:**

```
Pass 1 (Render): texture Undefined → ColorAttachment
Pass 2 (Sample): texture ColorAttachment → ShaderReadOnly
Pass 3 (Copy):   texture ShaderReadOnly → TransferSrc
                 buffer TransferWrite → VertexBuffer
```

The wgpu backend handles layout tracking internally, so this system is
Vulkan-specific.

### Automatic Buffer Barrier Placement (Vulkan Backend)

In addition to texture layout tracking, the Vulkan backend automatically generates
buffer memory barriers. Unlike textures, buffers don't have "layouts" but still
need barriers for memory coherence between passes.

**Buffer Access Modes:**

| Access Mode | Vulkan Stage | Access Flags |
|-------------|--------------|--------------|
| `VertexBuffer` | Vertex Input | Vertex Attribute Read |
| `IndexBuffer` | Vertex Input | Index Read |
| `UniformRead` | VS/FS | Uniform Read |
| `StorageRead` | VS/FS/CS | Shader Read |
| `StorageWrite` | VS/FS/CS | Shader Write |
| `IndirectRead` | Draw Indirect | Indirect Command Read |
| `TransferRead` | Transfer | Transfer Read |
| `TransferWrite` | Transfer | Transfer Write |

**Usage Inference:**

- **GraphicsPass**: Indirect draw buffers automatically detected
- **TransferPass**: Source and destination buffers for copy operations

The system is conservative - it may insert barriers where not strictly needed,
but guarantees correctness. Future optimization could add per-buffer state
tracking similar to the texture layout tracker.

### Frame Overlap (Pipelining)

With 2 frames in flight, the CPU and GPU work in parallel:

```
Frame 0: [CPU build] [submit] ─────────────────────────────────────────────►
                              [GPU execute frame 0] ───────────────────────►

Frame 1:              [CPU build] [submit] ────────────────────────────────►
                                           [GPU execute frame 1] ──────────►

Frame 2:                          [wait F0] [CPU build] [submit] ──────────►
                                                        [GPU execute F2] ──►

Time ──────────────────────────────────────────────────────────────────────►
```

- CPU doesn't wait for GPU unless it's reusing a frame slot
- GPU processes frames in order via semaphores
- Fences ensure we don't overwrite in-use resources

### Streaming Submission

Unlike batch submission (where all work is queued then submitted), streaming submission
sends graphs to the GPU immediately as they're ready:

```
Batch (traditional):
  [build shadow] [build depth] [build main] [submit all]
                                                  │
                                                  ▼ GPU starts here

Streaming (this engine):
  [build shadow] ──► [submit shadow] ──────────────────────────────►
                 │                       [GPU: shadow]
                 └► [build depth] ──► [submit depth] ──────────────►
                                  │              [GPU: depth]
                                  └► [build main] ──► [submit main]►
                                                            [GPU: main]
```

Benefits:
- GPU starts earlier while CPU continues building
- Better CPU-GPU parallelism
- Lower frame latency

### GPU Resource Lifetime Management

GPU resources (buffers, textures, samplers, fences, semaphores) have a critical lifetime constraint: they cannot be destroyed while the GPU is still using them. This is because GPU commands execute asynchronously - when you submit work, the CPU continues while the GPU processes commands 1-3 frames behind.

```
CPU Frame 0: Record commands using Buffer A → Submit → Continue to Frame 1
CPU Frame 1: Record commands using Buffer B → Submit → Continue to Frame 2
CPU Frame 2: User drops Buffer A (Arc refcount = 0)
                 ↓
GPU Frame 0: Still reading from Buffer A! ← PROBLEM
```

#### Deferred Destruction (Vulkan Backend)

The Vulkan backend implements a deferred destruction system to solve this problem. When a resource's `Arc` is dropped, instead of immediately destroying the Vulkan handle, it's queued for later destruction:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DeferredDestructor                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   Frame-indexed queues                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │  │
│  │  │ Frame 0  │  │ Frame 1  │  │ Frame 2  │  ...            │  │
│  │  │ pending  │  │ pending  │  │ pending  │                 │  │
│  │  └──────────┘  └──────────┘  └──────────┘                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Flow:**

1. **On Resource Drop**: Resource handle queued in current frame's pending list
2. **On Frame Boundary**: After fence wait in `begin_frame()`, oldest queue is processed
3. **Safe Destruction**: Resources destroyed only after `MAX_FRAMES_IN_FLIGHT` (3) frames

```
CPU Frame 0: Create Buffer A, submit commands → GPU starts
CPU Frame 2: Drop Buffer A → Queued for frame 2
CPU Frame 5: begin_frame() waits for frame 2 fence
             → fence signaled (GPU done with frame 2)
             → Buffer A safely destroyed
```

This is automatic - users don't need to manually manage resource lifetimes. The wgpu backend handles this internally.

#### Best Practices

1. **Avoid excessive resource churn**: Reuse buffers/textures across frames when possible
2. **Use object pools**: For frequently created/destroyed resources (particles, UI elements)
3. **Don't hold unnecessary references**: Drop `Arc` handles when no longer needed
4. **Trust the system**: Resources are automatically cleaned up safely

### Graceful Shutdown

When the application exits, call `FramePipeline::wait_idle()` before destroying resources:

```
[Window Close Event]
        │
        ▼
┌───────────────────┐
│  Stop rendering   │  Don't start new frames
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ pipeline.wait_idle│  Wait for all in-flight GPU work
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Drop resources   │  Safe to destroy GPU objects
└───────────────────┘
```

During shutdown, `wait_idle()` ensures all pending GPU work completes. The Vulkan backend then flushes all deferred destruction queues, safely destroying any pending resources before the device is destroyed.

### Typical Frame Flow

```rust
// Initialization
let instance = GraphicsInstance::new()?;
let device = instance.create_device()?;
let mut pipeline = device.create_pipeline(2);  // 2 frames in flight

// Main loop
while !window.should_close() {
    // begin_frame waits for frame slot AND returns a schedule
    let mut schedule = pipeline.begin_frame();

    // Build render graphs
    let shadow_graph = build_shadow_graph();
    let main_graph = build_main_graph();

    // Submit via streaming schedule
    let shadows = schedule.submit("shadows", shadow_graph.compile()?, &[]);
    let main = schedule.submit("main", main_graph.compile()?, &[shadows]);
    schedule.present("present", post_graph.compile()?, &[main]);

    // end_frame takes ownership of schedule
    pipeline.end_frame(schedule);
}

// Shutdown
pipeline.wait_idle();  // Wait for GPU before cleanup
```

### Window Resize Handling

Window resize requires special handling because the swapchain must be recreated.
Naive approaches that recreate on every resize event cause visible stuttering.

The `ResizeManager` provides debounced resize with configurable strategies:

```rust
use redlilium_graphics::resize::{ResizeManager, ResizeStrategy};

let mut resize_manager = ResizeManager::new(
    (1920, 1080),
    50,  // 50ms debounce
    ResizeStrategy::DynamicResolution { scale_during_resize: 0.5 },
);

// Event handling
match event {
    WindowEvent::Resized(size) => {
        resize_manager.on_resize_event(size.width, size.height);
    }
    _ => {}
}

// Each frame
if let Some(event) = resize_manager.update() {
    // Wait only for current slot (not wait_idle!)
    pipeline.wait_current_slot();
    surface.resize(event.width, event.height);
}

let render_size = resize_manager.render_size();
// Render at render_size (may be scaled during resize)
```

**Resize Strategies:**

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `Stretch` | Render at old size, OS stretches | Simplest, acceptable quality |
| `IntermediateTarget` | Render to fixed-size texture | Consistent quality |
| `DynamicResolution` | Reduced resolution during resize | Best UX, smoothest |

**Why `wait_current_slot()` instead of `wait_idle()`?**

- `wait_idle()` waits for ALL frame slots (2-3 frames = 33-50ms)
- `wait_current_slot()` waits for ONE slot (~16ms)
- Result: 2-3x faster resize response

## Coordinate System

RedLilium uses the **D3D/wgpu coordinate system convention** for consistency across backends:

### Conventions

| Aspect | Convention |
|--------|------------|
| **Depth Range (NDC)** | `[0, 1]` (near = 0, far = 1) |
| **Y-Axis (NDC)** | +Y points down |
| **Screen Origin** | Top-left corner |
| **Winding Order** | Counter-clockwise (CCW) front faces |

### Depth Range

The engine uses `[0, 1]` depth range, matching:
- Vulkan's native convention
- wgpu's cross-platform convention
- D3D and Metal conventions

This differs from OpenGL's `[-1, 1]` NDC depth range.

### Projection Matrices

When building projection matrices, use functions that output `[0, 1]` depth:

```rust
// glam - uses [0, 1] depth by default
let proj = glam::Mat4::perspective_rh(fov_y, aspect, near, far);

// nalgebra - use the explicit zero-to-one variant
let proj = nalgebra::Perspective3::new_zo(aspect, fov_y, near, far);
```

### Why [0, 1] Depth?

1. **Native Vulkan**: No shader transformation needed
2. **wgpu Compatibility**: Same convention across backends
3. **Industry Standard**: D3D, Metal, and modern APIs use this
4. **Better Precision**: Full depth buffer range from near to far

See `DECISIONS.md` ADR-015 for detailed rationale.

### Reverse-Z Support

For improved depth precision (especially with large view distances), you can use reverse-Z by swapping the depth range:

```rust
// Standard: near=0, far=1
let viewport = Viewport::new(0.0, 0.0, width, height);

// Reverse-Z: near=1, far=0 (better precision)
let viewport = Viewport::new(0.0, 0.0, width, height)
    .with_depth_range(1.0, 0.0);
```

Note: Reverse-Z also requires adjusting the depth comparison function to `GreaterEqual`.

## Multi-World Architecture

The engine supports multiple ECS worlds with shared rendering backend:

```
┌─────────────────────────────────────────────────────────┐
│                     Process                              │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  ECS World  │  │  ECS World  │  │  ECS World  │     │
│  │   (Game)    │  │  (Editor)   │  │  (Preview)  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         ▼                ▼                ▼             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │RenderGraph A│  │RenderGraph B│  │RenderGraph C│     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│              ┌───────────────────────┐                 │
│              │   Shared Backend      │                 │
│              │ (Vulkan/wgpu/Dummy)   │                 │
│              └───────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

## Reading Guide for AI Assistants

1. Start with this file for overall structure
2. Read individual crate READMEs for module details
3. Check `DECISIONS.md` for rationale behind choices
4. Use `cargo doc` for API reference
