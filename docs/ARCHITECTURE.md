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
├── scene/             # ECS-Rendering bridge
│   ├── mod.rs         # Scene module exports
│   ├── extracted.rs   # ExtractedTransform, ExtractedMesh, ExtractedMaterial
│   ├── render_world.rs # RenderWorld - extracted render data
│   └── scene_renderer.rs # SceneRenderer - ECS to render graph
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

### Typical Frame Flow

```rust
// Initialization
let mut pipeline = FramePipeline::new(2);  // 2 frames in flight

// Main loop
while !window.should_close() {
    pipeline.begin_frame();  // Wait for frame slot

    // Build render graphs
    let shadow_graph = build_shadow_graph();
    let main_graph = build_main_graph();

    // Submit via streaming schedule
    let mut schedule = FrameSchedule::new();
    let shadows = schedule.submit("shadows", shadow_graph.compile()?, &[]);
    let main = schedule.submit("main", main_graph.compile()?, &[shadows]);
    let fence = schedule.submit_and_present("present", post_graph.compile()?, &[main]);

    pipeline.end_frame(fence);
}

// Shutdown
pipeline.wait_idle();  // Wait for GPU before cleanup
```

## ECS-Rendering Integration

The engine uses a three-phase rendering approach:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   EXTRACT   │ ──▶ │   PREPARE   │ ──▶ │   RENDER    │
│   Phase     │     │   Phase     │     │   Phase     │
└─────────────┘     └─────────────┘     └─────────────┘
     │                    │                    │
Copy render data    Sort & batch         Execute render
from ECS World      for GPU upload       graph with backend
```

### Extract Phase
Queries ECS for entities with render components (GlobalTransform, RenderMesh, Material)
and copies relevant data into the RenderWorld.

### Prepare Phase
Sorts render items (front-to-back for opaque, back-to-front for transparent),
batches by material/mesh, and prepares instance data for GPU upload.

### Render Phase
Executes the compiled render graph through the backend, issuing draw calls
for all items in the RenderWorld.

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

See ADR-009 in DECISIONS.md for rationale.

## Reading Guide for AI Assistants

1. Start with this file for overall structure
2. Read individual crate READMEs for module details
3. Check `DECISIONS.md` for rationale behind choices
4. Use `cargo doc` for API reference
