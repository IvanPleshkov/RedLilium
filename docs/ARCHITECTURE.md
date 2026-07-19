# RedLilium Engine Architecture

This document describes the high-level architecture of RedLilium Engine.

## Overview

RedLilium Engine is structured as a Cargo workspace. The core layering:

```
┌─────────────────────────────────────────────────────────┐
│            redlilium-demos / redlilium-editor           │
│              (Runnable apps and tooling)                │
├──────────────────────────┬──────────────────────────────┤
│     redlilium-ecs        │     redlilium-graphics       │
│  (ECS; optional `rendering`│ (Rendering, Shaders, GPU)   │
│   feature → graphics)    │                              │
├──────────────────────────┴──────────────────────────────┤
│                    redlilium-core                       │
│               (Tools and common utilities)              │
└─────────────────────────────────────────────────────────┘
```

Supporting crates: `redlilium-app` (windowing/app loop), `redlilium-editor`,
`redlilium-vfs` (virtual filesystem), `redlilium-debug-drawer`, and `ecs-macro`
(derive macros). See the root `Cargo.toml` `[workspace] members` for the full list.

## Design Principles

### 1. Separation of Concerns

- **Core** handles tools and common parts with no graphics or ECS dependencies
- **ECS** is rendering-agnostic by default. Graphics integration (the
  `std::rendering` module) lives behind the optional `rendering` feature, which
  pulls in `redlilium-graphics` + `redlilium-debug-drawer`; the default ECS build
  has no rendering dependency.
- **Graphics** handles all rendering with no game logic
- **Demos / Editor** combine the crates to create runnable applications

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

Custom ECS with integrated async compute support. See [ecs/DESIGN.md](../ecs/DESIGN.md) for detailed architecture.

Key design decisions:
- **Sync systems + async compute**: ECS systems are synchronous (normal borrowing), compute tasks are async (own their data, yield at `.await` points)
- **Unified thread pool**: Sync systems and async tasks share one work-stealing pool. Idle cores automatically pick up background compute work.
- **Multiple worlds**: First-class support for independent ECS worlds sharing a thread pool
- **Sparse set storage**: Simple, fast enough for thousands of entities, easy to implement
- **Priority scheduling**: Critical (must finish this frame), High (should finish), Low (fills gaps)

#### Entity Visibility and Query Filtering

RedLilium uses flag-based exclusion masks to control which entities are visible to queries in different contexts:

**Entity flags** (bit fields on Entity struct):
- **DISABLED** (bit 0): Entity is manually disabled; children marked INHERITED_DISABLED (bit 1)
- **STATIC** (bit 2): Entity is static (rarely-changing); children marked INHERITED_STATIC (bit 3)
- **EDITOR** (bit 4): Entity is editor-only (grid, gizmos, camera); children marked INHERITED_EDITOR (bit 5)

**Query visibility**:
- **Default game queries** (Read<T>, Write<T>, Ref<T>, RefMut<T>): Exclude `DISABLED | STATIC | EDITOR`
  - Used by game systems to ensure they only see game entities
- **Infrastructure queries** (ReadAll<T>, WriteAll<T>): Exclude only `DISABLED`
  - Used by engine systems (transforms, asset loading, rendering) that must see editor and static entities
- **Unfiltered queries** (`_unfiltered` accessors): No automatic exclusion
  - Used by editor UI and special inspection code

For details on the design rationale, see [ADR-026: Default Query Masks and Entity Visibility](DECISIONS.md#adr-026-default-query-masks-and-entity-visibility).

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
│  Submits the frame's render graphs (one queue submit per graph, any     │
│  number per frame). Graphs run on the graphics queue unless they opt    │
│  into secondary-queue routing (set_queue_preference: async compute #47  │
│  or dedicated transfer #89 — honored only for compute/transfer-only     │
│  graphs on devices exposing the queue, with an automatic fallback       │
│  ladder transfer → async compute → graphics).                           │
│  Ordering is automatic: same-queue hazards become tracker-emitted       │
│  pipeline barriers (submission order), cross-queue hazards become       │
│  timeline-semaphore waits (#47). At most one graph per frame may write  │
│  the swapchain.                                                         │
│                                                                         │
│  Responsibilities:                                                      │
│  - Compile and submit each graph to the GPU (submit)                    │
│  - Signal one fence per submit                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                           RenderGraph                                   │
│  A set of passes (shadow, main, post, transfer/upload, UI overlay)      │
│  and their dependencies, submitted as one unit.                         │
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
| Graph → Graph (same queue) | Submission order | Barriers from the persistent trackers are valid across submits within a queue |
| Graph → Graph (cross-queue) | Timeline-semaphore waits | Emitted automatically by the queue-ownership trackers; resources are CONCURRENT-shared, no ownership transfers (#47) |
| Frame → Frame | Fences (timeline values) | CPU-GPU sync across frames (one per submit) |

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

### Profiling Support

The engine includes optional profiling support via [Tracy](https://github.com/wolfpld/tracy), a real-time frame profiler. Enable it with the `profiling` Cargo feature:

```toml
[dependencies]
redlilium-graphics = { version = "0.1", features = ["profiling"] }
```

Or at build time:

```bash
cargo run --features profiling
```

**Running Demos with Profiling:**

To run any demo with profiling enabled across all crates:

```bash
# Run window demo with profiling
cargo run -p redlilium-demos --bin window_demo --features profiling

# Run textured quad demo with profiling
cargo run -p redlilium-demos --bin textured_quad_demo --features profiling
```

The `profiling` feature on the demos crate automatically enables profiling in all dependent crates (core, ecs, graphics, app).

**Connecting Tracy:**

1. Download Tracy from https://github.com/wolfpld/tracy/releases
2. Run your application with the `profiling` feature enabled
3. Launch the Tracy GUI and connect to your running application
4. View real-time CPU zones, frame times, and performance metrics

**CPU Profiling Zones:**

Built-in profiling zones track key operations:
- Frame boundaries (`frame_mark!`)
- Frame begin/end (`begin_frame`, `end_frame`)
- Fence waits (`wait_fence`, `wait_idle`)
- Graph submission (`submit_graph`, `present`)
- Backend execution (`vulkan_execute_graph`, `wgpu_execute_graph`)
- Pass recording and queue submission

**Custom Profiling:**

Use the provided macros to instrument your code:

```rust
use redlilium_graphics::profiling::{profile_scope, profile_function, frame_mark};

fn game_update() {
    profile_function!();  // Profiles entire function

    {
        profile_scope!("physics_update");
        // Physics code...
    }

    {
        profile_scope!("ai_update");
        // AI code...
    }
}
```

**GPU Profiling (Advanced):**

Tracy supports GPU profiling via timestamp queries. The engine provides `GpuProfileContext` for this:

```rust
use redlilium_graphics::profiling::GpuProfileContext;

// Create during device initialization
let gpu_ctx = GpuProfileContext::new_vulkan("Main Queue", initial_timestamp, timestamp_period);

// Record zones around GPU work (requires manual timestamp query management)
```

GPU profiling requires backend-specific integration with timestamp queries. See the Tracy documentation for details on timestamp synchronization.

**Zero-Overhead:**

When the `profiling` feature is disabled (default), all profiling macros compile to no-ops with zero runtime overhead.

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

## Temporal Contract (#147)

The interface every temporal technique consumes — TAA (#148) first,
FSR2-class upscalers later. Four pieces, designed once:

- **Velocity** — the deferred G-buffer's `gbuffer_velocity` target
  (`Rg16Float`): per-pixel NDC motion `current − previous`, both computed
  from **unjittered** matrices (jitter in velocity would smear static
  imagery by the jitter amplitude). Cleared to zero; background pixels stay
  zero — the TAA resolve reprojects them from camera matrices alone.
  Previous-frame model matrices live in the `TemporalState` resource,
  rotated once per frame by `CameraRender` and filled by
  `VisibleScene::gather`; an entity's first visible frame reads
  `prev = current` (zero velocity, no spawn smear).
- **Jitter** — a camera opts in with the `TemporalJitter` component. The
  dispatcher offsets the projection by a Halton(2,3) sub-pixel amount
  (±0.5 px — fixed by the geometry of temporal supersampling, not a
  tunable; the cycle length is the only knob, stretched by upscalers).
  Applied as a clip-space translation `T(jx, jy) · VP`, so every object
  shifts by the same on-screen amount regardless of depth. Cameras without
  the component render **bit-identically** to the pre-temporal path —
  picking, gizmos, and the editor camera's goldens are untouched.
- **Depth** — the camera target's depth buffer (reversed-Z, ADR-038),
  already per-camera.
- **Exposure** — manual `CameraExposure` (#142), applied only in the
  `display_output` pass; everything upstream (including future TAA) works
  on pre-exposure scene-referred linear.

Uniform plumbing: `CameraUniforms` carries the jittered `view_projection`
(what rasterization uses) plus the unjittered current/previous pair;
`ModelUniforms` carries `model` + `prev_model`. Shaders that only rasterize
declare just the leading fields — a prefix of the block is a valid smaller
cbuffer.

The contract is guarded by `editor/src/golden.rs`: velocity must be exactly
zero on static frames (including with jitter enabled — the leak trap),
match the projected NDC delta while an entity moves, and return to zero the
frame after motion stops.

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

## Game Hosting and Play Mode

A game is a `Plugin` (`redlilium-runtime`) with three obligations:

- `register_types` — component registrations. Runs in **every** hosting world,
  including the editor's editing world, so scenes carrying game components can
  be inspected and serialized anywhere.
- `build` — resources, events, systems. Runs only in **game worlds**.
- `spawn_scene` — initial scene population. Game worlds only, first boot only
  (a warm reload restores the scene from a snapshot instead).

There is exactly **one game-world composition** — `App::new` (plus the
`App::boot` bootstrap around it). The standalone build and the editor's Play
mode both build the game through it, so parity is by construction:

- **Standalone build**: `App::boot(engine, plugin, aspect, start_scene)`;
  the window is the render target and the input source.
- **Editor Play** (`editor/src/play.rs`, `PlaySession`): the same boot,
  scoped to the hosted module's type generation, with three host parameters —
  start scene = the scene open for editing, render target = an offscreen
  camera target shown in the scene view, input source = the scene view.
  **Pause** = the play world is not ticked (game time freezes because nothing
  advances it). **Stop** = the play world is dropped whole. The editing world
  is never touched by any of this: no snapshots, no restore, no entity hiding.

The editing world hosts **only** the game's `register_types` (via `GameHost`);
no game system, resource, or entity ever enters it. GPU resource managers and
the asset database live in the host-owned `EngineContext` and are shared by
all worlds as the same `Arc`s, so a play world starting and dying does not
reload assets.

### The game owns the editor binary (ADR-037)

`redlilium-editor` is a **library**. A game project ships a small editor
binary that statically links its plugin:

```rust
// game-editor/src/main.rs — the whole binary
fn main() {
    redlilium_editor::hosting(CarGamePlugin)
        .behavior_reload("car-game")
        .run();
}
```

Authoring (inspector, scene save/load, undo) always runs against the static
image and can never be invalidated by module drift. Code iteration is tiered
by what changed:

- **Tier 1 — behavior** (`editor/src/behavior_reload.rs`): the editor
  watches the game package's sources (stale marker only — marking, never
  acting), rebuilds the game **cdylib** on request with the fixed invocation
  `cargo build -p <editor-pkg> -p <game-pkg>` (feature unification matches
  the running binary by construction), and hot-swaps it for **play worlds
  only**. The cdylib and the rlib inside the editor binary come from one
  rustc invocation, so their game `TypeId`s are identical — the behavior
  module registers under the authoring generation, and dylib liveness is
  enforced structurally (the play session stops before any swap). Component
  schemas are diffed by name (derive-generated `schema_hash`); a divergence
  means play runs the new code while authoring the changed fields needs a
  restart.
- **Tier 2 — data / editor tools / engine** (`editor/src/session.rs`): an
  exec-restart with session carry (open scene + editor camera pose in a
  one-shot `.redlilium/session.ron`). The trigger is the fingerprint gate:
  a freshly built cdylib carrying a different engine build id than the
  running binary is refused, and that refusal is the "restart required"
  signal. The engine build id is scoped to the engine crates a game module
  links (plus `Cargo.lock`), so commits to `editor/`/`demos/`/game crates
  do not invalidate modules.

The engine repo keeps a plain `redlilium-editor` binary (no game;
`REDLILIUM_GAME=<cdylib>` hosts one dynamically with the warm-restart reload
of ADR-020) for engine development.

### Shipping builds (dist)

`cargo xtask dist --target desktop|web` packages the standalone game
(#107/#108/#132/#133):

- **Profile.** Desktop builds use `[profile.dist]` (workspace `Cargo.toml`):
  inherits `release`, adds fat LTO, `codegen-units = 1`, symbol stripping.
  `panic` stays `"unwind"` — the runtime's in-image panic shield catches
  unwinds, `abort` would turn every game panic into a process kill. Web is
  pinned to wasm-pack's `release` profile; its size pass is `wasm-opt -Oz`
  (`[package.metadata.wasm-pack]` in `game/Cargo.toml`).
- **Mount resolution.** `GameConfig::mounts` holds relative directories
  compiled into the binary. The runtime (`EngineContext::new`) resolves them
  **against the executable's directory first, the working directory second**
  (on macOS the exe-dir probe also tries the sibling `Contents/Resources` —
  the `.app` layout). Exe-first makes the dist artifact self-contained (runs
  from any cwd, double-click included); dev runs fall through to the cwd
  because `target/debug/` holds no asset packs and `cargo run` starts at the
  workspace root. The editor is unaffected — it builds its own VFS and
  resolves its own directories.
- **Platform packaging (#133).** The desktop artifact is native to the host
  OS. macOS: a `Car Game.app` bundle (binary in `Contents/MacOS`, packs +
  `icon.icns` in `Contents/Resources`, generated `Info.plist`) wrapped in a
  `.dmg`; `codesign` runs when `REDLILIUM_SIGN_IDENTITY` is set, notarization
  + stapling when `REDLILIUM_NOTARY_PROFILE` names a stored `notarytool`
  keychain profile. Windows: flat zipped folder; the icon + version resource
  are embedded into the exe by `game/build.rs` (winresource). Linux: flat
  zipped folder plus a `car-game.desktop` launcher template and icon. The
  shared icon set lives in `game/icon/` and is regenerated by
  `scripts/gen-game-icon.py` (stdlib-only Python; `.icns`/`.ico` derivation
  needs macOS `sips`/`iconutil`).
- **Pack pruning (#134).** Dist ships only the referenced asset closure: the
  walk starts from the entry scenes declared in `game/dist-manifest.ron`,
  follows guid references through the merged `assets.db` and asset source
  text (`AssetDb::dependency_closure` in `redlilium-assets`), and adds the
  manifest's `keep` prefixes — assets loaded by *code* (hardcoded paths) that
  the data-driven walk cannot see; today that is the entire std pack. Each
  shipped pack's `assets.db` is regenerated to the shipped subset, and
  pruned files are printed by `xtask dist` for diagnosability.
- **Build stamping + releases (#135).** `game/build.rs` embeds the git commit
  into the binary; `car_game::BUILD_STAMP` (`<version>+<hash>`, `-dirty` when
  uncommitted, `unknown` without git) is printed on startup and by
  `xtask dist` — it identifies the build behind a bug report. Pushing a `v*`
  tag runs `.github/workflows/release.yml`: `xtask dist` across the
  macOS/Linux/Windows matrix plus web, artifacts attached to a GitHub
  release (no Slang SDK in CI — dist uses the committed baked table).

The engine supports multiple ECS worlds with shared rendering backend
(this is exactly how editor Play works — an editing world and a play world
side by side over one `EngineContext`):

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
