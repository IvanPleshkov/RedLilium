# RedLilium Graphics

Custom rendering engine for RedLilium Engine.

## Overview

This crate provides the rendering infrastructure built around an abstract **render graph** that enables:

- Declarative description of render passes and dependencies
- Automatic resource barrier and synchronization management
- Backend-agnostic rendering code
- Backend command encoding with automatic synchronization

## Architecture

### Render Graph

The render graph is the central abstraction for describing rendering operations:

```rust,ignore
use std::sync::Arc;
use redlilium_graphics::{
    BufferDescriptor, BufferUsage, TransferConfig, TransferOperation, TransferPass,
};

let buffer = device.create_buffer(&BufferDescriptor::new(
    256, BufferUsage::VERTEX | BufferUsage::COPY_DST,
))?;
let mut schedule = pipeline.begin_frame()?;
let mut graph = schedule.acquire_graph();
let mut upload = TransferPass::new("upload vertices".into());
upload.set_transfer_config(TransferConfig::new().with_operation(
    TransferOperation::write_buffer(buffer.clone(), 0, Arc::from(vertex_bytes)),
));
graph.add_transfer_pass(upload);
// Add graphics passes using buffer; the graph derives upload → draw ordering.
schedule.submit(graph)?;
pipeline.end_frame(schedule);
```

Resource transfers go through the graph; `GraphicsDevice` only creates resources.
`submit` reports failures through `Result`. Dropping a schedule automatically
returns its submitted resources to the pipeline for retirement after GPU completion.
There is one live frame pipeline per graphics instance and one active schedule per
pipeline. `submit_with_mode` selects Strict compilation when ambiguous writers
should be treated as errors instead of using addition order.

### Transfer validation

Before recording commands, `submit` validates every transfer operation in the
graph on all backends, including Dummy. Invalid parameters return
`GraphicsError::InvalidParameter` with the pass name and operation index; no
commands from that graph execute, and the schedule remains usable. Unsupported
mip generation returns `FeatureNotSupported`.

Validation checks resource ownership, usage flags, pending CPU mappings, buffer
ranges, mip/layer bounds, formats, sample counts, and copy layouts. The common
contract includes:

- Buffer copies and nonempty writes require 4-byte aligned offsets and sizes.
  CPU readback requires `MAP_READ`, an 8-byte aligned offset, and a 4-byte aligned
  size. Empty writes and readbacks are allowed within buffer bounds.
- Buffer/texture offsets must be aligned to both 4 bytes and the format block
  size. Copies spanning multiple block rows or images need a 256-byte aligned
  row pitch. The last row only needs its actual texel bytes, without trailing
  padding. Compressed copies may end at a mip edge smaller than a block.
- Copy regions must be nonempty and have nonzero extents. Array layers do not
  shrink with mip level; 3D depth does. Buffer/texture copies require one sample.
  Depth/stencil and multisampled texture copies cover the full mip width and
  height. Buffer copies of combined depth/stencil or `Depth24Plus`, and uploads
  of `Depth32Float`, are unsupported by the common contract.
- Texture copies require matching dimension classes, sample counts, and formats
  (linear/sRGB counterparts are compatible). Source and destination must be
  different resources, even for disjoint buffer ranges or texture subresources.
  In-place texture copies need subresource tracking, which is not implemented.
- Mip generation requires `COPY_SRC | COPY_DST` and, for multiple levels, a
  single-sampled 2D texture with a supported format. A single level is a no-op.

`upload_texture_data` and `upload_texture_level` prepare staging data with the
required padding; the destination upload is still an ordered graph operation.

### Public API boundary

Applications use `GraphicsDevice`, engine resources, `RenderGraph`, and
`FramePipeline`/`FrameSchedule`. Backend implementations, native handles, and
Vulkan layout/access conversions are internal. Uploads, copies, and readbacks
remain transfer operations in the graph.

`RenderTarget` has a private representation. Construct it with `from_texture`,
`from_texture_mip`, `from_texture_layer`, or `from_surface`. Inspect it through
`format`, `width`, `height`, `sample_count`, `is_surface`, `texture`, `mip_level`,
and `array_layer`. The last three return `None` for a surface target.

Migration from the previous API:

- Replace `RenderTarget::Texture { texture, mip_level, array_layer }` with
  `RenderTarget::from_texture_layer(texture, mip_level, array_layer)`.
- Obtain surface targets from an acquired `SurfaceTexture` using
  `RenderTarget::from_surface(&surface_texture)`.
- Replace variant matching with the accessors above. Resource `gpu_handle()`
  methods, `SurfaceTexture::gpu_texture()`, and `Blas::device_address()` are
  internal; there is currently no public native interop API.
- Vulkan validation counters are available through
  `diagnostics::vulkan::{validation_error_count, reset_validation_error_count}`
  with the `vulkan-backend` feature. They count errors on the calling thread;
  validation must be enabled and the validation layer available.

### Backend Support

The render graph supports three backends:

| Backend | Crate | Use Case |
|---------|-------|----------|
| Vulkan | `ash` | High-performance desktop rendering |
| wgpu | `wgpu` 28.0.0 | Cross-platform and web support |
| Dummy | - | Testing without GPU |

### Module Structure

```
redlilium-graphics
├── graph/           # Render graph infrastructure
│   ├── mod.rs       # Graph builder and compiler
│   ├── pass.rs      # Render pass definitions
│   └── resource_usage.rs # Inferred resource access
├── diagnostics.rs   # Public validation diagnostics
├── backend/         # Internal backend implementations
│   ├── mod.rs       # Enum-based backend dispatch
│   ├── vulkan/      # Vulkan backend (ash)
│   ├── wgpu_impl/   # wgpu backend
│   └── dummy.rs     # Dummy backend for testing
└── types/           # Common types and descriptors
```

## Building

```bash
# Build the crate
cargo build -p redlilium-graphics

# Run tests
cargo test -p redlilium-graphics

# Generate documentation
cargo doc -p redlilium-graphics --open
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `vulkan-backend` | Enable Vulkan backend (default on desktop) |
| `wgpu-backend` | Enable wgpu backend (default) |
| `dummy` | Enable dummy backend for testing |

## Thread Safety

The render graph is designed for multithreaded environments:

- Graph construction is single-threaded for determinism
- Frame submission is serialized by the pipeline; independent CPU work can run in parallel
- All public types implement `Send + Sync` where appropriate
