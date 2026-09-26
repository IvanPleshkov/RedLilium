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
├── backend/         # Backend implementations
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
