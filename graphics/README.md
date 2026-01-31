# RedLilium Graphics

Custom rendering engine for RedLilium Engine.

## Overview

This crate provides the rendering infrastructure built around an abstract **render graph** that enables:

- Declarative description of render passes and dependencies
- Automatic resource barrier and synchronization management
- Backend-agnostic rendering code
- Multithreaded command recording

## Architecture

### Render Graph

The render graph is the central abstraction for describing rendering operations:

```rust
use redlilium_graphics::{RenderGraph, TextureDescriptor, TextureFormat};

let mut graph = RenderGraph::new();

// Declare resources
let depth = graph.create_texture(TextureDescriptor {
    format: TextureFormat::Depth32Float,
    // ...
});

// Add passes with dependencies
graph.add_pass("geometry", |builder| {
    builder.write_depth(depth);
    // ...
});
```

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
│   └── resource.rs  # Resource handles and descriptors
├── backend/         # Backend implementations
│   ├── mod.rs       # Backend trait
│   ├── vulkan/      # Vulkan backend (ash)
│   ├── wgpu/        # wgpu backend
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
| `vulkan` | Enable Vulkan backend (default on desktop) |
| `wgpu` | Enable wgpu backend (default) |
| `dummy` | Enable dummy backend for testing |

## Thread Safety

The render graph is designed for multithreaded environments:

- Graph construction is single-threaded for determinism
- Command recording can happen in parallel per pass
- All public types implement `Send + Sync` where appropriate
