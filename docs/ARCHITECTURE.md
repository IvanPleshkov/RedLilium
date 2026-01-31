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
