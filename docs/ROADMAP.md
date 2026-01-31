# RedLilium Engine Roadmap

Development milestones for the RedLilium game and graphics engine.

## Milestone 1: Foundation ✅

**Goal**: Basic project structure and window creation

- [x] Cargo workspace setup
- [x] Core crate skeleton
- [x] Graphics crate skeleton
- [x] Demos crate with window demo
- [x] Web build support
- [x] Documentation structure

---

## Milestone 2: Graphics Architecture

**Goal:** Design and implement an abstract render graph with multiple backend support.

### Overview

The graphics system is built around an **abstract render graph** that defines high-level rendering operations while allowing the executor to handle low-level synchronization between passes and resource states. This architecture enables:

- Declarative description of render passes and their dependencies
- Automatic resource barrier/transition management
- Optimal scheduling and synchronization
- Backend-agnostic rendering logic

### Backend Support

The render graph abstraction supports three backends:

| Backend | Purpose | Target Platforms |
|---------|---------|------------------|
| **Vulkan** | High-performance rendering with full extension support | Windows, Linux |
| **wgpu** | Cross-platform compatibility, WebGPU support | All platforms including Web |
| **Dummy** | Testing and headless operation | All platforms |

### Architecture Components

#### 2.1 Core Abstractions

```
RenderGraph
├── RenderPass (compute, graphics, transfer)
├── Resource (buffer, texture, sampler)
├── ResourceHandle (lightweight reference)
└── PassDependency (explicit data flow)

Backend Trait
├── VulkanBackend
├── WgpuBackend
└── DummyBackend
```

#### 2.2 Render Graph System

The render graph is the central orchestration mechanism:

1. **Graph Construction Phase**
   - User defines passes and their resource dependencies
   - Resources are described (not allocated)
   - Pass execution order is implicit from dependencies

2. **Compilation Phase**
   - Topological sort of passes
   - Resource lifetime analysis
   - Memory aliasing opportunities identified
   - Synchronization points determined

3. **Execution Phase**
   - Resources allocated/reused
   - Barriers inserted automatically
   - Passes executed in optimal order
   - Command buffers recorded (potentially in parallel)

#### 2.3 Resource Management

Resources are managed through handles with automatic lifetime tracking:

- **Transient Resources:** Exist only within a frame, eligible for memory aliasing
- **Persistent Resources:** Survive across frames (textures, buffers)
- **Imported Resources:** External resources (swapchain images)

#### 2.4 Multithreading Design

The render graph is designed for multithreaded environments:

- **Graph construction:** Single-threaded (deterministic)
- **Command recording:** Parallel per-pass (thread-safe)
- **Resource access:** Immutable during execution
- **Backend submission:** Managed by executor

Thread-safety is achieved through:
- Immutable graph during execution
- Per-thread command buffer pools
- Arc-based resource sharing
- Lock-free handle allocation

### Implementation Phases

#### Phase 2.1: Core Traits and Types
- [ ] Define `Backend` trait with lifecycle methods
- [ ] Define `RenderPass` trait for pass abstraction
- [ ] Define resource types (Buffer, Texture, Sampler)
- [ ] Define handle types with generation-based validation
- [ ] Implement `DummyBackend` for testing

#### Phase 2.2: Render Graph Infrastructure
- [ ] Implement `RenderGraph` builder
- [ ] Implement pass dependency tracking
- [ ] Implement topological sorting
- [ ] Implement resource lifetime analysis
- [ ] Add graph validation and error reporting

#### Phase 2.3: wgpu Backend
- [ ] Implement `WgpuBackend` using wgpu 28.0.0
- [ ] Implement resource creation and management
- [ ] Implement command buffer recording
- [ ] Implement synchronization via wgpu's automatic barriers
- [ ] Test on native and web targets

#### Phase 2.4: Vulkan Backend
- [ ] Implement `VulkanBackend` using ash crate
- [ ] Implement explicit barrier management
- [ ] Implement command pool per-thread
- [ ] Implement pipeline cache
- [ ] Support relevant Vulkan extensions

#### Phase 2.5: Integration and Testing
- [ ] Create demo scenes exercising the render graph
- [ ] Performance benchmarks
- [ ] Multi-backend comparison tests
- [ ] Documentation and examples

### Dependencies

| Crate | Purpose |
|-------|---------|---------|
| wgpu | WebGPU/cross-platform backend |
| ash | Vulkan backend |
| gpu-allocator | Memory allocation for Vulkan |
| parking_lot | Fast synchronization primitives |

### Success Criteria

- [ ] Render graph can describe multi-pass rendering
- [ ] All three backends pass conformance tests
- [ ] Web demo works with wgpu backend
- [ ] Native demo works with Vulkan and wgpu backends
- [ ] Thread-safe command recording demonstrated
- [ ] Performance within 10% of hand-written backend code

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| wgpu API changes | Pin to specific version, monitor releases |
| Vulkan complexity | Start simple, add extensions incrementally |
| Cross-platform divergence | Extensive CI testing on all platforms |
| Performance overhead | Profiling, optional bypass for hot paths |

---
