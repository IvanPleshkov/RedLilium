# Introduction

**RedLilium** is a game and graphics engine written in Rust. It targets both native desktop platforms and the web via WebAssembly, providing a unified API across both.

## Workspace Crates

The engine is organized as a Cargo workspace:

| Crate | Description |
|-------|-------------|
| `redlilium-core` | Core types shared across the engine: meshes, textures, materials, samplers, scenes, and the glTF loader |
| `redlilium-ecs` | Entity Component System with async compute integration |
| `redlilium-graphics` | GPU rendering engine built on wgpu |
| `redlilium-app` | Application framework tying everything together |
| `redlilium-demos` | Demo scenes and examples |

## Design Goals

- **Performance first** -- sparse-set ECS with parallel system execution and work-stealing async compute
- **Cross-platform** -- identical API on native and WebAssembly; multi-threaded on native, single-threaded on web
- **Data-driven** -- property-based materials, format-agnostic scenes, runtime-composed entity archetypes
- **Incremental adoption** -- each crate can be used independently; the ECS doesn't require the renderer, the renderer doesn't require the ECS

## Getting Started

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
redlilium-ecs = { path = "ecs" }
redlilium-graphics = { path = "graphics" }
```

Then jump into the [ECS Overview](./ecs/overview.md) to learn about the entity component system, or browse the chapter list in the sidebar.
