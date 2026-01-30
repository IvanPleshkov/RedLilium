# RedLilium Engine Architecture

This document describes the high-level architecture of RedLilium Engine.

## Overview

RedLilium Engine is structured as a Cargo workspace with three main crates:

```
┌─────────────────────────────────────────────────────────┐
│                   redlilium-demos                       │
│              (Demo scenes and examples)                 │
├──────────────────────────┬──────────────────────────────┤
│     redlilium-core       │     redlilium-graphics       │
│                          │  (Rendering, Shaders, GPU)   │
└──────────────────────────┴──────────────────────────────┘
```

## Design Principles

### 1. Separation of Concerns

- **Core** handles tools and common parts with no graphics dependencies
- **Graphics** handles all rendering with no game logic
- **Demos** combines both to create runnable applications

### 2. Platform Abstraction

All crates support both native and web targets:
- Native: Windows, Linux, macOS
- Web: WebAssembly with WebGl2 and WebGPU

### 3. Data-Driven Design

- Configuration over code where possible
- Render graph for flexible pipeline configuration

## Module Overview

### redlilium-core

```
core/
```

### redlilium-graphics

```
graphics/
```

## Future Architecture (Planned)

## Reading Guide for AI Assistants

1. Start with this file for overall structure
2. Read individual crate READMEs for module details
3. Check `DECISIONS.md` for rationale behind choices
4. Use `cargo doc` for API reference
