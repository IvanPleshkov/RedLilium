# Architecture Decision Records

This document tracks significant architectural decisions for RedLilium Engine.

## ADR Format

Each decision follows this format:
- **Status**: Proposed | Accepted | Deprecated | Superseded
- **Context**: What is the issue?
- **Decision**: What was decided?
- **Consequences**: What are the trade-offs?

---

## ADR-001: Rust as Primary Language

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need to choose a language for building a custom game engine.

### Decision
Use Rust as the primary language.

### Consequences
- ✅ Memory safety without garbage collection
- ✅ Excellent WebAssembly support
- ✅ Strong type system catches bugs at compile time
- ✅ Modern package management with Cargo
- ⚠️ Steeper learning curve
- ⚠️ Longer compile times

---

## ADR-002: Workspace with Multiple Crates

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need to organize code for a game engine with multiple subsystems.

### Decision
Use a Cargo workspace with separate crates: `core`, `graphics`, `demos`.

### Consequences
- ✅ Clear separation of concerns
- ✅ Parallel compilation of independent crates
- ✅ Can publish crates independently
- ✅ Enforces API boundaries between modules
- ⚠️ More complex project structure
- ⚠️ Need to manage inter-crate dependencies

---

## ADR-003: winit for Window Management

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need a cross-platform window management library that supports both native and web.

### Decision
Use `winit` version 0.30.12.

### Consequences
- ✅ Cross-platform (Windows, Linux, macOS, Web)
- ✅ Well-maintained with active community
- ✅ Integrates well with wgpu
- ✅ Supports WebAssembly
- ⚠️ API changes between versions require updates

---

## ADR-004: Web Support via WebAssembly

**Date**: 2025-01-30
**Status**: Accepted

### Context
Want to support running demos in web browsers.

### Decision
Use wasm-pack to compile to WebAssembly with wasm-bindgen.

### Consequences
- ✅ Demos run in browsers without plugins
- ✅ Easy sharing via URLs
- ✅ Same codebase for native and web
- ⚠️ Some features unavailable on web (file system, threading)
- ⚠️ Performance may differ from native

---

## ADR-005: Documentation Strategy

**Date**: 2025-01-30
**Status**: Accepted

### Context
Need documentation that stays in sync with code and is useful for both humans and AI assistants.

### Decision
Use a layered documentation approach:
1. Rust doc comments for API documentation
2. `docs/` folder for architecture and decisions
3. Per-crate READMEs for module-specific info

### Consequences
- ✅ Doc comments are checked by compiler
- ✅ Examples in docs are tested via `cargo test --doc`
- ✅ AI can read markdown files for context
- ✅ Architecture docs separate from API docs
- ⚠️ Requires discipline to keep docs updated
