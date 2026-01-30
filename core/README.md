# RedLilium Engine Core

Core crate for RedLilium containing foundational common parts.

## Purpose

## Building
```bash
# Build the crate
cargo build -p redlilium-core

# Run tests
cargo test -p redlilium-core

# Generate documentation
cargo doc -p redlilium-core --open
```

## Design Decisions

- **No graphics dependencies** - Core is purely logic, graphics are separate
- **Minimal allocations** - Designed for performance-critical solutions
- **Platform agnostic** - Works on native and web targets
