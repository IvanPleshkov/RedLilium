# CLAUDE.md - AI Assistant Instructions for RedLilium Engine

This file provides instructions for AI assistants (particularly Claude Code) working on this project.

## Project Overview

RedLilium is a game and graphics engine written in Rust. It supports both native (desktop) and web (WebAssembly) targets.

### Workspace Structure

- `core/` - Core utilities and common functionality (`redlilium-core`)
- `graphics/` - Custom rendering engine (`redlilium-graphics`)
- `demos/` - Demo scenes and examples (`redlilium-demos`)
- `docs/` - Architecture and design documentation
- `scripts/` - Build and test automation scripts

## Testing After Changes

**IMPORTANT:** After making code changes, always run the test script to verify the project builds and passes all checks.

### Running Tests

**On Windows (PowerShell):**
```powershell
.\scripts\test-all.ps1
```

**On Linux/macOS:**
```bash
./scripts/test-all.sh
```

### What the Test Script Checks

1. **Native Build** - `cargo build --workspace`
2. **Web Build** - `wasm-pack build demos --target web --out-dir web/pkg`
3. **Unit Tests** - `cargo test --workspace`
4. **Clippy Linter** - `cargo clippy --workspace --all-targets -- -D warnings`

### Quick Test Options

If you only changed code (not build configuration), you can skip builds:
```powershell
# Windows
.\scripts\test-all.ps1 -SkipNative -SkipWeb

# Linux/macOS
./scripts/test-all.sh --skip-native --skip-web
```

If wasm-pack is not installed, skip web build:
```powershell
# Windows
.\scripts\test-all.ps1 -SkipWeb

# Linux/macOS
./scripts/test-all.sh --skip-web
```

## Development Guidelines

### Code Style

- Use Rust 2024 edition conventions
- Run `cargo clippy` before committing
- All warnings should be fixed (clippy runs with `-D warnings`)

### Documentation

- Update doc comments when changing public APIs
- Check `docs/ARCHITECTURE.md` for system design context
- Check `docs/DECISIONS.md` for architecture decision records

### Adding New Features

1. Read relevant crate README and `docs/ARCHITECTURE.md`
2. Implement the feature
3. Add unit tests
4. Run the full test suite
5. Update documentation if needed

### Common Commands

```bash
# Build all crates
cargo build --workspace

# Run the window demo
cargo run -p redlilium-demos --bin window_demo

# Run tests for a specific crate
cargo test -p redlilium-core
cargo test -p redlilium-graphics

# Generate documentation
cargo doc --workspace --no-deps --open

# Format code
cargo fmt --all

# Check without building
cargo check --workspace
```

## File Locations Reference

| Purpose | Location |
|---------|----------|
| Workspace config | `Cargo.toml` |
| Test scripts | `scripts/test-all.sh`, `scripts/test-all.ps1` |
| Architecture docs | `docs/ARCHITECTURE.md` |
| Decision records | `docs/DECISIONS.md` |
| Testing guide | `docs/TESTING.md` |
