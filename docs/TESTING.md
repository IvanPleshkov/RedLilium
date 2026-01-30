# Testing Guide

This document describes how to run tests for the RedLilium Engine project.

## Quick Start

### Linux / macOS

```bash
./scripts/test-all.sh
```

### Windows (PowerShell)

```powershell
.\scripts\test-all.ps1
```

## What the Test Script Does

The test script runs the following checks in order:

1. **Native Build** - Compiles all crates for the native target
2. **Web Build** - Compiles the demos crate to WebAssembly using wasm-pack
3. **Unit Tests** - Runs all unit tests across the workspace
4. **Clippy** - Runs the Clippy linter with warnings treated as errors

## Prerequisites

### Required Tools

- **Rust** (latest stable) - [Install](https://rustup.rs/)
- **Clippy** - Automatically installed if missing (via `rustup component add clippy`)

### Optional Tools (for web build)

- **wasm-pack** - [Install](https://rustwasm.github.io/wasm-pack/installer/)
  - On Linux/macOS: `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh`
  - On Windows: `cargo install wasm-pack` or download from the website

If wasm-pack is not installed, the web build test will be automatically skipped.

## Script Options

### Bash (Linux/macOS)

```bash
./scripts/test-all.sh [OPTIONS]

Options:
  --skip-native    Skip native build test
  --skip-web       Skip web build test
  --skip-tests     Skip unit tests
  --skip-clippy    Skip clippy linter
  --verbose        Show verbose output
  --help           Show help message
```

### PowerShell (Windows)

```powershell
.\scripts\test-all.ps1 [OPTIONS]

Options:
  -SkipNative    Skip native build test
  -SkipWeb       Skip web build test
  -SkipTests     Skip unit tests
  -SkipClippy    Skip clippy linter
  -Verbose       Show verbose output
  -Help          Show help message
```

## Examples

### Run all tests

```bash
# Linux/macOS
./scripts/test-all.sh

# Windows
.\scripts\test-all.ps1
```

### Skip web build (faster)

```bash
# Linux/macOS
./scripts/test-all.sh --skip-web

# Windows
.\scripts\test-all.ps1 -SkipWeb
```

### Only run unit tests and clippy

```bash
# Linux/macOS
./scripts/test-all.sh --skip-native --skip-web

# Windows
.\scripts\test-all.ps1 -SkipNative -SkipWeb
```

## Running Tests Manually

If you prefer to run tests manually:

### Native Build

```bash
cargo build --workspace
```

### Web Build

```bash
wasm-pack build demos --target web --out-dir web/pkg
```

### Unit Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p redlilium-core
cargo test -p redlilium-graphics
cargo test -p redlilium-demos
```

### Clippy

```bash
# Run with warnings as errors
cargo clippy --workspace --all-targets -- -D warnings

# Run without failing on warnings
cargo clippy --workspace --all-targets
```

## Exit Codes

The test script returns:
- `0` - All tests passed
- `1` - One or more tests failed

This makes it suitable for use in CI/CD pipelines and automation scripts.

## Adding New Tests

### Unit Tests

Add tests to your Rust source files using the standard `#[cfg(test)]` module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        assert!(true);
    }
}
```

### Integration Tests

Add integration tests in a `tests/` directory within each crate:

```
crate_name/
├── src/
│   └── lib.rs
└── tests/
    └── integration_test.rs
```
