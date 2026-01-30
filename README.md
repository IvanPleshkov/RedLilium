# RedLilium Engine

A custom game and graphics engine written in Rust.

## Project Structure

```
redlilium/
├── core/       # utilities and common parts
├── graphics/   # Custom rendering engine
├── demos/      # Demo scenes and examples
│   └── web/    # Web build files
└── docs/       # Architecture and design documentation
```

## Prerequisites

### Native Build

- [Rust](https://rustup.rs/) (latest stable, edition 2024)
- Platform-specific dependencies for winit (see [winit docs](https://docs.rs/winit))

### Web Build

- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- [miniserve](https://github.com/svenstaro/miniserve) - Install with `cargo install miniserve`

## Building & Running

### Native

```bash
# Build all crates
cargo build

# Run the window demo
cargo run -p redlilium-demos --bin window_demo
```

### Web

```bash
# Build for web
wasm-pack build demos --target web --out-dir web/pkg

# Serve the web files
miniserve demos/web --port 8080
# Then open http://localhost:8080/index.html in your browser
```

## Testing

Run all tests (native build, web build, unit tests, and clippy) with the cross-platform test script:

### Linux / macOS

```bash
./scripts/test-all.sh
```

### Windows (PowerShell)

```powershell
.\scripts\test-all.ps1
```

### Options

| Bash Flag | PowerShell Flag | Description |
|-----------|-----------------|-------------|
| `--skip-native` | `-SkipNative` | Skip native build test |
| `--skip-web` | `-SkipWeb` | Skip web build test |
| `--skip-tests` | `-SkipTests` | Skip unit tests |
| `--skip-clippy` | `-SkipClippy` | Skip clippy linter |
| `--verbose` | `-Verbose` | Show verbose output |
| `--help` | `-Help` | Show help message |

See [docs/TESTING.md](docs/TESTING.md) for detailed testing documentation.

## Documentation Strategy

This project uses a layered documentation approach designed for both human developers and AI assistants:

### 1. Code-Level Documentation (Rust Doc Comments)

- `//!` module-level docs explain the purpose and structure of each module
- `///` item-level docs describe functions, structs, and their usage
- Examples in doc comments serve as both documentation and tests

Generate docs with:
```bash
cargo doc --workspace --no-deps --open
```

### 2. Architecture Documentation (`docs/` folder)

The `docs/` folder contains high-level documentation:

- `docs/ARCHITECTURE.md` - System design and module interactions
- `docs/DECISIONS.md` - Architecture Decision Records (ADRs)
- `docs/ROADMAP.md` - Feature roadmap and milestones

### 3. Crate READMEs

Each crate has its own `README.md` with:
- Purpose and responsibilities
- Public API overview
- Usage examples
- Build instructions specific to that crate

### 4. Keeping Documentation Updated

**Best Practices:**

1. **Doc comments live with code** - When you change a function, update its doc comment in the same commit
2. **ADRs are append-only** - Never modify past decisions, add new ones that supersede
3. **Examples are tests** - Use `cargo test --doc` to verify documentation examples compile
4. **CI checks** - Run `cargo doc --workspace` in CI to catch broken doc links

**For AI Assistants:**

The documentation is structured so AI tools can:
- Read `docs/ARCHITECTURE.md` for high-level understanding
- Read crate READMEs for module-specific context
- Use `cargo doc` output for API reference
- Check `docs/DECISIONS.md` for rationale behind design choices

## License

MIT
