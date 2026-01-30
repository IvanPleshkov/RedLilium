# Contributing to RedLilium Engine

Thank you for your interest in contributing to RedLilium Engine! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) (for web builds)
- Git

### Setting Up Git Hooks

Before making any commits, you must set up the project's git hooks. These hooks ensure code quality by running automated checks before each commit.

**On Windows (PowerShell):**
```powershell
.\scripts\setup-hooks.ps1
```

**On Linux/macOS:**
```bash
./scripts/setup-hooks.sh
```

The setup script installs the following hooks:
- **pre-commit**: Runs `cargo fmt --check` to verify code formatting

## Code Formatting

**All Rust code must be formatted before each commit.** The pre-commit hook will automatically check this and reject commits with formatting issues.

To format your code:
```bash
cargo fmt --all
```

To check formatting without making changes:
```bash
cargo fmt --all -- --check
```
