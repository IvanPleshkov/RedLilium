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

### Canonical pre-commit gate

Run the **`preflight`** skill after any code change (it packages the exact
sequence and flags below and reports a pass/fail verdict). If running by hand
on Linux/macOS, the canonical invocation is:

```bash
cargo fmt --all
CARGO_INCREMENTAL=0 bash scripts/test-all.sh
git diff --stat -- std-assets/assets.db project-assets/assets.db   # must be empty
```

Load-bearing gotchas (getting any wrong wastes a run):

- **`bash` prefix**, not `./scripts/test-all.sh` — the script must be invoked
  through `bash`.
- **The web build is self-healing** — with `wasm-pack` present it runs the full
  packaging; without it, it falls back to a `cargo build --target
  wasm32-unknown-unknown` compile check. So `--skip-web` is now just a speed
  flag, not a requirement. (Baking needs the Slang SDK — `scripts/fetch-slang.sh`
  — else the bake staleness check self-skips.)
- **`CARGO_INCREMENTAL=0`** — avoids incremental artifacts skewing the run.
- **Asset DBs must stay clean** — `std-assets/assets.db` and
  `project-assets/assets.db` must not change as a side effect of unrelated
  work; check the diff before every commit.

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
- **IMPORTANT:** After making any changes to Rust code, always run `cargo fmt --all` to format the code before running tests or committing
- Run `cargo clippy` before committing
- All warnings should be fixed (clippy runs with `-D warnings`)
- A crate that instantiates wgpu-typed resources (e.g. links `redlilium-graphics`
  and constructs materials/pipelines) needs `#![recursion_limit = "256"]` at the
  crate root, or the `Send`/`Sync` auto-trait resolution overflows. Precedent:
  `ecs`, `graphics`, `editor`, `runtime`.

### Committing

- **Never commit without an explicit instruction** from the user. Running the
  `preflight` gate is not authorization to commit.
- Before committing, run `preflight` (or its manual equivalent) — green build,
  tests, clippy, and clean asset DBs.
- End every commit message with the co-author trailer for the model that did
  the work, e.g. `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  (or the Opus/Sonnet trailer when that tier owns the session).
- Reference issues in the message (`#N`); `Closes #N` auto-closes on push.
- If on the default branch (`main`), branch first unless told otherwise.

### Documentation

- Update doc comments when changing public APIs
- Check `docs/ARCHITECTURE.md` for system design context
- Check `docs/DECISIONS.md` for architecture decision records

### Task Tracking

- Tasks live in **GitHub Issues** (`IvanPleshkov/RedLilium`), not in markdown
  files — do NOT create TODO/roadmap .md files
- Delegate issue operations (create/digest/close) to the `gh-tasks` agent
  (model: haiku); issue bodies are written by the orchestrator, the agent
  only executes `gh` commands
- Reference issues in commits (`#N`); `Closes #N` auto-closes on push
- `docs/` holds durable design only (architecture, decisions, contracts)

### Agent Delegation

Specialized subagents live in `.claude/agents/`. Delegate instead of doing
these in the main context:

| Agent | Model | Use for |
|---|---|---|
| `editor-driver` | sonnet | Multi-step interactive editor sessions over the remote channel (`docs/REMOTE.md`, skill `editor-remote`) |
| `editor-logs` | haiku | Digesting editor logs — never pull raw logs into the main context |
| `gh-tasks` | haiku | Issue digests and batch operations (single create/close inline via `gh` is cheaper — the agent exists to keep bulk JSON out of context) |
| `Explore` (built-in) | — | Broad code searches across many files/conventions |

Validation sequences against the editor that need judgment (reading
screenshots, "does it look right") belong to the orchestrator or
`editor-driver`; mechanical regression checking is a test-infrastructure
concern, not an agent's job.

Note: new/edited agent definitions become invocable as named agent types
only from the next session.

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

# Run the car game's editor (statically hosted game, ADR-033; no REDLILIUM_GAME needed)
cargo run -p car-game-editor

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

## HARD RULES for Code Quality

These are non-negotiable invariants that affect architecture and code review:

### 1. Editor EditAction Invariant
**UI code must NEVER mutate World/resources directly.** Every edit (component change, asset update, hierarchy edit) goes through:
1. Create an `EditAction<World>` 
2. Push to `ActionQueue<World>`
3. Drained once per frame into `EditActionHistory`
4. Undo/redo works automatically

This is a red line enforced in code review. One direct `resource_mut()` write from UI breaks undo/redo silently. Asset-record edits must feed `ChangedAssets` + `DirtyMounts` in **both** apply AND undo.

**See:** `ecs/src/ui/component_inspector.rs` for pattern.

### 2. GPU Upload Through Frame Graph Only
**Phase 0 is DONE.** All GPU resource data uploads go through the per-frame `RenderGraph` via `TransferPass` operations. The following methods **do not exist** and must never be re-added to `GraphicsDevice`:
- `write_buffer`, `read_buffer`, `write_texture`, `create_mesh_from_cpu`, `create_texture_from_cpu`

When something needs data on/off the GPU and the sanctioned APIs feel inconvenient, **solve it by routing through the frame graph**, NOT by adding direct methods. The friction is intentional.

**Sanctioned APIs:** `device.create_mesh_deferred()`, `TransferOperation::write_buffer()`, `RingBuffer` for per-frame data, manager `flush_uploads()`.

**See:** `docs/ASSETS.md` (rev 3).

## Cost & Discipline

### Fable (Metered Tier) — Expert Review Only
Fable 5 is **expensive per-token**. Use only for hard architectural judgment calls — never for code generation, research, or fact-checking.

**Reserved use:** Design review, adversarial soundness checks, hidden-assumption spotting, cross-dylib ABI vetting.

**What NOT to do:** Mechanism research (read source code yourself), general exploration (web search first), task implementation (use Opus/Sonnet or main context).

**Explicit user authorization required** — you cannot spawn Fable autonomously.

### Task Tracking — GitHub Issues Only
- Tasks live in **GitHub Issues** (`IvanPleshkov/RedLilium`), never in markdown files
- Delegate CRUD to `gh-tasks` agent (haiku); you write issue bodies, it runs `gh`
- Reference issues in commits: `#N`; `Closes #N` auto-closes on push
- No TODO/roadmap .md files

## File Locations Reference

| Purpose | Location |
|---------|----------|
| Workspace config | `Cargo.toml` |
| Test scripts | `scripts/test-all.sh`, `scripts/test-all.ps1` |
| Architecture docs | `docs/ARCHITECTURE.md` |
| Decision records | `docs/DECISIONS.md` |
| Testing guide | `docs/TESTING.md` |
