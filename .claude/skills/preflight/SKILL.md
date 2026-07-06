---
name: preflight
description: Run the full pre-commit gate for RedLilium — format, the workspace test-all script (build + unit tests + clippy -D warnings), and the asset-DB guard — then report a pass/fail verdict and the change footprint. Use after any code change, before proposing a commit.
---

# Preflight — the pre-commit gate

Run these steps in order. Stop and report on the first failure; do not commit
if any step fails. This gate is identical regardless of which model owns the
session — run it the same way every time.

## 1. Format

```bash
cargo fmt --all
```

Formatting mutates files, so it must run **before** the test script (a later
`fmt` change would leave an uncommitted diff behind an already-green run).

## 2. Full test script

```bash
CARGO_INCREMENTAL=0 bash scripts/test-all.sh --skip-web
```

Gotchas (all load-bearing):

- **`bash` prefix**, not `./scripts/test-all.sh` — the script relies on being
  invoked through `bash`.
- **`--skip-web`** — `wasm-pack` is not installed in this environment; without
  the flag the web build step fails spuriously.
- **`CARGO_INCREMENTAL=0`** — keeps incremental artifacts from skewing the run.
- The script runs native build, unit tests, and **clippy with `-D warnings`**
  (zero-warnings policy). A single clippy warning fails the gate.

If only code changed (no build config), a faster local check is
`... test-all.sh --skip-native --skip-web`, but run the full form before a
commit.

## 3. Asset-DB guard

```bash
git diff --stat -- std-assets/assets.db project-assets/assets.db
```

These databases must **not** change as a side effect of unrelated work. If the
diff is non-empty and you did not intentionally edit assets, investigate before
committing — a stray rehash or scan usually means something ran that should not
have.

## 4. Report

State the verdict plainly:

- **PASS** — every step green, asset DBs clean. Report `git status --short` and
  `git diff --stat` as the change footprint.
- **FAIL** — name the failing step and paste the relevant output (compile
  error, failing test, clippy warning). Do not soften or omit failures.

Do **not** commit here. Committing is a separate, explicitly-authorized step
(see CLAUDE.md → Committing).
