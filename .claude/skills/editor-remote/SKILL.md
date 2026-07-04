---
name: editor-remote
description: Drive the RedLilium editor through its remote-control channel — inspect the scene, make undoable edits, take scene screenshots. Use when validating editor/rendering changes or manipulating the demo scene.
---

# Editor remote control

Protocol reference: `docs/REMOTE.md`. Everything below uses the CLI.

## Launch

```bash
rm -f .redlilium/editor.port
REDLILIUM_REMOTE=1 cargo run -p redlilium-editor --bin redlilium-editor > /tmp/editor.log 2>&1 &
# wait for .redlilium/editor.port to appear (up to ~60s on first build)
```

Kill with `pkill -f redlilium-editor` when done. If a command reports
"connection closed", the editor likely crashed — check /tmp/editor.log.

## CLI

`cargo run -q -p redlilium-editor --bin editor_ctl -- <cmd>`:

- `state` — entities (id `index@tick`, name, parent, components) + selection
- `inspect <entity>` — component values as natural RON
- `edit <entity> <Component> '<RON>'` — undoable edit; response = fresh snapshot
- `add|remove <entity> <Component>`, `select <entity>…`, `undo`, `redo`
- `wait-assets [timeout_ms]` — until the asset pipeline is calm (use after
  edits that trigger loads/hot-reload, BEFORE verifying)
- `wait-frames <n>`, `logs [since_seq]`
- `screenshot <path.png>` — the scene render target, then Read the PNG
- `raw '(id: 1, cmd: "…")'` — arbitrary envelope

## Recipes

Verify a visual change:
```bash
editor_ctl edit 6@0 Transform '(translation: (0,2,0), rotation: (0,0,0,1), scale: (1,1,1))'
editor_ctl wait-frames 3          # or wait-assets if assets are involved
editor_ctl screenshot /tmp/check.png   # then Read the image
```

Verify hot reload: edit the asset file / `asset_settings`, `wait-assets`,
screenshot, compare.

Notes: entity ids are stable per session only — always start from `state`.
Component data uses natural RON; tuple-struct components address fields as
`{"0": …}`. Writes respond AFTER applying (one frame) — no need to re-inspect.
