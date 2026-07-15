# Plugin Lifecycle Design for #45 (Approved)

**Date**: 2026-07-12  
**Status**: APPROVED by Fable 5 (Architectural Review); **partially superseded
by ADR-032 (2026-07-16)** — the editor rebuild removed in-world Play→Stop
transitions, so `Plugin::on_stop` and every PlayModeAware/snapshot-restore
reference below no longer exist. What still stands: `on_unload` (reload
cleanup before World drop), the host-owned plugin registry drop-coupled to
the dylib, and the task-quiescence requirement. The plugin contract also
gained `register_types` (see ADR-032).  
**Selected Option**: A (on_stop + on_unload callbacks) + Mitigations  
**Blocking**: #45 implementation

---

## Executive Summary

Plugins cannot use `PlayModeAwareRegistry` (dylib-unsafe function pointers). Instead:

1. **Stop Cleanup:** Engine-mediated `plugin.on_stop(&mut app)` callbacks, called **before** despawn/restore
2. **Reload Cleanup:** Engine-mediated `plugin.on_unload(&mut app)` callbacks, called **before** World drop
3. **Plugin Registry:** Host-owned (in `EngineContext`/`GameModule`), drop-coupled to dylib lifetime
4. **Task Quiescence:** Async tasks must be joined/drained before dylib unload

---

## Plugin Trait Extension

```rust
pub trait Plugin {
    /// Register components, resources, events, and systems. 
    /// Runs once per world generation (first boot and every reload).
    fn build(&self, app: &mut App);

    /// Populate the initial scene. Called only on first boot.
    fn spawn_scene(&self, app: &mut App) {
        let _ = app;
    }

    /// Stop cleanup: called when Play → Stop transition fires.
    /// Called BEFORE engine hooks (despawn, restore, etc.).
    /// Use for: muting audio, canceling tasks, flushing pending state.
    /// 
    /// **Invariant:** plugin runs this while game world is still live;
    /// World is dropped immediately after (via ManagePlayModeTransitions).
    fn on_stop(&self, app: &mut App) {
        let _ = app;
    }

    /// Reload cleanup: called before World drop during reload.
    /// Called AFTER scene serialization, BEFORE World drop, BEFORE dylib unload.
    /// Use for: joining long-lived tasks, flushing external resources (files, network).
    /// 
    /// **Invariant:** World is available but about to be dropped;
    /// this is the last chance to access game state before dylib unload.
    fn on_unload(&self, app: &mut App) {
        let _ = app;
    }
}
```

---

## Stop Transition Sequence (Corrected)

**NEW RULE:** `on_stop` runs **before any engine-side transition logic**.

```
1. ManagePlayModeTransitions system detects Stop-transition request (in PreUpdate)

2. **PLUGIN CLEANUP PHASE** (Host-side, before schedules run):
   for plugin in host.plugin_registry:
       catch_unwind(|| plugin.on_stop(&mut app))
           on panic: log error, continue to next plugin
   
3. apply_transition(world, PlayState::Playing → PlayState::Stopped):
   - emit PlayModeTransition(Playing, Stopped) event
   - dispatch PlayModeAware hooks (engine-side)
   - despawn game entities
   - restore snapshot resources (Score, RNG, Physics.paused=false)
   - hide editor entities

4. Set GameActive = false (gate off game systems)

5. frame() resumes (game schedules skip; UI schedules continue)
```

**Key:** on_stop runs while World is still live and consistent. No despawn/restore has occurred yet.

---

## Reload Sequence (Corrected)

```
1. Editor state: (GameModule, App, World) all live
   
2. Capture scene:
   snapshot = world.serialize_world()

3. **PLUGIN UNLOAD CLEANUP PHASE** (Host-side):
   for plugin in host.plugin_registry:
       catch_unwind(|| plugin.on_unload(&mut app))
           on panic: log error, continue to next plugin

4. Drop World (triggers drop glue, component dtors)

5. Drop App (triggers schedule drop, system dtors)

6. **TASK QUIESCENCE (NEW)**:
   block on ComputePool::drain()
   → all plugin-spawned tasks must complete or be canceled
   → no dylib code is still running

7. unload_dylib() (dlclose; plugin code unmaps)

8. load_dylib(new path) (dlopen; fingerprint gate)

9. new App + World created

10. call plugin.build(&mut app) (re-register types/systems)

11. deserialize_world_into(&mut world, snapshot)

12. app.run_frame() (using new plugin's systems)
```

**New Step 6:** Task drain is CRITICAL. If any plugin task is still executing dylib code when dlclose fires, instant UB.

---

## Plugin Registry: Host-Owned

### Placement

```rust
pub struct GameModule {
    plugin: Box<dyn Plugin>,
    #[cfg(not(target_arch = "wasm32"))]
    _library: Option<libloading::Library>,
}

pub struct EngineContext {
    // ...
    plugins: Vec<Box<dyn Plugin>>,  // NEW: host-owned registry
}
```

**Invariant:** `plugins` Vec lives as long as `GameModule` + library handle. Drop order (by field order in the containing struct) ensures:
1. Plugins Vec is available during on_unload
2. World/App drop before plugins Vec
3. Plugins Vec drops, triggering drop glue in dylib code
4. Library unmaps

### Why Not World-Resident?

If `plugins` is a World resource:
- Plugins Vec is dropped by World (arbitrary order, not coupled to library)
- Library might unmap before `plugins` drop glue runs dylib code
- Same hazard as `PlayModeAwareRegistry` inside World → UB

**Solution:** Host-side lifetime coupling is the fix.

---

## Task Quiescence on Reload

### Problem

Plugin systems can spawn async tasks:
```rust
pub struct GameSystem;
impl System for GameSystem {
    fn run(&self, ctx: &SystemContext) {
        let pool = ctx.lock::<Res<ComputePool>>();
        pool.spawn(|| {
            // This task captures `||` closure in dylib code
            // On reload, dylib unmaps while task is running
            // → UB when task tries to jump back into dylib code
        });
    }
}
```

### Mitigation: Cooperative Cancel + Block on Drain

1. **Mark reload as pending:** Set a flag before unload phase
2. **Tasks check flag:** In periodic checkpoints, tasks cooperatively exit
3. **Block on drain:** Before dlclose, wait for all tasks to finish
   ```rust
   compute_pool.drain();  // Blocks until all tasks complete or timeout
   ```

### Implementation

In App reload sequence:

```rust
pub fn reload(app: &mut App, new_plugin: Box<dyn Plugin>) {
    // 1. Snapshot
    let snapshot = app.world.serialize_world();
    
    // 2. Plugin cleanup (on_unload)
    for plugin in &app.plugins {
        std::panic::catch_unwind(...|| plugin.on_unload(app));
    }
    
    // 3. Drop World
    drop(app.world);
    
    // 4. Drain all async tasks (BLOCKING)
    app.compute_pool.drain();  // <-- CRITICAL STEP
    
    // 5. Now safe to drop App (systems, plugins) + unload dylib
    drop(app);
    drop(old_plugin_library);
    
    // 6. Load new dylib + rebuild
    let new_lib = GameModule::load(new_path)?;
    let new_app = new_lib.plugin().build_app();
    new_app.world.deserialize_world_into(snapshot);
}
```

---

## Panic Handling in Callbacks

### Rule: Catch + Continue

```rust
for plugin in &self.plugins {
    match std::panic::catch_unwind(
        std::panic::AssertUnwindSafe(|| plugin.on_stop(&mut app))
    ) {
        Ok(()) => {},
        Err(payload) => {
            log::error!("plugin.on_stop panicked (payload dropped); continuing");
            // Don't propagate; let next plugin run
        }
    }
}
```

**Rationale:** One plugin's panic must not prevent others from cleaning up.

---

## Snapshot Schema Safety (Fable Note)

### Problem

Self-describing formats (JSON, RON) tolerate field reordering. Positional encodings (bincode) do not — deserializing into a reordered struct layout silently produces garbage.

### Mitigation: Per-Component Schema Hash

Add to snapshot metadata:

```rust
pub struct SnapshotMetadata {
    engine_fingerprint: String,
    timestamp: u64,
    component_schemas: HashMap<ComponentName, SchemaHash>,
        // E.g., ("Transform", hash_of_field_order_and_types)
}

fn deserialize_world_into(world: &mut World, snapshot: &SerializedWorld) {
    for (name, data, schema) in &snapshot.components {
        let current_schema = registry.schema_hash(name);
        if current_schema != schema {
            return Err(format!("Schema mismatch for {}: {} vs {}", name, schema, current_schema));
        }
        // Safe to deserialize; schema is certified stable
        registry.deserialize(world, name, data);
    }
}
```

**Consequence:** Schema changes (e.g., adding a field) are caught as errors, not silent corruption.

---

## Multi-Plugin Composition

### Scope: All-or-Nothing Reload

**Decision:** Partial reload (unload Plugin B while keeping Plugin A) is **NOT supported**.

**Reason:** Schedules intermix plugins' systems (Plugin A's SystemX → Plugin B's SystemY edges). Unloading Plugin B breaks the schedule DAG and leaves Plugin A's edges pointing to unloaded code.

**Implication:** Reload always tears down all plugins and reloads all.

### Ordering: Insertion Order

Plugins are registered in load order (host initialization sequence):

```rust
host.load_plugin("game-core.dylib");      // Registered first
host.load_plugin("game-audio.dylib");     // Registered second
host.load_plugin("game-logic.dylib");     // Registered third
```

Callbacks run in order: on_stop(core), on_stop(audio), on_stop(logic).

System ordering is determined by `add_edge` in `build()` — explicit DAG, not implicit.

---

## Lifecycle Diagram (All Paths)

```
First Boot:
  build(core) → build(audio) → build(logic)
    ↓ spawn_scene() (only first boot)
    ↓ run_frame()

Play → Stop:
  on_stop(core, audio, logic) [host-side, before despawn]
    ↓ apply_transition (engine hooks, despawn, restore)
    ↓ set GameActive = false
    ↓ game systems skip (but always-on systems run, e.g., UI)

Reload:
  on_unload(core, audio, logic) [host-side, World still live]
    ↓ drop World
    ↓ drain async tasks [BLOCKING]
    ↓ unload dylib
    ↓ load new dylib
    ↓ build(core') → build(audio') → build(logic') [fresh registration]
    ↓ deserialize_world (scene restored)
    ↓ run_frame()

Shutdown:
  (terminate; drop is sufficient)
```

---

## Testing Strategy

### Unit Tests

- `test_on_stop_called_before_despawn()` — verify order
- `test_on_stop_with_multiple_plugins()` — verify all called despite panic in one
- `test_on_unload_with_live_world()` — verify World is accessible
- `test_schema_mismatch_detected()` — verify snapshot validation

### Integration Tests

- Full Play → Stop cycle with example plugin
- Full Reload cycle (serialize, unload, load, deserialize)
- Task drain blocking before unload

---

## Risk Mitigation Summary

| Risk | Mitigation |
|------|-----------|
| Async tasks in-flight during reload | drain() blocks before dlclose |
| on_stop ordering vs. despawn/restore | Called pre-transition; World is live |
| Plugin registry inside World (UB) | Host-owned registry drop-coupled to lib handle |
| One plugin panic blocks others | catch_unwind + continue-on-panic |
| Schema corruption (field reorder) | Per-component schema hash in snapshot metadata |
| Partial reload hazard | Explicitly not supported; all-or-nothing only |

---

## References

- **ADR-020:** Game Code Authoring — Rust Plugins over Shared Engine Dylib
- **ADR-024:** Resource Lifecycle Management — Hybrid Hook + Event Model
- **Fable 5 Review:** Architectural soundness check (2026-07-12)
- **#45:** runtime: game cdylib loading + warm-restart reload
- **#65:** Resource Lifecycle Management (completed; PlayModeAware + SnapshotResource)
