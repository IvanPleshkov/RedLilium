//! Hosting a game module inside the editor (#58, ADR-020 slice C3).
//!
//! The editing world hosts **only the game's type registrations**
//! (`Plugin::register_types`): the inspector and scene serialization must
//! understand game components, but no game system, resource, or entity ever
//! enters the editing world — running the game is the play world's job
//! (EDITOR_REBUILD.md). Even so, registrations are `fn` pointers into the
//! module's mapped image (component meta, serialize/restore hooks, storage
//! drop glue), so a reload still cannot swap the dylib under a live world.
//!
//! The reload here follows the **proven `App::reload` shape** instead of
//! surgical in-place unloading: capture a whole-world snapshot, tear the old
//! world down *while the old image is still mapped*, swap the module (from a
//! fresh temp copy — see [`GameModule::load_fresh_copy`]), stand up a
//! replacement world (resources + schedules, zero entities), re-run
//! `Plugin::register_types` under a fresh [`SourceId`] generation, and restore
//! the snapshot. Editor entities (camera pose included) ride the same
//! snapshot; shell-owned resources (remote transport, egui, debug renderer)
//! are *adopted* across as the same `Arc`s. Undo history and selection reset
//! by design.

use std::path::PathBuf;
use std::time::Duration;

use redlilium_ecs::{Camera, EcsRunner, Entity, SourceId, World};
use redlilium_runtime::{EngineContext, GameModule};

use crate::core::{EditorWorld, EditorWorldParams, create_editor_world_base, spawn_editor_camera};
use crate::scene_view::SceneViewState;

/// How long a reload waits for in-flight compute tasks before aborting.
/// Mirrors the runtime's `prepare_for_reload` timeout: unmapping the old
/// image under a live task is UB, aborting and keeping the old module mapped
/// is a bounded leak.
const QUIESCE_TIMEOUT: Duration = Duration::from_secs(5);

/// A game module hosted in the editor: the mapped dylib (or a statically
/// linked plugin), the [`SourceId`] generation its types registered under,
/// and the paths needed to reload it.
///
/// **Lifetime invariant** (see [`GameModule`]): the host must outlive every
/// world its plugin touched. Shells own the `GameHost` *outside* the
/// `EditorWorld` and tear worlds down first (reload does this internally;
/// shutdown relies on drop order in the shell).
pub struct GameHost {
    /// `None` only transiently inside [`swap`](Self::swap).
    module: Option<GameModule>,
    generation: SourceId,
    /// The original cdylib path — reloads re-copy from here. `None` for a
    /// statically linked plugin (tests, built-in games).
    source_path: Option<PathBuf>,
    /// The unique temp copy currently mapped (removed on swap/drop).
    temp_path: Option<PathBuf>,
}

impl GameHost {
    /// Load a game cdylib from `path` (via a unique temp copy) and register
    /// its types into `ew` under a fresh generation. Nothing else of the game
    /// enters the editing world — no systems, no resources, no scene.
    ///
    /// # Safety
    ///
    /// Same contract as [`GameModule::load`]: the cdylib must come from the
    /// same `cargo build` as this editor (fingerprint-gated), default
    /// allocator on both sides, produced by `redlilium_game_module!`.
    pub unsafe fn load(
        path: impl Into<PathBuf>,
        engine: &EngineContext,
        ew: &mut EditorWorld,
    ) -> Result<Self, String> {
        let path = path.into();
        let (module, temp_path) =
            unsafe { GameModule::load_fresh_copy(&path) }.map_err(|e| e.to_string())?;
        let generation = engine.generation_registry().write().allocate_generation();
        let host = Self {
            module: Some(module),
            generation,
            source_path: Some(path),
            temp_path: Some(temp_path),
        };
        host.register_into(ew);
        log::info!(
            "game module loaded (generation {:?}, image {:?})",
            host.generation,
            host.temp_path
        );
        Ok(host)
    }

    /// Host a statically linked plugin (no dylib). The generation machinery
    /// runs identically — tests use this to exercise the reload flow without
    /// building a cdylib.
    #[cfg_attr(not(test), allow(dead_code))] // test seam + future built-in games
    pub fn from_static(
        plugin: Box<dyn redlilium_runtime::Plugin>,
        engine: &EngineContext,
        ew: &mut EditorWorld,
    ) -> Self {
        let generation = engine.generation_registry().write().allocate_generation();
        let host = Self {
            module: Some(GameModule::from_static(plugin)),
            generation,
            source_path: None,
            temp_path: None,
        };
        host.register_into(ew);
        host
    }

    /// The generation the current module's types registered under.
    #[cfg_attr(not(test), allow(dead_code))] // test seam; production reads logs
    pub fn generation(&self) -> SourceId {
        self.generation
    }

    /// Boot a play world from the hosted module: the standalone composition
    /// ([`redlilium_runtime::App::boot`]), with the game's registrations
    /// scoped to this host's generation — the same one the editing world
    /// knows the types under.
    ///
    /// The returned world's systems point into the module's mapped image:
    /// the host must outlive every world built from it (a reload requires
    /// the play session stopped first).
    pub fn boot_play_world(
        &self,
        engine: &EngineContext,
        aspect: f32,
        start_scene: Option<&str>,
    ) -> redlilium_runtime::App {
        let module = self.module.as_ref().expect("module present outside swap");
        redlilium_runtime::App::boot_scoped(
            engine,
            module.plugin(),
            aspect,
            start_scene,
            self.generation,
        )
    }

    /// Run `Plugin::register_types` against `ew`'s world, scoped to this
    /// host's generation. This is the editing world's *entire* exposure to
    /// the game module: type registrations for inspection and serialization.
    /// `Plugin::build`/`spawn_scene` run only in game worlds (`App::boot`).
    pub fn register_into(&self, ew: &mut EditorWorld) {
        let module = self.module.as_ref().expect("module present outside swap");
        let mut world = std::mem::take(&mut ew.world);
        world.with_registration_source(self.generation, |scoped| {
            module.plugin().register_types(scoped);
        });
        ew.world = world;
    }

    /// Swap in a freshly built image: unmap the old module and load a new
    /// temp copy of `source_path` under a new generation.
    ///
    /// Only valid between worlds — the caller (reload) must have dropped
    /// every world the old plugin touched. Static hosts use
    /// [`swap_static`](Self::swap_static) instead.
    fn swap(&mut self, engine: &EngineContext) -> Result<(), String> {
        let Some(source_path) = self.source_path.clone() else {
            return Err("static game module cannot be reloaded from disk".into());
        };
        // Unmap the old image first: `load_fresh_copy` maps a new unique
        // temp path either way, but the old mapping has no live referents
        // (worlds are down) and holding it is pure leak.
        self.module = None;
        if let Some(old_temp) = self.temp_path.take() {
            let _ = std::fs::remove_file(&old_temp);
        }
        // SAFETY: same-build contract as the original load; enforced by the
        // fingerprint gate inside.
        let (module, temp_path) =
            unsafe { GameModule::load_fresh_copy(&source_path) }.map_err(|e| e.to_string())?;
        self.module = Some(module);
        self.temp_path = Some(temp_path);
        self.generation = engine.generation_registry().write().allocate_generation();
        Ok(())
    }

    /// Swap in a replacement statically linked plugin (test seam for the
    /// reload flow — models "the rebuilt module" without a dylib).
    #[cfg_attr(not(test), allow(dead_code))] // test seam
    pub fn swap_static(
        &mut self,
        plugin: Box<dyn redlilium_runtime::Plugin>,
        engine: &EngineContext,
    ) {
        self.module = Some(GameModule::from_static(plugin));
        self.generation = engine.generation_registry().write().allocate_generation();
    }
}

impl Drop for GameHost {
    fn drop(&mut self) {
        // Drop the mapped module before unlinking its temp file (unlinking a
        // mapped image is fine on unix, refused on Windows — order it anyway).
        self.module = None;
        if let Some(temp) = self.temp_path.take() {
            let _ = std::fs::remove_file(&temp);
        }
    }
}

/// Options for [`reload_game`] — which shell-owned resources to adopt into
/// the replacement world (same `Arc`s, not recreated).
pub struct ReloadOptions {
    pub params: EditorWorldParams,
    pub aspect: f32,
}

/// Warm-reload the hosted game module inside the editor (#58).
///
/// Consumes the old [`EditorWorld`] and returns the replacement; the reload
/// outcome rides alongside so a failure still leaves the editor with a
/// usable world (worst case: the old module stays loaded, or the game is
/// absent but the scene survived).
///
/// Sequence (the order is load-bearing, see `GameModule`'s lifetime notes):
///
/// Callers stop the play session before reloading (both shells do) — there
/// is no play-state check here, since Play now runs in a wholly separate
/// world (`editor/src/play.rs`) that the caller has already dropped.
///
/// 1. snapshot the **whole** world — editor and game entities alike
/// 2. build the replacement world (resources + schedules, zero entities) and
///    adopt shell-owned resources from the old world
/// 3. drop the old world/history — the old image is still mapped, so game
///    storage drop glue and system destructors are sound
/// 4. quiesce the compute pool (abort on timeout: keep the old module)
/// 5. swap the module image (fresh temp copy, fresh generation)
/// 6. re-run `Plugin::register_types` into the replacement (the scene comes
///    from the snapshot; game systems never live in the editing world)
/// 7. restore the snapshot (schema-validated) and re-resolve the editor camera
///
/// On reload the undo history and selection reset by design.
pub fn reload_game(
    host: &mut GameHost,
    old: EditorWorld,
    engine: &EngineContext,
    scene_view: &mut SceneViewState,
    runner: &EcsRunner,
    opts: &ReloadOptions,
    swap: impl FnOnce(&mut GameHost, &EngineContext) -> Result<(), String>,
) -> (EditorWorld, Result<(), String>) {
    let mut old = old;

    // 1. Whole-world snapshot: editor entities (camera pose included), game
    // entities, and opted-in snapshot resources.
    let snapshot = match old.world.serialize_world() {
        Ok(s) => s,
        Err(e) => return (old, Err(format!("snapshot capture failed: {e}"))),
    };

    // 2. Replacement world. Built while the old world is still alive so
    // shell-owned resources can be adopted (same Arcs); the base world binds
    // no remote transport of its own (`remote: false` semantics come from
    // adoption), so there is no port collision.
    let mut params = EditorWorldParams {
        remote: false,
        egui: opts.params.egui,
    };
    // If the old world has no transport to adopt but the shell wants one,
    // let the base create it.
    let adopt_transport = old.world.has_resource::<redlilium_ecs::RemoteTransport>();
    params.remote = opts.params.remote && !adopt_transport;
    let mut fresh = create_editor_world_base(&params, engine, scene_view);
    if adopt_transport {
        fresh
            .world
            .adopt_resource_from::<redlilium_ecs::RemoteTransport>(&mut old.world);
    }
    // Shell-owned GPU-backed renderers ride across as the same Arcs; absent
    // ones (other shell) are simply skipped.
    fresh
        .world
        .adopt_resource_from::<redlilium_debug_drawer::DebugDrawerRenderer>(&mut old.world);
    fresh
        .world
        .adopt_resource_from::<redlilium_graphics::egui::EguiController>(&mut old.world);

    // 3. Tear the old world down while the old image is mapped.
    drop(old);

    // 4. Game tasks must finish before the image unmaps (their futures'
    // code lives inside it).
    let elapsed = runner.compute().quiesce(QUIESCE_TIMEOUT);
    if elapsed >= QUIESCE_TIMEOUT {
        // Fail closed: keep the old module mapped (bounded leak, no UB) and
        // rebuild the world against it so the editor stays usable.
        host.register_into(&mut fresh);
        let restore = restore_into(&mut fresh, &snapshot, opts.aspect);
        let msg = format!(
            "task quiescence timeout after {QUIESCE_TIMEOUT:?} — reload aborted, \
             old module kept mapped{}",
            restore
                .as_ref()
                .err()
                .map(|e| format!("; restore: {e}"))
                .unwrap_or_default()
        );
        return (fresh, Err(msg));
    }

    // 5. Swap the image (dylib: unmap + fresh temp copy + new generation).
    if let Err(e) = swap(host, engine) {
        // Old module still mapped (swap fails before unmapping only for
        // static hosts / IO errors after which `module` was reloaded or
        // kept). Rebuild against whatever module the host still holds.
        host.register_into(&mut fresh);
        let _ = restore_into(&mut fresh, &snapshot, opts.aspect);
        return (fresh, Err(format!("module swap failed: {e}")));
    }

    // 6. Fresh registrations under the new generation; the scene comes from
    // the snapshot.
    host.register_into(&mut fresh);

    // 7. Snapshot restore + camera re-resolution.
    let restore = restore_into(&mut fresh, &snapshot, opts.aspect);
    (fresh, restore)
}

/// The dylib swap used by production callers of [`reload_game`].
pub fn swap_from_disk(host: &mut GameHost, engine: &EngineContext) -> Result<(), String> {
    host.swap(engine)
}

fn restore_into(
    fresh: &mut EditorWorld,
    snapshot: &redlilium_ecs::serialize::SerializedWorld,
    aspect: f32,
) -> Result<(), String> {
    fresh
        .world
        .deserialize_world_into(snapshot)
        .map_err(|e| format!("snapshot restore failed: {e}"))?;
    fresh.schedules.mark_startup_done();
    fresh.editor_camera = find_editor_camera(&fresh.world)
        .unwrap_or_else(|| spawn_editor_camera(&mut fresh.world, aspect));
    Ok(())
}

/// The restored editor camera: the EDITOR-flagged entity carrying a `Camera`.
fn find_editor_camera(world: &World) -> Option<Entity> {
    world.iter_entities().find(|&e| {
        world.get_entity_flags(e) & Entity::EDITOR != 0 && world.get::<Camera>(e).is_some()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EditorWorldParams, create_editor_world};
    use crate::scene_view::SceneViewState;
    use redlilium_ecs::Component;
    use redlilium_graphics::{GraphicsInstance, TextureFormat};
    use redlilium_runtime::Plugin;
    use redlilium_vfs::Vfs;

    /// The hosted game's component: registered by `register_types`, authored
    /// into the editing world by the editor, serialized by its derive like
    /// any std component.
    #[derive(Clone, Component)]
    struct GameBlip {
        tag: u32,
    }

    /// A game module under the v2 contract: `register_types` declares the
    /// component; `build`/`spawn_scene` belong to game worlds and must never
    /// run in the editing world.
    struct GameV1;
    impl Plugin for GameV1 {
        fn register_types(&self, world: &mut World) {
            world.register_inspector::<GameBlip>();
        }
        fn build(&self, _app: &mut redlilium_runtime::App) {
            panic!("build must not run in the editing world (registrations only)");
        }
        fn spawn_scene(&self, _app: &mut redlilium_runtime::App) {
            panic!("spawn_scene must not run in the editing world");
        }
    }

    /// v2 — "the rebuilt module": same types, same contract.
    struct GameV2;
    impl Plugin for GameV2 {
        fn register_types(&self, world: &mut World) {
            world.register_inspector::<GameBlip>();
        }
        fn build(&self, _app: &mut redlilium_runtime::App) {
            panic!("build must not run in the editing world (registrations only)");
        }
        fn spawn_scene(&self, _app: &mut redlilium_runtime::App) {
            panic!("spawn_scene must not run in the editing world");
        }
    }

    fn blip_tags(world: &World) -> Vec<u32> {
        let blips = world.read_all::<GameBlip>().unwrap();
        blips.iter().map(|(_, b)| b.tag).collect()
    }

    /// #58/#59 end-to-end (static-module seam; the dylib load/temp-copy path
    /// is exercised by `demos/src/bin/reload_harness.rs`): hosting a module
    /// gives the editing world the game's *types and nothing else*; authored
    /// game components survive a warm reload into the new generation.
    #[test]
    fn editor_hosts_registrations_only_and_reloads() {
        let instance = GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("graphics device");
        let engine = EngineContext::with_vfs(device.clone(), Vfs::new());
        let mut scene_view = SceneViewState::new(device.clone(), TextureFormat::Bgra8UnormSrgb);
        let runner = EcsRunner::single_thread();
        let params = EditorWorldParams {
            remote: false,
            egui: false,
        };

        let mut ew = create_editor_world(&params, &engine, &mut scene_view, 1.0);
        let editor_entities = ew.world.iter_entities().count();
        assert!(editor_entities > 0, "demo scene spawned");

        // --- Host the game module: registrations only. The plugin's
        // build/spawn_scene panic if called, so from_static completing at all
        // proves the editing world never runs them. ---
        let mut host = GameHost::from_static(Box::new(GameV1), &engine, &mut ew);
        let gen1 = host.generation();
        assert_ne!(gen1, SourceId::HOST, "game types under a real generation");
        assert_eq!(
            ew.world.iter_entities().count(),
            editor_entities,
            "hosting adds no entities to the editing world"
        );

        // The type is genuinely known: the editor can author a game component
        // and a frame ticks without any game system existing.
        let e = ew.world.spawn();
        ew.world.insert(e, GameBlip { tag: 7 }).unwrap();
        ew.schedules.run_frame(&mut ew.world, &runner, 1.0 / 60.0);
        assert_eq!(blip_tags(&ew.world), vec![7], "authored game component");

        // --- Warm reload: swap in "the rebuilt module" (v2). ---
        let opts = ReloadOptions {
            params,
            aspect: 1.0,
        };
        let (ew, result) = reload_game(
            &mut host,
            ew,
            &engine,
            &mut scene_view,
            &runner,
            &opts,
            |host, engine| {
                host.swap_static(Box::new(GameV2), engine);
                Ok(())
            },
        );
        result.expect("reload succeeds");

        // Generation advanced; the whole scene survived exactly once.
        assert_ne!(host.generation(), gen1, "fresh generation after reload");
        assert_eq!(
            blip_tags(&ew.world),
            vec![7],
            "authored game component restored, not respawned"
        );
        assert_eq!(
            ew.world.iter_entities().count(),
            editor_entities + 1,
            "no duplication of editor or game entities"
        );
        // The editor camera was re-resolved from the restored entities.
        assert!(
            ew.world.is_alive(ew.editor_camera),
            "editor camera restored"
        );
        // Undo history and selection reset by design.
        assert!(!ew.history.can_undo(), "undo history reset");
        assert!(
            ew.world
                .resource::<redlilium_ecs::ui::Selection>()
                .entities()
                .is_empty(),
            "selection reset"
        );
    }
}
