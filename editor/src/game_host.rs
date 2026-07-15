//! Hosting a game module inside the editor (#58, ADR-020 slice C3).
//!
//! The editor owns a long-lived world with editor-only state (undo history,
//! selection, remote transport, GPU-backed renderers). A game module's
//! `Plugin::build` plants systems, component registrations, and closures —
//! all pointing into the module's mapped image — into that world. A reload
//! therefore cannot swap the dylib under a live world.
//!
//! The reload here follows the **proven `App::reload` shape** instead of
//! surgical in-place unloading: capture a whole-world snapshot, tear the old
//! world down *while the old image is still mapped*, swap the module (from a
//! fresh temp copy — see [`GameModule::load_fresh_copy`]), stand up a
//! replacement world (resources + schedules, zero entities), re-run
//! `Plugin::build` under a fresh [`SourceId`] generation, and restore the
//! snapshot. Editor entities (camera pose included) ride the same snapshot;
//! shell-owned resources (remote transport, egui, debug renderer) are
//! *adopted* across as the same `Arc`s. Undo history and selection reset by
//! design.

use std::path::PathBuf;
use std::time::Duration;

use redlilium_ecs::{Camera, EcsRunner, Entity, PlayControl, PlayState, SourceId, World};
use redlilium_runtime::{App, EngineContext, GameModule};

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
    /// Load a game cdylib from `path` (via a unique temp copy), register its
    /// types under a fresh generation, build its systems into `ew`, and spawn
    /// its scene.
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
        aspect: f32,
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
        host.build_into(ew, aspect, true);
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
        aspect: f32,
    ) -> Self {
        let generation = engine.generation_registry().write().allocate_generation();
        let host = Self {
            module: Some(GameModule::from_static(plugin)),
            generation,
            source_path: None,
            temp_path: None,
        };
        host.build_into(ew, aspect, true);
        host
    }

    /// The generation the current module's types registered under.
    #[cfg_attr(not(test), allow(dead_code))] // test seam; production reads logs
    pub fn generation(&self) -> SourceId {
        self.generation
    }

    /// Re-run `Plugin::build` (and optionally `spawn_scene`) against `ew`,
    /// with registrations scoped to this host's generation. Mirrors
    /// `App::reload`'s scoped-build: the world and schedules are moved into a
    /// temporary [`App`] for the plugin's benefit and moved back after.
    pub fn build_into(&self, ew: &mut EditorWorld, aspect: f32, spawn_scene: bool) {
        let module = self.module.as_ref().expect("module present outside swap");
        let mut world = std::mem::take(&mut ew.world);
        let mut schedules = std::mem::take(&mut ew.schedules);
        let window_input = ew.window_input.clone();
        world.with_registration_source(self.generation, |scoped| {
            module.plugin().register_types(scoped);
            let mut app = App::from_parts(
                std::mem::take(scoped),
                std::mem::take(&mut schedules),
                window_input.clone(),
                aspect,
            );
            module.plugin().build(&mut app);
            if spawn_scene {
                module.plugin().spawn_scene(&mut app);
            }
            let (w, s) = app.into_parts();
            *scoped = w;
            schedules = s;
        });
        ew.world = world;
        ew.schedules = schedules;
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
/// 1. refuse outside `Stopped` (play-state restore interplay is out of scope)
/// 2. snapshot the **whole** world — editor and game entities alike
/// 3. build the replacement world (resources + schedules, zero entities) and
///    adopt shell-owned resources from the old world
/// 4. drop the old world/history — the old image is still mapped, so game
///    storage drop glue and system destructors are sound
/// 5. quiesce the compute pool (abort on timeout: keep the old module)
/// 6. swap the module image (fresh temp copy, fresh generation)
/// 7. re-run `Plugin::build` into the replacement (scene comes from the
///    snapshot — `spawn_scene` is deliberately skipped)
/// 8. restore the snapshot (schema-validated) and re-resolve the editor camera
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
    // 1. Only from Stopped.
    if old.world.resource::<PlayControl>().state() != PlayState::Stopped {
        return (old, Err("reload_game requires PlayState::Stopped".into()));
    }
    let mut old = old;

    // 2. Whole-world snapshot: editor entities (camera pose included), game
    // entities, and opted-in snapshot resources.
    let snapshot = match old.world.serialize_world() {
        Ok(s) => s,
        Err(e) => return (old, Err(format!("snapshot capture failed: {e}"))),
    };

    // 3. Replacement world. Built while the old world is still alive so
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

    // 4. Tear the old world down while the old image is mapped.
    drop(old);

    // 5. Game tasks must finish before the image unmaps (their futures'
    // code lives inside it).
    let elapsed = runner.compute().quiesce(QUIESCE_TIMEOUT);
    if elapsed >= QUIESCE_TIMEOUT {
        // Fail closed: keep the old module mapped (bounded leak, no UB) and
        // rebuild the world against it so the editor stays usable.
        host.build_into(&mut fresh, opts.aspect, false);
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

    // 6. Swap the image (dylib: unmap + fresh temp copy + new generation).
    if let Err(e) = swap(host, engine) {
        // Old module still mapped (swap fails before unmapping only for
        // static hosts / IO errors after which `module` was reloaded or
        // kept). Rebuild against whatever module the host still holds.
        host.build_into(&mut fresh, opts.aspect, false);
        let _ = restore_into(&mut fresh, &snapshot, opts.aspect);
        return (fresh, Err(format!("module swap failed: {e}")));
    }

    // 7. Fresh build under the new generation; the scene comes from the
    // snapshot, so `spawn_scene` is skipped (same contract as `App::reload`).
    host.build_into(&mut fresh, opts.aspect, false);

    // 8. Snapshot restore + camera re-resolution.
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
    use redlilium_ecs::{Component, GameActiveCondition, Update};
    use redlilium_graphics::{GraphicsInstance, TextureFormat};
    use redlilium_runtime::Plugin;
    use redlilium_vfs::Vfs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// The hosted game's component: registered by `build`, spawned by
    /// `spawn_scene`, serialized by its derive like any std component.
    #[derive(Clone, Component)]
    struct GameBlip {
        tag: u32,
    }

    /// Game system gated on `GameActiveCondition` — counts its runs so the
    /// test can assert the Play/Pause/Stop gating end-to-end.
    struct GameTick(Arc<AtomicU32>);
    impl redlilium_ecs::System for GameTick {
        type Result = ();
        fn run<'a>(
            &'a self,
            _ctx: &'a redlilium_ecs::SystemContext<'a>,
        ) -> Result<(), redlilium_ecs::SystemError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// v1 of the game module.
    struct GameV1(Arc<AtomicU32>);
    impl Plugin for GameV1 {
        fn build(&self, app: &mut App) {
            app.register_component::<GameBlip>();
            let counter = self.0.clone();
            let update = app.schedule_mut::<Update>();
            update.add_condition(GameActiveCondition);
            update.add(GameTick(counter));
            update
                .add_edge::<GameActiveCondition, GameTick>()
                .expect("no cycle");
        }
        fn spawn_scene(&self, app: &mut App) {
            let world = app.world_mut();
            let e = world.spawn();
            world.insert(e, GameBlip { tag: 7 }).unwrap();
        }
    }

    /// v2 — "the rebuilt module": same types, same registration, no
    /// spawn_scene on reload (the snapshot is the scene).
    struct GameV2(Arc<AtomicU32>);
    impl Plugin for GameV2 {
        fn build(&self, app: &mut App) {
            app.register_component::<GameBlip>();
            let counter = self.0.clone();
            let update = app.schedule_mut::<Update>();
            update.add_condition(GameActiveCondition);
            update.add(GameTick(counter));
            update
                .add_edge::<GameActiveCondition, GameTick>()
                .expect("no cycle");
        }
        fn spawn_scene(&self, _app: &mut App) {
            panic!("spawn_scene must not run on reload — the snapshot is the scene");
        }
    }

    fn blip_tags(world: &World) -> Vec<u32> {
        let blips = world.read_all::<GameBlip>().unwrap();
        blips.iter().map(|(_, b)| b.tag).collect()
    }

    /// #58/#59/#62 end-to-end (static-module seam; the dylib load/temp-copy
    /// path is exercised by `demos/src/bin/reload_harness.rs`): host a game
    /// in the editor world, drive the full Play→Pause→Resume→Stop transition
    /// cycle, then warm-reload the module and assert the scene (game AND
    /// editor entities) survived into the new generation.
    #[test]
    fn editor_hosts_game_through_transitions_and_reload() {
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

        // --- Host the game module (build + spawn_scene). ---
        let v1_ticks = Arc::new(AtomicU32::new(0));
        let mut host =
            GameHost::from_static(Box::new(GameV1(v1_ticks.clone())), &engine, &mut ew, 1.0);
        let gen1 = host.generation();
        assert_ne!(gen1, SourceId::HOST, "game types under a real generation");
        assert_eq!(blip_tags(&ew.world), vec![7], "game scene spawned");
        assert_eq!(
            ew.world.iter_entities().count(),
            editor_entities + 1,
            "game added exactly its own entity"
        );

        let dt = 1.0 / 60.0;

        // --- Full transition cycle (#62): the gated game system runs only
        // while the game is active. ---
        ew.schedules.run_frame(&mut ew.world, &runner, dt); // stopped
        assert_eq!(v1_ticks.load(Ordering::SeqCst), 0, "inactive while stopped");

        ew.world.resource_mut::<PlayControl>().play();
        ew.schedules.run_frame(&mut ew.world, &runner, dt); // playing
        assert_eq!(v1_ticks.load(Ordering::SeqCst), 1, "runs while playing");

        ew.world.resource_mut::<PlayControl>().pause();
        ew.schedules.run_frame(&mut ew.world, &runner, dt); // paused
        assert_eq!(
            v1_ticks.load(Ordering::SeqCst),
            2,
            "game systems keep running while paused (freeze is zero game-time)"
        );

        ew.world.resource_mut::<PlayControl>().resume();
        ew.schedules.run_frame(&mut ew.world, &runner, dt); // playing again
        assert_eq!(v1_ticks.load(Ordering::SeqCst), 3, "runs after resume");

        ew.world.resource_mut::<PlayControl>().stop();
        ew.schedules.run_frame(&mut ew.world, &runner, dt); // stopped
        let ticks_at_stop = v1_ticks.load(Ordering::SeqCst);
        ew.schedules.run_frame(&mut ew.world, &runner, dt);
        assert_eq!(
            v1_ticks.load(Ordering::SeqCst),
            ticks_at_stop,
            "gated off after stop"
        );
        assert_eq!(
            blip_tags(&ew.world),
            vec![7],
            "scene intact after the cycle"
        );

        // --- Warm reload: swap in "the rebuilt module" (v2). ---
        let v2_ticks = Arc::new(AtomicU32::new(0));
        let opts = ReloadOptions {
            params,
            aspect: 1.0,
        };
        let (mut ew, result) = reload_game(
            &mut host,
            ew,
            &engine,
            &mut scene_view,
            &runner,
            &opts,
            |host, engine| {
                host.swap_static(Box::new(GameV2(v2_ticks.clone())), engine);
                Ok(())
            },
        );
        result.expect("reload succeeds");

        // Generation advanced; the whole scene survived exactly once.
        assert_ne!(host.generation(), gen1, "fresh generation after reload");
        assert_eq!(
            blip_tags(&ew.world),
            vec![7],
            "game scene restored, not respawned"
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

        // The new module's systems are live: play drives v2's counter.
        ew.world.resource_mut::<PlayControl>().play();
        ew.schedules.run_frame(&mut ew.world, &runner, dt);
        assert_eq!(
            v2_ticks.load(Ordering::SeqCst),
            1,
            "v2 systems run after reload"
        );
        assert_eq!(
            v1_ticks.load(Ordering::SeqCst),
            ticks_at_stop,
            "v1 systems are gone"
        );
    }
}
