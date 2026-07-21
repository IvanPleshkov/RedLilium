//! The editor's play world (EDITOR_REBUILD.md §4.3).
//!
//! Play builds a **game world through the same composition as the standalone
//! build** ([`App::boot`]): same resources, same schedules, same ungated game
//! systems — parity with the shipped game by construction. The differences
//! are host parameters only: the start scene is the scene open for editing,
//! the camera renders to an offscreen target the shell shows in the scene
//! view, and input comes from the scene view instead of a window.
//!
//! Pause = the game schedules don't tick; instead the host materializes an
//! **inspection overlay** over the frozen world: the editing world's
//! EDITOR-flagged entities are instantiated into the play world *as data*
//! (whatever the editor apparatus is, it transfers by construction — no
//! hardcoded spawn list), and a fresh host-owned kit of systems flies the
//! camera over it. Both die whole on resume, so overlay state can never
//! dangle into the running game. Stop = drop the play world. The editing
//! world is never touched by any of this — there is nothing to snapshot,
//! restore, or hide.

use std::sync::Arc;

use redlilium_core::abstract_editor::{ActionQueue, EditActionHistory};
use redlilium_ecs::sync::RwLock;
use redlilium_ecs::ui::Selection;
use redlilium_ecs::{
    Camera, CameraOutput, CameraTarget, EcsRunner, Entity, FreeFlyCamera, MainViewport,
    OutputFormat, Render, RenderSchedule, Schedules, SizePolicy, SystemsContainer, Transform,
    UpdateCameraMatrices, UpdateFreeFlyCamera, UpdateGlobalTransforms, WindowInput, World,
};
use redlilium_graphics::{RenderGraph, TextureFormat};
use redlilium_runtime::{AppControl, EngineContext};

use crate::game_host::GameHost;

/// Clear color of the play camera's target — the standalone build's default
/// (`GameConfig::default().clear_color`), so Play looks like the game.
const CLEAR_COLOR: [f32; 4] = [0.02, 0.02, 0.03, 1.0];

/// The pause-inspection overlay: everything the host adds to a paused play
/// world, created fresh on every pause and dropped whole on resume.
///
/// Fresh-per-pause is the invariant, not an implementation detail: while the
/// game runs, any retained entity id or hover state goes stale (entities die,
/// indexes recycle). Killing the whole kit on resume makes stale references
/// unrepresentable instead of "checked" — the same lifecycle argument as
/// dropping the play world on Stop.
struct PauseOverlay {
    /// Host-owned systems ticked against the paused world in place of the
    /// game schedules (flyover camera → transform propagation → camera
    /// matrices). Never merged into the game's own schedules.
    systems: SystemsContainer,
    /// Entities instantiated from the editing world's EDITOR apparatus;
    /// despawned on resume. Their EDITOR flags ride the serialized records,
    /// so game queries never see them even mid-pause.
    spawned: Vec<Entity>,
    /// The flyover camera among [`spawned`](Self::spawned).
    camera: Option<Entity>,
}

/// The overlay's system kit — one composition function, mirroring how
/// [`redlilium_runtime::App::new`] is the one composition of a game world.
fn compose_pause_overlay() -> SystemsContainer {
    let mut systems = SystemsContainer::new();
    systems.add(UpdateFreeFlyCamera);
    systems.add(UpdateGlobalTransforms);
    systems.add(UpdateCameraMatrices);
    systems
        .add_edge::<UpdateFreeFlyCamera, UpdateGlobalTransforms>()
        .expect("no cycle");
    systems
        .add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>()
        .expect("no cycle");
    systems
}

/// The editing world's EDITOR apparatus as data — the inverse of
/// [`redlilium_ecs::scene_from_world`] (which strips editor entities from a
/// scene): only EDITOR-flagged entities, no resources. Taken from the *live*
/// editing world at pause time — the editing world survives Play, so there
/// is no ahead-of-time copy to keep fresh.
fn editor_apparatus_from(
    world: &World,
) -> Result<redlilium_ecs::serialize::SerializedWorld, redlilium_ecs::serialize::SerializeError> {
    let mut snapshot = world.serialize_world()?;
    snapshot
        .entities
        .retain(|e| e.entity_flags & (Entity::EDITOR | Entity::INHERITED_EDITOR) != 0);
    snapshot.resources.clear();
    Ok(snapshot)
}

/// A running play session: the game world + schedules exactly as the
/// standalone runtime builds them, dropped whole on Stop.
///
/// **Lifetime:** the game's systems and component storages point into the
/// hosted module's mapped image, so the [`GameHost`] must outlive the
/// session. Shells own the host outside the session and drop the session
/// first; a module reload requires the session to be stopped.
pub struct PlaySession {
    world: World,
    schedules: Schedules,
    /// Play-world input — the shell feeds scene-view input here.
    window_input: Arc<RwLock<WindowInput>>,
    /// While `true` the game schedules are not ticked at all (game time
    /// freezes); the pause overlay ticks instead.
    paused: bool,
    /// The inspection overlay, present only while paused (and only if the
    /// apparatus transfer succeeded).
    overlay: Option<PauseOverlay>,
    /// Flyover pose carried across pauses of this session — plain values
    /// copied out before the overlay dies, never references into it.
    saved_flyover: Option<(FreeFlyCamera, Transform)>,
    /// History the pause-mode inspector drains play-world edits through
    /// ([`dock_targets`](Self::dock_targets)). Always in **discarding** mode:
    /// the play world is a throwaway copy dropped on Stop, so edits apply but
    /// nothing accumulates and the world is never "dirty". Keeping the panels'
    /// `EditAction` → `execute` path (not a direct `resource_mut`) preserves the
    /// editor's mutate-only-through-actions invariant even here.
    pause_history: EditActionHistory<World>,
}

impl PlaySession {
    /// Boot a play world from the hosted module: the standalone composition
    /// ([`App::boot`] — `register_types` + `build` + `spawn_scene`), with the
    /// editing session's scene as the host's start-scene override.
    ///
    /// `start_scene` is mount-relative (what [`redlilium_ecs::SceneManager`]
    /// addresses); `None` lets the game start at its own default (menu).
    pub fn start(
        host: &GameHost,
        engine: &EngineContext,
        runner: &EcsRunner,
        aspect: f32,
        start_scene: Option<&str>,
    ) -> Self {
        log::info!(
            "play: booting game world (start scene: {})",
            start_scene.unwrap_or("<game default>")
        );
        let app = host.boot_play_world(engine, aspect, start_scene);
        let window_input = app.window_input();
        let (mut world, mut schedules) = app.into_parts();
        schedules.run_startup(&mut world, runner);
        Self {
            world,
            schedules,
            window_input,
            paused: false,
            overlay: None,
            saved_flyover: None,
            pause_history: EditActionHistory::discarding(),
        }
    }

    /// Advance the session by one frame. `view_size` is the scene view's
    /// physical size — the play world's "window". While paused, the game
    /// schedules do not run; the pause overlay's kit ticks instead, flying
    /// the inspection camera over the frozen world.
    pub fn tick(&mut self, runner: &EcsRunner, dt: f64, view_size: (f32, f32)) {
        {
            let mut input = self.window_input.write();
            input.window_width = view_size.0;
            input.window_height = view_size.1;
        }
        if self.paused {
            if let Some(overlay) = &self.overlay {
                for err in runner.run(&mut self.world, &overlay.systems) {
                    log::error!("pause-overlay system error: {err}");
                }
                self.window_input.write().begin_frame();
            }
            return;
        }
        self.schedules.run_frame(&mut self.world, runner, dt);
        // Clear per-frame input deltas after the game consumed them (the
        // standalone host does the same at the end of its frame).
        self.window_input.write().begin_frame();
    }

    /// Render the play world into `graph`: sync the game camera's offscreen
    /// output spec to the scene view size, run the `Render` schedule (uploads,
    /// `EnsureCameraTargets`, forward pass), and hand back the graph plus any
    /// asset-upload transfer graphs (the shell submits those first, then the
    /// graph — before the editor's own frame graph, so the egui pass sampling
    /// the play texture executes after the forward pass wrote it).
    pub fn render(
        &mut self,
        runner: &EcsRunner,
        graph: RenderGraph,
        width: u32,
        height: u32,
        surface_format: TextureFormat,
    ) -> (RenderGraph, Vec<RenderGraph>) {
        let (width, height) = (width.max(1), height.max(1));
        self.world.insert_resource(MainViewport::new(width, height));
        if let Some(camera) = self.view_camera() {
            let desired = CameraOutput::offscreen(SizePolicy::Fixed(width, height), None)
                .with_clear_color(CLEAR_COLOR)
                .with_format(OutputFormat::matching_surface(surface_format));
            let stale = self
                .world
                .get::<CameraOutput>(camera)
                .is_none_or(|current| *current != desired);
            if stale {
                let _ = self.world.insert(camera, desired);
            }
        }
        self.world.resource_mut::<RenderSchedule>().set(graph);
        self.schedules
            .run_schedule::<Render>(&mut self.world, runner);
        let mut schedule_res = self.world.resource_mut::<RenderSchedule>();
        let transfers = schedule_res.take_transfer_graphs();
        let graph = schedule_res
            .take()
            .expect("RenderSchedule must hold the graph after the Render schedule");
        drop(schedule_res);
        (graph, transfers)
    }

    /// The color texture of the game camera's derived target — what the shell
    /// shows in the scene view. `None` until the first `Render` schedule run
    /// derives the target (the panel is blank for one frame, same as the
    /// editing world after a resize).
    pub fn scene_color(&self) -> Option<Arc<redlilium_graphics::Texture>> {
        let camera = self.view_camera()?;
        self.world
            .get::<CameraTarget>(camera)
            .map(|t| t.color.clone())
    }

    /// The game requested an exit (`AppControl`) — the shell maps this to Stop.
    pub fn exit_requested(&self) -> bool {
        self.world.resource::<AppControl>().exit_requested()
    }

    /// Handle to the play world's input resource (shells forward scene-view
    /// input through it).
    pub fn window_input(&self) -> Arc<RwLock<WindowInput>> {
        self.window_input.clone()
    }

    /// Read access to the play world (play-world *inspection* is deliberately
    /// not exposed to the panels yet).
    #[cfg_attr(not(test), allow(dead_code))] // test seam + future inspection surface
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Pause the session and stand up the inspection overlay: the editing
    /// world's EDITOR apparatus (flyover camera and friends) is instantiated
    /// into the play world as data, plus a fresh system kit to fly it. If the
    /// transfer fails, the session still pauses — just without the flyover.
    ///
    /// `editing_world` is read live at pause time; nothing is captured ahead
    /// of Play.
    pub fn pause(&mut self, editing_world: &World) {
        if self.paused {
            return;
        }
        self.paused = true;

        // Stand up the pause-mode inspection surface on the play world: the
        // panels edit through an ActionQueue like any editing world, and carry
        // their own Selection. Both are inert while the game runs (nothing
        // ticks them) and the play world is dropped on Stop, so they need no
        // teardown on resume. The history stays discarding (see the field).
        self.world.insert_resource(ActionQueue::<World>::new());
        self.world.insert_resource(Selection::new());
        self.pause_history = EditActionHistory::discarding();

        let apparatus = match editor_apparatus_from(editing_world) {
            Ok(a) => a,
            Err(e) => {
                log::error!("pause: capturing the editor apparatus failed: {e} (no flyover)");
                return;
            }
        };
        let spawned = match self.world.deserialize_world_into(&apparatus) {
            Ok(s) => s,
            Err(e) => {
                log::error!("pause: instantiating the editor apparatus failed: {e} (no flyover)");
                return;
            }
        };
        let camera = spawned
            .iter()
            .copied()
            .find(|&e| self.world.get::<Camera>(e).is_some());
        // Continue the flyover where the previous pause of this session left
        // it (the instantiated copy carries the editing world's pose).
        if let (Some(camera), Some((fly, transform))) = (camera, self.saved_flyover) {
            let _ = self.world.insert(camera, fly);
            let _ = self.world.insert(camera, transform);
        }
        self.overlay = Some(PauseOverlay {
            systems: compose_pause_overlay(),
            spawned,
            camera,
        });
    }

    /// Resume the game: the overlay dies whole — its entities despawn, its
    /// kit drops. Only the flyover pose survives, as a value.
    pub fn resume(&mut self) {
        if let Some(overlay) = self.overlay.take() {
            if let Some(camera) = overlay.camera {
                let fly = self.world.get::<FreeFlyCamera>(camera).copied();
                let transform = self.world.get::<Transform>(camera).copied();
                if let (Some(fly), Some(transform)) = (fly, transform) {
                    self.saved_flyover = Some((fly, transform));
                }
            }
            for entity in overlay.spawned {
                self.world.despawn(entity);
            }
        }
        self.paused = false;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// The paused play world plus the history the inspector edits it through —
    /// what the shell feeds the dock **instead of** the editing world while
    /// paused, so the hierarchy/inspector observe and edit the *real* running
    /// entities (the game camera, spawned actors, …). Disjoint field borrows,
    /// so the dock gets a mutable world and a shared history at once.
    ///
    /// Only meaningful while [`is_paused`](Self::is_paused); off-pause the
    /// ActionQueue the panels expect may be absent.
    pub fn dock_targets(&mut self) -> (&mut World, &EditActionHistory<World>) {
        (&mut self.world, &self.pause_history)
    }

    /// Drain the play world's [`ActionQueue`] into the discarding history,
    /// applying this frame's pause-mode edits. Mirrors
    /// [`crate::core::EditorWorld::drain_actions`] for the editing world. A
    /// no-op unless paused with the queue present.
    pub fn drain_pause_actions(&mut self) {
        if !self.paused || !self.world.has_resource::<ActionQueue<World>>() {
            return;
        }
        let actions = self.world.resource::<ActionQueue<World>>().drain();
        for action in actions {
            if let Err(e) = self.pause_history.execute(action, &mut self.world) {
                log::warn!("pause-mode edit failed: {e}");
            }
        }
    }

    /// The camera the scene view shows: the flyover camera while paused with
    /// an overlay, the game's own camera otherwise.
    fn view_camera(&self) -> Option<Entity> {
        if self.paused
            && let Some(overlay) = &self.overlay
            && overlay.camera.is_some()
        {
            return overlay.camera;
        }
        self.game_camera()
    }

    /// The first non-EDITOR camera of the game world — the same resolution
    /// rule the standalone host uses for its primary camera. (Games with
    /// several outputs need a declared primary output — a shared host-layer
    /// track, standalone and editor alike.)
    fn game_camera(&self) -> Option<Entity> {
        let cameras = self.world.read_all::<Camera>().ok()?;
        for (idx, _) in cameras.iter() {
            let Some(entity) = self.world.entity_at_index(idx) else {
                continue;
            };
            if self.world.get_entity_flags(entity) & (Entity::EDITOR | Entity::INHERITED_EDITOR)
                == 0
            {
                return Some(entity);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{EditorWorldParams, create_editor_world};
    use crate::game_host::GameHost;
    use crate::scene_view::SceneViewState;
    use redlilium_ecs::{Component, Update};
    use redlilium_graphics::GraphicsInstance;
    use redlilium_runtime::{App, Plugin};
    use redlilium_vfs::Vfs;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Clone, Component)]
    struct Blip {
        tag: u32,
    }

    struct Tick(Arc<AtomicU32>);
    impl redlilium_ecs::System for Tick {
        type Result = ();
        fn run<'a>(
            &'a self,
            _ctx: &'a redlilium_ecs::SystemContext<'a>,
        ) -> Result<(), redlilium_ecs::SystemError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// A real game under the v2 contract: types for every world, systems and
    /// scene for game worlds only.
    struct TestGame(Arc<AtomicU32>);
    impl Plugin for TestGame {
        fn register_types(&self, world: &mut World) {
            world.register_inspector::<Blip>();
        }
        fn build(&self, app: &mut App) {
            app.add_system::<Update, _>(Tick(self.0.clone()));
        }
        fn spawn_scene(&self, app: &mut App) {
            let world = app.world_mut();
            let e = world.spawn();
            world.insert(e, Blip { tag: 7 }).unwrap();
        }
    }

    /// The heart of the rebuild: Play in the editor stands up a *game* world
    /// (spawned scene, ungated game systems ticking) while the editing world
    /// is untouched; Pause stops ticking; Stop drops the world and leaves the
    /// editing world exactly as it was.
    #[test]
    fn play_session_runs_the_game_and_leaves_editing_world_alone() {
        let instance = GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("graphics device");
        let engine = redlilium_runtime::EngineContext::with_vfs(device.clone(), Vfs::new());
        let mut scene_view = SceneViewState::new(
            device.clone(),
            redlilium_graphics::TextureFormat::Bgra8UnormSrgb,
        );
        let runner = EcsRunner::single_thread();
        let params = EditorWorldParams {
            remote: false,
            egui: false,
        };
        let mut ew = create_editor_world(&params, &engine, &mut scene_view, 1.0);
        let ticks = Arc::new(AtomicU32::new(0));
        let host = GameHost::from_static(Box::new(TestGame(ticks.clone())), &engine, &mut ew);
        let editor_entities = ew.world.iter_entities().count();

        // Play: a game world exists, its scene is spawned, its systems tick
        // without any gating — and the editing world gained nothing.
        let mut play = PlaySession::start(&host, &engine, &runner, 1.0, None);
        assert_eq!(
            play.world().read_all::<Blip>().unwrap().iter().count(),
            1,
            "game scene spawned in the play world"
        );
        assert_eq!(
            ew.world.iter_entities().count(),
            editor_entities,
            "editing world untouched by Play"
        );

        let dt = 1.0 / 60.0;
        play.tick(&runner, dt, (640.0, 480.0));
        play.tick(&runner, dt, (640.0, 480.0));
        assert_eq!(ticks.load(Ordering::SeqCst), 2, "game systems run in Play");

        // Pause: the game does not tick, and the editing world's EDITOR
        // apparatus arrives as data — the flyover camera exists in the play
        // world, invisible to game queries by its flags.
        play.pause(&ew.world);
        play.tick(&runner, dt, (640.0, 480.0));
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            2,
            "paused world does not tick"
        );
        let flyover = {
            let cams = play.world().read_all::<FreeFlyCamera>().unwrap();
            let idx = cams.iter().next().map(|(idx, _)| idx);
            idx.and_then(|idx| play.world().entity_at_index(idx))
        }
        .expect("flyover camera instantiated on pause");
        assert_ne!(
            play.world().get_entity_flags(flyover) & Entity::EDITOR,
            0,
            "flyover camera carries the EDITOR flag into the play world"
        );

        // Fly it: hold W, tick — the camera moves while the game stays frozen.
        let before = play.world().get::<Transform>(flyover).copied().unwrap();
        play.window_input()
            .write()
            .on_key_pressed(redlilium_core::input::KeyCode::W);
        play.tick(&runner, dt, (640.0, 480.0));
        let after = play.world().get::<Transform>(flyover).copied().unwrap();
        assert_ne!(
            before.translation, after.translation,
            "flyover camera moves while paused"
        );
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            2,
            "game stays frozen while flying"
        );
        play.window_input()
            .write()
            .on_key_released(redlilium_core::input::KeyCode::W);

        // Resume: the overlay dies whole; the game ticks again.
        play.resume();
        assert_eq!(
            play.world()
                .read_all::<FreeFlyCamera>()
                .unwrap()
                .iter()
                .count(),
            0,
            "overlay entities despawned on resume"
        );
        play.tick(&runner, dt, (640.0, 480.0));
        assert_eq!(ticks.load(Ordering::SeqCst), 3, "resume ticks again");

        // Second pause: the flyover pose survived as a value.
        play.pause(&ew.world);
        let flyover2 = {
            let cams = play.world().read_all::<FreeFlyCamera>().unwrap();
            let idx = cams.iter().next().map(|(idx, _)| idx);
            idx.and_then(|idx| play.world().entity_at_index(idx))
        }
        .expect("flyover camera instantiated on second pause");
        assert_eq!(
            play.world()
                .get::<Transform>(flyover2)
                .copied()
                .unwrap()
                .translation,
            after.translation,
            "flyover pose carried across pauses"
        );
        play.resume();

        // Stop: drop the play world; the editing world still works.
        drop(play);
        ew.schedules.run_frame(&mut ew.world, &runner, dt);
        assert_eq!(
            ew.world.iter_entities().count(),
            editor_entities,
            "editing world identical after Stop"
        );
    }

    /// A component edit an inspector would push, targeting a real play-world
    /// entity.
    #[derive(Debug)]
    struct SetBlipTag {
        entity: Entity,
        tag: u32,
    }
    impl redlilium_core::abstract_editor::EditAction<World> for SetBlipTag {
        fn apply(
            &mut self,
            world: &mut World,
        ) -> redlilium_core::abstract_editor::EditActionResult {
            let _ = world.insert(self.entity, Blip { tag: self.tag });
            Ok(())
        }
        fn undo(
            &mut self,
            _world: &mut World,
        ) -> redlilium_core::abstract_editor::EditActionResult {
            unreachable!("discarding history never undoes");
        }
        fn description(&self) -> &str {
            "set blip tag"
        }
    }

    /// Pause exposes the **real** play world to the panels: the inspection
    /// ActionQueue + Selection appear on the play world, an edit pushed like the
    /// inspector does lands on the running entity via the frame drain, and the
    /// pause history retains nothing (ephemeral, no-undo) — the abstraction
    /// leak (panels bound to the editing world in pause) is closed.
    #[test]
    fn pause_binds_panels_to_the_real_play_world_ephemerally() {
        let instance = GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("graphics device");
        let engine = redlilium_runtime::EngineContext::with_vfs(device.clone(), Vfs::new());
        let mut scene_view = SceneViewState::new(
            device.clone(),
            redlilium_graphics::TextureFormat::Bgra8UnormSrgb,
        );
        let runner = EcsRunner::single_thread();
        let params = EditorWorldParams {
            remote: false,
            egui: false,
        };
        let mut ew = create_editor_world(&params, &engine, &mut scene_view, 1.0);
        let ticks = Arc::new(AtomicU32::new(0));
        let host = GameHost::from_static(Box::new(TestGame(ticks.clone())), &engine, &mut ew);
        let mut play = PlaySession::start(&host, &engine, &runner, 1.0, None);

        // The game spawned Blip{tag:7} — a runtime entity that lives *only* in
        // the play world (the editing world never has it).
        let blip = {
            let blips = play.world().read_all::<Blip>().unwrap();
            let idx = blips.iter().next().map(|(idx, _)| idx);
            idx.and_then(|idx| play.world().entity_at_index(idx))
        }
        .expect("game spawned a Blip");
        assert_eq!(play.world().get::<Blip>(blip).unwrap().tag, 7);
        // Off pause the inspection surface is absent.
        assert!(!play.world().has_resource::<ActionQueue<World>>());

        play.pause(&ew.world);
        assert!(
            play.world().has_resource::<ActionQueue<World>>(),
            "pause stands up the inspection ActionQueue on the play world"
        );
        assert!(play.world().has_resource::<Selection>());

        // Push an edit the way the inspector does, then run the frame drain.
        {
            let (world, _history) = play.dock_targets();
            world
                .resource::<ActionQueue<World>>()
                .push(Box::new(SetBlipTag {
                    entity: blip,
                    tag: 42,
                }));
        }
        play.drain_pause_actions();
        assert_eq!(
            play.world().get::<Blip>(blip).unwrap().tag,
            42,
            "the edit landed on the real running entity"
        );

        // Ephemeral: nothing retained, world never reported dirty.
        let (_world, history) = play.dock_targets();
        assert!(!history.can_undo(), "pause edits are not undoable");
        assert!(!history.has_unsaved_changes());

        // Resume + Stop leave the editing world innocent of the game entity.
        play.resume();
        drop(play);
        assert_eq!(
            ew.world.read_all::<Blip>().unwrap().iter().count(),
            0,
            "the game entity never touched the editing world"
        );
    }
}
