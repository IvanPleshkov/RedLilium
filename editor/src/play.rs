//! The editor's play world (EDITOR_REBUILD.md §4.3).
//!
//! Play builds a **game world through the same composition as the standalone
//! build** ([`App::boot`]): same resources, same schedules, same ungated game
//! systems — parity with the shipped game by construction. The differences
//! are host parameters only: the start scene is the scene open for editing,
//! the camera renders to an offscreen target the shell shows in the scene
//! view, and input comes from the scene view instead of a window.
//!
//! Pause = don't tick the world. Stop = drop it. The editing world is never
//! touched by any of this — there is nothing to snapshot, restore, or hide.

use std::sync::Arc;

use redlilium_ecs::sync::RwLock;
use redlilium_ecs::{
    Camera, CameraOutput, CameraTarget, EcsRunner, Entity, MainViewport, OutputFormat, Render,
    RenderSchedule, Schedules, SizePolicy, WindowInput, World,
};
use redlilium_graphics::{RenderGraph, TextureFormat};
use redlilium_runtime::{AppControl, EngineContext};

use crate::game_host::GameHost;

/// Clear color of the play camera's target — the standalone build's default
/// (`GameConfig::default().clear_color`), so Play looks like the game.
const CLEAR_COLOR: [f32; 4] = [0.02, 0.02, 0.03, 1.0];

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
    /// While `true` the world is not ticked at all (game time freezes).
    paused: bool,
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
        }
    }

    /// Advance the game by one frame (no-op while paused). `view_size` is the
    /// scene view's physical size — the play world's "window".
    pub fn tick(&mut self, runner: &EcsRunner, dt: f64, view_size: (f32, f32)) {
        if self.paused {
            return;
        }
        {
            let mut input = self.window_input.write();
            input.window_width = view_size.0;
            input.window_height = view_size.1;
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
        if let Some(camera) = self.first_camera() {
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
        let camera = self.first_camera()?;
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

    /// Read access to the play world (remote screenshots point here while
    /// playing; play-world *inspection* is deliberately not exposed yet).
    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// The first (and in practice only) camera of the game world — the same
    /// resolution rule the standalone host uses for its primary camera.
    fn first_camera(&self) -> Option<Entity> {
        let idx = {
            let cameras = self.world.read_all::<Camera>().ok()?;
            cameras.iter().next().map(|(idx, _)| idx)?
        };
        self.world.entity_at_index(idx)
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

        // Pause: the world is not ticked at all.
        play.pause();
        play.tick(&runner, dt, (640.0, 480.0));
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            2,
            "paused world does not tick"
        );
        play.resume();
        play.tick(&runner, dt, (640.0, 480.0));
        assert_eq!(ticks.load(Ordering::SeqCst), 3, "resume ticks again");

        // Stop: drop the play world; the editing world still works.
        drop(play);
        ew.schedules.run_frame(&mut ew.world, &runner, dt);
        assert_eq!(
            ew.world.iter_entities().count(),
            editor_entities,
            "editing world identical after Stop"
        );
    }
}
