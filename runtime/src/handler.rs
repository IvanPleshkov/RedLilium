//! The [`AppHandler`] gluing the window/GPU loop to the game world.

use std::sync::Arc;

use redlilium_ecs::sync::RwLock;

use redlilium_app::{AppContext, AppHandler, DrawContext, input::map_winit_key};
use redlilium_ecs::{
    Camera, CameraOutput, CameraTarget, EcsRunner, MainViewport, OutputFormat, Render,
    RenderSchedule, ScenePass, WindowInput, World,
};
use redlilium_graphics::FrameSchedule;
use winit::event::{KeyEvent, MouseButton};

use crate::blit::PresentBlit;
use crate::{App, EngineContext, GameConfig, Plugin};

/// Everything created once the graphics device exists.
struct GameState {
    /// Persistent engine state; outlives the world by design (ADR-020).
    _engine: EngineContext,
    app: App,
    runner: EcsRunner,
    window_input: Arc<RwLock<WindowInput>>,
    blit: PresentBlit,
    /// Primary-camera output format, derived from the surface's color space.
    output_format: OutputFormat,
    /// Plugin module: owns plugin(s) + dylib library handle.
    /// Field order is load-bearing: app drops first, module drops last.
    /// This ensures plugin drop glue runs while dylib is still mapped (ADR-020, #45).
    module: crate::GameModule,
}

pub(crate) struct RuntimeHandler<P: Plugin + 'static> {
    config: GameConfig,
    plugin: Option<P>,
    state: Option<GameState>,
}

impl<P: Plugin + 'static> RuntimeHandler<P> {
    pub(crate) fn new(config: GameConfig, plugin: P) -> Self {
        Self {
            config,
            plugin: Some(plugin),
            state: None,
        }
    }
}

impl<P: Plugin + 'static> AppHandler for RuntimeHandler<P> {
    fn on_init(&mut self, ctx: &mut AppContext) {
        let engine = EngineContext::new(ctx.device().clone(), &self.config.mounts);
        // First boot: register the game's types, then populate the initial scene.
        // A reload (ADR-020, #45) re-runs `build` but restores the scene from a
        // snapshot instead of calling `spawn_scene` (see `App::reload`).
        let plugin = self.plugin.take().expect("plugin taken once");
        let mut app = App::boot(&engine, &plugin, ctx.aspect_ratio());

        let runner = EcsRunner::single_thread();
        {
            let (world, schedules) = app.parts_mut();
            schedules.run_startup(world, &runner);
        }

        let blit = PresentBlit::new(ctx.device(), ctx.surface_format());
        let window_input = app.window_input();

        // Create GameModule: owns plugin(s) and (eventually) dylib library handle.
        // For static linking: _library is None; for dynamic loading it holds libloading::Library.
        // Field order in GameState ensures: app drops first, module drops last (drop order enforced).
        let module = crate::GameModule::from_static(Box::new(plugin) as Box<dyn crate::Plugin>);

        self.state = Some(GameState {
            _engine: engine,
            app,
            runner,
            window_input,
            blit,
            output_format: OutputFormat::matching_surface(ctx.surface_format()),
            module,
        });
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        let Some(state) = self.state.as_mut() else {
            return true;
        };
        {
            let mut input = state.window_input.write();
            input.window_width = ctx.width() as f32;
            input.window_height = ctx.height() as f32;
        }

        // Phase 2: Plugin Stop cleanup (before transition applies).
        // Check if a Stop transition is pending; if so, run all plugins' on_stop
        // callbacks while the game world is still live.
        let should_stop = {
            let world = state.app.world();
            if world.has_resource::<redlilium_ecs::PlayControl>() {
                world.resource::<redlilium_ecs::PlayControl>().pending()
                    == Some(redlilium_ecs::PlayState::Stopped)
            } else {
                false
            }
        };

        if should_stop {
            let plugins = state.module.plugins();
            for plugin in plugins {
                // Catch panics in plugin cleanup; one plugin's error shouldn't
                // block others from running their cleanup.
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    plugin.on_stop(&mut state.app)
                })) {
                    Ok(()) => {}
                    Err(_payload) => {
                        log::error!("plugin.on_stop panicked; continuing to next plugin");
                        // CRITICAL: Payload's drop glue may live in dylib.
                        // It drops here at end of match arm, not inside closure.
                        // Must drop BEFORE dylib unload (guaranteed by prepare_for_reload sequencing).
                    }
                }
            }
        }

        let (world, schedules) = state.app.parts_mut();
        schedules.run_frame(world, &state.runner, ctx.delta_time() as f64);
        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        let Some(state) = self.state.as_mut() else {
            let graph = ctx.acquire_graph();
            return ctx.render(graph);
        };

        let (width, height) = (ctx.width(), ctx.height());
        let clear = self.config.clear_color;
        let (world, schedules) = state.app.parts_mut();

        ensure_camera_output(world, width, height, clear, state.output_format);

        // Render bracket: hand the frame graph to the ECS `Render` schedule,
        // take it back, then composite the scene onto the swapchain.
        let graph = ctx.acquire_graph();
        world.resource_mut::<RenderSchedule>().set(graph);
        schedules.run_schedule::<Render>(world, &state.runner);
        let mut graph = world
            .resource_mut::<RenderSchedule>()
            .take()
            .expect("RenderSchedule must hold the graph after the Render schedule");

        state.blit.flush_uploads(&mut graph);
        let scene_pass = world.resource::<ScenePass>().0;
        let source = world
            .read_all::<CameraTarget>()
            .ok()
            .and_then(|targets| targets.iter().next().map(|(_, t)| t.color.clone()));
        state.blit.encode(
            &mut graph,
            ctx.swapchain_texture(),
            source,
            scene_pass,
            clear,
        );

        state.window_input.write().begin_frame();
        ctx.render(graph)
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        let Some(state) = self.state.as_ref() else {
            return;
        };
        if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key
            && let Some(key) = map_winit_key(code)
        {
            let mut input = state.window_input.write();
            if event.state.is_pressed() {
                input.on_key_pressed(key);
            } else {
                input.on_key_released(key);
            }
        }
    }

    fn on_mouse_move(&mut self, _ctx: &mut AppContext, x: f64, y: f64) {
        if let Some(state) = self.state.as_ref() {
            state.window_input.write().on_mouse_move(x, y);
        }
    }

    fn on_mouse_button(&mut self, _ctx: &mut AppContext, button: MouseButton, pressed: bool) {
        let Some(state) = self.state.as_ref() else {
            return;
        };
        let index = match button {
            MouseButton::Left => 0,
            MouseButton::Right => 1,
            MouseButton::Middle => 2,
            _ => return,
        };
        state.window_input.write().on_mouse_button(index, pressed);
    }

    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, delta_x: f32, delta_y: f32) {
        if let Some(state) = self.state.as_ref() {
            state.window_input.write().on_scroll(delta_x, delta_y);
        }
    }
}

impl<P: Plugin + 'static> RuntimeHandler<P> {
    /// Phase 3: Prepare for warm-restart reload. Sequence (Fable-corrected):
    /// 1. Capture scene snapshot (FIRST, captures pre-cleanup running state)
    /// 2. Loop plugins: call on_unload(&mut app) with panic catching (after snapshot)
    /// 3. Drop app (triggers World drop, clears PlayTasks, cancels task tokens)
    /// 4. Quiesce ComputePool until all tasks complete (timeout: 5s, AFTER app drop)
    ///
    /// CRITICAL: If quiesce times out, ABORT RELOAD (fail closed). Leaking the dylib
    /// is safer than unmapping it with tasks still running.
    ///
    /// Returns the captured snapshot. Caller then unloads dylib and reloads.
    #[allow(dead_code)] // Will be called when reload is requested via remote
    fn prepare_for_reload(
        &mut self,
    ) -> Result<redlilium_ecs::serialize::SerializedWorld, redlilium_ecs::serialize::SerializeError>
    {
        self.prepare_for_reload_with_timeout(std::time::Duration::from_secs(5))
    }

    /// [`prepare_for_reload`](Self::prepare_for_reload) with an explicit
    /// quiescence timeout — the production path uses 5s; tests use a short
    /// timeout to exercise the abort branch without a 5-second stall.
    fn prepare_for_reload_with_timeout(
        &mut self,
        quiesce_timeout: std::time::Duration,
    ) -> Result<redlilium_ecs::serialize::SerializedWorld, redlilium_ecs::serialize::SerializeError>
    {
        let Some(state) = self.state.as_mut() else {
            return Err(redlilium_ecs::serialize::SerializeError::FormatError(
                "reload called without active game state".to_string(),
            ));
        };

        // Phase 3a: Capture scene snapshot FIRST (captures pre-cleanup running state)
        let snapshot = state.app.capture()?;

        // Phase 3b: Plugin unload cleanup (after snapshot, before World drop)
        let plugins = state.module.plugins();
        for plugin in plugins {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                plugin.on_unload(&mut state.app)
            })) {
                Ok(()) => {}
                Err(_payload) => {
                    log::error!("plugin.on_unload panicked; continuing to next plugin");
                    // CRITICAL: Payload's drop glue may live in dylib.
                    // It drops here at end of match arm. Must drop BEFORE dylib unload.
                }
            }
        }

        // Phase 3c: Drop app (World drop triggers PlayTasks cleanup + token cancellation)
        // When state goes out of scope at the end of this function, App drops, which
        // triggers World drop and PlayTasks token cancellation.

        // Phase 3d: Task quiescence (block until all tasks complete or timeout)
        // CRITICAL: Only NOW, after World drop cancels tokens, do tasks finish.
        // Before World drop, long-lived tasks may legitimately still be running.
        let quiesce_result = match &state.runner {
            redlilium_ecs::EcsRunner::SingleThread(r) => r.compute().quiesce(quiesce_timeout),
            #[cfg(not(target_arch = "wasm32"))]
            redlilium_ecs::EcsRunner::MultiThread(r) => r.compute().quiesce(quiesce_timeout),
        };

        if quiesce_result >= quiesce_timeout {
            log::error!(
                "Task quiescence timeout: some plugin tasks still executing after {:?}. \
                 Unloading dylib now would cause UB (tasks jump into unmapped memory). \
                 ABORTING RELOAD and leaking dylib (bounded memory leak vs process UB).",
                quiesce_timeout
            );
            return Err(redlilium_ecs::serialize::SerializeError::FormatError(
                "Task quiescence timeout — reload aborted to prevent UB".to_string(),
            ));
        }

        Ok(snapshot)
    }
}

/// Feed the ECS side what it needs to derive the primary camera's target
/// (ADR-029): publish the window size as [`MainViewport`] and give the first
/// camera a `CameraOutput::screen()` spec when the game didn't author one.
/// The GPU textures themselves are created by the `EnsureCameraTargets`
/// system inside the Render schedule — no host-side texture management.
fn ensure_camera_output(
    world: &mut World,
    width: u32,
    height: u32,
    clear_color: [f32; 4],
    format: OutputFormat,
) {
    world.insert_resource(MainViewport::new(width.max(1), height.max(1)));

    let camera = {
        let Ok(cameras) = world.read_all::<Camera>() else {
            return;
        };
        cameras
            .iter()
            .next()
            .and_then(|(idx, _)| world.entity_at_index(idx))
    };
    let Some(camera) = camera else {
        return;
    };
    if world.get::<CameraOutput>(camera).is_none() {
        let _ = world.insert(
            camera,
            CameraOutput::screen()
                .with_clear_color(clear_color)
                .with_format(format),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{App, EngineContext, GameConfig, Plugin};
    use redlilium_graphics::GraphicsInstance;
    use redlilium_vfs::Vfs;
    use std::time::Duration;

    /// Minimal no-op plugin — the test exercises the reload sequencing, not
    /// game logic.
    struct NoopPlugin;
    impl Plugin for NoopPlugin {
        fn build(&self, _app: &mut App) {}
    }

    /// Builds a `RuntimeHandler` with a live `GameState`, bypassing `on_init`
    /// (which needs a real window loop). Mirrors `on_init`'s wiring:
    /// engine → boot → runner/module, headless device, no swapchain.
    fn test_handler() -> RuntimeHandler<NoopPlugin> {
        let instance = GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("graphics device");
        let engine = EngineContext::with_vfs(device.clone(), Vfs::new());

        let plugin = NoopPlugin;
        let app = App::boot(&engine, &plugin, 1.0);
        let runner = EcsRunner::single_thread();
        let window_input = app.window_input();
        let blit = PresentBlit::new(&device, redlilium_graphics::TextureFormat::Rgba8Unorm);
        let module = crate::GameModule::from_static(Box::new(plugin) as Box<dyn crate::Plugin>);

        RuntimeHandler {
            config: GameConfig::default(),
            plugin: None,
            state: Some(GameState {
                _engine: engine,
                app,
                runner,
                window_input,
                blit,
                output_format: OutputFormat::default(),
                module,
            }),
        }
    }

    /// #81 (deferred #45 Phase 6, Step 5): a task that never finishes must
    /// make `prepare_for_reload` abort with a timeout error instead of
    /// letting the caller proceed to unmap the dylib under a live task.
    #[test]
    fn reload_aborts_on_task_quiescence_timeout() {
        let mut handler = test_handler();

        // A task that ignores cancellation and never completes — the worst
        // case for unload safety. Keep the handle alive so nothing reaps it.
        let _stuck = {
            let state = handler.state.as_ref().unwrap();
            state
                .runner
                .compute()
                .spawn(redlilium_ecs::Priority::Low, |_ctx| {
                    std::future::pending::<()>()
                })
        };

        let result = handler.prepare_for_reload_with_timeout(Duration::from_millis(100));

        // Abort, with the timeout named in the error.
        match result {
            Err(redlilium_ecs::serialize::SerializeError::FormatError(msg)) => {
                assert!(
                    msg.contains("quiescence timeout"),
                    "error must name the quiescence timeout, got: {msg}"
                );
            }
            Ok(_) => panic!("reload must abort when a task is still running"),
            Err(e) => panic!("expected FormatError with timeout message, got: {e:?}"),
        }

        // Fail closed: the module (and with it the dylib handle) must remain
        // loaded — the caller never gets a snapshot to proceed with.
        let state = handler.state.as_ref().unwrap();
        assert_eq!(
            state.module.plugins().len(),
            1,
            "module must stay loaded after an aborted reload"
        );
    }

    /// The happy path with the same short timeout: no live tasks → quiesce
    /// returns immediately and the snapshot comes back.
    #[test]
    fn reload_succeeds_when_tasks_are_quiet() {
        let mut handler = test_handler();

        let result = handler.prepare_for_reload_with_timeout(Duration::from_millis(100));
        assert!(
            result.is_ok(),
            "reload must proceed when no tasks are pending: {:?}",
            result.err()
        );
    }
}
