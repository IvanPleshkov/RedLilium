//! The [`AppHandler`] gluing the window/GPU loop to the game world.

use std::sync::Arc;

use redlilium_ecs::sync::RwLock;

use redlilium_app::{AppContext, AppHandler, DrawContext, input::map_winit_key};
use redlilium_ecs::{
    Camera, CameraOutput, CameraTarget, EcsRunner, MainViewport, OutputFormat, Render,
    RenderSchedule, ScenePass, WindowInput, World,
};
use redlilium_graphics::egui::{EguiApp, EguiController};
use redlilium_graphics::{FrameSchedule, RenderTarget};
use winit::event::{KeyEvent, MouseButton};

use crate::blit::PresentBlit;
use crate::{App, EngineContext, GameConfig, GameUi, Plugin};

/// The runtime builds its UI from game systems (via the [`GameUi`] resource),
/// not through the controller's [`EguiApp`] callback — this stub satisfies the
/// constructor.
struct NoopUi;
impl EguiApp for NoopUi {
    fn update(&mut self, _ctx: &redlilium_graphics::egui::egui::Context) {}
}

/// Everything created once the graphics device exists.
struct GameState {
    /// Persistent engine state; outlives the world by design (ADR-020).
    _engine: EngineContext,
    app: App,
    runner: EcsRunner,
    window_input: Arc<RwLock<WindowInput>>,
    blit: PresentBlit,
    /// In-game UI layer (#100): the host brackets each frame around the game
    /// schedules so systems can draw through the [`GameUi`] resource.
    egui: EguiController,
    /// Whether an egui pass is open (`begin_frame` ran without a matching
    /// `end_frame`) — the draw phase can be skipped (outdated surface, busy
    /// frame slot), so the bracket must tolerate an unpaired update.
    ui_frame_open: bool,
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
    /// Smoothed display headroom H (#154), polled from the window screen each
    /// frame and published to the game world's [`DisplayHeadroom`] so the
    /// deferred display pass rolls highlights off to the real panel range.
    /// `1.0` (SDR) until the first poll resolves an HDR screen.
    display_headroom: f32,
}

impl<P: Plugin + 'static> RuntimeHandler<P> {
    pub(crate) fn new(config: GameConfig, plugin: P) -> Self {
        Self {
            config,
            plugin: Some(plugin),
            state: None,
            display_headroom: redlilium_graphics::display::SDR_HEADROOM,
        }
    }
}

impl<P: Plugin + 'static> AppHandler for RuntimeHandler<P> {
    fn on_init(&mut self, ctx: &mut AppContext) {
        let engine = EngineContext::new(
            ctx.device().clone(),
            &self.config.mounts,
            &self.config.embedded_packs,
        );
        // First boot: register the game's types, then populate the initial scene.
        // A reload (ADR-020, #45) re-runs `build` but restores the scene from a
        // snapshot instead of calling `spawn_scene` (see `App::reload`).
        let plugin = self.plugin.take().expect("plugin taken once");
        let mut app = App::boot(
            &engine,
            &plugin,
            ctx.aspect_ratio(),
            self.config.start_scene.as_deref(),
        );

        let runner = EcsRunner::single_thread();
        {
            let (world, schedules) = app.parts_mut();
            schedules.run_startup(world, &runner);
        }

        let blit = PresentBlit::new(ctx.device(), ctx.surface_format());
        let window_input = app.window_input();

        // In-game UI layer (#100): the controller renders straight to the
        // swapchain on top of the present blit; game systems build the UI
        // through the world-resident GameUi handle.
        let egui = EguiController::new(
            ctx.device().clone(),
            Arc::new(parking_lot::RwLock::new(NoopUi)),
            ctx.width(),
            ctx.height(),
            ctx.scale_factor(),
            ctx.surface_format(),
        );
        app.world_mut()
            .insert_resource(GameUi::new(egui.context().clone()));

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
            egui,
            ui_frame_open: false,
            output_format: OutputFormat::matching_surface(ctx.surface_format()),
            module,
        });
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        if let Some(state) = self.state.as_mut() {
            state.egui.on_resize(ctx.width(), ctx.height());
        }
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

        // Open the UI frame before the game schedules so systems can draw
        // through GameUi. If the previous frame's draw was skipped (outdated
        // surface, busy frame slot), the stale pass is still open — discard it
        // (atlas deltas survive, shapes are dropped) before opening a new one.
        if state.ui_frame_open {
            state.egui.discard_frame();
        }
        state.egui.begin_frame(ctx.elapsed_time() as f64);
        state.ui_frame_open = true;

        // Poll the window screen's EDR headroom (#154), ease it (maxEDR tracks
        // the brightness slider live, so smoothing keeps highlights from
        // pumping), and publish it so the deferred display pass rolls
        // highlights off to the real panel range. 1.0 (no-op) off macOS / SDR.
        let raw = redlilium_graphics::display::window_display_headroom(ctx.window().as_ref());
        self.display_headroom += (raw - self.display_headroom) * 0.1;

        let (world, schedules) = state.app.parts_mut();
        world.insert_resource(redlilium_ecs::DisplayHeadroom(self.display_headroom));
        // The unpaced CPU delta, for frame-timing HUDs — `run_frame` gets the
        // paced one that motion must use (see redlilium-app's `pacing`).
        world.insert_resource(redlilium_ecs::RawFrameDelta(ctx.raw_delta_time() as f64));
        schedules.run_frame(world, &state.runner, ctx.delta_time() as f64);

        // Game-requested exit (#100). On wasm a browser tab has no process to
        // exit, so the request is ignored there.
        #[cfg(not(target_arch = "wasm32"))]
        if state
            .app
            .world()
            .resource::<crate::AppControl>()
            .exit_requested()
        {
            log::info!("game requested exit (AppControl)");
            return false;
        }
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
        let mut schedule_res = world.resource_mut::<RenderSchedule>();
        // Asset-upload transfer graphs (#89) go first, each as its own
        // submit — on hardware with a transfer queue they overlap the frame.
        let transfer_graphs = schedule_res.take_transfer_graphs();
        let mut graph = schedule_res
            .take()
            .expect("RenderSchedule must hold the graph after the Render schedule");
        drop(schedule_res);
        for transfer in transfer_graphs {
            ctx.submit(transfer);
        }

        state.blit.flush_uploads(&mut graph);
        let scene_pass = world.resource::<ScenePass>().0;
        let source = world
            .read_all::<CameraTarget>()
            .ok()
            .and_then(|targets| targets.iter().next().map(|(_, t)| t.color.clone()));
        let blit_handle = state.blit.encode(
            &mut graph,
            ctx.swapchain_texture(),
            source,
            scene_pass,
            clear,
        );

        // Close the UI frame opened in on_update and composite it over the
        // blit (#100). The egui pass loads (not clears) the swapchain, so the
        // dependency edge is what keeps it on top.
        if state.ui_frame_open {
            state.ui_frame_open = false;
            let target = RenderTarget::from_surface(ctx.swapchain_texture());
            state.egui.flush_uploads(&mut graph);
            if let Some(pass) = state.egui.end_frame(&target, width, height) {
                let handle = graph.add_graphics_pass(pass);
                graph.add_dependency(handle, blit_handle);
            }
        }

        {
            let mut input = state.window_input.write();
            input.ui_wants_input =
                state.egui.wants_pointer_input || state.egui.wants_keyboard_input;
            input.begin_frame();
        }
        ctx.render(graph)
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let ui_wants = state.egui.on_key(event);
        if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key
            && let Some(key) = map_winit_key(code)
        {
            let mut input = state.window_input.write();
            input.ui_wants_input = ui_wants;
            if event.state.is_pressed() {
                input.on_key_pressed(key);
            } else {
                input.on_key_released(key);
            }
        }
    }

    fn on_mouse_move(&mut self, _ctx: &mut AppContext, x: f64, y: f64) {
        if let Some(state) = self.state.as_mut() {
            let ui_wants = state.egui.on_mouse_move(x, y);
            let mut input = state.window_input.write();
            input.on_mouse_move(x, y);
            input.ui_wants_input = ui_wants;
        }
    }

    fn on_mouse_button(&mut self, _ctx: &mut AppContext, button: MouseButton, pressed: bool) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let ui_wants = state.egui.on_mouse_button(button, pressed);
        let index = match button {
            MouseButton::Left => 0,
            MouseButton::Right => 1,
            MouseButton::Middle => 2,
            _ => return,
        };
        let mut input = state.window_input.write();
        input.on_mouse_button(index, pressed);
        input.ui_wants_input = ui_wants;
    }

    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, delta_x: f32, delta_y: f32) {
        if let Some(state) = self.state.as_mut() {
            let ui_wants = state
                .egui
                .on_mouse_scroll(winit::event::MouseScrollDelta::LineDelta(delta_x, delta_y));
            let mut input = state.window_input.write();
            input.on_scroll(delta_x, delta_y);
            input.ui_wants_input = ui_wants;
        }
    }

    fn on_modifiers_changed(
        &mut self,
        _ctx: &mut AppContext,
        modifiers: winit::keyboard::ModifiersState,
    ) {
        if let Some(state) = self.state.as_mut() {
            state.egui.on_modifiers_changed(modifiers);
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
    // Standalone-runtime reload entry, currently unwired: the editor hosts
    // games through its own GameHost world-rebuild path (#58) instead. This
    // stays as the runtime-side counterpart (plus its timeout tests below)
    // until the standalone runtime grows a reload trigger.
    #[allow(dead_code)]
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

    // Keep the projection's aspect in step with the viewport. Without this the
    // standalone game renders forever with the aspect captured at boot
    // (`App::initial_aspect`), so any resize that changes the window's
    // proportions leaves a stretched image and a frustum that no longer matches
    // the target. The editor patched its own camera; nothing did it here.
    let aspect = width.max(1) as f32 / height.max(1) as f32;
    if let Ok(mut cameras) = world.write_all::<Camera>()
        && let Some(mut cam) = cameras.get_mut(camera.index())
    {
        sync_projection_aspect(&mut cam.projection_matrix, aspect);
    }
}

/// Rewrite a perspective projection's aspect in place, preserving its field of
/// view, near/far planes and depth convention.
///
/// [`Camera`] stores only matrices — there is no fov/near/far to rebuild from —
/// but the aspect enters a perspective matrix solely as `m00 = m11 / aspect`, so
/// rewriting that single term is exactly equivalent to rebuilding the matrix
/// with the new aspect (the test pins that equivalence). Orthographic
/// projections encode extent rather than aspect and are left untouched.
fn sync_projection_aspect(projection: &mut redlilium_core::math::Mat4, aspect: f32) {
    // `m33 == 0` marks a perspective projection.
    if !aspect.is_finite() || aspect <= 0.0 || projection[(3, 3)].abs() >= 1e-6 {
        return;
    }
    // `copysign` so a future Y-flip convention (negative `m11`) could not
    // silently mirror X.
    let desired = (projection[(1, 1)].abs() / aspect).copysign(projection[(0, 0)]);
    if (projection[(0, 0)] - desired).abs() > 1e-6 {
        projection[(0, 0)] = desired;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{App, EngineContext, GameConfig, Plugin};
    use redlilium_graphics::GraphicsInstance;
    use redlilium_vfs::Vfs;
    use std::time::Duration;

    /// Rewriting the single aspect term must be indistinguishable from
    /// rebuilding the projection outright — that equivalence is what lets the
    /// camera keep matrices only, with no fov/near/far to rebuild from.
    #[test]
    fn syncing_aspect_matches_a_rebuilt_projection() {
        use redlilium_core::math::DepthConvention;
        use std::f32::consts::FRAC_PI_4;

        for convention in [DepthConvention::ReversedZ, DepthConvention::Classic] {
            let mut projection = convention.perspective(FRAC_PI_4, 16.0 / 9.0, 0.1, 500.0);
            sync_projection_aspect(&mut projection, 4.0 / 3.0);
            let rebuilt = convention.perspective(FRAC_PI_4, 4.0 / 3.0, 0.1, 500.0);
            for row in 0..4 {
                for col in 0..4 {
                    assert!(
                        (projection[(row, col)] - rebuilt[(row, col)]).abs() < 1e-5,
                        "{convention:?} m[{row}][{col}]: {} != {}",
                        projection[(row, col)],
                        rebuilt[(row, col)]
                    );
                }
            }
        }
    }

    /// An orthographic projection encodes extent, not aspect — rewriting `m00`
    /// there would silently rescale the view instead of retargeting it.
    #[test]
    fn syncing_aspect_leaves_orthographic_alone() {
        use redlilium_core::math::DepthConvention;

        let before = DepthConvention::ReversedZ.orthographic(-10.0, 10.0, -10.0, 10.0, 0.1, 500.0);
        let mut projection = before;
        sync_projection_aspect(&mut projection, 4.0 / 3.0);
        assert_eq!(projection, before);
    }

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
        let app = App::boot(&engine, &plugin, 1.0, None);
        let runner = EcsRunner::single_thread();
        let window_input = app.window_input();
        let blit = PresentBlit::new(&device, redlilium_graphics::TextureFormat::Rgba8Unorm);
        let egui = EguiController::new(
            device.clone(),
            Arc::new(parking_lot::RwLock::new(NoopUi)),
            1,
            1,
            1.0,
            redlilium_graphics::TextureFormat::Rgba8Unorm,
        );
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
                egui,
                ui_frame_open: false,
                output_format: OutputFormat::default(),
                module,
            }),
            display_headroom: redlilium_graphics::display::SDR_HEADROOM,
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
