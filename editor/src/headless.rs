//! Headless editor shell: the same editor core (world, assets, undo history,
//! remote protocol) without a window, swapchain, or egui.
//!
//! Enabled by `REDLILIUM_HEADLESS=1` (implies the remote channel — a headless
//! editor is unreachable without it). The scene renders into the editor
//! camera's off-screen `CameraTarget`, so `screenshot` works exactly as in
//! the windowed shell.
//!
//! Frames are **ticked on demand**: while remote work is parked (`step N`,
//! queued writes, `wait`, a screenshot) or the asset pipeline is busy, the
//! loop ticks with a fixed dt; once everything has been calm for a few ticks
//! it blocks on the transport until the next command arrives. An idle
//! headless editor costs nothing.

use std::time::Duration;

use redlilium_debug_drawer::DebugDrawerRenderer;
use redlilium_ecs::{EcsRunner, RemoteTransport, Render, RenderSchedule};
use redlilium_graphics::{GraphicsInstance, InstanceParameters, TextureFormat};
use redlilium_vfs::{FileSystemProvider, Vfs};

use crate::core::{EditorWorld, EditorWorldParams, create_editor_world};
use crate::remote_commands::{self, RemoteCommands};
use crate::scene_view::SceneViewState;

/// Fixed timestep per on-demand tick — headless frames are not wall-clock
/// paced, so a deterministic dt keeps stepped simulations reproducible.
const FIXED_DT: f64 = 1.0 / 60.0;

/// Consecutive quiet ticks (no parked remote work, assets idle) before the
/// loop stops free-running and blocks on the transport. Mirrors the
/// `assets_idle` calm-frames logic: one quiet tick can be a gap between an
/// asset stage draining and the next manager registering demand.
const CALM_TICKS: u32 = 3;

/// Scene color format (no surface to negotiate with); sRGB so screenshot
/// PNGs come out gamma-correct. The play world's camera target uses the same
/// format.
const COLOR_FORMAT: TextureFormat = TextureFormat::Bgra8UnormSrgb;

pub fn run() {
    redlilium_core::init();
    redlilium_graphics::init();

    let (width, height) = target_size();
    log::info!("headless editor: {width}x{height}, tick-on-demand");

    let instance =
        GraphicsInstance::with_parameters(InstanceParameters::new()).expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    log::info!("headless device: {}", device.name());

    // The same local mounts as the windowed shell (see `Editor::new`),
    // project.toml `scan = true` mounts included (#105), minus the browser's
    // file watcher — external file edits are picked up by the startup scan,
    // in-editor edits by the ChangedAssets flow. Only the config is read
    // here (no `build_vfs`): plain/sftp mounts are a windowed concern.
    let config = crate::project::load_project(std::path::Path::new("project.toml")).ok();
    let local_mounts = crate::project::local_mounts(config.as_ref());
    let mut vfs = Vfs::new();
    for &(name, dir) in &local_mounts {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::error!("failed to create mount dir '{dir}': {e}");
        }
        vfs.mount(name, FileSystemProvider::new(dir));
    }

    // The scene color format is a constant here (no surface to negotiate
    // with); sRGB so screenshot PNGs come out gamma-correct.
    let color_format = COLOR_FORMAT;
    let mut scene_view = SceneViewState::new(device.clone(), color_format);
    // Picking works in scene-image space: the entity-index target matches the
    // scene size and the viewport covers it fully, so remote pick coordinates
    // equal screenshot pixels with no offset.
    scene_view.resize_if_needed(width, height);
    scene_view.set_viewport(
        egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(width as f32, height as f32),
        ),
        1.0,
    );

    // Multi-threaded runner enabled after Track 2 MT-hardening (#45).
    let runner = EcsRunner::multi_thread(
        std::thread::available_parallelism()
            .map(|p| p.get().saturating_sub(2).max(1))
            .unwrap_or(1),
    );
    // Persistent engine state + the startup mount scan (ADR-020).
    let engine = redlilium_runtime::EngineContext::with_vfs(device.clone(), vfs.clone());
    crate::core::scan_local_mounts(&engine, &local_mounts);
    // Declared before `ew` and `play` so it drops after them — the mapped game
    // image must outlive every world its plugin touched (ADR-020, #58).
    let mut game_host: Option<crate::game_host::GameHost> = None;
    // The play session (EDITOR_REBUILD.md §4.3): a full game world booted from
    // the hosted module on `play`, dropped on `stop`.
    let mut play: Option<crate::play::PlaySession> = None;
    let mut ew = create_editor_world(
        &EditorWorldParams {
            remote: true,
            egui: false,
        },
        &engine,
        &mut scene_view,
        width as f32 / height as f32,
    );
    ew.world.insert_resource(DebugDrawerRenderer::new(
        device.clone(),
        color_format,
        Some(TextureFormat::Depth32Float),
    ));
    ew.world
        .insert_resource(redlilium_gizmo::GizmoRenderer::new(
            device.clone(),
            color_format,
        ));
    // Headless: the scene image is the whole frame.
    ew.world
        .insert_resource(crate::gizmo_system::SceneViewRect {
            offset: [0.0, 0.0],
            size: [width as f32, height as f32],
        });
    ew.schedules.run_startup(&mut ew.world, &runner);

    let aspect = width as f32 / height as f32;
    if let Ok(path) = std::env::var("REDLILIUM_GAME") {
        // SAFETY: the operator points this at a game cdylib built in the same
        // `cargo build` as this editor (the fingerprint gate enforces engine
        // parity; the same-build contract is documented on GameModule::load).
        match unsafe { crate::game_host::GameHost::load(&path, &engine, &mut ew) } {
            Ok(host) => game_host = Some(host),
            Err(e) => log::error!("failed to load game module '{path}': {e}"),
        }
    }

    let mut pipeline = device.create_pipeline(2);
    let mut rc = RemoteCommands::default();

    let mut calm = 0u32;
    let mut render_failures = 0u32;
    let mut shutdown = false;
    while !shutdown {
        // A live, unpaused play session keeps the loop free-running: the game
        // must tick whether or not remote work is parked.
        let playing = play.as_ref().is_some_and(|p| !p.is_paused());
        let busy = playing || rc.has_pending() || !remote_commands::assets_idle(&ew.world);
        calm = if busy { 0 } else { calm + 1 };
        if calm >= CALM_TICKS {
            // Quiescent: sleep on the transport until the next command. The
            // timeout only bounds the poll for ctrl-c responsiveness.
            let arrived = ew
                .world
                .resource::<RemoteTransport>()
                .wait_incoming(Duration::from_millis(250));
            if !arrived {
                continue;
            }
        }

        let outcome = tick(
            &mut ew,
            &mut rc,
            &mut scene_view,
            &mut pipeline,
            &runner,
            &local_mounts,
            width,
            height,
            play.as_mut(),
        );
        shutdown = outcome.shutdown;

        // Play-session lifecycle (EDITOR_REBUILD.md §4.3), applied between
        // frames like the reload below.
        if let Some(request) = rc.take_play_request() {
            use remote_commands::PlayRequest;
            match request {
                PlayRequest::Play => match (&play, game_host.as_ref()) {
                    (Some(_), _) => log::warn!("play: session already running"),
                    (None, None) => log::error!(
                        "play: no game module hosted (launch with REDLILIUM_GAME=<cdylib>)"
                    ),
                    (None, Some(host)) => {
                        // SceneManager addresses scenes mount-relative; the
                        // editing session's CurrentScene is a VFS path
                        // ("mount/rel") — strip the mount.
                        let start_scene = ew
                            .world
                            .resource::<crate::core::CurrentScene>()
                            .0
                            .as_deref()
                            .and_then(|p| p.split_once('/'))
                            .map(|(_, rel)| rel.to_string());
                        play = Some(crate::play::PlaySession::start(
                            host,
                            &engine,
                            &runner,
                            width as f32 / height as f32,
                            start_scene.as_deref(),
                        ));
                    }
                },
                PlayRequest::Pause => match play.as_mut() {
                    Some(p) => p.pause(&ew.world),
                    None => log::warn!("pause: no play session"),
                },
                PlayRequest::Resume => match play.as_mut() {
                    Some(p) => p.resume(),
                    None => log::warn!("resume: no play session"),
                },
                PlayRequest::Stop => {
                    if play.take().is_none() {
                        log::warn!("stop: no play session");
                    }
                }
            }
        }
        // The game requested an exit (AppControl) — same as `stop`.
        if play.as_ref().is_some_and(|p| p.exit_requested()) {
            log::info!("play: game requested exit — stopping session");
            play = None;
        }

        if rc.take_reload() {
            // A reload swaps the mapped image every world's game systems point
            // into — the play world must die first.
            if play.take().is_some() {
                log::info!("reload_game: play session stopped");
            }
            match game_host.as_mut() {
                None => log::error!(
                    "reload_game: no game module hosted (launch with REDLILIUM_GAME=<cdylib>)"
                ),
                Some(host) => {
                    let opts = crate::game_host::ReloadOptions {
                        params: EditorWorldParams {
                            remote: true,
                            egui: false,
                        },
                        aspect,
                    };
                    let (fresh, result) = crate::game_host::reload_game(
                        host,
                        ew,
                        &engine,
                        &mut scene_view,
                        &runner,
                        &opts,
                        crate::game_host::swap_from_disk,
                    );
                    ew = fresh;
                    match result {
                        Ok(()) => log::info!("game module reloaded (scene restored)"),
                        Err(e) => log::error!("game reload failed: {e}"),
                    }
                }
            }
        }

        if outcome.rendered {
            render_failures = 0;
            // Even a busy headless editor has no vsync to pace it — yield a
            // little so asset waits don't spin a core at full tilt. A playing
            // game is paced to roughly its fixed dt (real-time-ish).
            let pace = if play.as_ref().is_some_and(|p| !p.is_paused()) {
                16
            } else {
                1
            };
            std::thread::sleep(Duration::from_millis(pace));
        } else {
            // Rendering failed (device error, wedged fence). Without vsync or
            // an asset-idle signal this loop would spin at ~1 kHz retrying
            // (the #46 CPU burn). Back off exponentially, capped at 250 ms —
            // still responsive to commands, and recovers immediately once a
            // frame succeeds again.
            render_failures = render_failures.saturating_add(1);
            let backoff = (1u64 << render_failures.min(8)).min(250);
            std::thread::sleep(Duration::from_millis(backoff));
        }
    }

    // The shutdown ack is queued on the IO runtime's writer task; give it a
    // beat to reach the socket before tearing the process down.
    std::thread::sleep(Duration::from_millis(150));
    if let Err(e) = pipeline.wait_idle() {
        log::error!("wait_idle failed during shutdown: {e}");
    }
    let _ = std::fs::remove_file(".redlilium/editor.port");
    log::info!("headless editor: shutdown");
}

/// One editor frame: the same sequence as the windowed shell's
/// `on_update` + `on_draw`, minus input, picking, and egui. A live play
/// session ticks and renders in the same shell frame, after the editing
/// world (EDITOR_REBUILD.md §4.3).
#[allow(clippy::too_many_arguments)]
fn tick(
    ew: &mut EditorWorld,
    rc: &mut RemoteCommands,
    scene_view: &mut SceneViewState,
    pipeline: &mut redlilium_graphics::FramePipeline,
    runner: &EcsRunner,
    local_mounts: &[(&'static str, &'static str)],
    width: u32,
    height: u32,
    mut play: Option<&mut crate::play::PlaySession>,
) -> TickOutcome {
    ew.persist_dirty_mounts(local_mounts);
    ew.debug_drawer.read().advance_tick();

    // Resolve last tick's pick readback into its remote response.
    if let Some(hit) = scene_view.resolve_pick() {
        remote_commands::complete_point_pick(rc, &ew.world, hit);
    }
    if let Some(indices) = scene_view.resolve_rect_pick() {
        remote_commands::complete_rect_pick(rc, &ew.world, &indices);
    }

    // Actions queued by last tick's remote commands apply here…
    ew.drain_actions();
    // …and the pump completes their parked responses, then dispatches newly
    // arrived commands (their actions drain next tick).
    remote_commands::pump(rc, &mut ew.world, &mut ew.history);
    let shutdown = rc.take_shutdown();

    // Submit a freshly arrived pick — coordinates are already scene-image
    // space here (viewport == scene texture, no offset).
    if let Some(request) = remote_commands::take_pick_request(rc) {
        use remote_commands::PickRequest;
        match request {
            PickRequest::Point { x, y } if x < width && y < height => {
                scene_view.request_pick(x, y);
            }
            PickRequest::Rect { x, y, w, h } if x < width && y < height => {
                scene_view.request_rect_pick(x, y, w.min(width - x), h.min(height - y));
            }
            _ => remote_commands::fail_pick(
                rc,
                &ew.world,
                "pick coordinates outside the scene image",
            ),
        }
    }

    ew.schedules.run_frame(&mut ew.world, runner, FIXED_DT);

    // The play world ticks after the editing world, in the same shell frame.
    if let Some(play) = play.as_deref_mut() {
        play.tick(runner, FIXED_DT, (width as f32, height as f32));
    }

    // Render into the camera's off-screen target (created on first tick).
    // On fence-wait failure skip rendering this tick — the fence stays in its
    // slot and the next tick retries the wait.
    let mut schedule = match pipeline.begin_frame() {
        Ok(schedule) => schedule,
        Err(e) => {
            log::error!("begin_frame failed, skipping frame: {e}");
            ew.window_input.write().begin_frame();
            return TickOutcome {
                shutdown,
                rendered: false,
            };
        }
    };
    let mut graph = schedule.acquire_graph();
    scene_view.sync_camera_output(&mut ew.world, ew.editor_camera, width, height);
    ew.world.resource_mut::<RenderSchedule>().set(graph);
    ew.schedules.run_schedule::<Render>(&mut ew.world, runner);
    let mut schedule_res = ew.world.resource_mut::<RenderSchedule>();
    // Asset-upload transfer graphs (#89) go first, each as its own submit.
    let transfer_graphs = schedule_res.take_transfer_graphs();
    graph = schedule_res
        .take()
        .expect("RenderSchedule must hold the graph after the Render schedule");
    drop(schedule_res);
    for transfer in transfer_graphs {
        schedule.submit(transfer);
    }

    // The play world renders into its own graph, submitted BEFORE the main
    // graph: cross-submit synchronization is automatic (same queue), and the
    // screenshot pass below — which reads the play texture while playing —
    // lives in the main graph.
    if let Some(play) = play.as_deref_mut() {
        let play_graph = schedule.acquire_graph();
        let (play_graph, play_transfers) =
            play.render(runner, play_graph, width, height, COLOR_FORMAT);
        for transfer in play_transfers {
            schedule.submit(transfer);
        }
        schedule.submit(play_graph);
    }

    // While a session is live, `screenshot` captures its view camera — the
    // game camera during Play, the flyover camera during pause. Exactly what
    // the windowed scene view shows.
    let source = play.as_deref().and_then(|p| p.scene_color());
    remote_commands::inject_screenshot_pass(rc, &ew.world, scene_view.device(), &mut graph, source);

    // Entity-index pass + readback, only while a remote pick is in flight
    // (mirrors the windowed shell's on_draw picking block).
    let pending_pick = scene_view.take_pending_pick();
    let pending_rect = scene_view.take_pending_rect_pick();
    if pending_pick.is_some() || pending_rect.is_some() {
        scene_view.fill_picking_rings(&ew.world);
        if let Some(ei_pass) = scene_view.build_entity_index_pass(&ew.world) {
            let ei_handle = graph.add_graphics_pass(ei_pass);
            if let Some([px, py]) = pending_pick {
                let readback = scene_view.build_pick_readback(px, py);
                let handle = graph.add_transfer_pass(readback);
                graph.add_dependency(handle, ei_handle);
            }
            if let Some([rx, ry, rw, rh]) = pending_rect {
                let readback = scene_view.build_rect_readback(rx, ry, rw, rh);
                let handle = graph.add_transfer_pass(readback);
                graph.add_dependency(handle, ei_handle);
            }
        }
        if let Some([_, _, rw, rh]) = pending_rect {
            scene_view.set_rect_layout(rw, rh);
        }
    }

    schedule.render(graph);
    pipeline.end_frame(schedule);

    ew.window_input.write().begin_frame();
    TickOutcome {
        shutdown,
        rendered: true,
    }
}

/// Result of one headless [`tick`].
struct TickOutcome {
    /// A remote `shutdown` command arrived this tick.
    shutdown: bool,
    /// The frame reached the GPU (`begin_frame` succeeded). `false` means the
    /// frame was skipped (fence-wait failure) — the loop backs off so a
    /// persistent failure does not spin a core.
    rendered: bool,
}

/// Scene target size: `REDLILIUM_HEADLESS_SIZE=WxH`, default 1280x720.
fn target_size() -> (u32, u32) {
    if let Ok(spec) = std::env::var("REDLILIUM_HEADLESS_SIZE")
        && let Some((w, h)) = spec.split_once('x')
        && let (Ok(w), Ok(h)) = (w.trim().parse::<u32>(), h.trim().parse::<u32>())
        && w > 0
        && h > 0
    {
        return (w, h);
    }
    (1280, 720)
}
