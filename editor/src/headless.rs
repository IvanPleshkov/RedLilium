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

pub fn run() {
    redlilium_core::init();
    redlilium_graphics::init();

    let (width, height) = target_size();
    log::info!("headless editor: {width}x{height}, tick-on-demand");

    let instance =
        GraphicsInstance::with_parameters(InstanceParameters::new()).expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    log::info!("headless device: {}", device.name());

    // The same local mounts as the windowed shell (see `Editor::new`), minus
    // the browser's file watcher — external file edits are picked up by the
    // startup scan, in-editor edits by the ChangedAssets flow.
    let local_mounts: Vec<(&'static str, &'static str)> =
        vec![("std", "std-assets"), ("project", "project-assets")];
    let mut vfs = Vfs::new();
    for &(name, dir) in &local_mounts {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::error!("failed to create mount dir '{dir}': {e}");
        }
        vfs.mount(name, FileSystemProvider::new(dir));
    }

    // The scene color format is a constant here (no surface to negotiate
    // with); sRGB so screenshot PNGs come out gamma-correct.
    let color_format = TextureFormat::Bgra8UnormSrgb;
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
    // Declared before `ew` so it drops after it — the mapped game image must
    // outlive every world its plugin touched (ADR-020, #58).
    let mut game_host: Option<crate::game_host::GameHost> = None;
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
    ew.schedules.run_startup(&mut ew.world, &runner);

    let aspect = width as f32 / height as f32;
    if let Ok(path) = std::env::var("REDLILIUM_GAME") {
        // SAFETY: the operator points this at a game cdylib built in the same
        // `cargo build` as this editor (the fingerprint gate enforces engine
        // parity; the same-build contract is documented on GameModule::load).
        match unsafe { crate::game_host::GameHost::load(&path, &engine, &mut ew, aspect) } {
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
        let busy = rc.has_pending() || !remote_commands::assets_idle(&ew.world);
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
        );
        shutdown = outcome.shutdown;

        if rc.take_reload() {
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
            // little so asset waits don't spin a core at full tilt.
            std::thread::sleep(Duration::from_millis(1));
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
/// `on_update` + `on_draw`, minus input, picking, and egui.
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
    scene_view.ensure_camera_target(&mut ew.world, ew.editor_camera, width, height);
    ew.world.resource_mut::<RenderSchedule>().set(graph);
    ew.schedules.run_schedule::<Render>(&mut ew.world, runner);
    graph = ew
        .world
        .resource_mut::<RenderSchedule>()
        .take()
        .expect("RenderSchedule must hold the graph after the Render schedule");
    remote_commands::inject_screenshot_pass(rc, &ew.world, scene_view.device(), &mut graph);

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
