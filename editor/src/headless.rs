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

    let runner = EcsRunner::single_thread();
    let mut ew = create_editor_world(
        &EditorWorldParams {
            vfs: &vfs,
            local_mounts: &local_mounts,
            remote: true,
            egui: false,
        },
        &mut scene_view,
        width as f32 / height as f32,
    );
    ew.world.insert_resource(DebugDrawerRenderer::new(
        device.clone(),
        color_format,
        Some(TextureFormat::Depth32Float),
    ));
    ew.schedules.run_startup(&mut ew.world, &runner);

    let mut pipeline = device.create_pipeline(2);
    let mut rc = RemoteCommands::default();

    let mut calm = 0u32;
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

        shutdown = tick(
            &mut ew,
            &mut rc,
            &mut scene_view,
            &mut pipeline,
            &runner,
            &local_mounts,
            width,
            height,
        );

        // Even a busy headless editor has no vsync to pace it — yield a
        // little so asset waits don't spin a core at full tilt.
        std::thread::sleep(Duration::from_millis(1));
    }

    // The shutdown ack is queued on the IO runtime's writer task; give it a
    // beat to reach the socket before tearing the process down.
    std::thread::sleep(Duration::from_millis(150));
    pipeline.wait_idle();
    let _ = std::fs::remove_file(".redlilium/editor.port");
    log::info!("headless editor: shutdown");
}

/// One editor frame: the same sequence as the windowed shell's
/// `on_update` + `on_draw`, minus input, picking, and egui. Returns `true`
/// when a remote `shutdown` arrived.
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
) -> bool {
    ew.persist_dirty_mounts(local_mounts);
    ew.debug_drawer.read().advance_tick();

    // Actions queued by last tick's remote commands apply here…
    ew.drain_actions();
    // …and the pump completes their parked responses, then dispatches newly
    // arrived commands (their actions drain next tick).
    remote_commands::pump(rc, &mut ew.world, &mut ew.history);
    let shutdown = rc.take_shutdown();

    ew.schedules.run_frame(&mut ew.world, runner, FIXED_DT);

    // Render into the camera's off-screen target (created on first tick).
    let mut schedule = pipeline.begin_frame();
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
    schedule.render(graph);
    pipeline.end_frame(schedule);

    ew.window_input.write().begin_frame();
    shutdown
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
