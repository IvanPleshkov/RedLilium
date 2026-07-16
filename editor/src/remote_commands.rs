//! Remote-control protocol layer: parses command lines from the
//! [`RemoteTransport`], executes them, and replies (`docs/REMOTE.md`).
//!
//! Contract highlights:
//! - **Everything through the editor guarantee**: writes construct the same
//!   `EditAction`s the UI would and push them to the `ActionQueue` — they land
//!   in the undo history. Responses to writes are sent **after** the action
//!   executed (the next frame's queue drain), carrying a fresh snapshot.
//! - **Natural RON** on the wire: component payloads are plain RON values
//!   (converted through `serialize::natural`), not the internal tagged tree.
//! - Asset asynchrony is explicit: `wait(for: "assets_idle")` parks until the
//!   asset pipeline has been calm for a couple of frames.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use redlilium_assets::AssetProcessor;
use redlilium_core::abstract_editor::{ActionQueue, EditAction, EditActionHistory};
use redlilium_ecs::serialize::{
    SerializedComponent, find_entity, natural_to_value, value_to_natural,
};
use redlilium_ecs::ui::{
    ActionRegistry, AddComponentAction, ImportComponentAction, RemoveComponentAction, SpawnReport,
};
use redlilium_ecs::{
    CameraTarget, ChangedAssets, Entity, Name, Parent, RemoteTransport, ScenePass, World,
};
use redlilium_graphics::{
    Buffer, BufferDescriptor, BufferTextureCopyRegion, BufferTextureLayout, BufferUsage,
    GraphicsDevice, RenderGraph, TextureCopyLocation, TextureOrigin, TransferConfig,
    TransferOperation, TransferPass,
};
use serde::Serialize;

const PROTO_VERSION: u32 = 1;
const COMMANDS: &[&str] = &[
    "hello",
    "state",
    "inspect",
    "edit_component",
    "add_component",
    "remove_component",
    "select",
    "undo",
    "redo",
    "logs",
    "wait",
    "step",
    "screenshot",
    "shutdown",
    "action",
    "actions",
    "new_asset",
    "save_prefab",
    "spawn_prefab",
    "save_scene",
    "load_scene",
    "pick",
    "pick_rect",
    "reload_game",
    "set_gizmo_mode",
    "play",
    "pause",
    "resume",
    "stop",
];

/// A play-session lifecycle request (`play` / `pause` / `resume` / `stop`),
/// applied by the shell between frames — dispatch cannot create or drop the
/// play world from inside the editing world's frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayRequest {
    Play,
    Pause,
    Resume,
    Stop,
}

/// The hosted game's status, surfaced in the `state` response (ADR-033).
/// The shell refreshes this snapshot once per frame before dispatch.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct GameStatus {
    /// `"static"` (game-owned editor binary), `"dylib"` (`REDLILIUM_GAME`),
    /// or `"none"`.
    pub hosted: &'static str,
    /// Play worlds boot from a Tier-1 behavior dylib (vs the hosted module).
    pub behavior_override: bool,
    /// Game sources changed since the behavior module was (re)built.
    pub stale: bool,
    /// A `cargo build` is running in the background.
    pub rebuilding: bool,
    /// The last build relinked the editor binary itself — a Tier-2 process
    /// restart is required; no swap happened.
    pub restart_required: bool,
    /// The behavior module's component schemas diverge from the authoring
    /// (static) image: play runs the new code, authoring the new fields
    /// needs a restart.
    pub schema_diverged: bool,
}

/// Per-editor remote protocol state (parked writes/waits, screenshot job).
#[derive(Default)]
pub struct RemoteCommands {
    parked: Vec<Parked>,
    screenshot: Option<ScreenshotJob>,
    pick: Option<PickJob>,
    shutdown: bool,
    /// A game-module reload was requested; the shell executes it between
    /// frames (dispatch cannot — the reload replaces the whole world).
    reload_game: bool,
    /// A play-session request; the shell applies it between frames.
    play_request: Option<PlayRequest>,
    /// Hosted-game status for the `state` response (shell-refreshed).
    pub game_status: Option<GameStatus>,
}

impl RemoteCommands {
    /// Whether parked work (writes, waits, a screenshot, a pick) still needs
    /// frames to resolve. The headless shell keeps ticking while this is true.
    pub fn has_pending(&self) -> bool {
        !self.parked.is_empty() || self.screenshot.is_some() || self.pick.is_some()
    }

    /// Take the `shutdown` request, if one arrived. The shell decides how to
    /// exit (the headless loop breaks; the windowed editor closes the window).
    pub fn take_shutdown(&mut self) -> bool {
        std::mem::take(&mut self.shutdown)
    }

    /// Take the `reload_game` request, if one arrived. The shell runs the
    /// reload (see `game_host::reload_game`) between frames.
    pub fn take_reload(&mut self) -> bool {
        std::mem::take(&mut self.reload_game)
    }

    /// Take the play-session request, if one arrived. The shell applies it
    /// between frames (see `crate::play::PlaySession`).
    pub fn take_play_request(&mut self) -> Option<PlayRequest> {
        self.play_request.take()
    }
}

// ---------------------------------------------------------------------------
// Picking
// ---------------------------------------------------------------------------

/// A pending `pick` / `pick_rect`, in **scene-image coordinates** (the same
/// pixel space `screenshot` produces — what a remote agent actually sees).
/// The shell translates to its own pick space (the windowed editor offsets by
/// the SceneView panel origin) and drives the entity-index readback.
#[derive(Clone, Copy)]
pub enum PickRequest {
    Point { x: u32, y: u32 },
    Rect { x: u32, y: u32, w: u32, h: u32 },
}

struct PickJob {
    conn: u64,
    id: i64,
    request: PickRequest,
    /// Set once the shell submitted the readback to the scene view.
    in_flight: bool,
}

/// Take the not-yet-submitted pick request, marking it in flight. The shell
/// calls this each frame after `pump` and forwards to the scene view.
pub fn take_pick_request(rc: &mut RemoteCommands) -> Option<PickRequest> {
    let job = rc.pick.as_mut()?;
    if job.in_flight {
        return None;
    }
    job.in_flight = true;
    Some(job.request)
}

/// Whether a remote pick is waiting on the GPU readback — the shell routes
/// scene-view pick results here instead of into its own selection.
pub fn pick_in_flight(rc: &RemoteCommands) -> bool {
    rc.pick.as_ref().is_some_and(|j| j.in_flight)
}

/// Fail the current pick (scene view hidden, out-of-bounds coordinates, …).
pub fn fail_pick(rc: &mut RemoteCommands, world: &World, error: &str) {
    if let Some(job) = rc.pick.take() {
        send_err(world, job.conn, job.id, error);
    }
}

/// Map a picked entity index to the entity, mirroring the editor's click
/// selection: only indices belonging to a renderable count as hits (the
/// clear value / stale indices resolve to `None`).
fn pick_index_to_entity(world: &World, index: u32) -> Option<Entity> {
    let renderable = world
        .read::<redlilium_ecs::MeshRenderer>()
        .ok()
        .is_some_and(|renderers| renderers.get(index).is_some());
    if renderable {
        world.entity_at_index(index)
    } else {
        None
    }
}

/// Complete a point pick with the readback's hit (`None` = empty space).
pub fn complete_point_pick(rc: &mut RemoteCommands, world: &World, hit: Option<u32>) {
    let Some(job) = rc.pick.take() else { return };
    #[derive(Serialize)]
    struct PickResp {
        id: i64,
        ok: bool,
        entity: Option<String>,
    }
    send(
        world,
        job.conn,
        &PickResp {
            id: job.id,
            ok: true,
            entity: hit
                .and_then(|index| pick_index_to_entity(world, index))
                .map(entity_spec),
        },
    );
}

/// Complete a rect pick with the readback's entity indices.
pub fn complete_rect_pick(rc: &mut RemoteCommands, world: &World, indices: &[u32]) {
    let Some(job) = rc.pick.take() else { return };
    #[derive(Serialize)]
    struct PickRectResp {
        id: i64,
        ok: bool,
        entities: Vec<String>,
    }
    send(
        world,
        job.conn,
        &PickRectResp {
            id: job.id,
            ok: true,
            entities: indices
                .iter()
                .filter_map(|&idx| pick_index_to_entity(world, idx))
                .map(entity_spec)
                .collect(),
        },
    );
}

/// The apply outcome of a queued action, filled by [`ReportedAction`] when
/// the drain executes it. `None` until then.
type ExecResult = Arc<Mutex<Option<Result<(), String>>>>;

/// Decorator that captures the apply result of the wrapped action so the
/// parked response can report failure instead of a false ack. Everything
/// else delegates — history semantics (recording, merging, undo) are the
/// inner action's.
struct ReportedAction {
    inner: Box<dyn EditAction<World>>,
    result: ExecResult,
}

impl std::fmt::Debug for ReportedAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl EditAction<World> for ReportedAction {
    fn apply(&mut self, target: &mut World) -> redlilium_core::abstract_editor::EditActionResult {
        let result = self.inner.apply(target);
        if let Ok(mut slot) = self.result.lock() {
            *slot = Some(result.clone().map_err(|e| e.to_string()));
        }
        result
    }

    fn undo(&mut self, target: &mut World) -> redlilium_core::abstract_editor::EditActionResult {
        self.inner.undo(target)
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn merge(&mut self, other: Box<dyn EditAction<World>>) -> Option<Box<dyn EditAction<World>>> {
        self.inner.merge(other)
    }

    fn is_recorded(&self) -> bool {
        self.inner.is_recorded()
    }

    fn breaks_merge(&self) -> bool {
        self.inner.breaks_merge()
    }

    fn modifies_content(&self) -> bool {
        self.inner.modifies_content()
    }
}

enum Parked {
    /// A queued write — respond after the next queue drain applies it.
    /// `result` carries the apply outcome (error → error response);
    /// `component` set → the response carries that component's snapshot;
    /// `spawned` set → it carries the entities the action created.
    Write {
        conn: u64,
        id: i64,
        entity: Option<Entity>,
        component: Option<String>,
        spawned: Option<SpawnReport>,
        result: ExecResult,
    },
    /// `wait(for: "assets_idle")` — resolved after the asset pipeline has been
    /// idle for [`CALM_FRAMES`] consecutive frames.
    WaitAssets {
        conn: u64,
        id: i64,
        deadline: Instant,
        calm: u32,
    },
    /// `wait(for: "frames", n: N)`.
    WaitFrames { conn: u64, id: i64, remaining: u64 },
}

/// Consecutive idle frames before `assets_idle` resolves — a manager may
/// register fresh demand one frame after the processor drains.
const CALM_FRAMES: u32 = 3;

struct ScreenshotJob {
    conn: u64,
    id: i64,
    path: String,
    /// Filled by the GPU readback (via the frame graph).
    result: Arc<Mutex<Vec<u8>>>,
    /// `(width, height, padded_bytes_per_row, bgra)` — set once the readback
    /// pass is injected.
    layout: Option<(u32, u32, u32, bool)>,
    /// Keeps the readback buffer alive until the transfer completes.
    _buffer: Option<Arc<Buffer>>,
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ErrResp<'a> {
    id: i64,
    ok: bool,
    error: &'a str,
}

#[derive(Serialize)]
struct OkResp {
    id: i64,
    ok: bool,
}

fn send_err(world: &World, conn: u64, id: i64, error: &str) {
    let resp = ron::to_string(&ErrResp {
        id,
        ok: false,
        error,
    })
    .unwrap_or_else(|_| format!("(id: {id}, ok: false)"));
    world.resource::<RemoteTransport>().send(conn, resp);
}

fn send<T: Serialize>(world: &World, conn: u64, resp: &T) {
    match ron::to_string(resp) {
        Ok(text) => world.resource::<RemoteTransport>().send(conn, text),
        Err(e) => log::error!("remote: response serialization failed: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Pump
// ---------------------------------------------------------------------------

/// Run once per frame right after the `ActionQueue` drain: completes parked
/// responses (their actions have just applied), advances waits and the
/// screenshot job, then executes newly arrived commands.
pub fn pump(rc: &mut RemoteCommands, world: &mut World, history: &mut EditActionHistory<World>) {
    if !world.has_resource::<RemoteTransport>() {
        return;
    }

    complete_parked(rc, world);
    complete_screenshot(rc, world);

    let lines = world.resource::<RemoteTransport>().try_recv();
    for line in lines {
        dispatch(rc, world, history, line.conn, &line.line);
    }
}

fn complete_parked(rc: &mut RemoteCommands, world: &World) {
    let parked = std::mem::take(&mut rc.parked);
    for entry in parked {
        match entry {
            Parked::Write {
                conn,
                id,
                entity,
                component,
                spawned,
                result,
            } => {
                // The action drained this frame; a captured apply failure is
                // the response (a false ack hides a no-op edit). `None` means
                // the queue rejected/merged it away without running apply —
                // report that too rather than pretend it applied.
                match result.lock().ok().and_then(|slot| slot.clone()) {
                    Some(Err(e)) => {
                        send_err(world, conn, id, &format!("apply failed: {e}"));
                        continue;
                    }
                    Some(Ok(())) => {}
                    None => {
                        send_err(world, conn, id, "action was not executed");
                        continue;
                    }
                }
                // Success: report the resulting state.
                match (entity, component, spawned) {
                    (Some(entity), Some(name), _) => {
                        send_component_snapshot(world, conn, id, entity, &name)
                    }
                    (_, _, Some(report)) => {
                        #[derive(Serialize)]
                        struct SpawnedResp {
                            id: i64,
                            ok: bool,
                            entities: Vec<String>,
                        }
                        let entities = report
                            .lock()
                            .map(|e| e.iter().map(|e| entity_spec(*e)).collect())
                            .unwrap_or_default();
                        send(
                            world,
                            conn,
                            &SpawnedResp {
                                id,
                                ok: true,
                                entities,
                            },
                        );
                    }
                    _ => send(world, conn, &OkResp { id, ok: true }),
                }
            }
            Parked::WaitAssets {
                conn,
                id,
                deadline,
                mut calm,
            } => {
                let idle = assets_idle(world);
                calm = if idle { calm + 1 } else { 0 };
                if calm >= CALM_FRAMES {
                    send(world, conn, &OkResp { id, ok: true });
                } else if Instant::now() > deadline {
                    send_err(world, conn, id, "timeout waiting for assets_idle");
                } else {
                    rc.parked.push(Parked::WaitAssets {
                        conn,
                        id,
                        deadline,
                        calm,
                    });
                }
            }
            Parked::WaitFrames {
                conn,
                id,
                remaining,
            } => {
                if remaining <= 1 {
                    send(world, conn, &OkResp { id, ok: true });
                } else {
                    rc.parked.push(Parked::WaitFrames {
                        conn,
                        id,
                        remaining: remaining - 1,
                    });
                }
            }
        }
    }
}

/// Whether the asset pipeline is quiet: no in-flight processor work and no
/// pending hot-reload notifications. One calm frame is not proof the pipeline
/// is done (see [`CALM_FRAMES`]) — used both for `wait(assets_idle)` and for
/// the headless shell's quiescence check.
pub fn assets_idle(world: &World) -> bool {
    let processor_idle =
        !world.has_resource::<AssetProcessor>() || world.resource::<AssetProcessor>().is_idle();
    let no_pending_reloads =
        !world.has_resource::<ChangedAssets>() || world.resource::<ChangedAssets>().is_empty();
    processor_idle && no_pending_reloads
}

fn complete_screenshot(rc: &mut RemoteCommands, world: &World) {
    let Some(job) = &rc.screenshot else { return };
    let Some((w, h, padded_bpr, bgra)) = job.layout else {
        return; // pass not injected yet
    };
    let data = {
        let Ok(mut guard) = job.result.lock() else {
            return;
        };
        if guard.is_empty() {
            return; // GPU readback still in flight
        }
        std::mem::take(&mut *guard)
    };
    let job = rc.screenshot.take().expect("checked above");

    // Un-pad rows and encode.
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h {
        let start = (row * padded_bpr) as usize;
        let end = start + (w * 4) as usize;
        if end > data.len() {
            send_err(world, job.conn, job.id, "readback shorter than expected");
            return;
        }
        pixels.extend_from_slice(&data[start..end]);
    }
    if bgra {
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    let result = image::RgbaImage::from_raw(w, h, pixels)
        .ok_or_else(|| "pixel buffer size mismatch".to_string())
        .and_then(|img| img.save(&job.path).map_err(|e| e.to_string()));

    #[derive(Serialize)]
    struct ScreenshotResp<'a> {
        id: i64,
        ok: bool,
        path: &'a str,
        width: u32,
        height: u32,
    }
    match result {
        Ok(()) => send(
            world,
            job.conn,
            &ScreenshotResp {
                id: job.id,
                ok: true,
                path: &job.path,
                width: w,
                height: h,
            },
        ),
        Err(e) => send_err(world, job.conn, job.id, &format!("png encode: {e}")),
    }
}

/// Inject the pending screenshot's readback pass into this frame's graph.
/// Call after the `Render` schedule ran (the scene pass is in the graph);
/// the readback depends on the scene pass so it copies this frame's image.
pub fn inject_screenshot_pass(
    rc: &mut RemoteCommands,
    world: &World,
    device: &Arc<GraphicsDevice>,
    graph: &mut RenderGraph,
    source: Option<Arc<redlilium_graphics::Texture>>,
) {
    let Some(job) = &mut rc.screenshot else {
        return;
    };
    if job.layout.is_some() {
        return; // already injected
    }
    // The texture to capture. `Some` = the play session's view camera (the
    // game camera, or the pause flyover) — it was rendered in an earlier
    // submit of this frame, so cross-submit sync covers the copy and no
    // intra-graph edge is needed. `None` = the editing world's scene camera:
    // first CameraTarget, ordered after this frame's scene pass.
    let (color, after) = match source {
        Some(color) => (color, None),
        None => {
            let Some(scene_handle) = world
                .has_resource::<ScenePass>()
                .then(|| world.resource::<ScenePass>().0)
                .flatten()
            else {
                return; // no scene pass this frame — try again next frame
            };
            let Some(color) = world
                .read_all::<CameraTarget>()
                .ok()
                .and_then(|t| t.iter().next().map(|(_, target)| target.color.clone()))
            else {
                return;
            };
            (color, Some(scene_handle))
        }
    };

    let size = color.size();
    let (w, h) = (size.width, size.height);
    let padded_bpr = (w * 4).div_ceil(256) * 256;
    let total = padded_bpr as u64 * h as u64;
    let buffer = match device.create_buffer(&BufferDescriptor::new(
        total,
        BufferUsage::COPY_DST | BufferUsage::MAP_READ,
    )) {
        Ok(b) => b,
        Err(e) => {
            log::error!("remote: screenshot buffer failed: {e}");
            return;
        }
    };
    let bgra = format!("{:?}", color.format()).starts_with("Bgra");

    let region = BufferTextureCopyRegion::new(
        BufferTextureLayout::new(0, Some(padded_bpr), Some(h)),
        TextureCopyLocation::new(0, TextureOrigin::new(0, 0, 0)),
        redlilium_graphics::Extent3d {
            width: w,
            height: h,
            depth: 1,
        },
    );
    let mut pass = TransferPass::new("remote_screenshot".into());
    pass.set_transfer_config(TransferConfig::new().with_operations(vec![
        TransferOperation::readback_texture(color, buffer.clone(), vec![region]),
        TransferOperation::readback_buffer(buffer.clone(), 0..total as usize, job.result.clone()),
    ]));
    let handle = graph.add_transfer_pass(pass);
    if let Some(scene_handle) = after {
        graph.add_dependency(handle, scene_handle);
    }

    job.layout = Some((w, h, padded_bpr, bgra));
    job._buffer = Some(buffer);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn dispatch(
    rc: &mut RemoteCommands,
    world: &mut World,
    history: &mut EditActionHistory<World>,
    conn: u64,
    line: &str,
) {
    let parsed: ron::Value = match ron::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            send_err(world, conn, 0, &format!("parse: {e}"));
            return;
        }
    };
    let ron::Value::Map(map) = &parsed else {
        send_err(world, conn, 0, "expected a (key: value, …) envelope");
        return;
    };
    let get = |key: &str| {
        map.iter()
            .find(|(k, _)| **k == ron::Value::String(key.into()))
    };
    let id = match get("id").map(|(_, v)| v) {
        Some(ron::Value::Number(ron::Number::Integer(i))) => *i,
        _ => 0,
    };
    let Some((_, ron::Value::String(cmd))) = get("cmd") else {
        send_err(world, conn, id, "missing 'cmd'");
        return;
    };
    let cmd = cmd.clone();

    let str_param = |key: &str| -> Option<String> {
        match get(key).map(|(_, v)| v) {
            Some(ron::Value::String(s)) => Some(s.clone()),
            _ => None,
        }
    };
    let int_param = |key: &str| -> Option<i64> {
        match get(key).map(|(_, v)| v) {
            Some(ron::Value::Number(ron::Number::Integer(i))) => Some(*i),
            _ => None,
        }
    };

    match cmd.as_str() {
        "hello" => {
            #[derive(Serialize)]
            struct HelloResp<'a> {
                id: i64,
                ok: bool,
                proto: u32,
                commands: &'a [&'a str],
            }
            send(
                world,
                conn,
                &HelloResp {
                    id,
                    ok: true,
                    proto: PROTO_VERSION,
                    commands: COMMANDS,
                },
            );
        }
        "state" => cmd_state(world, conn, id, rc.game_status),
        "inspect" => {
            let Some(entity) = str_param("entity").and_then(|s| find_entity(world, &s)) else {
                return send_err(world, conn, id, "unknown entity");
            };
            send_inspect(world, conn, id, entity);
        }
        "edit_component" => {
            let Some(entity) = str_param("entity").and_then(|s| find_entity(world, &s)) else {
                return send_err(world, conn, id, "unknown entity");
            };
            let Some(component) = str_param("component") else {
                return send_err(world, conn, id, "missing 'component'");
            };
            let Some((_, data)) = get("data") else {
                return send_err(world, conn, id, "missing 'data'");
            };
            let value = match natural_to_value(data) {
                Ok(v) => v,
                Err(e) => return send_err(world, conn, id, &format!("data: {e}")),
            };
            let serialized = SerializedComponent {
                type_name: component.clone(),
                schema_version: 1,
                data: value,
            };
            let result = push_action(
                world,
                Box::new(ImportComponentAction::new(entity, serialized)),
            );
            rc.parked.push(Parked::Write {
                conn,
                id,
                entity: Some(entity),
                component: Some(component),
                spawned: None,
                result,
            });
        }
        "add_component" | "remove_component" => {
            let Some(entity) = str_param("entity").and_then(|s| find_entity(world, &s)) else {
                return send_err(world, conn, id, "unknown entity");
            };
            let Some(name) = str_param("component") else {
                return send_err(world, conn, id, "missing 'component'");
            };
            let action: Box<dyn EditAction<World>> = if cmd == "add_component" {
                Box::new(AddComponentAction { entity, name })
            } else {
                Box::new(RemoveComponentAction::new(entity, name))
            };
            let result = push_action(world, action);
            rc.parked.push(Parked::Write {
                conn,
                id,
                entity: Some(entity),
                component: None,
                spawned: None,
                result,
            });
        }
        "select" => {
            let entities: Vec<Entity> = match get("entities").map(|(_, v)| v) {
                Some(ron::Value::Seq(items)) => items
                    .iter()
                    .filter_map(|v| match v {
                        ron::Value::String(s) => find_entity(world, s),
                        _ => None,
                    })
                    .collect(),
                _ => return send_err(world, conn, id, "missing 'entities' list"),
            };
            let result = push_action(
                world,
                Box::new(redlilium_ecs::ui::SelectAction::set(entities)),
            );
            // SelectAction is transient (not recorded) but still applies via
            // the queue next frame; ack after it does.
            rc.parked.push(Parked::Write {
                conn,
                id,
                entity: None,
                component: None,
                spawned: None,
                result,
            });
        }
        "undo" | "redo" => {
            let result = if cmd == "undo" {
                history.undo(world)
            } else {
                history.redo(world)
            };
            match result {
                Ok(()) => send(world, conn, &OkResp { id, ok: true }),
                Err(e) => send_err(world, conn, id, &e.to_string()),
            }
        }
        "logs" => {
            let since = int_param("since").unwrap_or(0) as u64;
            cmd_logs(world, conn, id, since);
        }
        "wait" => match str_param("for").as_deref() {
            Some("assets_idle") => {
                let timeout = int_param("timeout_ms").unwrap_or(10_000) as u64;
                rc.parked.push(Parked::WaitAssets {
                    conn,
                    id,
                    deadline: Instant::now() + Duration::from_millis(timeout),
                    calm: 0,
                });
            }
            Some("frames") => {
                let n = int_param("n").unwrap_or(1).max(1) as u64;
                rc.parked.push(Parked::WaitFrames {
                    conn,
                    id,
                    remaining: n,
                });
            }
            other => send_err(
                world,
                conn,
                id,
                &format!("unknown wait condition {other:?} (assets_idle | frames)"),
            ),
        },
        // Tick-on-demand: advance N frames, ack when they ran. In the headless
        // shell this *drives* the frames (the loop ticks while work is
        // parked); in the windowed shell frames run anyway, so it degenerates
        // to `wait(for: "frames")`.
        "step" => {
            let n = int_param("n").unwrap_or(1).max(1) as u64;
            rc.parked.push(Parked::WaitFrames {
                conn,
                id,
                remaining: n,
            });
        }
        "shutdown" => {
            rc.shutdown = true;
            send(world, conn, &OkResp { id, ok: true });
        }
        // Switch the transform gizmo's manipulation mode (#85 v2):
        // translate | rotate | scale. Immediate — the mode is plain editor
        // state, not an undoable edit.
        "set_gizmo_mode" => {
            let Some(mode) = str_param("mode") else {
                return send_err(world, conn, id, "missing 'mode'");
            };
            let mode = match mode.as_str() {
                "translate" => redlilium_gizmo::GizmoMode::Translate,
                "rotate" => redlilium_gizmo::GizmoMode::Rotate,
                "scale" => redlilium_gizmo::GizmoMode::Scale,
                other => {
                    return send_err(
                        world,
                        conn,
                        id,
                        &format!("unknown mode '{other}' (translate|rotate|scale)"),
                    );
                }
            };
            if world.has_resource::<redlilium_gizmo::TransformGizmo>() {
                world
                    .resource_mut::<redlilium_gizmo::TransformGizmo>()
                    .set_mode(mode);
                send(world, conn, &OkResp { id, ok: true });
            } else {
                send_err(world, conn, id, "gizmo not available");
            }
        }
        // Queue a game-module warm reload; the shell executes it between
        // frames (the reload replaces the whole world, which dispatch cannot
        // do from inside it). `ok: true` means *queued* — the outcome lands
        // in the editor log. Requires a hosted game and the Stopped state.
        "reload_game" => {
            rc.reload_game = true;
            send(world, conn, &OkResp { id, ok: true });
        }
        // Play-session lifecycle (EDITOR_REBUILD.md §4.3): the shell boots a
        // game world (the standalone composition) from the hosted module,
        // seeded with the current scene; `stop` drops it. `ok: true` means
        // *queued* — applied between frames; requires REDLILIUM_GAME. While
        // playing, `screenshot` captures the play world's camera.
        "play" | "pause" | "resume" | "stop" => {
            rc.play_request = Some(match cmd.as_str() {
                "play" => PlayRequest::Play,
                "pause" => PlayRequest::Pause,
                "resume" => PlayRequest::Resume,
                _ => PlayRequest::Stop,
            });
            send(world, conn, &OkResp { id, ok: true });
        }
        // The generic path: any action in the ActionRegistry, invoked by name
        // with a natural-RON parameter map. Same queue -> history route as
        // the specialized commands; spawn-style actions report the created
        // entities in the response.
        "action" => {
            let Some(name) = str_param("name") else {
                return send_err(world, conn, id, "missing 'name'");
            };
            let empty = ron::Value::Map(ron::Map::new());
            let params = get("params").map(|(_, v)| v).unwrap_or(&empty);
            let constructed = {
                let registry = world.resource::<ActionRegistry>();
                registry.construct(&name, world, params)
            };
            match constructed {
                Ok(registered) => {
                    let result = push_action(world, registered.action);
                    rc.parked.push(Parked::Write {
                        conn,
                        id,
                        entity: None,
                        component: None,
                        spawned: registered.spawned,
                        result,
                    });
                }
                Err(e) => send_err(world, conn, id, &format!("{name}: {e}")),
            }
        }
        // Create an asset file + DB record — the same (non-undoable) flow as
        // the browser's "New asset": file ops sit outside the history, only
        // record edits are actions. The VFS write is a local-mount file;
        // blocking on it keeps this a one-response command.
        "new_asset" => {
            let (Some(source), Some(kind)) = (str_param("source"), str_param("kind")) else {
                return send_err(world, conn, id, "missing 'source' or 'kind'");
            };
            let dir = str_param("dir").unwrap_or_default();
            // A material instance parents to the std opaque material by
            // default (mirrors the browser flow).
            let parent = if kind == "material_instance" {
                world.resource::<redlilium_assets::AssetDb>().guid_of(
                    &redlilium_assets::AssetPath::new("std", "materials/opaque.material"),
                )
            } else {
                None
            };
            let Some(spec) = redlilium_ecs::new_asset_spec(&kind, parent) else {
                return send_err(world, conn, id, &format!("unknown asset kind '{kind}'"));
            };
            let path = match str_param("name") {
                Some(name) => {
                    let path = if dir.is_empty() {
                        format!("{name}.{}", spec.extension)
                    } else {
                        format!("{dir}/{name}.{}", spec.extension)
                    };
                    let taken = world
                        .resource::<redlilium_assets::AssetDb>()
                        .guid_of(&redlilium_assets::AssetPath::new(&source, &path))
                        .is_some();
                    if taken {
                        return send_err(world, conn, id, &format!("'{path}' already exists"));
                    }
                    path
                }
                None => crate::core::unique_asset_path(world, &source, &dir, spec.extension),
            };
            let guid = {
                let mut db = world.resource_mut::<redlilium_assets::AssetDb>();
                let guid =
                    db.register_path(redlilium_assets::AssetPath::new(&source, &path), &kind, 0);
                db.set_settings(&guid, spec.settings);
                guid
            };
            let vfs_path = format!("{source}/{path}");
            let write = pollster::block_on(
                world
                    .resource::<AssetProcessor>()
                    .vfs()
                    .write(&vfs_path, spec.content),
            );
            if let Err(e) = write {
                world
                    .resource_mut::<redlilium_assets::AssetDb>()
                    .remove(&guid);
                return send_err(world, conn, id, &format!("write {vfs_path}: {e}"));
            }
            world
                .resource_mut::<redlilium_ecs::DirtyMounts>()
                .push(&source);
            log::info!("remote: created asset {vfs_path} ({guid})");

            #[derive(Serialize)]
            struct NewAssetResp {
                id: i64,
                ok: bool,
                guid: String,
                path: String,
            }
            send(
                world,
                conn,
                &NewAssetResp {
                    id,
                    ok: true,
                    guid: guid.to_string(),
                    path: vfs_path,
                },
            );
        }
        // Serialize an entity subtree to a `.prefab` file (RON) — the wire
        // twin of the browser's prefab export.
        "save_prefab" => {
            let Some(entity) = str_param("entity").and_then(|s| find_entity(world, &s)) else {
                return send_err(world, conn, id, "unknown entity");
            };
            let Some(path) = str_param("path") else {
                return send_err(world, conn, id, "missing 'path'");
            };
            let data = match world
                .serialize_prefab_asset(entity)
                .map_err(|e| e.to_string())
                .and_then(|prefab| {
                    redlilium_ecs::serialize::encode(&prefab, redlilium_ecs::serialize::Format::Ron)
                        .map_err(|e| e.to_string())
                }) {
                Ok(data) => data,
                Err(e) => return send_err(world, conn, id, &format!("serialize: {e}")),
            };
            let write =
                pollster::block_on(world.resource::<AssetProcessor>().vfs().write(&path, data));
            match write {
                Ok(()) => {
                    log::info!("remote: saved prefab {path}");
                    send(world, conn, &OkResp { id, ok: true });
                }
                Err(e) => send_err(world, conn, id, &format!("write {path}: {e}")),
            }
        }
        // Persist the world's scene content as a `.scene` asset (#2) — the
        // wire twin of File > Save. `path` (VFS `"mount/path"`) is optional
        // and defaults to the world's current scene; the chosen path is
        // recorded back as current and returned.
        "save_scene" => {
            let path = str_param("path")
                .or_else(|| world.resource::<crate::core::CurrentScene>().0.clone());
            let Some(path) = path else {
                return send_err(world, conn, id, "missing 'path' (and no current scene)");
            };
            let ron = match redlilium_ecs::serialize_scene_ron(world) {
                Ok(ron) => ron,
                Err(e) => return send_err(world, conn, id, &format!("serialize: {e}")),
            };
            let write = pollster::block_on(
                world
                    .resource::<AssetProcessor>()
                    .vfs()
                    .write(&path, ron.into_bytes()),
            );
            match write {
                Ok(()) => {
                    world.resource_mut::<crate::core::CurrentScene>().0 = Some(path.clone());
                    log::info!("remote: saved scene {path}");
                    #[derive(Serialize)]
                    struct SceneResp {
                        id: i64,
                        ok: bool,
                        path: String,
                    }
                    send(world, conn, &SceneResp { id, ok: true, path });
                }
                Err(e) => send_err(world, conn, id, &format!("write {path}: {e}")),
            }
        }
        // Open a `.scene` asset (#2): an undoable replace of the world's
        // scene content (editor entities survive; undo restores the previous
        // content). The response lands after the queued action executes.
        "load_scene" => {
            let Some(path) = str_param("path") else {
                return send_err(world, conn, id, "missing 'path'");
            };
            let data =
                match pollster::block_on(world.resource::<AssetProcessor>().vfs().read(&path)) {
                    Ok(data) => data,
                    Err(e) => return send_err(world, conn, id, &format!("read {path}: {e}")),
                };
            let scene = match redlilium_ecs::serialize::decode::<
                redlilium_ecs::serialize::SerializedWorld,
            >(&data, redlilium_ecs::serialize::Format::Ron)
            {
                Ok(scene) => scene,
                Err(e) => return send_err(world, conn, id, &format!("decode {path}: {e}")),
            };
            let action = redlilium_ecs::ReplaceSceneAction::new(scene, &path);
            let result = push_action(world, Box::new(action));
            world.resource_mut::<crate::core::CurrentScene>().0 = Some(path);
            rc.parked.push(Parked::Write {
                conn,
                id,
                entity: None,
                component: None,
                spawned: None,
                result,
            });
        }
        // Entity under a point / all entities in a region, in scene-image
        // coordinates (what `screenshot` shows). Resolved by the entity-index
        // GPU pass a frame later.
        "pick" | "pick_rect" => {
            if rc.pick.is_some() {
                return send_err(world, conn, id, "a pick is already in flight");
            }
            let (Some(x), Some(y)) = (int_param("x"), int_param("y")) else {
                return send_err(world, conn, id, "missing 'x'/'y'");
            };
            let request = if cmd == "pick" {
                PickRequest::Point {
                    x: x.max(0) as u32,
                    y: y.max(0) as u32,
                }
            } else {
                let (Some(w), Some(h)) = (int_param("w"), int_param("h")) else {
                    return send_err(world, conn, id, "missing 'w'/'h'");
                };
                PickRequest::Rect {
                    x: x.max(0) as u32,
                    y: y.max(0) as u32,
                    w: w.max(1) as u32,
                    h: h.max(1) as u32,
                }
            };
            rc.pick = Some(PickJob {
                conn,
                id,
                request,
                in_flight: false,
            });
        }
        // Spawn a `.prefab` file (undoable; response carries the new ids).
        "spawn_prefab" => {
            let Some(path) = str_param("path") else {
                return send_err(world, conn, id, "missing 'path'");
            };
            let parent = match str_param("parent") {
                Some(spec) => match find_entity(world, &spec) {
                    Some(e) => Some(e),
                    None => return send_err(world, conn, id, &format!("unknown parent '{spec}'")),
                },
                None => None,
            };
            let data =
                match pollster::block_on(world.resource::<AssetProcessor>().vfs().read(&path)) {
                    Ok(data) => data,
                    Err(e) => return send_err(world, conn, id, &format!("read {path}: {e}")),
                };
            let prefab = match redlilium_ecs::serialize::decode::<
                redlilium_ecs::serialize::SerializedPrefab,
            >(&data, redlilium_ecs::serialize::Format::Ron)
            {
                Ok(p) => p,
                Err(e) => return send_err(world, conn, id, &format!("decode {path}: {e}")),
            };
            let report = SpawnReport::default();
            let action = redlilium_ecs::ui::SpawnPrefabAction::new(prefab, parent)
                .with_report(report.clone());
            let result = push_action(world, Box::new(action));
            rc.parked.push(Parked::Write {
                conn,
                id,
                entity: None,
                component: None,
                spawned: Some(report),
                result,
            });
        }
        "actions" => {
            #[derive(Serialize)]
            struct ActionsResp {
                id: i64,
                ok: bool,
                actions: Vec<(String, String)>,
            }
            let actions = world
                .resource::<ActionRegistry>()
                .list()
                .into_iter()
                .map(|(n, u)| (n.to_owned(), u.to_owned()))
                .collect();
            send(
                world,
                conn,
                &ActionsResp {
                    id,
                    ok: true,
                    actions,
                },
            );
        }
        "screenshot" => {
            let Some(path) = str_param("path") else {
                return send_err(world, conn, id, "missing 'path'");
            };
            match str_param("target").as_deref() {
                Some("scene") | None => {}
                other => {
                    return send_err(world, conn, id, &format!("unknown target {other:?}"));
                }
            }
            if rc.screenshot.is_some() {
                return send_err(world, conn, id, "a screenshot is already in flight");
            }
            rc.screenshot = Some(ScreenshotJob {
                conn,
                id,
                path,
                result: Arc::new(Mutex::new(Vec::new())),
                layout: None,
                _buffer: None,
            });
        }
        other => send_err(world, conn, id, &format!("unknown command '{other}'")),
    }
}

/// Queue `action` wrapped in a [`ReportedAction`]; the returned slot holds
/// the apply outcome once the next frame's drain executes it.
fn push_action(world: &World, action: Box<dyn EditAction<World>>) -> ExecResult {
    let result = ExecResult::default();
    world
        .resource::<ActionQueue<World>>()
        .push(Box::new(ReportedAction {
            inner: action,
            result: result.clone(),
        }));
    result
}

fn entity_spec(e: Entity) -> String {
    format!("{}@{}", e.index(), e.spawn_tick())
}

fn cmd_state(world: &World, conn: u64, id: i64, game: Option<GameStatus>) {
    #[derive(Serialize)]
    struct EntityRow {
        entity: String,
        name: Option<String>,
        parent: Option<String>,
        components: Vec<String>,
    }
    #[derive(Serialize)]
    struct StateResp {
        id: i64,
        ok: bool,
        /// The scene asset the world content belongs to (VFS `"mount/path"`,
        /// #112) — `None` before the first save/load picks one.
        scene: Option<String>,
        entities: Vec<EntityRow>,
        selection: Vec<String>,
        /// Hosted-game status (ADR-033): stale/rebuilding/restart markers.
        game: Option<GameStatus>,
    }
    let entities: Vec<EntityRow> = world
        .iter_entities()
        .map(|e| EntityRow {
            entity: entity_spec(e),
            name: world.get::<Name>(e).map(|n| n.0.clone()),
            parent: world.get::<Parent>(e).map(|p| entity_spec(p.0)),
            components: world
                .inspectable_components_of(e)
                .into_iter()
                .map(str::to_owned)
                .collect(),
        })
        .collect();
    let selection = if world.has_resource::<redlilium_ecs::ui::Selection>() {
        world
            .resource::<redlilium_ecs::ui::Selection>()
            .entities()
            .iter()
            .map(|e| entity_spec(*e))
            .collect()
    } else {
        Vec::new()
    };
    send(
        world,
        conn,
        &StateResp {
            id,
            ok: true,
            scene: world.resource::<crate::core::CurrentScene>().0.clone(),
            entities,
            selection,
            game,
        },
    );
}

fn send_inspect(world: &World, conn: u64, id: i64, entity: Entity) {
    #[derive(Serialize)]
    struct InspectResp {
        id: i64,
        ok: bool,
        entity: String,
        components: Vec<(String, ron::Value)>,
    }
    let mut components = Vec::new();
    for name in world.inspectable_components_of(entity) {
        match world.serialize_component_by_name(entity, name) {
            Ok(Some(serialized)) => {
                components.push((name.to_owned(), value_to_natural(&serialized.data)));
            }
            Ok(None) => {}
            Err(e) => {
                components.push((name.to_owned(), ron::Value::String(format!("<error: {e}>"))));
            }
        }
    }
    send(
        world,
        conn,
        &InspectResp {
            id,
            ok: true,
            entity: entity_spec(entity),
            components,
        },
    );
}

/// Respond to an applied component write with the component's fresh snapshot.
fn send_component_snapshot(world: &World, conn: u64, id: i64, entity: Entity, component: &str) {
    #[derive(Serialize)]
    struct SnapshotResp {
        id: i64,
        ok: bool,
        entity: String,
        component: String,
        data: ron::Value,
    }
    match world.serialize_component_by_name(entity, component) {
        Ok(Some(serialized)) => send(
            world,
            conn,
            &SnapshotResp {
                id,
                ok: true,
                entity: entity_spec(entity),
                component: component.to_owned(),
                data: value_to_natural(&serialized.data),
            },
        ),
        Ok(None) => send_err(world, conn, id, "component absent after apply"),
        Err(e) => send_err(world, conn, id, &format!("snapshot: {e}")),
    }
}

fn cmd_logs(world: &World, conn: u64, id: i64, since: u64) {
    #[derive(Serialize)]
    struct LogRow {
        seq: u64,
        level: String,
        target: String,
        message: String,
    }
    #[derive(Serialize)]
    struct LogsResp {
        id: i64,
        ok: bool,
        entries: Vec<LogRow>,
    }
    let buffer = crate::log_capture::log_buffer();
    let entries: Vec<LogRow> = {
        let guard = buffer.lock().expect("log buffer poisoned");
        guard
            .entries()
            .iter()
            .filter(|e| e.seq > since)
            .map(|e| LogRow {
                seq: e.seq,
                level: e.level.to_string(),
                target: e.target.clone(),
                message: e.message.clone(),
            })
            .collect()
    };
    send(
        world,
        conn,
        &LogsResp {
            id,
            ok: true,
            entries,
        },
    );
}
