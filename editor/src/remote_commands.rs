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
use redlilium_ecs::serialize::{SerializedComponent, natural_to_value, value_to_natural};
use redlilium_ecs::ui::{AddComponentAction, ImportComponentAction, RemoveComponentAction};
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
];

/// Per-editor remote protocol state (parked writes/waits, screenshot job).
#[derive(Default)]
pub struct RemoteCommands {
    parked: Vec<Parked>,
    screenshot: Option<ScreenshotJob>,
    shutdown: bool,
}

impl RemoteCommands {
    /// Whether parked work (writes, waits, a screenshot) still needs frames to
    /// resolve. The headless shell keeps ticking while this is true.
    pub fn has_pending(&self) -> bool {
        !self.parked.is_empty() || self.screenshot.is_some()
    }

    /// Take the `shutdown` request, if one arrived. The shell decides how to
    /// exit (the headless loop breaks; the windowed editor closes the window).
    pub fn take_shutdown(&mut self) -> bool {
        std::mem::take(&mut self.shutdown)
    }
}

enum Parked {
    /// A queued write — respond after the next queue drain applies it.
    /// `component` set → the response carries that component's snapshot.
    Write {
        conn: u64,
        id: i64,
        entity: Option<Entity>,
        component: Option<String>,
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
            } => {
                // The action drained this frame; report the resulting state.
                match (entity, component) {
                    (Some(entity), Some(name)) => {
                        send_component_snapshot(world, conn, id, entity, &name)
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
) {
    let Some(job) = &mut rc.screenshot else {
        return;
    };
    if job.layout.is_some() {
        return; // already injected
    }
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
    graph.add_dependency(handle, scene_handle);

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
        "state" => cmd_state(world, conn, id),
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
                data: value,
            };
            push_action(
                world,
                Box::new(ImportComponentAction::new(entity, serialized)),
            );
            rc.parked.push(Parked::Write {
                conn,
                id,
                entity: Some(entity),
                component: Some(component),
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
            push_action(world, action);
            rc.parked.push(Parked::Write {
                conn,
                id,
                entity: Some(entity),
                component: None,
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
            push_action(
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

fn push_action(world: &World, action: Box<dyn EditAction<World>>) {
    world.resource::<ActionQueue<World>>().push(action);
}

fn find_entity(world: &World, spec: &str) -> Option<Entity> {
    let (index, tick) = redlilium_ecs::serialize::parse_entity_spec(spec).ok()?;
    world
        .iter_entities()
        .find(|e| e.index() == index && e.spawn_tick() == tick)
}

fn entity_spec(e: Entity) -> String {
    format!("{}@{}", e.index(), e.spawn_tick())
}

fn cmd_state(world: &World, conn: u64, id: i64) {
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
        entities: Vec<EntityRow>,
        selection: Vec<String>,
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
            entities,
            selection,
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
