//! Tier-2 session carry (ADR-033, #127): the state an editor process writes
//! down right before an exec-restart so its successor comes up where the
//! user left off — the open scene and the editor camera's pose.
//!
//! Deliberately small and lossy: unsaved scene edits, undo history, and
//! selection do **not** survive a restart (the windowed shell routes restart
//! through the unsaved-changes dialog, so nothing is lost silently). The
//! file is one-shot: consumed (and deleted) by the next startup, so a later
//! cold launch is not haunted by a stale session.

use serde::{Deserialize, Serialize};

/// Where the carry lives — next to `editor.port` in the project-local
/// `.redlilium/` scratch dir.
const SESSION_PATH: &str = ".redlilium/session.ron";

/// The editor camera's free-fly pose (`target`/`distance`/`yaw`/`pitch` —
/// the transform is derived from these every frame, so they ARE the pose).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CameraPose {
    pub target: [f32; 3],
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

/// What survives an exec-restart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCarry {
    /// The scene open for editing (VFS `"mount/path"`), `None` if unsaved.
    pub scene: Option<String>,
    /// The editor camera pose, `None` if the camera was not resolvable.
    pub camera: Option<CameraPose>,
}

/// Capture the carry from the live editing world: the current scene and the
/// editor camera's free-fly pose.
pub fn capture(ew: &crate::core::EditorWorld) -> SessionCarry {
    let scene = ew.world.resource::<crate::core::CurrentScene>().0.clone();
    let camera = ew
        .world
        .get::<redlilium_ecs::FreeFlyCamera>(ew.editor_camera)
        .map(|fly| CameraPose {
            target: [fly.target.x, fly.target.y, fly.target.z],
            distance: fly.distance,
            yaw: fly.yaw,
            pitch: fly.pitch,
        });
    SessionCarry { scene, camera }
}

/// Apply a carried pose to the freshly spawned editor camera. Only the pose
/// fields move — sensitivities and limits keep the fresh defaults. The
/// camera's transform is derived from these by `UpdateFreeFlyCamera` on the
/// next frame.
pub fn apply_camera_pose(ew: &mut crate::core::EditorWorld, pose: CameraPose) {
    let Some(fly) = ew
        .world
        .get::<redlilium_ecs::FreeFlyCamera>(ew.editor_camera)
    else {
        return;
    };
    let mut fly = *fly;
    fly.target = redlilium_core::math::Vec3::new(pose.target[0], pose.target[1], pose.target[2]);
    fly.distance = pose.distance;
    fly.yaw = pose.yaw;
    fly.pitch = pose.pitch;
    let _ = ew.world.insert(ew.editor_camera, fly);
}

/// Persist the carry for the successor process. Failure is logged, not
/// fatal — a restart without carry is a cold start, not an error.
pub fn write(carry: &SessionCarry) {
    let write = || -> Result<(), String> {
        let text =
            ron::ser::to_string_pretty(carry, Default::default()).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(".redlilium").map_err(|e| e.to_string())?;
        std::fs::write(SESSION_PATH, text).map_err(|e| e.to_string())
    };
    match write() {
        Ok(()) => log::info!("session carry written ({SESSION_PATH})"),
        Err(e) => log::warn!("session carry write failed ({e}); restarting cold"),
    }
}

/// Consume the carry left by a predecessor process, if any. Reads **and
/// deletes** the file — the carry applies to exactly one startup.
pub fn take() -> Option<SessionCarry> {
    let text = std::fs::read_to_string(SESSION_PATH).ok()?;
    let _ = std::fs::remove_file(SESSION_PATH);
    match ron::from_str::<SessionCarry>(&text) {
        Ok(carry) => {
            log::info!(
                "session carry restored (scene: {:?}, camera: {})",
                carry.scene,
                if carry.camera.is_some() { "yes" } else { "no" }
            );
            Some(carry)
        }
        Err(e) => {
            log::warn!("session carry unreadable ({e}); starting cold");
            None
        }
    }
}

/// Whether the process should exec-restart after its main loop exits.
/// Set by a shell handling a restart request; read once by
/// `crate::launch` after `App::run`/`headless::run` return (every world,
/// GPU resource, and mapped game image is torn down by then).
static RESTART_AFTER_EXIT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Request an exec-restart once the main loop winds down.
pub fn request_restart() {
    RESTART_AFTER_EXIT.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a restart was requested (consumed by `crate::launch`).
pub fn restart_requested() -> bool {
    RESTART_AFTER_EXIT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Replace this process with a fresh image of the same binary, same argv and
/// environment. Unix: `exec` (same PID, remote clients just reconnect via
/// the re-published port file). Elsewhere: spawn + exit.
pub fn exec_restart() -> ! {
    let exe = std::env::current_exe().expect("current_exe for restart");
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    log::info!("restarting: exec {exe:?}");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = std::process::Command::new(&exe).args(&args).exec();
        // exec only returns on failure.
        panic!("exec-restart failed: {err}");
    }
    #[cfg(not(unix))]
    {
        std::process::Command::new(&exe)
            .args(&args)
            .spawn()
            .expect("spawn for restart");
        std::process::exit(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The carry round-trips through RON (shape stability for the file).
    #[test]
    fn carry_roundtrips() {
        let carry = SessionCarry {
            scene: Some("game/scenes/level1.scene".into()),
            camera: Some(CameraPose {
                target: [1.0, 2.0, 3.0],
                distance: 7.5,
                yaw: 0.3,
                pitch: -0.2,
            }),
        };
        let text = ron::ser::to_string_pretty(&carry, Default::default()).unwrap();
        let back: SessionCarry = ron::from_str(&text).unwrap();
        assert_eq!(back.scene.as_deref(), Some("game/scenes/level1.scene"));
        let cam = back.camera.unwrap();
        assert_eq!(cam.target, [1.0, 2.0, 3.0]);
        assert_eq!(cam.distance, 7.5);
    }
}
