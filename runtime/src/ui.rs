//! Game-facing UI and host-control resources (#100).
//!
//! The standalone runtime owns an [`EguiController`] and brackets each frame
//! around the game schedules: `begin_frame` before `Schedules::run_frame`,
//! `end_frame` in the draw phase, compositing the UI over the present blit.
//! Game systems draw between the two through the [`GameUi`] resource.
//!
//! [`EguiController`]: redlilium_graphics::egui::EguiController

use redlilium_graphics::egui::egui;

/// World resource exposing the in-game egui context to game systems.
///
/// Inserted by the standalone runtime host; a world hosted elsewhere (the
/// editor's `GameHost`) may not provide it, so game systems must treat it as
/// optional (`world.has_resource::<GameUi>()`).
///
/// ```ignore
/// let world = ctx.raw_world();
/// if world.has_resource::<GameUi>() {
///     let ui = world.resource::<GameUi>();
///     egui::Window::new("HUD").show(ui.ctx(), |ui| { /* widgets */ });
/// }
/// ```
pub struct GameUi {
    ctx: egui::Context,
}

impl GameUi {
    pub(crate) fn new(ctx: egui::Context) -> Self {
        Self { ctx }
    }

    /// The egui context. Valid for UI building only while the host's frame
    /// bracket is open — i.e. from any system inside the frame schedules.
    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }
}

/// World resource for controlling the host application from game code.
///
/// Always present (inserted by [`App::new`](crate::App)); hosts honor it where
/// it makes sense — the standalone runtime exits at the end of the frame,
/// wasm ignores it (a browser tab has no process to exit), and the editor's
/// play mode may map it to a Stop instead.
#[derive(Default)]
pub struct AppControl {
    exit_requested: bool,
}

impl AppControl {
    /// Request the host application to exit. Takes effect at the end of the
    /// current frame; no-op on wasm.
    pub fn request_exit(&mut self) {
        self.exit_requested = true;
    }

    /// Whether game code requested an exit.
    pub fn exit_requested(&self) -> bool {
        self.exit_requested
    }
}
