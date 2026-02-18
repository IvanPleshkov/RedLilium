use std::sync::{Arc, RwLock};

use egui_dock::DockState;
use redlilium_app::{AppContext, AppHandler, DrawContext};
use redlilium_ecs::ui::InspectorState;
use redlilium_ecs::{EcsRunner, Schedules, World};
use redlilium_graphics::egui::{EguiApp, EguiController};
use redlilium_graphics::{FrameSchedule, RenderTarget};
use winit::event::{KeyEvent, MouseButton, MouseScrollDelta};

use crate::dock::{self, EditorTabViewer, Tab};
#[cfg(not(target_os = "macos"))]
use crate::menu;
#[cfg(target_os = "macos")]
use crate::menu::NativeMenu;
use crate::toolbar::{self, PlayState};

/// A minimal EguiApp that does nothing.
///
/// All actual UI rendering happens in [`Editor::on_draw`] using the egui
/// context directly between `begin_frame` / `end_frame`.
struct NullEguiApp;

impl EguiApp for NullEguiApp {
    fn update(&mut self, _ctx: &egui::Context) {}

    fn setup(&mut self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.window_corner_radius = egui::CornerRadius::same(4);
        ctx.set_style(style);
    }
}

pub struct Editor {
    // ECS
    world: World,
    schedules: Schedules,
    runner: EcsRunner,

    // UI
    egui_controller: Option<EguiController>,
    dock_state: DockState<Tab>,
    inspector_state: InspectorState,
    play_state: PlayState,
    #[cfg(target_os = "macos")]
    native_menu: Option<NativeMenu>,
}

impl Editor {
    pub fn new() -> Self {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);

        Self {
            world,
            schedules: Schedules::new(),
            runner: EcsRunner::single_thread(),
            egui_controller: None,
            dock_state: dock::create_default_layout(),
            inspector_state: InspectorState::new(),
            play_state: PlayState::Editing,
            #[cfg(target_os = "macos")]
            native_menu: None,
        }
    }
}

impl AppHandler for Editor {
    fn on_init(&mut self, ctx: &mut AppContext) {
        log::info!("Editor initialized");

        let null_app: Arc<RwLock<dyn EguiApp>> = Arc::new(RwLock::new(NullEguiApp));
        self.egui_controller = Some(EguiController::new(
            ctx.device().clone(),
            null_app,
            ctx.width(),
            ctx.height(),
            ctx.scale_factor(),
            ctx.surface_format(),
        ));

        // Create native menu after the event loop / NSApplication is initialized
        #[cfg(target_os = "macos")]
        {
            self.native_menu = Some(NativeMenu::new());
        }

        self.schedules.run_startup(&mut self.world, &self.runner);
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_resize(ctx.width(), ctx.height());
        }
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        // Poll native menu events (macOS only)
        #[cfg(target_os = "macos")]
        if let Some(menu) = &self.native_menu
            && let Some(action) = menu.poll_event()
        {
            log::info!("Menu action: {:?}", action);
        }

        if self.play_state == PlayState::Playing {
            self.schedules
                .run_frame(&mut self.world, &self.runner, ctx.delta_time() as f64);
        }
        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        let mut graph = ctx.acquire_graph();

        if let Some(egui) = &mut self.egui_controller {
            let width = ctx.width();
            let height = ctx.height();
            let elapsed = ctx.elapsed_time() as f64;
            let render_target = RenderTarget::from_surface(ctx.swapchain_texture());

            egui.begin_frame(elapsed);

            let egui_ctx = egui.context().clone();

            // Menu bar (egui fallback for non-macOS platforms)
            #[cfg(not(target_os = "macos"))]
            menu::draw_menu_bar(&egui_ctx);

            // Toolbar (below menu bar)
            self.play_state = toolbar::draw_toolbar(&egui_ctx, self.play_state);

            // Dock area fills remaining space
            egui::CentralPanel::default().show(&egui_ctx, |ui| {
                let mut tab_viewer = EditorTabViewer {
                    world: &mut self.world,
                    inspector_state: &mut self.inspector_state,
                };
                egui_dock::DockArea::new(&mut self.dock_state).show_inside(ui, &mut tab_viewer);
            });

            if let Some(egui_pass) = egui.end_frame(&render_target, width, height) {
                graph.add_graphics_pass(egui_pass);
            }
        }

        let _handle = ctx.submit("editor_ui", graph, &[]);
        ctx.finish(&[])
    }

    fn on_mouse_move(&mut self, _ctx: &mut AppContext, x: f64, y: f64) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_move(x, y);
        }
    }

    fn on_mouse_button(&mut self, _ctx: &mut AppContext, button: MouseButton, pressed: bool) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_button(button, pressed);
        }
    }

    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, dx: f32, dy: f32) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_scroll(MouseScrollDelta::LineDelta(dx, dy));
        }
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_key(event);
        }
    }
}
