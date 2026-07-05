//! The [`AppHandler`] gluing the window/GPU loop to the game world.

use std::sync::Arc;

use parking_lot::RwLock;

use redlilium_app::{AppContext, AppHandler, DrawContext, input::map_winit_key};
use redlilium_ecs::{
    Camera, CameraTarget, EcsRunner, Render, RenderSchedule, ScenePass, WindowInput, World,
};
use redlilium_graphics::{
    FrameSchedule, GraphicsDevice, TextureDescriptor, TextureFormat, TextureUsage,
};
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
    /// Current size of the main camera's render target.
    target_size: (u32, u32),
    color_format: TextureFormat,
}

pub(crate) struct RuntimeHandler<P: Plugin> {
    config: GameConfig,
    plugin: Option<P>,
    state: Option<GameState>,
}

impl<P: Plugin> RuntimeHandler<P> {
    pub(crate) fn new(config: GameConfig, plugin: P) -> Self {
        Self {
            config,
            plugin: Some(plugin),
            state: None,
        }
    }
}

impl<P: Plugin> AppHandler for RuntimeHandler<P> {
    fn on_init(&mut self, ctx: &mut AppContext) {
        let engine = EngineContext::new(ctx.device().clone(), &self.config.mounts);
        let mut app = App::new(&engine, ctx.aspect_ratio());
        if let Some(plugin) = self.plugin.take() {
            plugin.build(&mut app);
        }

        let runner = EcsRunner::single_thread();
        {
            let (world, schedules) = app.parts_mut();
            schedules.run_startup(world, &runner);
        }

        let blit = PresentBlit::new(ctx.device(), ctx.surface_format());
        let window_input = app.window_input();
        self.state = Some(GameState {
            _engine: engine,
            app,
            runner,
            window_input,
            blit,
            target_size: (0, 0),
            color_format: ctx.surface_format(),
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
        let device = ctx.device().clone();
        let clear = self.config.clear_color;
        let (world, schedules) = state.app.parts_mut();

        ensure_camera_target(
            world,
            &device,
            width,
            height,
            state.color_format,
            &mut state.target_size,
            clear,
        );

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

/// Ensure the first camera has a [`CameraTarget`] sized to the window,
/// recreating the color/depth textures on resize.
fn ensure_camera_target(
    world: &mut World,
    device: &Arc<GraphicsDevice>,
    width: u32,
    height: u32,
    color_format: TextureFormat,
    size: &mut (u32, u32),
    clear_color: [f32; 4],
) {
    let width = width.max(1);
    let height = height.max(1);
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
    let stale = (width, height) != *size || world.get::<CameraTarget>(camera).is_none();
    if !stale {
        return;
    }

    let color = device
        .create_texture(
            &TextureDescriptor::new_2d(
                width,
                height,
                color_format,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            )
            .with_label("game_camera_color"),
        )
        .expect("failed to create game camera color target");
    let depth = device
        .create_texture(
            &TextureDescriptor::new_2d(
                width,
                height,
                TextureFormat::Depth32Float,
                TextureUsage::RENDER_ATTACHMENT,
            )
            .with_label("game_camera_depth"),
        )
        .expect("failed to create game camera depth target");
    let _ = world.insert(camera, CameraTarget::new(color, depth, clear_color));
    *size = (width, height);
}
