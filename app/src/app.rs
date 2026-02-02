//! Main application struct and event loop.

use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{Window, WindowId};

use redlilium_graphics::{GraphicsInstance, InstanceParameters, PresentMode, SurfaceConfiguration};

use crate::args::{AppArgs, WindowMode};
use crate::context::{AppContext, DrawContext};
use crate::handler::AppHandler;

/// Main application struct that manages the window and graphics.
///
/// The `App` struct is generic over:
/// - `H`: The handler type that implements [`AppHandler`]
/// - `A`: The arguments type that implements [`AppArgs`]
///
/// # Example
///
/// ```ignore
/// use redlilium_app::{App, AppHandler, DefaultAppArgs, DrawContext};
///
/// struct MyApp;
///
/// impl AppHandler for MyApp {
///     fn on_draw(&mut self, ctx: DrawContext) -> redlilium_graphics::FrameSchedule {
///         // Render frame
///         ctx.finish(&[])
///     }
/// }
///
/// fn main() {
///     let args = DefaultAppArgs::parse();
///     App::run(MyApp, args);
/// }
/// ```
pub struct App<H, A>
where
    H: AppHandler,
    A: AppArgs,
{
    handler: H,
    args: A,
    window: Option<Window>,
    context: Option<AppContext>,
    start_time: Instant,
    last_frame_time: Instant,
    running: bool,
    initialized: bool,
}

impl<H, A> App<H, A>
where
    H: AppHandler + 'static,
    A: AppArgs + 'static,
{
    /// Create a new application.
    pub fn new(handler: H, args: A) -> Self {
        Self {
            handler,
            args,
            window: None,
            context: None,
            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            running: true,
            initialized: false,
        }
    }

    /// Run the application with the given handler and arguments.
    ///
    /// This is the main entry point for the application. It creates the
    /// event loop, window, and graphics context, then runs the main loop.
    ///
    /// # Panics
    ///
    /// Panics if the event loop or window cannot be created.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(handler: H, args: A) {
        // Initialize logging
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

        redlilium_core::init();
        redlilium_graphics::init();
        crate::init();

        #[cfg(target_os = "windows")]
        let event_loop = EventLoop::builder()
            .with_any_thread(true)
            .build()
            .expect("Failed to create event loop");

        #[cfg(not(target_os = "windows"))]
        let event_loop = EventLoop::new().expect("Failed to create event loop");

        let mut app = Self::new(handler, args);
        event_loop.run_app(&mut app).expect("Event loop error");
    }

    /// Run the application (WASM version).
    #[cfg(target_arch = "wasm32")]
    pub fn run(handler: H, args: A) {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).expect("Failed to initialize logger");

        redlilium_core::init();
        redlilium_graphics::init();
        crate::init();

        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let mut app = Self::new(handler, args);
        event_loop.run_app(&mut app).expect("Event loop error");
    }

    /// Initialize graphics after window creation.
    fn init_graphics(&mut self) -> bool {
        let window = match &self.window {
            Some(w) => w,
            None => {
                log::error!("No window available for graphics init");
                return false;
            }
        };

        // Create graphics instance with parameters from args
        let params = InstanceParameters::new()
            .with_backend(self.args.backend())
            .with_wgpu_backend(self.args.wgpu_backend())
            .with_validation(self.args.validation());

        let instance = match GraphicsInstance::with_parameters(params) {
            Ok(i) => i,
            Err(e) => {
                log::error!("Failed to create graphics instance: {}", e);
                return false;
            }
        };

        // Create surface
        let surface = match instance.create_surface(window) {
            Ok(s) => s,
            Err(e) => {
                log::error!("Failed to create surface: {}", e);
                return false;
            }
        };

        // Create device compatible with the surface
        let device = match instance.create_device_for_surface(&surface) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Failed to create graphics device: {}", e);
                return false;
            }
        };

        // Configure surface
        let present_mode = if self.args.vsync() {
            PresentMode::Fifo
        } else {
            PresentMode::Immediate
        };

        let config = SurfaceConfiguration::new(self.args.window_width(), self.args.window_height())
            .with_format(surface.preferred_format())
            .with_present_mode(present_mode);

        if let Err(e) = surface.configure(&device, &config) {
            log::error!("Failed to configure surface: {}", e);
            return false;
        }

        // Create frame pipeline
        let pipeline = device.create_pipeline(2);

        log::info!(
            "Graphics initialized: {} ({}x{})",
            device.name(),
            self.args.window_width(),
            self.args.window_height()
        );

        self.context = Some(AppContext {
            instance,
            device,
            surface,
            pipeline,
            width: self.args.window_width(),
            height: self.args.window_height(),
            frame_number: 0,
            delta_time: 0.0,
            elapsed_time: 0.0,
        });

        true
    }

    /// Handle resize.
    fn handle_resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        if let Some(ctx) = &mut self.context {
            if ctx.width == width && ctx.height == height {
                return;
            }

            ctx.width = width;
            ctx.height = height;

            // Wait for current slot before reconfiguring
            ctx.pipeline.wait_current_slot();

            // Reconfigure surface
            let present_mode = if self.args.vsync() {
                PresentMode::Fifo
            } else {
                PresentMode::Immediate
            };

            let config = SurfaceConfiguration::new(width, height)
                .with_format(ctx.surface.preferred_format())
                .with_present_mode(present_mode);

            if let Err(e) = ctx.surface.configure(&ctx.device, &config) {
                log::error!("Failed to reconfigure surface: {}", e);
            }

            // Notify handler
            self.handler.on_resize(ctx);
        }
    }

    /// Render a frame.
    fn render_frame(&mut self) {
        let now = Instant::now();
        let delta_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // We need to split the borrow of self to handle the handler and context separately
        let ctx = match &mut self.context {
            Some(c) => c,
            None => return,
        };

        ctx.delta_time = delta_time;
        ctx.elapsed_time = now.duration_since(self.start_time).as_secs_f32();

        // Call update
        if !self.handler.on_update(ctx) {
            self.running = false;
            return;
        }

        // Acquire swapchain texture
        let swapchain_texture = match ctx.surface.acquire_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Failed to acquire swapchain texture: {}", e);
                return;
            }
        };

        // Begin frame
        let schedule = ctx.pipeline.begin_frame();

        // Create draw context
        let draw_ctx = DrawContext {
            app: ctx,
            schedule,
            swapchain_texture,
        };

        // Call draw - handler returns the schedule after finishing
        let schedule = self.handler.on_draw(draw_ctx);

        // End frame with the returned schedule
        if let Some(ctx) = &mut self.context {
            ctx.pipeline.end_frame(schedule);
            ctx.frame_number += 1;

            // Check max frames limit
            if let Some(max_frames) = self.args.max_frames()
                && ctx.frame_number >= max_frames
            {
                log::info!("Reached max frames limit ({}), exiting", max_frames);
                self.running = false;
            }
        }
    }
}

impl<H, A> ApplicationHandler for App<H, A>
where
    H: AppHandler + 'static,
    A: AppArgs + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            // Create window
            let mut window_attributes = Window::default_attributes()
                .with_title(self.args.window_title())
                .with_inner_size(winit::dpi::LogicalSize::new(
                    self.args.window_width(),
                    self.args.window_height(),
                ));

            // Apply window mode
            match self.args.window_mode() {
                WindowMode::Windowed => {}
                WindowMode::Borderless => {
                    window_attributes = window_attributes
                        .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                }
                WindowMode::Fullscreen => {
                    // Get primary monitor's video mode
                    if let Some(monitor) = event_loop.primary_monitor()
                        && let Some(video_mode) = monitor.video_modes().next()
                    {
                        window_attributes = window_attributes.with_fullscreen(Some(
                            winit::window::Fullscreen::Exclusive(video_mode),
                        ));
                    }
                }
            }

            match event_loop.create_window(window_attributes) {
                Ok(window) => {
                    log::info!("Window created");
                    self.window = Some(window);

                    if !self.init_graphics() {
                        log::error!("Failed to initialize graphics");
                        event_loop.exit();
                        return;
                    }

                    // Initialize handler
                    if let Some(ctx) = &mut self.context {
                        self.handler.on_init(ctx);
                    }
                    self.initialized = true;
                }
                Err(e) => {
                    log::error!("Failed to create window: {}", e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested");
                self.running = false;

                // Shutdown handler
                if let Some(ctx) = &mut self.context {
                    self.handler.on_shutdown(ctx);
                }

                // Wait for GPU before exiting
                if let Some(ctx) = &self.context {
                    ctx.pipeline.wait_idle();
                }

                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                self.handle_resize(size.width, size.height);
            }

            WindowEvent::RedrawRequested => {
                if self.initialized && self.running {
                    self.render_frame();
                }

                if !self.running {
                    event_loop.exit();
                } else if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(ctx) = &mut self.context {
                    self.handler.on_key(ctx, &event);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Some(ctx) = &mut self.context {
                    self.handler.on_mouse_move(ctx, position.x, position.y);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(ctx) = &mut self.context {
                    let pressed = state == ElementState::Pressed;
                    self.handler.on_mouse_button(ctx, button, pressed);
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(ctx) = &mut self.context {
                    let (dx, dy) = match delta {
                        MouseScrollDelta::LineDelta(x, y) => (x, y),
                        MouseScrollDelta::PixelDelta(pos) => (pos.x as f32, pos.y as f32),
                    };
                    self.handler.on_mouse_scroll(ctx, dx, dy);
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
