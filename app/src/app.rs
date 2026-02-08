//! Main application struct and event loop.

use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{Window, WindowId};

use redlilium_graphics::{
    GraphicsInstance, InstanceParameters, PresentMode, ResizeManager, ResizeStrategy, Surface,
    SurfaceConfiguration, TextureFormat,
};

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

        // Get scale factor and physical size from window
        let scale_factor = window.scale_factor();
        let physical_size = window.inner_size();

        // Configure surface with physical dimensions
        let present_mode = if self.args.vsync() {
            PresentMode::Fifo
        } else {
            PresentMode::Immediate
        };

        // Determine surface format - use HDR if requested and supported
        let (surface_format, hdr_active) = self.select_surface_format(&surface);

        let config = SurfaceConfiguration::new(physical_size.width, physical_size.height)
            .with_format(surface_format)
            .with_present_mode(present_mode);

        if let Err(e) = surface.configure(&device, &config) {
            log::error!("Failed to configure surface: {}", e);
            return false;
        }

        // Create frame pipeline
        let pipeline = device.create_pipeline(2);

        log::info!(
            "Graphics initialized: {} ({}x{} physical, scale_factor={}, format={:?}, hdr={})",
            device.name(),
            physical_size.width,
            physical_size.height,
            scale_factor,
            surface_format,
            hdr_active
        );

        let resize_manager = ResizeManager::new(
            (physical_size.width, physical_size.height),
            self.args.resize_debounce_ms(),
            ResizeStrategy::Stretch,
        );

        self.context = Some(AppContext {
            instance,
            device,
            surface,
            pipeline,
            width: physical_size.width,
            height: physical_size.height,
            scale_factor,
            frame_number: 0,
            delta_time: 0.0,
            elapsed_time: 0.0,
            surface_format,
            hdr_active,
            resize_manager,
        });

        true
    }

    /// Select the surface format based on HDR preference and availability.
    ///
    /// Returns (format, hdr_active) tuple.
    fn select_surface_format(&self, surface: &Surface) -> (TextureFormat, bool) {
        if self.args.hdr() {
            // Try to use HDR format
            let supported = surface.supported_formats();
            let hdr_formats = surface.supported_hdr_formats();

            if !hdr_formats.is_empty() {
                // Prefer Rgba10a2Unorm (HDR10) as it's widely supported
                let preferred = surface.preferred_hdr_format();
                if supported.contains(&preferred) {
                    log::info!("HDR enabled: using {:?}", preferred);
                    return (preferred, true);
                }
                // Fall back to first available HDR format
                let format = hdr_formats[0];
                log::info!("HDR enabled: using {:?}", format);
                return (format, true);
            }

            log::warn!("HDR requested but no HDR formats available, falling back to SDR");
        }

        // Use standard SDR format
        let format = surface.preferred_format();
        log::info!("Using SDR format: {:?}", format);
        (format, false)
    }

    /// Handle a resize event from the OS.
    ///
    /// Before any frame has been rendered, the resize is applied immediately
    /// (matching the pre-debounce behavior) so the swapchain size is correct
    /// for the first frame. After that, events are buffered in the
    /// ResizeManager and applied after the debounce period.
    fn handle_resize_event(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        // Before the first frame, apply immediately so the initial window
        // size is correct for on_init resources. This matches the old
        // immediate-resize behavior for startup.
        let first_frame = self.context.as_ref().is_some_and(|c| c.frame_number == 0);
        if first_frame {
            self.apply_resize(width, height);
            // Sync resize manager's internal size so it doesn't re-apply later
            if let Some(ctx) = &mut self.context {
                ctx.resize_manager.on_resize_event(width, height);
                ctx.resize_manager.force_resize();
            }
            return;
        }

        if let Some(ctx) = &mut self.context {
            ctx.resize_manager.on_resize_event(width, height);
        }
    }

    /// Apply any pending debounced resize.
    ///
    /// Called at the top of [`render_frame`] to check if the debounce period has
    /// elapsed and a resize should be applied.
    fn apply_pending_resize(&mut self) {
        let (width, height) = {
            let ctx = match &mut self.context {
                Some(c) => c,
                None => return,
            };
            match ctx.resize_manager.update() {
                Some(e) => (e.width, e.height),
                None => return,
            }
        };
        self.apply_resize(width, height);
    }

    /// Reconfigure the swapchain and notify the handler of a resize.
    fn apply_resize(&mut self, width: u32, height: u32) {
        let ctx = match &mut self.context {
            Some(c) => c,
            None => return,
        };

        if ctx.width == width && ctx.height == height {
            return;
        }

        ctx.width = width;
        ctx.height = height;

        // Wait for current slot before reconfiguring
        ctx.pipeline.wait_current_slot();

        // Reconfigure surface with the same format
        let present_mode = if self.args.vsync() {
            PresentMode::Fifo
        } else {
            PresentMode::Immediate
        };

        let config = SurfaceConfiguration::new(width, height)
            .with_format(ctx.surface_format)
            .with_present_mode(present_mode);

        if let Err(e) = ctx.surface.configure(&ctx.device, &config) {
            log::error!("Failed to reconfigure surface: {}", e);
        }

        // Notify handler
        self.handler.on_resize(ctx);
    }

    /// Render a frame.
    fn render_frame(&mut self) {
        // Apply any pending debounced resize before rendering
        self.apply_pending_resize();

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
                self.handle_resize_event(size.width, size.height);
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(ctx) = &mut self.context {
                    ctx.scale_factor = scale_factor;
                    log::info!("Scale factor changed to {}", scale_factor);
                }
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
