//! Main application struct and event loop.

use std::collections::VecDeque;
use std::sync::Arc;

use web_time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
#[cfg(target_os = "macos")]
use winit::platform::macos::WindowAttributesExtMacOS;
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
use crate::pacing::FramePacer;

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
    window: Option<Arc<Window>>,
    context: Option<AppContext>,
    /// A graphics instance created before the window (wasm/WebGPU, #33): device
    /// creation is async there and must finish before `spawn_app`, so it is
    /// built up front and consumed by `init_graphics`. `None` on native, where
    /// `init_graphics` creates the instance synchronously.
    prebuilt_instance: Option<Arc<GraphicsInstance>>,
    start_time: Instant,
    last_frame_time: Instant,
    /// Turns the CPU-sampled frame delta into the interval the frame is
    /// actually displayed for (see [`crate::pacing`]).
    pacer: FramePacer,
    /// Last few raw frame deltas, logged around a hitch so the cadence that
    /// led up to it (present-queue refill bursts) is visible at debug level.
    recent_raw_deltas: VecDeque<f32>,
    /// Previous frame's phase timings in ms — update, acquire, fence wait,
    /// draw+present — logged with a hitch to show where that frame slept.
    last_phase_ms: [f32; 4],
    /// Frames since the display period was last read from the monitor.
    frames_since_monitor_probe: u32,
    running: bool,
    initialized: bool,
}

/// How often to re-read the monitor's refresh rate, in frames — often enough to
/// follow a monitor or mode change, rarely enough to stay off the frame path.
const MONITOR_PROBE_INTERVAL: u32 = 60;

/// Raw deltas kept for the hitch log — long enough to show the cadence around
/// a hitch (the frame before it and the queue-refill burst after a previous
/// one), short enough to read in a log line.
const RAW_DELTA_LOG_WINDOW: usize = 8;

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
            prebuilt_instance: None,
            start_time: Instant::now(),
            last_frame_time: Instant::now(),
            pacer: FramePacer::new(),
            recent_raw_deltas: VecDeque::with_capacity(RAW_DELTA_LOG_WINDOW),
            last_phase_ms: [0.0; 4],
            frames_since_monitor_probe: MONITOR_PROBE_INTERVAL,
            running: true,
            initialized: false,
        }
    }

    /// Create an application with a graphics instance built ahead of the window.
    ///
    /// Used by the wasm entry (#33), where the wgpu device is created inside an
    /// async task before `spawn_app`; `init_graphics` then reuses this instance
    /// instead of creating one (which would need a blocking request the browser
    /// thread cannot make).
    pub fn new_with_instance(handler: H, args: A, instance: Arc<GraphicsInstance>) -> Self {
        Self {
            prebuilt_instance: Some(instance),
            ..Self::new(handler, args)
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
        // Use try_init so callers can set a custom logger before App::run.
        let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init();

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
    ///
    /// The browser thread cannot block, so device creation is done in an async
    /// task first (#33); only once the instance exists do we build the event loop
    /// and hand the app to winit's non-blocking `spawn_app` (which drives frames
    /// via `requestAnimationFrame`). `run_app` would panic on web.
    #[cfg(target_arch = "wasm32")]
    pub fn run(handler: H, args: A) {
        use winit::platform::web::EventLoopExtWebSys;

        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).expect("Failed to initialize logger");

        redlilium_core::init();
        redlilium_graphics::init();
        crate::init();

        wasm_bindgen_futures::spawn_local(async move {
            let params = InstanceParameters::new()
                .with_backend(args.backend())
                .with_wgpu_backend(args.wgpu_backend())
                .with_validation(args.validation());

            let instance = match GraphicsInstance::with_parameters_async(params).await {
                Ok(i) => i,
                Err(e) => {
                    log::error!("Failed to create graphics instance: {e}");
                    return;
                }
            };

            let event_loop = EventLoop::new().expect("Failed to create event loop");
            let app = Self::new_with_instance(handler, args, instance);
            // Non-blocking: returns immediately, driving the app from rAF.
            event_loop.spawn_app(app);
        });
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

        // Reuse an instance built ahead of the window (wasm async path, #33), or
        // create one synchronously (native).
        let instance = match self.prebuilt_instance.take() {
            Some(instance) => instance,
            None => {
                let params = InstanceParameters::new()
                    .with_backend(self.args.backend())
                    .with_wgpu_backend(self.args.wgpu_backend())
                    .with_validation(self.args.validation());
                match GraphicsInstance::with_parameters(params) {
                    Ok(i) => i,
                    Err(e) => {
                        log::error!("Failed to create graphics instance: {}", e);
                        return false;
                    }
                }
            }
        };

        // Create surface (owned Arc<Window> so the surface keeps the window alive)
        let surface = match instance.create_surface(window.clone()) {
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
        // On web the canvas can be 0×0 before the page's first layout, and WebGPU
        // rejects a zero-sized `configure`. Clamp to at least 1×1; the first real
        // `Resized` event corrects it through the frame-0 immediate-resize path
        // (#33).
        let width = physical_size.width.max(1);
        let height = physical_size.height.max(1);

        // Configure surface with physical dimensions
        let present_mode = if self.args.vsync() {
            PresentMode::Fifo
        } else {
            PresentMode::Immediate
        };

        // Determine surface format - use HDR if requested and supported
        let (surface_format, hdr_active) = self.select_surface_format(&surface);

        let config = SurfaceConfiguration::new(width, height)
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
            width,
            height,
            scale_factor,
            surface_format,
            hdr_active
        );

        let resize_manager = ResizeManager::new(
            (width, height),
            self.args.resize_debounce_ms(),
            ResizeStrategy::Stretch,
        );

        self.context = Some(AppContext {
            window: self.window.as_ref().unwrap().clone(),
            custom_titlebar: self.args.custom_titlebar(),
            instance,
            device,
            surface,
            pipeline,
            width,
            height,
            scale_factor,
            frame_number: 0,
            delta_time: 0.0,
            raw_delta_time: 0.0,
            elapsed_time: 0.0,
            surface_format,
            hdr_active,
            resize_manager,
            surface_outdated: false,
        });

        true
    }

    /// Select the surface format: HDR preferred over sRGB (on by default,
    /// `--no-hdr` opts out), falling back to the sRGB SDR format.
    ///
    /// Only `Rgba16Float` qualifies as HDR here — it pairs with the
    /// extended-sRGB-linear color space, the one contract the render path can
    /// drive (shaders output linear with 1.0 = SDR white). 10-bit formats the
    /// display may also offer (`Rgba10a2Unorm` = HDR10) require a PQ encode
    /// no shader performs yet, so auto-selecting them would render wrong
    /// colors — they are deliberately skipped.
    /// Returns a (format, hdr_active) tuple.
    fn select_surface_format(&self, surface: &Surface) -> (TextureFormat, bool) {
        if self.args.hdr() {
            let hdr_formats = surface.supported_hdr_formats();
            if hdr_formats.contains(&TextureFormat::Rgba16Float) {
                log::info!("HDR surface: Rgba16Float (extended linear)");
                return (TextureFormat::Rgba16Float, true);
            }
            if !hdr_formats.is_empty() {
                log::info!(
                    "display offers HDR formats {hdr_formats:?} but not Rgba16Float; \
                     the render path has no PQ encode — using SDR"
                );
            }
        }

        // Standard SDR format (sRGB-typed when available).
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
        {
            let ctx = match &mut self.context {
                Some(c) => c,
                None => return,
            };

            if ctx.width == width && ctx.height == height {
                return;
            }

            ctx.width = width;
            ctx.height = height;
        }

        if !self.reconfigure_surface() {
            return;
        }

        // Notify handler
        if let Some(ctx) = &mut self.context {
            self.handler.on_resize(ctx);
        }
    }

    /// Reconfigure the surface with the current context dimensions.
    ///
    /// Used both for resize events and for `SurfaceOutdated` reports from
    /// acquire/present (where the size may be unchanged but the swapchain no
    /// longer matches the surface). Returns `false` only if the surface was
    /// not touched at all (no context, or in-flight work could not be
    /// drained); a failed `configure` still returns `true` so callers behave
    /// as after any other reconfigure — a failing surface reports outdated on
    /// the next acquire and is retried there.
    fn reconfigure_surface(&mut self) -> bool {
        let ctx = match &mut self.context {
            Some(c) => c,
            None => return false,
        };

        // Wait for ALL in-flight frames before reconfiguring the surface.
        // The surface is shared across all frame slots, so any slot with
        // pending GPU work could still reference the old swapchain textures.
        // On failure the GPU may still be using them — skip the reconfigure
        // entirely (the next resize event or outdated report retries) rather
        // than recycling graphs the GPU still reads.
        //
        // wasm: `wait_idle` blocks on a fence whose completion callback can only
        // fire from the event loop we're on — a guaranteed hang. WebGPU has no
        // persistent back-buffer set (each frame's texture comes from
        // `getCurrentTexture` and is released after present), and destroying a
        // resource the GPU still reads is safe by construction there, so
        // reconfiguring without draining is fine. The per-slot fence in
        // `try_begin_frame` still gates re-recording a pooled graph. (#33)
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(e) = ctx.pipeline.wait_idle() {
            log::error!("wait_idle failed during resize, skipping surface reconfigure: {e}");
            return false;
        }

        // Recycle all submitted graphs to release Arc<TextureView> references
        // to the old swapchain back buffers.
        ctx.pipeline.recycle_all_graphs();

        // Reconfigure surface with the same format
        let present_mode = if self.args.vsync() {
            PresentMode::Fifo
        } else {
            PresentMode::Immediate
        };

        let config = SurfaceConfiguration::new(ctx.width, ctx.height)
            .with_format(ctx.surface_format)
            .with_present_mode(present_mode);

        ctx.surface_outdated = false;
        if let Err(e) = ctx.surface.configure(&ctx.device, &config) {
            log::error!("Failed to reconfigure surface: {}", e);
        }
        true
    }

    /// Render a frame.
    fn render_frame(&mut self) {
        // Apply any pending debounced resize before rendering
        self.apply_pending_resize();

        // If the previous frame reported the surface outdated (resize,
        // monitor change), recreate the swapchain before acquiring.
        if self.context.as_ref().is_some_and(|c| c.surface_outdated) {
            self.reconfigure_surface();
        }

        let now = Instant::now();
        let raw_delta = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // Re-read the display period now and then (monitor or mode change, and
        // it is not worth a query every frame).
        self.frames_since_monitor_probe += 1;
        if self.frames_since_monitor_probe >= MONITOR_PROBE_INTERVAL {
            self.frames_since_monitor_probe = 0;
            let probe_start = Instant::now();
            let period = self
                .window
                .as_ref()
                .and_then(|w| w.current_monitor())
                .and_then(|m| m.refresh_rate_millihertz())
                .filter(|mhz| *mhz > 0)
                .map(|mhz| 1000.0 / mhz as f32);
            self.pacer.set_display_period(period);
            // The probe sits on the frame path once a second; if the OS query
            // ever gets expensive it becomes a recurring hitch seed.
            let probe_ms = probe_start.elapsed().as_secs_f32() * 1000.0;
            if probe_ms > 1.0 {
                log::debug!("monitor refresh-rate probe took {probe_ms:.2} ms");
            }
        }

        // What the simulation needs is the interval this frame is *displayed*
        // for, not the interval the CPU measured between frame starts. Under
        // vsync those differ by up to ±13% frame to frame, which shows up as
        // the world subtly speeding up and slowing down (see `pacing`).
        self.pacer.set_vsync(self.args.vsync());
        let delta_time = self.pacer.pace(raw_delta);

        // Hitch forensics: one debug line with the cadence leading up to a
        // long frame — enough to spot present-queue refill bursts (short raw
        // deltas trailing a hitch, see `pacing`) without a profiler attached.
        if let Some(period) = self.pacer.display_period()
            && raw_delta > period * 1.5
        {
            let recent_ms: Vec<f32> = self
                .recent_raw_deltas
                .iter()
                .map(|d| (d * 1e5).round() / 100.0)
                .collect();
            let [update, acquire, fence, draw] = self.last_phase_ms;
            log::debug!(
                "frame hitch: raw delta {:.2} ms vs {:.2} ms display period; \
                 preceding raw deltas {recent_ms:?} ms; previous frame spent \
                 update {update:.2} / acquire {acquire:.2} / fence {fence:.2} \
                 / draw+present {draw:.2} ms",
                raw_delta * 1000.0,
                period * 1000.0,
            );
        }
        if self.recent_raw_deltas.len() == RAW_DELTA_LOG_WINDOW {
            self.recent_raw_deltas.pop_front();
        }
        self.recent_raw_deltas.push_back(raw_delta);

        // We need to split the borrow of self to handle the handler and context separately
        let ctx = match &mut self.context {
            Some(c) => c,
            None => return,
        };

        ctx.delta_time = delta_time;
        ctx.raw_delta_time = raw_delta;
        ctx.elapsed_time = now.duration_since(self.start_time).as_secs_f32();

        // Call update
        let phase_start = Instant::now();
        if !self.handler.on_update(ctx) {
            self.running = false;
            return;
        }
        self.last_phase_ms[0] = phase_start.elapsed().as_secs_f32() * 1000.0;

        // Acquire the swapchain texture and begin the frame. The order differs by
        // platform. On native, acquire first so a `SurfaceOutdated` is caught
        // before the blocking `begin_frame` (which is also the frame pacing). On
        // wasm, check the frame slot (non-blocking) FIRST and skip the rAF tick if
        // it isn't ready — `getCurrentTexture` followed by no present flashes the
        // canvas transparent-black, so we must not acquire on a skipped tick (#33).
        #[cfg(not(target_arch = "wasm32"))]
        let (swapchain_texture, schedule) = {
            let phase_start = Instant::now();
            let swapchain_texture = match ctx.surface.acquire_texture() {
                Ok(t) => t,
                Err(
                    e @ (redlilium_graphics::GraphicsError::SurfaceOutdated
                    | redlilium_graphics::GraphicsError::SurfaceLost),
                ) => {
                    // Skip this frame; the swapchain is recreated at the top of
                    // the next one.
                    log::debug!("Surface needs reconfiguration: {e}");
                    ctx.surface_outdated = true;
                    return;
                }
                Err(e) => {
                    log::warn!("Failed to acquire swapchain texture: {}", e);
                    return;
                }
            };
            self.last_phase_ms[1] = phase_start.elapsed().as_secs_f32() * 1000.0;

            // On fence-wait failure (GPU hang, device lost) the slot's resources
            // must not be recycled — skip this frame; the fence stays in its slot
            // and the next frame retries the wait.
            let phase_start = Instant::now();
            let schedule = match ctx.pipeline.begin_frame() {
                Ok(schedule) => schedule,
                Err(e) => {
                    log::error!("begin_frame failed, skipping frame: {e}");
                    return;
                }
            };
            self.last_phase_ms[2] = phase_start.elapsed().as_secs_f32() * 1000.0;
            (swapchain_texture, schedule)
        };

        #[cfg(target_arch = "wasm32")]
        let (swapchain_texture, schedule) = {
            // Non-blocking: if this slot's previous GPU work isn't finished, skip
            // this tick (completion arrives via the fence callback on a later rAF).
            let schedule = match ctx.pipeline.try_begin_frame() {
                Ok(Some(schedule)) => schedule,
                Ok(None) => return,
                Err(e) => {
                    log::error!("try_begin_frame failed, skipping frame: {e}");
                    return;
                }
            };

            let swapchain_texture = match ctx.surface.acquire_texture() {
                Ok(t) => t,
                Err(
                    e @ (redlilium_graphics::GraphicsError::SurfaceOutdated
                    | redlilium_graphics::GraphicsError::SurfaceLost),
                ) => {
                    log::debug!("Surface needs reconfiguration: {e}");
                    ctx.surface_outdated = true;
                    return;
                }
                Err(e) => {
                    log::warn!("Failed to acquire swapchain texture: {}", e);
                    return;
                }
            };
            (swapchain_texture, schedule)
        };

        // Create draw context
        let draw_ctx = DrawContext {
            app: ctx,
            schedule,
            swapchain_texture,
        };

        // Call draw - handler returns the schedule after finishing
        let phase_start = Instant::now();
        let schedule = self.handler.on_draw(draw_ctx);

        // End frame with the returned schedule
        if let Some(ctx) = &mut self.context {
            ctx.pipeline.end_frame(schedule);
            self.last_phase_ms[3] = phase_start.elapsed().as_secs_f32() * 1000.0;
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
            let mut window_attributes =
                Window::default_attributes().with_title(self.args.window_title());
            // On web the page owns the canvas layout: an explicit inner_size makes winit
            // PIN the canvas CSS size (it stops tracking the container), so an embedded
            // canvas (e.g. a VSCode webview panel) renders at a fixed 1280×720 patch that
            // ignores the panel's real shape and resizes. Without it, winit follows the
            // page CSS via its ResizeObserver and reports honest `Resized` events.
            #[cfg(not(target_arch = "wasm32"))]
            {
                window_attributes =
                    window_attributes.with_inner_size(winit::dpi::LogicalSize::new(
                        self.args.window_width(),
                        self.args.window_height(),
                    ));
            }

            // Apply custom titlebar
            if self.args.custom_titlebar() {
                #[cfg(target_os = "macos")]
                {
                    window_attributes = window_attributes
                        .with_titlebar_transparent(true)
                        .with_title_hidden(true)
                        .with_fullsize_content_view(true);
                }

                #[cfg(not(target_os = "macos"))]
                {
                    window_attributes = window_attributes.with_decorations(false);
                }
            }

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

            // On web, attach the winit window to the page's existing canvas
            // (winit does NOT insert a canvas into the DOM), so the WebGPU
            // surface renders where the page laid it out (#33).
            #[cfg(target_arch = "wasm32")]
            let window_attributes = {
                use wasm_bindgen::JsCast;
                use winit::platform::web::WindowAttributesExtWebSys;

                let canvas = web_sys::window()
                    .and_then(|w| w.document())
                    .and_then(|d| d.get_element_by_id("redlilium-canvas"))
                    .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok());
                if canvas.is_none() {
                    log::error!(
                        "canvas #redlilium-canvas not found; add <canvas id=\"redlilium-canvas\"> \
                         to the page"
                    );
                }
                window_attributes.with_canvas(canvas)
            };

            match event_loop.create_window(window_attributes) {
                Ok(window) => {
                    log::info!("Window created");
                    self.window = Some(Arc::new(window));

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

                // Let the handler decide whether to close immediately.
                let allow_close = self
                    .context
                    .as_mut()
                    .is_none_or(|ctx| self.handler.on_close_requested(ctx));

                if allow_close {
                    self.running = false;

                    // Shutdown handler
                    if let Some(ctx) = &mut self.context {
                        self.handler.on_shutdown(ctx);
                    }

                    // Wait for GPU before exiting; on failure we exit anyway —
                    // the process is going down. wasm: a browser tab does not
                    // really "close" here, and `wait_idle` would hang on a fence
                    // serviced only by the event loop we're in — so drain on
                    // native only (#33).
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(ctx) = &self.context
                        && let Err(e) = ctx.pipeline.wait_idle()
                    {
                        log::error!("wait_idle failed during shutdown: {e}");
                    }

                    event_loop.exit();
                }
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

            WindowEvent::ModifiersChanged(modifiers) => {
                if let Some(ctx) = &mut self.context {
                    self.handler.on_modifiers_changed(ctx, modifiers.state());
                }
            }

            WindowEvent::DroppedFile(path) => {
                if let Some(ctx) = &mut self.context {
                    self.handler.on_file_dropped(ctx, path);
                }
            }

            WindowEvent::HoveredFile(path) => {
                if let Some(ctx) = &mut self.context {
                    self.handler.on_file_hovered(ctx, path);
                }
            }

            WindowEvent::HoveredFileCancelled => {
                if let Some(ctx) = &mut self.context {
                    self.handler.on_file_hover_cancelled(ctx);
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
