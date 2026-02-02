//! Command line arguments trait and default implementation.

use redlilium_graphics::{BackendType, WgpuBackendType};

/// Window mode enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowMode {
    /// Windowed mode with decorations.
    #[default]
    Windowed,
    /// Borderless fullscreen.
    Borderless,
    /// Exclusive fullscreen.
    Fullscreen,
}

/// Trait for parsing command line arguments.
///
/// Implement this trait to customize how your application handles
/// command line arguments. The trait provides defaults for all methods,
/// making it easy to override only the options you need.
///
/// # Example
///
/// ```ignore
/// use redlilium_app::{AppArgs, WindowMode};
/// use redlilium_graphics::BackendType;
///
/// struct MyArgs {
///     fullscreen: bool,
///     scene_path: String,
/// }
///
/// impl AppArgs for MyArgs {
///     fn parse() -> Self {
///         let args: Vec<String> = std::env::args().collect();
///         Self {
///             fullscreen: args.contains(&"--fullscreen".to_string()),
///             scene_path: args.get(1).cloned().unwrap_or_default(),
///         }
///     }
///
///     fn window_mode(&self) -> WindowMode {
///         if self.fullscreen {
///             WindowMode::Borderless
///         } else {
///             WindowMode::Windowed
///         }
///     }
/// }
/// ```
pub trait AppArgs: Sized {
    /// Parse command line arguments.
    fn parse() -> Self;

    /// Get the graphics backend to use.
    ///
    /// Default: `BackendType::Auto` (automatically select best available)
    fn backend(&self) -> BackendType {
        BackendType::Auto
    }

    /// Get the wgpu backend type (when using wgpu backend).
    ///
    /// Default: `WgpuBackendType::Auto`
    fn wgpu_backend(&self) -> WgpuBackendType {
        WgpuBackendType::Auto
    }

    /// Get the window mode.
    ///
    /// Default: `WindowMode::Windowed`
    fn window_mode(&self) -> WindowMode {
        WindowMode::Windowed
    }

    /// Get the initial window width.
    ///
    /// Default: 1280
    fn window_width(&self) -> u32 {
        1280
    }

    /// Get the initial window height.
    ///
    /// Default: 720
    fn window_height(&self) -> u32 {
        720
    }

    /// Get the window title.
    ///
    /// Default: "RedLilium App"
    fn window_title(&self) -> &str {
        "RedLilium App"
    }

    /// Get whether VSync is enabled.
    ///
    /// Default: true
    fn vsync(&self) -> bool {
        true
    }

    /// Get the maximum number of frames to process before auto-exit.
    ///
    /// This is useful for AI agents and automated testing to verify
    /// that the application can start and render without errors.
    ///
    /// Default: `None` (run indefinitely)
    fn max_frames(&self) -> Option<u64> {
        None
    }

    /// Get whether validation layers should be enabled.
    ///
    /// Default: `cfg!(debug_assertions)` (enabled in debug builds)
    fn validation(&self) -> bool {
        cfg!(debug_assertions)
    }
}

/// Default command line arguments implementation.
///
/// Parses standard command line arguments:
/// - `--backend=<vulkan|wgpu|dummy>` - Graphics backend
/// - `--wgpu-backend=<vulkan|metal|dx12|gl|auto>` - wgpu backend type
/// - `--fullscreen` - Run in borderless fullscreen
/// - `--width=<N>` - Window width
/// - `--height=<N>` - Window height
/// - `--no-vsync` - Disable VSync
/// - `--max-frames=<N>` - Exit after N frames
/// - `--validation` - Enable validation layers
/// - `--no-validation` - Disable validation layers
#[derive(Debug, Clone)]
pub struct DefaultAppArgs {
    backend: BackendType,
    wgpu_backend: WgpuBackendType,
    window_mode: WindowMode,
    width: u32,
    height: u32,
    title: String,
    vsync: bool,
    max_frames: Option<u64>,
    validation: bool,
}

impl Default for DefaultAppArgs {
    fn default() -> Self {
        Self {
            backend: BackendType::Auto,
            wgpu_backend: WgpuBackendType::Auto,
            window_mode: WindowMode::Windowed,
            width: 1280,
            height: 720,
            title: "RedLilium App".to_string(),
            vsync: true,
            max_frames: None,
            validation: cfg!(debug_assertions),
        }
    }
}

impl DefaultAppArgs {
    /// Create new default args with a custom title.
    pub fn with_title(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    /// Set the window size.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the graphics backend.
    pub fn with_backend(mut self, backend: BackendType) -> Self {
        self.backend = backend;
        self
    }

    /// Set the maximum number of frames.
    pub fn with_max_frames(mut self, max_frames: u64) -> Self {
        self.max_frames = Some(max_frames);
        self
    }
}

impl AppArgs for DefaultAppArgs {
    fn parse() -> Self {
        let mut args = Self::default();
        let env_args: Vec<String> = std::env::args().collect();

        for arg in &env_args[1..] {
            if let Some(value) = arg.strip_prefix("--backend=") {
                args.backend = match value.to_lowercase().as_str() {
                    "vulkan" => BackendType::Vulkan,
                    "wgpu" => BackendType::Wgpu,
                    "dummy" => BackendType::Dummy,
                    _ => BackendType::Auto,
                };
            } else if let Some(value) = arg.strip_prefix("--wgpu-backend=") {
                // Note: WgpuBackendType variants are platform-conditional
                // Just parse what we can and fallback to Auto
                args.wgpu_backend = parse_wgpu_backend(value);
            } else if arg == "--fullscreen" {
                args.window_mode = WindowMode::Borderless;
            } else if let Some(value) = arg.strip_prefix("--width=") {
                if let Ok(w) = value.parse() {
                    args.width = w;
                }
            } else if let Some(value) = arg.strip_prefix("--height=") {
                if let Ok(h) = value.parse() {
                    args.height = h;
                }
            } else if arg == "--no-vsync" {
                args.vsync = false;
            } else if let Some(value) = arg.strip_prefix("--max-frames=") {
                if let Ok(n) = value.parse() {
                    args.max_frames = Some(n);
                }
            } else if arg == "--validation" {
                args.validation = true;
            } else if arg == "--no-validation" {
                args.validation = false;
            }
        }

        args
    }

    fn backend(&self) -> BackendType {
        self.backend
    }

    fn wgpu_backend(&self) -> WgpuBackendType {
        self.wgpu_backend
    }

    fn window_mode(&self) -> WindowMode {
        self.window_mode
    }

    fn window_width(&self) -> u32 {
        self.width
    }

    fn window_height(&self) -> u32 {
        self.height
    }

    fn window_title(&self) -> &str {
        &self.title
    }

    fn vsync(&self) -> bool {
        self.vsync
    }

    fn max_frames(&self) -> Option<u64> {
        self.max_frames
    }

    fn validation(&self) -> bool {
        self.validation
    }
}

/// Parse wgpu backend type from string, handling platform-specific variants.
fn parse_wgpu_backend(value: &str) -> WgpuBackendType {
    match value.to_lowercase().as_str() {
        #[cfg(any(
            target_os = "linux",
            target_os = "windows",
            target_os = "android",
            target_os = "freebsd"
        ))]
        "vulkan" => WgpuBackendType::Vulkan,

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        "metal" => WgpuBackendType::Metal,

        #[cfg(target_os = "windows")]
        "dx12" => WgpuBackendType::Dx12,

        #[cfg(not(target_arch = "wasm32"))]
        "gl" | "opengl" => WgpuBackendType::Gl,

        _ => WgpuBackendType::Auto,
    }
}
