//! Command line arguments trait and default implementation.
//!
//! Uses clap for proper CLI parsing on native targets with:
//! - Help text (`--help`)
//! - Validation and clear error messages
//! - Platform-specific backend availability warnings

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

// ============================================================================
// CLI Backend Selection (clap enums with clearer naming)
// ============================================================================

/// Graphics backend selection for CLI.
///
/// This enum provides clearer naming for CLI users than the internal `BackendType`.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum CliBackend {
    /// Automatically select the best available backend (wgpu preferred).
    #[default]
    Auto,
    /// Cross-platform backend via wgpu (recommended for most use cases).
    /// Supports full rendering features including draw commands.
    Wgpu,
    /// Native Vulkan backend via ash (advanced/experimental).
    /// Note: Currently limited to transfer operations, no draw commands.
    #[value(name = "vulkan-native")]
    VulkanNative,
    /// No-op backend for testing and CI environments.
    Dummy,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<CliBackend> for BackendType {
    fn from(cli: CliBackend) -> Self {
        match cli {
            CliBackend::Auto => BackendType::Auto,
            CliBackend::Wgpu => BackendType::Wgpu,
            CliBackend::VulkanNative => BackendType::Vulkan,
            CliBackend::Dummy => BackendType::Dummy,
        }
    }
}

/// GPU API selection for wgpu backend.
///
/// Controls which underlying graphics API wgpu uses.
/// Only relevant when using the wgpu backend.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum CliGpuApi {
    /// Platform-appropriate default:
    /// - macOS/iOS: Metal
    /// - Linux: Vulkan
    /// - Windows: DirectX 12
    /// - Web: WebGL
    #[default]
    Auto,
    /// Vulkan (available on Linux, Windows, Android).
    #[cfg(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "android",
        target_os = "freebsd"
    ))]
    Vulkan,
    /// Metal (available on macOS and iOS only).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Metal,
    /// DirectX 12 (available on Windows only).
    #[cfg(target_os = "windows")]
    Dx12,
    /// OpenGL/WebGL (cross-platform fallback).
    Gl,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<CliGpuApi> for WgpuBackendType {
    fn from(cli: CliGpuApi) -> Self {
        match cli {
            CliGpuApi::Auto => WgpuBackendType::Auto,
            #[cfg(any(
                target_os = "linux",
                target_os = "windows",
                target_os = "android",
                target_os = "freebsd"
            ))]
            CliGpuApi::Vulkan => WgpuBackendType::Vulkan,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            CliGpuApi::Metal => WgpuBackendType::Metal,
            #[cfg(target_os = "windows")]
            CliGpuApi::Dx12 => WgpuBackendType::Dx12,
            CliGpuApi::Gl => WgpuBackendType::Gl,
        }
    }
}

// ============================================================================
// Default App Args (with clap on native)
// ============================================================================

/// Default command line arguments implementation.
///
/// On native platforms, uses clap for proper CLI parsing with help text.
/// On WASM, uses simple environment-based defaults.
///
/// # Examples
///
/// ```bash
/// # Show help
/// ./my_app --help
///
/// # Use wgpu with Vulkan API
/// ./my_app --backend wgpu --gpu-api vulkan
///
/// # Use native Vulkan (limited features)
/// ./my_app --backend vulkan-native
///
/// # Run in fullscreen with validation disabled
/// ./my_app --fullscreen --no-validation
///
/// # Run for 100 frames then exit (useful for testing)
/// ./my_app --max-frames 100
/// ```
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

    /// Set the window title.
    pub fn with_title_str(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }
}

// ============================================================================
// Native implementation using clap
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::*;
    use clap::Parser;

    /// RedLilium Engine application arguments.
    #[derive(Parser, Debug)]
    #[command(
        name = "RedLilium App",
        about = "RedLilium Engine application",
        long_about = "A graphics application powered by the RedLilium Engine.\n\n\
            BACKEND SELECTION:\n\
            The engine supports multiple graphics backends:\n\
            \n\
            • wgpu (recommended): Cross-platform abstraction supporting Metal, Vulkan, DX12, OpenGL.\n\
              Use --gpu-api to select the underlying API.\n\
            \n\
            • vulkan-native: Direct Vulkan via ash. Currently limited to transfer operations.\n\
              Only use this for advanced debugging or development purposes.\n\
            \n\
            • dummy: No-op backend for testing without GPU.\n\
            \n\
            EXAMPLES:\n\
              # Use wgpu with platform default (recommended)\n\
              ./app --backend wgpu\n\
            \n\
              # Use wgpu with explicit Vulkan API\n\
              ./app --backend wgpu --gpu-api vulkan\n\
            \n\
              # Run headless test\n\
              ./app --backend dummy --max-frames 10",
        version
    )]
    pub(super) struct ClapArgs {
        /// Graphics backend to use.
        #[arg(long, default_value = "auto", value_enum)]
        pub backend: CliBackend,

        /// GPU API for wgpu backend.
        /// Only applies when --backend is 'wgpu' or 'auto'.
        #[arg(long, default_value = "auto", value_enum)]
        pub gpu_api: CliGpuApi,

        /// Run in borderless fullscreen mode.
        #[arg(long)]
        pub fullscreen: bool,

        /// Initial window width in pixels.
        #[arg(long, default_value = "1280")]
        pub width: u32,

        /// Initial window height in pixels.
        #[arg(long, default_value = "720")]
        pub height: u32,

        /// Disable vertical sync (may cause tearing).
        #[arg(long)]
        pub no_vsync: bool,

        /// Exit after rendering N frames (useful for testing).
        #[arg(long)]
        pub max_frames: Option<u64>,

        /// Enable GPU validation layers (slower but helps catch bugs).
        #[arg(long, conflicts_with = "no_validation")]
        pub validation: bool,

        /// Disable GPU validation layers (faster but less safe).
        #[arg(long, conflicts_with = "validation")]
        pub no_validation: bool,
    }

    impl From<ClapArgs> for DefaultAppArgs {
        fn from(args: ClapArgs) -> Self {
            // Warn if gpu-api is set but backend is not wgpu
            if args.gpu_api != CliGpuApi::Auto
                && args.backend != CliBackend::Wgpu
                && args.backend != CliBackend::Auto
            {
                log::warn!(
                    "--gpu-api has no effect when --backend is '{:?}'. \
                    The --gpu-api option only applies to the wgpu backend.",
                    args.backend
                );
            }

            // Warn about vulkan-native limitations
            if args.backend == CliBackend::VulkanNative {
                log::warn!(
                    "Using vulkan-native backend. Note: This backend currently only \
                    supports transfer operations (no draw commands). \
                    Consider using '--backend wgpu --gpu-api vulkan' for full Vulkan support."
                );
            }

            // Determine validation setting: explicit flags override default
            // --validation forces on, --no-validation forces off, otherwise use debug default
            let validation = args.validation || (!args.no_validation && cfg!(debug_assertions));

            Self {
                backend: args.backend.into(),
                wgpu_backend: args.gpu_api.into(),
                window_mode: if args.fullscreen {
                    WindowMode::Borderless
                } else {
                    WindowMode::Windowed
                },
                width: args.width,
                height: args.height,
                title: "RedLilium App".to_string(),
                vsync: !args.no_vsync,
                max_frames: args.max_frames,
                validation,
            }
        }
    }
}

impl AppArgs for DefaultAppArgs {
    fn parse() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use clap::Parser;
            let clap_args = native::ClapArgs::parse();
            clap_args.into()
        }

        #[cfg(target_arch = "wasm32")]
        {
            // On WASM, just use defaults (no CLI)
            Self::default()
        }
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

// ============================================================================
// Backward compatibility: keep the old parsing function available
// ============================================================================

/// Parse wgpu backend type from string (for backward compatibility).
///
/// This function is kept for custom AppArgs implementations that may want
/// to parse backend strings manually.
#[deprecated(since = "0.2.0", note = "Use CliGpuApi with clap instead")]
#[allow(dead_code)]
pub fn parse_wgpu_backend(value: &str) -> WgpuBackendType {
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

        "gl" | "opengl" => WgpuBackendType::Gl,

        _ => {
            if value != "auto" {
                log::warn!(
                    "Unknown GPU API '{}', falling back to auto. \
                    Valid options on this platform: auto, gl{}",
                    value,
                    platform_gpu_apis()
                );
            }
            WgpuBackendType::Auto
        }
    }
}

/// Parse backend type from string (for backward compatibility).
///
/// This function is kept for custom AppArgs implementations that may want
/// to parse backend strings manually.
#[deprecated(since = "0.2.0", note = "Use CliBackend with clap instead")]
#[allow(dead_code)]
pub fn parse_backend(value: &str) -> BackendType {
    match value.to_lowercase().as_str() {
        "wgpu" => BackendType::Wgpu,
        "vulkan" | "vulkan-native" => BackendType::Vulkan,
        "dummy" => BackendType::Dummy,
        "auto" => BackendType::Auto,
        _ => {
            log::warn!(
                "Unknown backend '{}', falling back to auto. \
                Valid options: auto, wgpu, vulkan-native, dummy",
                value
            );
            BackendType::Auto
        }
    }
}

/// Get a string listing available GPU APIs for the current platform.
#[allow(dead_code)]
fn platform_gpu_apis() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        ", metal"
    }
    #[cfg(target_os = "windows")]
    {
        ", vulkan, dx12"
    }
    #[cfg(target_os = "linux")]
    {
        ", vulkan"
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        ""
    }
}
