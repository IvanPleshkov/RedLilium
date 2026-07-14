//! Graphics instance.
//!
//! The [`GraphicsInstance`] is the top-level entry point for the graphics system.
//! It manages one or more [`GraphicsDevice`]s.

use std::sync::{Arc, RwLock, Weak};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::backend;
use crate::device::GraphicsDevice;
use crate::error::GraphicsError;
use crate::swapchain::Surface;

// ============================================================================
// Instance Parameters
// ============================================================================

/// Backend selection for the graphics instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendType {
    /// Automatically select the best available backend: native Vulkan (via
    /// ash) on every desktop platform including macOS (MoltenVK), falling back
    /// to wgpu and then dummy only if Vulkan cannot be created. wgpu is never
    /// the default here — request it explicitly with [`Wgpu`](Self::Wgpu).
    #[default]
    Auto,
    /// Use the dummy backend (no actual GPU operations).
    Dummy,
    /// Use the wgpu backend (cross-platform via wgpu).
    Wgpu,
    /// Use the native Vulkan backend (via ash).
    Vulkan,
}

/// wgpu-specific backend selection.
///
/// When using the wgpu backend, this controls which underlying graphics API is used.
/// Backend variants are conditionally compiled based on target platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WgpuBackendType {
    /// Automatically select the best available backend for the current platform:
    /// - macOS: Metal
    /// - Linux: Vulkan
    /// - Windows: DX12
    /// - Web: WebGL
    #[default]
    Auto,
    /// Use Vulkan backend (Linux, Windows, Android).
    #[cfg(any(
        target_os = "linux",
        target_os = "windows",
        target_os = "android",
        target_os = "freebsd"
    ))]
    Vulkan,
    /// Use DirectX 12 backend (Windows only).
    #[cfg(target_os = "windows")]
    Dx12,
    /// Use Metal backend (macOS/iOS only).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Metal,
    /// Use OpenGL/WebGL backend.
    Gl,
    /// Use WebGPU backend (browser only).
    #[cfg(target_arch = "wasm32")]
    WebGpu,
}

impl WgpuBackendType {
    /// Convert to wgpu::Backends flags.
    ///
    /// For `Auto` mode, selects the platform-appropriate backend:
    /// - macOS: Metal
    /// - Linux: Vulkan
    /// - Windows: DX12
    /// - Web: WebGL (GL backend)
    #[cfg(feature = "wgpu-backend")]
    pub(crate) fn to_wgpu_backends(self) -> wgpu::Backends {
        match self {
            #[cfg(target_os = "macos")]
            Self::Auto => wgpu::Backends::METAL,
            #[cfg(target_os = "linux")]
            Self::Auto => wgpu::Backends::VULKAN,
            #[cfg(target_os = "windows")]
            Self::Auto => wgpu::Backends::DX12,
            // Web target is WebGPU-only (#33): the wgpu backend exists for the
            // browser, and WebGL2 has no compute and caps limits well below a
            // HiDPI canvas. Auto on wasm selects WebGPU, not GL (the WG-L7 bug).
            #[cfg(target_arch = "wasm32")]
            Self::Auto => wgpu::Backends::BROWSER_WEBGPU,
            // Fallback for other platforms
            #[cfg(not(any(
                target_os = "macos",
                target_os = "linux",
                target_os = "windows",
                target_arch = "wasm32"
            )))]
            Self::Auto => wgpu::Backends::all(),

            #[cfg(any(
                target_os = "linux",
                target_os = "windows",
                target_os = "android",
                target_os = "freebsd"
            ))]
            Self::Vulkan => wgpu::Backends::VULKAN,

            #[cfg(target_os = "windows")]
            Self::Dx12 => wgpu::Backends::DX12,

            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Self::Metal => wgpu::Backends::METAL,

            Self::Gl => wgpu::Backends::GL,

            #[cfg(target_arch = "wasm32")]
            Self::WebGpu => wgpu::Backends::BROWSER_WEBGPU,
        }
    }
}

/// Stable identity of an adapter for explicit selection (ADR-028).
///
/// Never an enumeration index: indices change with driver updates, eGPU
/// hotplug, and enumeration order. PCI ids survive all of those; a name
/// substring is the human-friendly fallback (config files, env override).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterId {
    /// PCI `vendor:device` pair (hex in the textual form, e.g. `1002:744c`).
    Pci { vendor_id: u32, device_id: u32 },
    /// Case-insensitive substring of the adapter name (e.g. `radeon`).
    NameContains(String),
}

impl AdapterId {
    /// Parse the textual form used by config values and `REDLILIUM_ADAPTER`:
    /// `vvvv:dddd` (hex PCI ids) or any other string as a name substring.
    pub fn parse(s: &str) -> Self {
        if let Some((v, d)) = s.split_once(':')
            && let (Ok(vendor_id), Ok(device_id)) = (
                u32::from_str_radix(v.trim_start_matches("0x"), 16),
                u32::from_str_radix(d.trim_start_matches("0x"), 16),
            )
        {
            return Self::Pci {
                vendor_id,
                device_id,
            };
        }
        Self::NameContains(s.to_string())
    }

    /// Whether an adapter with these ids/name matches this identity.
    pub fn matches(&self, vendor_id: u32, device_id: u32, name: &str) -> bool {
        match self {
            Self::Pci {
                vendor_id: v,
                device_id: d,
            } => *v == vendor_id && *d == device_id,
            Self::NameContains(needle) => name.to_lowercase().contains(&needle.to_lowercase()),
        }
    }
}

/// Which adapter to select at instance creation (ADR-028).
///
/// A declared *policy*, not an index — the backend resolves it against
/// whatever the machine actually has. Selection happens once, at instance
/// creation; changing it requires recreating the instance (in practice: an
/// application restart), which is the industry-standard contract.
///
/// The `REDLILIUM_ADAPTER` environment variable (PCI `vvvv:dddd` or a name
/// substring) overrides whatever the application configured — for CI,
/// multi-GPU dev machines, and support scenarios.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AdapterPreference {
    /// Same as [`HighPerformance`](Self::HighPerformance) — the default:
    /// predictable capabilities everywhere, no editor-vs-game GPU split.
    #[default]
    Auto,
    /// Prefer the discrete GPU.
    HighPerformance,
    /// Prefer the integrated GPU (battery life over throughput). A deliberate
    /// opt-in at graphics initialization, never chosen automatically.
    LowPower,
    /// Exactly this adapter; instance creation fails loudly if it is absent
    /// or below the baseline tier (ADR-027).
    Explicit(AdapterId),
}

/// Whether the Vulkan backend records per-pass GPU crash breadcrumbs (#97).
///
/// Breadcrumbs mark which pass the GPU was executing when a
/// `VK_ERROR_DEVICE_LOST` is reported, turning a bare `DeviceLost` into a
/// named "the GPU died in pass X" post-mortem. They add one marker write per
/// pass; the happy path with them off adds zero encode work.
///
/// `Auto` (the default) follows validation — on for dev/editor builds (where
/// validation is enabled), off otherwise. The `REDLILIUM_BREADCRUMBS`
/// environment variable overrides whatever the application configured
/// (`1`/`true`/`on` → on, `0`/`false`/`off` → off), mirroring the
/// `REDLILIUM_ADAPTER` precedent (ADR-028).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BreadcrumbsMode {
    /// Follow validation: on when validation is enabled, off otherwise.
    #[default]
    Auto,
    /// Always record breadcrumbs.
    On,
    /// Never record breadcrumbs.
    Off,
}

impl BreadcrumbsMode {
    /// Resolve to an on/off decision given whether validation is enabled.
    pub fn resolve(self, validation: bool) -> bool {
        match self {
            Self::Auto => validation,
            Self::On => true,
            Self::Off => false,
        }
    }
}

/// Configuration parameters for creating a graphics instance.
///
/// Use the builder pattern to configure the instance:
///
/// ```ignore
/// let params = InstanceParameters::new()
///     .with_backend(BackendType::Wgpu)
///     .with_wgpu_backend(WgpuBackendType::Auto)
///     .with_validation(true)
///     .with_debug(true);
///
/// let instance = GraphicsInstance::new(params)?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct InstanceParameters {
    /// Which backend to use.
    pub backend: BackendType,
    /// Which wgpu backend to use (only relevant when backend is Wgpu or Auto).
    pub wgpu_backend: WgpuBackendType,
    /// Enable GPU validation layers for debugging.
    pub validation: bool,
    /// Enable debug mode (additional logging, debug names).
    pub debug: bool,
    /// Which adapter to select (ADR-028). Overridden by `REDLILIUM_ADAPTER`.
    pub adapter: AdapterPreference,
    /// Per-pass GPU crash breadcrumbs (#97). Overridden by
    /// `REDLILIUM_BREADCRUMBS`.
    pub breadcrumbs: BreadcrumbsMode,
}

impl InstanceParameters {
    /// Create new default instance parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the backend type to use.
    pub fn with_backend(mut self, backend: BackendType) -> Self {
        self.backend = backend;
        self
    }

    /// Set the wgpu backend type (only used when backend is Wgpu or Auto).
    pub fn with_wgpu_backend(mut self, wgpu_backend: WgpuBackendType) -> Self {
        self.wgpu_backend = wgpu_backend;
        self
    }

    /// Enable or disable GPU validation layers.
    ///
    /// Validation layers help catch API misuse but have a performance cost.
    /// Recommended for development, disabled for release builds.
    pub fn with_validation(mut self, validation: bool) -> Self {
        self.validation = validation;
        self
    }

    /// Enable or disable debug mode.
    ///
    /// Debug mode enables additional logging and debug names for resources.
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Set the adapter selection policy (ADR-028).
    pub fn with_adapter_preference(mut self, adapter: AdapterPreference) -> Self {
        self.adapter = adapter;
        self
    }

    /// Set the GPU crash breadcrumb mode (#97).
    pub fn with_breadcrumbs(mut self, breadcrumbs: BreadcrumbsMode) -> Self {
        self.breadcrumbs = breadcrumbs;
        self
    }

    /// Apply the `REDLILIUM_ADAPTER` and `REDLILIUM_BREADCRUMBS` environment
    /// overrides, if set.
    ///
    /// Called by instance creation so the overrides work regardless of how
    /// the application built its parameters.
    fn with_env_overrides(mut self) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(value) = std::env::var("REDLILIUM_ADAPTER")
            && !value.is_empty()
        {
            let id = AdapterId::parse(&value);
            log::info!("Adapter override via REDLILIUM_ADAPTER: {value} -> {id:?}");
            self.adapter = AdapterPreference::Explicit(id);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(value) = std::env::var("REDLILIUM_BREADCRUMBS") {
            match value.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "on" | "yes" => {
                    log::info!("GPU breadcrumbs forced ON via REDLILIUM_BREADCRUMBS");
                    self.breadcrumbs = BreadcrumbsMode::On;
                }
                "0" | "false" | "off" | "no" => {
                    log::info!("GPU breadcrumbs forced OFF via REDLILIUM_BREADCRUMBS");
                    self.breadcrumbs = BreadcrumbsMode::Off;
                }
                "" => {}
                other => log::warn!(
                    "ignoring unrecognized REDLILIUM_BREADCRUMBS value {other:?} \
                     (expected 1/true/on or 0/false/off)"
                ),
            }
        }
        self
    }
}

/// Information about a graphics adapter.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    /// Adapter name.
    pub name: String,
    /// Adapter vendor.
    pub vendor: String,
    /// Device type (discrete, integrated, etc.).
    pub device_type: AdapterType,
    /// PCI vendor id (0 when the backend has none, e.g. dummy).
    pub vendor_id: u32,
    /// PCI device id (0 when the backend has none, e.g. dummy).
    pub device_id: u32,
}

impl AdapterInfo {
    /// The stable identity of this adapter for [`AdapterPreference::Explicit`]
    /// (persist this in settings, not an enumeration index).
    pub fn id(&self) -> AdapterId {
        if self.vendor_id != 0 || self.device_id != 0 {
            AdapterId::Pci {
                vendor_id: self.vendor_id,
                device_id: self.device_id,
            }
        } else {
            AdapterId::NameContains(self.name.clone())
        }
    }
}

/// Type of graphics adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdapterType {
    /// Discrete GPU (dedicated graphics card).
    Discrete,
    /// Integrated GPU (shared with CPU).
    Integrated,
    /// Software renderer.
    Software,
    /// Unknown adapter type.
    Unknown,
}

/// Human-readable vendor name for a PCI vendor id, used by backends when
/// filling [`AdapterInfo`]. Unknown ids format as `vendor 0x….`.
pub(crate) fn vendor_name(id: u32) -> String {
    match id {
        0x1002 => "AMD".to_string(),
        0x10DE => "NVIDIA".to_string(),
        0x8086 => "Intel".to_string(),
        0x106B => "Apple".to_string(),
        0x13B5 => "ARM".to_string(),
        0x5143 => "Qualcomm".to_string(),
        0x1010 => "Imagination".to_string(),
        0x10005 => "Mesa".to_string(),
        _ => format!("vendor {id:#06x}"),
    }
}

/// The graphics instance manages devices and adapters.
///
/// This is the top-level entry point for the graphics system. Create an instance
/// to enumerate available adapters and create devices.
///
/// # Thread Safety
///
/// `GraphicsInstance` is `Send + Sync` and can be safely shared across threads.
///
/// # Example
///
/// ```ignore
/// let instance = GraphicsInstance::new()?;
/// let device = instance.create_device()?;
/// ```
pub struct GraphicsInstance {
    /// Weak self-reference for creating devices.
    self_ref: RwLock<Weak<GraphicsInstance>>,
    /// Number of currently alive [`GraphicsDevice`]s created by this
    /// instance (incremented on create, decremented by the device's `Drop`).
    /// Only used as a guard signal (see `create_device_for_surface`); the
    /// instance deliberately holds NO references to its devices.
    live_devices: std::sync::atomic::AtomicUsize,
    /// GPU backend for this instance (wrapped in RwLock for interior mutability).
    ///
    /// OWNERSHIP: the instance does not own or track the devices it creates —
    /// ownership flows strictly upward (resources → `GraphicsDevice` →
    /// `GraphicsInstance`), Vulkan-style. Each device holds a strong `Arc`
    /// back to the instance, so the backend outlives everything created from
    /// it and drops (running real teardown: `device_wait_idle`, pool/cache
    /// destruction) once the last device and resource are gone. An
    /// instance→device link would recreate the reference cycle fixed in #50.
    backend: RwLock<backend::GpuBackend>,
}

impl GraphicsInstance {
    /// Create a new graphics instance with default parameters.
    ///
    /// This is equivalent to `GraphicsInstance::with_parameters(InstanceParameters::default())`.
    ///
    /// # Errors
    ///
    /// Returns an error if the graphics system cannot be initialized.
    pub fn new() -> Result<Arc<Self>, GraphicsError> {
        Self::with_parameters(InstanceParameters::default())
    }

    /// Create a new graphics instance with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - Configuration parameters for the instance
    ///
    /// # Errors
    ///
    /// Returns an error if the graphics system cannot be initialized.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use redlilium_graphics::{GraphicsInstance, InstanceParameters, BackendType};
    ///
    /// let params = InstanceParameters::new()
    ///     .with_backend(BackendType::Wgpu)
    ///     .with_validation(true);
    ///
    /// let instance = GraphicsInstance::with_parameters(params)?;
    /// ```
    pub fn with_parameters(params: InstanceParameters) -> Result<Arc<Self>, GraphicsError> {
        let params = params.with_env_overrides();
        log::info!("Creating GraphicsInstance with params: {:?}", params);

        // Create the GPU backend based on parameters
        let backend = backend::create_backend_with_params(&params)?;
        Ok(Self::from_backend(backend))
    }

    /// Create a graphics instance, awaiting device creation.
    ///
    /// The browser entry point (#33): on wasm the wgpu adapter/device requests
    /// are JS promises that only resolve while the event loop runs, so instance
    /// creation cannot block. Call this from an async task
    /// (`wasm_bindgen_futures::spawn_local`) before starting the window loop.
    /// Equivalent to [`with_parameters`](Self::with_parameters) on native.
    #[cfg(feature = "wgpu-backend")]
    pub async fn with_parameters_async(
        params: InstanceParameters,
    ) -> Result<Arc<Self>, GraphicsError> {
        let params = params.with_env_overrides();
        log::info!(
            "Creating GraphicsInstance (async) with params: {:?}",
            params
        );
        let backend = backend::create_backend_with_params_async(&params).await?;
        Ok(Self::from_backend(backend))
    }

    /// Wrap a created backend in a reference-counted instance with its weak
    /// self-reference wired up. Shared by the blocking and async constructors.
    fn from_backend(backend: backend::GpuBackend) -> Arc<Self> {
        log::info!("Using GPU backend: {}", backend.name());

        let instance = Arc::new(Self {
            self_ref: RwLock::new(Weak::new()),
            live_devices: std::sync::atomic::AtomicUsize::new(0),
            backend: RwLock::new(backend),
        });

        // Store self-reference
        if let Ok(mut self_ref) = instance.self_ref.write() {
            *self_ref = Arc::downgrade(&instance);
        }

        instance
    }

    /// Get the GPU backend for reading (internal use only).
    pub(crate) fn backend(&self) -> std::sync::RwLockReadGuard<'_, backend::GpuBackend> {
        self.backend.read().expect("backend lock poisoned")
    }

    /// Get the GPU backend for writing (internal use only).
    pub(crate) fn backend_mut(&self) -> std::sync::RwLockWriteGuard<'_, backend::GpuBackend> {
        self.backend.write().expect("backend lock poisoned")
    }

    /// Get the strong self-reference.
    fn arc_self(&self) -> Option<Arc<GraphicsInstance>> {
        self.self_ref.read().ok().and_then(|r| r.upgrade())
    }

    /// Enumerate available graphics adapters.
    ///
    /// Returns the adapter the backend was created on. The backend commits to
    /// one physical device at instance creation, so today this is always a
    /// single-element list with honest name/vendor/type — full multi-GPU
    /// enumeration and selection is future work (#38 / XB-L3).
    pub fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        vec![self.backend().adapter_info()]
    }

    /// Create a graphics device.
    ///
    /// Creates a device using the default (best available) adapter.
    ///
    /// # Errors
    ///
    /// Returns an error if device creation fails.
    pub fn create_device(&self) -> Result<Arc<GraphicsDevice>, GraphicsError> {
        self.create_device_with_adapter(0)
    }

    /// Create a graphics device with a specific adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter_index` - Index of the adapter to use (from `enumerate_adapters`)
    ///
    /// # Errors
    ///
    /// Returns an error if the adapter index is invalid or device creation fails.
    pub fn create_device_with_adapter(
        &self,
        adapter_index: usize,
    ) -> Result<Arc<GraphicsDevice>, GraphicsError> {
        let adapters = self.enumerate_adapters();
        if adapter_index >= adapters.len() {
            return Err(GraphicsError::InvalidParameter(format!(
                "adapter index {adapter_index} out of range ({})",
                adapters.len()
            )));
        }

        let adapter = &adapters[adapter_index];
        log::info!("Creating device on adapter: {}", adapter.name);

        let instance = self.arc_self().ok_or_else(|| {
            GraphicsError::ResourceCreationFailed("instance has been dropped".to_string())
        })?;
        // The device owns a strong Arc to the instance; the instance keeps no
        // reference back (see the `backend` field docs), only a live counter.
        self.live_devices
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Arc::new(GraphicsDevice::new(
            instance,
            adapter.name.clone(),
        )))
    }

    /// Called by [`GraphicsDevice`]'s `Drop`.
    pub(crate) fn device_dropped(&self) {
        self.live_devices
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Number of currently alive devices created by this instance.
    pub fn device_count(&self) -> usize {
        self.live_devices.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Create a surface for presenting to a window.
    ///
    /// # Arguments
    ///
    /// * `window` - A window that implements the raw-window-handle traits
    ///
    /// # Errors
    ///
    /// Returns an error if surface creation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let surface = instance.create_surface(&window)?;
    /// surface.configure(&device, &SurfaceConfiguration::new(800, 600));
    /// ```
    #[allow(clippy::arc_with_non_send_sync)] // Surface is intentionally !Send+!Sync for window safety
    pub fn create_surface<W>(&self, window: Arc<W>) -> Result<Arc<Surface>, GraphicsError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        let instance = self.arc_self().ok_or_else(|| {
            GraphicsError::ResourceCreationFailed("instance has been dropped".to_string())
        })?;

        let surface = Surface::new(instance, window)?;
        Ok(Arc::new(surface))
    }

    /// Create a graphics device that is compatible with the given surface.
    ///
    /// This method ensures that the selected adapter/device can present to the
    /// given surface. On some systems, not all adapters can present to all surfaces.
    ///
    /// # Arguments
    ///
    /// * `surface` - The surface that the device must be compatible with
    ///
    /// # Errors
    ///
    /// Returns an error if no compatible adapter is found or device creation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let surface = instance.create_surface(&window)?;
    /// let device = instance.create_device_for_surface(&surface)?;
    /// surface.configure(&device, &SurfaceConfiguration::new(800, 600))?;
    /// ```
    pub fn create_device_for_surface(
        &self,
        surface: &Surface,
    ) -> Result<Arc<GraphicsDevice>, GraphicsError> {
        // Ensure the backend is compatible with the surface
        {
            let mut backend = self.backend_mut();

            // On wgpu an incompatible adapter is fixed by swapping the
            // backend's device — safe only while nothing has been created
            // yet. Resources made through an existing GraphicsDevice belong
            // to the old wgpu device; after a swap the first draw touching
            // them is a cross-device validation error. Refuse instead.
            if self.device_count() > 0 && !backend.surface_compatible(surface.gpu_surface()) {
                return Err(GraphicsError::InvalidParameter(
                    "surface requires a different GPU adapter, but graphics devices (and \
                     potentially their resources) already exist on the current one; create \
                     the surface and its device before any other device"
                        .to_string(),
                ));
            }

            backend.ensure_compatible_with_surface(surface.gpu_surface())?;
        }

        // Now create the device using the (potentially updated) backend
        self.create_device()
    }
}

impl std::fmt::Debug for GraphicsInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphicsInstance")
            .field("device_count", &self.device_count())
            .finish()
    }
}

// Ensure GraphicsInstance is Send + Sync
static_assertions::assert_impl_all!(GraphicsInstance: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_creation() {
        let instance = GraphicsInstance::new().unwrap();
        assert_eq!(instance.device_count(), 0);
    }

    #[test]
    fn test_enumerate_adapters() {
        let instance = GraphicsInstance::new().unwrap();
        let adapters = instance.enumerate_adapters();
        assert!(!adapters.is_empty());
    }

    /// ADR-028: the textual adapter identity — PCI pair when it parses as
    /// `hex:hex`, name substring otherwise.
    #[test]
    fn test_adapter_id_parse() {
        assert_eq!(
            AdapterId::parse("1002:744c"),
            AdapterId::Pci {
                vendor_id: 0x1002,
                device_id: 0x744c
            }
        );
        assert_eq!(
            AdapterId::parse("0x10de:0x2684"),
            AdapterId::Pci {
                vendor_id: 0x10de,
                device_id: 0x2684
            }
        );
        assert_eq!(
            AdapterId::parse("Radeon"),
            AdapterId::NameContains("Radeon".to_string())
        );
        // A colon with non-hex parts is a name, not a broken PCI id.
        assert_eq!(
            AdapterId::parse("weird:name"),
            AdapterId::NameContains("weird:name".to_string())
        );
    }

    #[test]
    fn test_adapter_id_matches() {
        let pci = AdapterId::parse("1002:744c");
        assert!(pci.matches(0x1002, 0x744c, "whatever"));
        assert!(!pci.matches(0x1002, 0x744d, "whatever"));

        let name = AdapterId::parse("radeon");
        assert!(name.matches(0, 0, "AMD Radeon RX 9070 XT"));
        assert!(!name.matches(0, 0, "NVIDIA GeForce RTX 4090"));
    }

    /// `AdapterInfo::id()` prefers the stable PCI pair; falls back to the
    /// name only when the backend reports no ids (dummy).
    #[test]
    fn test_adapter_info_id() {
        let real = AdapterInfo {
            name: "Some GPU".to_string(),
            vendor: "AMD".to_string(),
            device_type: AdapterType::Discrete,
            vendor_id: 0x1002,
            device_id: 0x744c,
        };
        assert_eq!(
            real.id(),
            AdapterId::Pci {
                vendor_id: 0x1002,
                device_id: 0x744c
            }
        );

        let dummy = AdapterInfo {
            name: "Dummy Adapter".to_string(),
            vendor: "RedLilium".to_string(),
            device_type: AdapterType::Software,
            vendor_id: 0,
            device_id: 0,
        };
        assert_eq!(
            dummy.id(),
            AdapterId::NameContains("Dummy Adapter".to_string())
        );
    }

    #[test]
    fn test_create_device() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        // The device carries the real adapter's name (XB-L3), which matches
        // whatever `enumerate_adapters` honestly reports.
        assert_eq!(device.name(), instance.enumerate_adapters()[0].name);
        assert_eq!(instance.device_count(), 1);
    }

    #[test]
    fn test_create_multiple_devices() {
        let instance = GraphicsInstance::new().unwrap();
        let _device1 = instance.create_device().unwrap();
        let _device2 = instance.create_device().unwrap();
        assert_eq!(instance.device_count(), 2);
    }

    #[test]
    fn test_invalid_adapter_index() {
        let instance = GraphicsInstance::new().unwrap();
        let result = instance.create_device_with_adapter(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_device_has_instance_reference() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        // Device holds a strong reference to instance
        assert!(Arc::ptr_eq(device.instance(), &instance));
    }

    /// Regression test for #50: the instance must hold no references to its
    /// devices, or the instance↔device cycle keeps both alive forever and
    /// backend teardown (`device_wait_idle`, pool/cache destruction) becomes
    /// unreachable dead code.
    #[test]
    fn test_no_instance_device_cycle() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let weak_instance = Arc::downgrade(&instance);
        let weak_device = Arc::downgrade(&device);

        drop(device);
        assert!(weak_device.upgrade().is_none(), "device leaked");
        assert_eq!(instance.device_count(), 0);

        drop(instance);
        assert!(
            weak_instance.upgrade().is_none(),
            "instance leaked — something re-introduced an instance→device reference"
        );
    }
}
