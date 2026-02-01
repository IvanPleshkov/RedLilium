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
    /// Automatically select the best available backend.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WgpuBackendType {
    /// Automatically select the best available backend.
    #[default]
    Auto,
    /// Use Vulkan backend.
    Vulkan,
    /// Use DirectX 12 backend (Windows only).
    Dx12,
    /// Use Metal backend (macOS/iOS only).
    Metal,
    /// Use OpenGL backend.
    Gl,
    /// Use WebGPU backend (browser only).
    WebGpu,
}

impl WgpuBackendType {
    /// Convert to wgpu::Backends flags.
    #[cfg(feature = "wgpu-backend")]
    pub(crate) fn to_wgpu_backends(self) -> wgpu::Backends {
        match self {
            Self::Auto => wgpu::Backends::all(),
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Metal => wgpu::Backends::METAL,
            Self::Gl => wgpu::Backends::GL,
            Self::WebGpu => wgpu::Backends::BROWSER_WEBGPU,
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
///     .with_wgpu_backend(WgpuBackendType::Vulkan)
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
    /// Devices created by this instance.
    devices: RwLock<Vec<Arc<GraphicsDevice>>>,
    /// GPU backend for this instance.
    backend: backend::GpuBackend,
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
        log::info!("Creating GraphicsInstance with params: {:?}", params);

        // Create the GPU backend based on parameters
        let backend = backend::create_backend_with_params(&params)?;
        log::info!("Using GPU backend: {}", backend.name());

        let instance = Arc::new(Self {
            self_ref: RwLock::new(Weak::new()),
            devices: RwLock::new(Vec::new()),
            backend,
        });

        // Store self-reference
        if let Ok(mut self_ref) = instance.self_ref.write() {
            *self_ref = Arc::downgrade(&instance);
        }

        Ok(instance)
    }

    /// Get the GPU backend (internal use only).
    pub(crate) fn backend(&self) -> &backend::GpuBackend {
        &self.backend
    }

    /// Get the strong self-reference.
    fn arc_self(&self) -> Option<Arc<GraphicsInstance>> {
        self.self_ref.read().ok().and_then(|r| r.upgrade())
    }

    /// Enumerate available graphics adapters.
    ///
    /// Returns information about all available graphics adapters on the system.
    #[cfg(feature = "dummy")]
    pub fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        // Dummy implementation returns a single software adapter
        vec![AdapterInfo {
            name: "Dummy Adapter".to_string(),
            vendor: "RedLilium".to_string(),
            device_type: AdapterType::Software,
        }]
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
        let device = Arc::new(GraphicsDevice::new(instance, adapter.name.clone()));

        // Track the device
        if let Ok(mut devices) = self.devices.write() {
            devices.push(device.clone());
        }

        Ok(device)
    }

    /// Get all devices created by this instance.
    pub fn devices(&self) -> Vec<Arc<GraphicsDevice>> {
        self.devices
            .read()
            .map(|d| d.clone())
            .unwrap_or_else(|_| Vec::new())
    }

    /// Get the number of devices created by this instance.
    pub fn device_count(&self) -> usize {
        self.devices.read().map(|d| d.len()).unwrap_or(0)
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
    pub fn create_surface<W>(&self, window: &W) -> Result<Arc<Surface>, GraphicsError>
    where
        W: HasWindowHandle + HasDisplayHandle + Sync,
    {
        let instance = self.arc_self().ok_or_else(|| {
            GraphicsError::ResourceCreationFailed("instance has been dropped".to_string())
        })?;

        let surface = Surface::new(instance, window)?;
        Ok(Arc::new(surface))
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

    #[test]
    fn test_create_device() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        assert_eq!(device.name(), "Dummy Adapter");
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
}
