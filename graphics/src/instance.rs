//! Graphics instance.
//!
//! The [`GraphicsInstance`] is the top-level entry point for the graphics system.
//! It manages one or more [`GraphicsDevice`]s.

use std::sync::{Arc, RwLock, Weak};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::backend::{self, GpuBackend};
use crate::device::GraphicsDevice;
use crate::error::GraphicsError;
use crate::swapchain::Surface;

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
    backend: Arc<dyn GpuBackend>,
}

impl GraphicsInstance {
    /// Create a new graphics instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the graphics system cannot be initialized.
    pub fn new() -> Result<Arc<Self>, GraphicsError> {
        log::info!("Creating GraphicsInstance");

        // Create the GPU backend
        let backend = backend::create_backend()?;
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
    pub(crate) fn backend(&self) -> &Arc<dyn GpuBackend> {
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
    pub fn create_surface<W>(&self, window: &W) -> Result<Arc<Surface>, GraphicsError>
    where
        W: HasWindowHandle + HasDisplayHandle,
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
