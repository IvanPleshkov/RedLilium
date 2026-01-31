//! Graphics instance.
//!
//! The [`GraphicsInstance`] is the top-level entry point for the graphics system.
//! It manages one or more [`GraphicsDevice`]s.

use std::sync::{Arc, RwLock, Weak};

use crate::device::GraphicsDevice;
use crate::error::GraphicsError;

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
}

impl GraphicsInstance {
    /// Create a new graphics instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the graphics system cannot be initialized.
    pub fn new() -> Result<Arc<Self>, GraphicsError> {
        log::info!("Creating GraphicsInstance");

        let instance = Arc::new(Self {
            self_ref: RwLock::new(Weak::new()),
            devices: RwLock::new(Vec::new()),
        });

        // Store self-reference
        if let Ok(mut self_ref) = instance.self_ref.write() {
            *self_ref = Arc::downgrade(&instance);
        }

        Ok(instance)
    }

    /// Get the weak self-reference.
    fn weak_self(&self) -> Weak<GraphicsInstance> {
        self.self_ref
            .read()
            .map(|r| r.clone())
            .unwrap_or_else(|_| Weak::new())
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

        let device = Arc::new(GraphicsDevice::new(self.weak_self(), adapter.name.clone()));

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
        self.devices
            .read()
            .map(|d| d.len())
            .unwrap_or(0)
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
        assert!(device.instance().is_some());
    }
}
