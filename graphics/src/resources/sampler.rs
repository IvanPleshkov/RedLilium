//! GPU sampler resource.

use std::sync::Arc;

use crate::device::GraphicsDevice;
use crate::types::SamplerDescriptor;

/// A GPU texture sampler.
///
/// Samplers are created by [`GraphicsDevice::create_sampler`] and are reference-counted.
/// They hold a strong reference to their parent device, keeping it alive.
///
/// # Example
///
/// ```ignore
/// let sampler = device.create_sampler(&SamplerDescriptor::linear())?;
/// ```
pub struct Sampler {
    device: Arc<GraphicsDevice>,
    descriptor: SamplerDescriptor,
}

impl Sampler {
    /// Create a new sampler (called by GraphicsDevice).
    pub(crate) fn new(device: Arc<GraphicsDevice>, descriptor: SamplerDescriptor) -> Self {
        Self { device, descriptor }
    }

    /// Get the parent device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Get the sampler descriptor.
    pub fn descriptor(&self) -> &SamplerDescriptor {
        &self.descriptor
    }

    /// Get the sampler label, if set.
    pub fn label(&self) -> Option<&str> {
        self.descriptor.label.as_deref()
    }
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("mag_filter", &self.descriptor.mag_filter)
            .field("min_filter", &self.descriptor.min_filter)
            .field("label", &self.descriptor.label)
            .finish()
    }
}

// Ensure Sampler is Send + Sync
static_assertions::assert_impl_all!(Sampler: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;

    fn create_test_device() -> Arc<GraphicsDevice> {
        let instance = GraphicsInstance::new().unwrap();
        instance.create_device().unwrap()
    }

    #[test]
    fn test_sampler_debug() {
        let desc = SamplerDescriptor::linear();
        let sampler = Sampler::new(create_test_device(), desc);
        let debug = format!("{:?}", sampler);
        assert!(debug.contains("Sampler"));
        assert!(debug.contains("Linear"));
    }

    #[test]
    fn test_sampler_label() {
        let desc = SamplerDescriptor::linear().with_label("test_sampler");
        let sampler = Sampler::new(create_test_device(), desc);
        assert_eq!(sampler.label(), Some("test_sampler"));
    }
}
