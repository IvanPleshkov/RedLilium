//! Material instance with bound resources.
//!
//! A [`MaterialInstance`] contains the actual GPU resources bound for rendering.
//! Multiple instances can share the same [`Material`].

use std::sync::Arc;

use crate::resources::{Buffer, Sampler, Texture};

use super::bindings::BindingFrequency;
use super::material::Material;

/// A bound resource for a specific binding slot.
#[derive(Debug, Clone)]
pub enum BoundResource {
    /// A uniform or storage buffer.
    Buffer(Arc<Buffer>),

    /// A texture resource.
    Texture(Arc<Texture>),

    /// A sampler resource.
    Sampler(Arc<Sampler>),

    /// Combined texture and sampler.
    CombinedTextureSampler {
        /// The texture.
        texture: Arc<Texture>,
        /// The sampler.
        sampler: Arc<Sampler>,
    },
}

impl BoundResource {
    /// Create a buffer binding.
    pub fn buffer(buffer: Arc<Buffer>) -> Self {
        Self::Buffer(buffer)
    }

    /// Create a texture binding.
    pub fn texture(texture: Arc<Texture>) -> Self {
        Self::Texture(texture)
    }

    /// Create a sampler binding.
    pub fn sampler(sampler: Arc<Sampler>) -> Self {
        Self::Sampler(sampler)
    }

    /// Create a combined texture+sampler binding.
    pub fn combined(texture: Arc<Texture>, sampler: Arc<Sampler>) -> Self {
        Self::CombinedTextureSampler { texture, sampler }
    }
}

/// A binding entry with its slot and resource.
#[derive(Debug, Clone)]
pub struct BindingEntry {
    /// The binding slot index.
    pub binding: u32,

    /// The bound resource.
    pub resource: BoundResource,
}

impl BindingEntry {
    /// Create a new binding entry.
    pub fn new(binding: u32, resource: BoundResource) -> Self {
        Self { binding, resource }
    }
}

/// A group of bindings for a specific frequency.
#[derive(Debug, Clone)]
pub struct BindingGroup {
    /// The frequency/group this belongs to.
    pub frequency: BindingFrequency,

    /// The bound entries.
    pub entries: Vec<BindingEntry>,
}

impl BindingGroup {
    /// Create a new binding group.
    pub fn new(frequency: BindingFrequency) -> Self {
        Self {
            frequency,
            entries: Vec::new(),
        }
    }

    /// Add a binding entry.
    pub fn with_entry(mut self, binding: u32, resource: BoundResource) -> Self {
        self.entries.push(BindingEntry::new(binding, resource));
        self
    }

    /// Add a buffer binding.
    pub fn with_buffer(self, binding: u32, buffer: Arc<Buffer>) -> Self {
        self.with_entry(binding, BoundResource::buffer(buffer))
    }

    /// Add a texture binding.
    pub fn with_texture(self, binding: u32, texture: Arc<Texture>) -> Self {
        self.with_entry(binding, BoundResource::texture(texture))
    }

    /// Add a sampler binding.
    pub fn with_sampler(self, binding: u32, sampler: Arc<Sampler>) -> Self {
        self.with_entry(binding, BoundResource::sampler(sampler))
    }

    /// Add a combined texture+sampler binding.
    pub fn with_combined(self, binding: u32, texture: Arc<Texture>, sampler: Arc<Sampler>) -> Self {
        self.with_entry(binding, BoundResource::combined(texture, sampler))
    }
}

/// A material instance with bound resources for rendering.
///
/// The instance references a [`Material`] and contains the actual GPU resources
/// needed for rendering. Multiple instances can share the same material but
/// have different textures, buffers, etc.
///
/// # Binding Groups
///
/// Bindings are organized by frequency:
/// - **PerFrame** (group 0): Shared across all draws (camera, lights)
/// - **PerMaterial** (group 1): Shared per material (textures, properties)
/// - **PerObject** (group 2): Per draw call (transforms)
///
/// # Example
///
/// ```ignore
/// let instance = MaterialInstance::new(material.clone())
///     .with_binding_group(BindingGroup::new(BindingFrequency::PerMaterial)
///         .with_buffer(0, properties_buffer)
///         .with_combined(1, albedo_texture, linear_sampler));
/// ```
pub struct MaterialInstance {
    material: Arc<Material>,
    binding_groups: Vec<BindingGroup>,
    label: Option<String>,
}

impl MaterialInstance {
    /// Create a new material instance.
    pub fn new(material: Arc<Material>) -> Self {
        Self {
            material,
            binding_groups: Vec::new(),
            label: None,
        }
    }

    /// Add a binding group.
    pub fn with_binding_group(mut self, group: BindingGroup) -> Self {
        // Replace existing group with same frequency, or add new one
        if let Some(existing) = self
            .binding_groups
            .iter_mut()
            .find(|g| g.frequency == group.frequency)
        {
            *existing = group;
        } else {
            self.binding_groups.push(group);
        }
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the parent material.
    pub fn material(&self) -> &Arc<Material> {
        &self.material
    }

    /// Get all binding groups.
    pub fn binding_groups(&self) -> &[BindingGroup] {
        &self.binding_groups
    }

    /// Get a binding group by frequency.
    pub fn binding_group(&self, frequency: BindingFrequency) -> Option<&BindingGroup> {
        self.binding_groups
            .iter()
            .find(|g| g.frequency == frequency)
    }

    /// Get the instance label, if set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Set a binding group (mutable version).
    pub fn set_binding_group(&mut self, group: BindingGroup) {
        if let Some(existing) = self
            .binding_groups
            .iter_mut()
            .find(|g| g.frequency == group.frequency)
        {
            *existing = group;
        } else {
            self.binding_groups.push(group);
        }
    }
}

impl std::fmt::Debug for MaterialInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterialInstance")
            .field("material", &self.material.label())
            .field("binding_group_count", &self.binding_groups.len())
            .field("label", &self.label)
            .finish()
    }
}

// Ensure MaterialInstance is Send + Sync
static_assertions::assert_impl_all!(MaterialInstance: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;
    use crate::materials::{MaterialDescriptor, ShaderSource};
    use crate::types::{BufferDescriptor, BufferUsage};

    fn create_test_material() -> Arc<Material> {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let desc = MaterialDescriptor::new()
            .with_shader(ShaderSource::vertex(b"vs".to_vec(), "main"))
            .with_label("test_material");
        Arc::new(Material::new(device, desc))
    }

    fn create_test_buffer() -> Arc<Buffer> {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        device
            .create_buffer(&BufferDescriptor::new(256, BufferUsage::UNIFORM))
            .unwrap()
    }

    #[test]
    fn test_material_instance_creation() {
        let material = create_test_material();
        let instance = MaterialInstance::new(material.clone()).with_label("test_instance");

        assert!(Arc::ptr_eq(instance.material(), &material));
        assert_eq!(instance.label(), Some("test_instance"));
    }

    #[test]
    fn test_binding_group() {
        let buffer = create_test_buffer();
        let group = BindingGroup::new(BindingFrequency::PerMaterial).with_buffer(0, buffer);

        assert_eq!(group.frequency, BindingFrequency::PerMaterial);
        assert_eq!(group.entries.len(), 1);
    }

    #[test]
    fn test_material_instance_with_bindings() {
        let material = create_test_material();
        let buffer = create_test_buffer();

        let instance = MaterialInstance::new(material).with_binding_group(
            BindingGroup::new(BindingFrequency::PerMaterial).with_buffer(0, buffer),
        );

        assert!(
            instance
                .binding_group(BindingFrequency::PerMaterial)
                .is_some()
        );
        assert!(instance.binding_group(BindingFrequency::PerFrame).is_none());
    }

    #[test]
    fn test_binding_group_replacement() {
        let material = create_test_material();
        let buffer1 = create_test_buffer();
        let buffer2 = create_test_buffer();

        let mut instance = MaterialInstance::new(material).with_binding_group(
            BindingGroup::new(BindingFrequency::PerMaterial).with_buffer(0, buffer1),
        );

        // Replace the binding group
        instance.set_binding_group(
            BindingGroup::new(BindingFrequency::PerMaterial)
                .with_buffer(0, buffer2.clone())
                .with_buffer(1, buffer2),
        );

        let group = instance
            .binding_group(BindingFrequency::PerMaterial)
            .unwrap();
        assert_eq!(group.entries.len(), 2);
    }
}
