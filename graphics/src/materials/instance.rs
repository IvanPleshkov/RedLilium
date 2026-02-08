//! Material instance with bound resources.
//!
//! A [`MaterialInstance`] contains the actual GPU resources bound for rendering.
//! Multiple instances can share the same [`Material`].
//!
//! Binding groups are wrapped in `Arc` to enable efficient batching - the renderer
//! can compare `Arc` pointers to group draw calls that share the same bindings.

use std::sync::Arc;

use crate::resources::{Buffer, Sampler, Texture};

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

/// A group of bound resources.
///
/// Binding groups are typically wrapped in `Arc` and shared between material instances
/// to enable efficient batching by pointer comparison.
#[derive(Debug, Clone)]
pub struct BindingGroup {
    /// The bound entries.
    pub entries: Vec<BindingEntry>,

    /// Optional label for debugging.
    pub label: Option<String>,
}

impl BindingGroup {
    /// Create a new empty binding group.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            label: None,
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

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl Default for BindingGroup {
    fn default() -> Self {
        Self::new()
    }
}

// Ensure BindingGroup is Send + Sync for use in Arc
static_assertions::assert_impl_all!(BindingGroup: Send, Sync);

/// A material instance with bound resources for rendering.
///
/// The instance references a [`Material`] and contains the actual GPU resources
/// needed for rendering. Multiple instances can share the same material but
/// have different textures, buffers, etc.
///
/// # Binding Groups
///
/// Binding groups are stored as `Arc<BindingGroup>` to enable efficient batching.
/// The renderer can compare `Arc` pointers to group draw calls that share bindings,
/// minimizing GPU state changes.
///
/// # Example
///
/// ```ignore
/// let binding_group = Arc::new(BindingGroup::new()
///     .with_buffer(0, properties_buffer)
///     .with_combined(1, albedo_texture, linear_sampler));
///
/// let instance = MaterialInstance::new(material.clone())
///     .with_binding_group(binding_group);
/// ```
pub struct MaterialInstance {
    material: Arc<Material>,
    binding_groups: Vec<Arc<BindingGroup>>,
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
    pub fn with_binding_group(mut self, group: Arc<BindingGroup>) -> Self {
        self.binding_groups.push(group);
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
    pub fn binding_groups(&self) -> &[Arc<BindingGroup>] {
        &self.binding_groups
    }

    /// Get a binding group by index.
    pub fn binding_group(&self, index: usize) -> Option<&Arc<BindingGroup>> {
        self.binding_groups.get(index)
    }

    /// Get the instance label, if set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Add a binding group (mutable version).
    pub fn add_binding_group(&mut self, group: Arc<BindingGroup>) {
        self.binding_groups.push(group);
    }

    /// Set binding groups, replacing all existing ones.
    pub fn set_binding_groups(&mut self, groups: Vec<Arc<BindingGroup>>) {
        self.binding_groups = groups;
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
        Arc::new(Material::new(
            device,
            desc,
            crate::backend::GpuPipeline::Dummy,
        ))
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
        let group = BindingGroup::new()
            .with_buffer(0, buffer)
            .with_label("test_group");

        assert_eq!(group.entries.len(), 1);
        assert_eq!(group.label, Some("test_group".to_string()));
    }

    #[test]
    fn test_material_instance_with_bindings() {
        let material = create_test_material();
        let buffer = create_test_buffer();

        let group = Arc::new(BindingGroup::new().with_buffer(0, buffer));
        let instance = MaterialInstance::new(material).with_binding_group(group);

        assert_eq!(instance.binding_groups().len(), 1);
        assert!(instance.binding_group(0).is_some());
        assert!(instance.binding_group(1).is_none());
    }

    #[test]
    fn test_binding_group_sharing() {
        let material = create_test_material();
        let buffer = create_test_buffer();

        // Create a shared binding group
        let shared_group = Arc::new(BindingGroup::new().with_buffer(0, buffer));

        let instance1 =
            MaterialInstance::new(material.clone()).with_binding_group(shared_group.clone());
        let instance2 = MaterialInstance::new(material).with_binding_group(shared_group.clone());

        // Both instances share the same binding group
        assert!(Arc::ptr_eq(
            instance1.binding_group(0).unwrap(),
            instance2.binding_group(0).unwrap()
        ));
    }
}
