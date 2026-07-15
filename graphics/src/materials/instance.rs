//! Material instance with bound resources.
//!
//! A [`MaterialInstance`] contains the actual GPU resources bound for rendering.
//! Multiple instances can share the same [`Material`].
//!
//! Binding groups are wrapped in `Arc` to enable efficient batching - the renderer
//! can compare `Arc` pointers to group draw calls that share the same bindings.

use std::sync::Arc;

use crate::backend::GpuBindingGroup;
use crate::device::GraphicsDevice;
use crate::resources::{Buffer, Sampler, Texture, Tlas};

use super::bindings::BindingLayout;
use super::material::Material;

/// A bound resource for a specific binding slot.
#[derive(Debug, Clone)]
pub enum BoundResource {
    /// A uniform or storage buffer (binds the whole buffer).
    Buffer(Arc<Buffer>),

    /// A sub-range of a buffer: `offset..offset+size` bytes.
    ///
    /// Used for ring-allocated dynamic uniforms: bind with `offset = 0` and
    /// `size = <one element>`; the per-draw
    /// [`dynamic_offsets`](crate::DrawCommand::dynamic_offsets) select the
    /// element. The binding's layout type must be
    /// [`DynamicUniformBuffer`](crate::BindingType::DynamicUniformBuffer).
    BufferRange {
        /// The buffer.
        buffer: Arc<Buffer>,
        /// Byte offset of the bound range.
        offset: u64,
        /// Size of the bound range in bytes (one dynamic element).
        size: u64,
    },

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

    /// A top-level acceleration structure for ray queries (#110). The group
    /// keeps the TLAS (and, transitively, the BLASes of its last written
    /// instances) alive.
    AccelerationStructure(Arc<Tlas>),
}

impl BoundResource {
    /// Create a buffer binding (whole buffer).
    pub fn buffer(buffer: Arc<Buffer>) -> Self {
        Self::Buffer(buffer)
    }

    /// Create a sub-range buffer binding.
    pub fn buffer_range(buffer: Arc<Buffer>, offset: u64, size: u64) -> Self {
        Self::BufferRange {
            buffer,
            offset,
            size,
        }
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

    /// Create a top-level acceleration structure binding (#110).
    pub fn acceleration_structure(tlas: Arc<Tlas>) -> Self {
        Self::AccelerationStructure(tlas)
    }
}

/// The image layout a sampled depth texture will be in at draw time.
///
/// A binding group's GPU descriptor set is written **once** at creation (issue
/// #40), recording each sampled image's layout. That layout must match the
/// image's actual layout when the group is bound. For a depth texture the
/// correct sampled layout depends on how the pass uses it, so the intent is
/// declared here, at creation time, and the render graph transitions the image
/// to exactly this layout before any pass that binds the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SampledDepthLayout {
    /// Pure sampling (e.g. a shadow map sampled in a later pass):
    /// `SHADER_READ_ONLY_OPTIMAL`. The default for every texture binding.
    #[default]
    ShaderReadOnly,

    /// The depth texture is also bound as a **read-only depth attachment**
    /// (sampling it in the *same* pass is allowed): `DEPTH_STENCIL_READ_ONLY_OPTIMAL`.
    ///
    /// This is the only layout valid for a simultaneous read-only depth
    /// attachment + shader read. It is also a legal pure-sampling layout, so a
    /// group marked this way stays valid even in a pass that only samples.
    /// Valid only on a binding whose bound texture has a depth format.
    DepthStencilReadOnly,
}

/// A binding entry with its slot and resource.
#[derive(Debug, Clone)]
pub struct BindingEntry {
    /// The binding slot index.
    pub binding: u32,

    /// The bound resource.
    pub resource: BoundResource,

    /// For a sampled depth texture, the layout it will be sampled in. Ignored
    /// for buffers, samplers, and non-depth textures. Defaults to
    /// [`ShaderReadOnly`](SampledDepthLayout::ShaderReadOnly).
    pub sampled_depth_layout: SampledDepthLayout,
}

impl BindingEntry {
    /// Create a new binding entry.
    pub fn new(binding: u32, resource: BoundResource) -> Self {
        Self {
            binding,
            resource,
            sampled_depth_layout: SampledDepthLayout::ShaderReadOnly,
        }
    }

    /// Set the sampled depth layout (see [`SampledDepthLayout`]).
    pub fn with_sampled_depth_layout(mut self, layout: SampledDepthLayout) -> Self {
        self.sampled_depth_layout = layout;
        self
    }
}

/// A description of the resources to bind in a group.
///
/// This is the *un-compiled* form: a plain builder holding the entries and an
/// optional label. It is handed to
/// [`GraphicsDevice::create_binding_group`](crate::GraphicsDevice::create_binding_group),
/// which compiles it into an immutable [`BindingGroup`] resource (the GPU
/// descriptor set / bind group is created eagerly there, exactly as a
/// [`Material`] is the compiled form of a `MaterialDescriptor`).
#[derive(Debug, Clone)]
pub struct BindingGroupDescriptor {
    /// The bound entries.
    pub entries: Vec<BindingEntry>,

    /// Optional label for debugging.
    pub label: Option<String>,
}

impl BindingGroupDescriptor {
    /// Create a new empty binding group descriptor.
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

    /// Add a sub-range buffer binding (for dynamic ring uniforms).
    pub fn with_buffer_range(
        self,
        binding: u32,
        buffer: Arc<Buffer>,
        offset: u64,
        size: u64,
    ) -> Self {
        self.with_entry(binding, BoundResource::buffer_range(buffer, offset, size))
    }

    /// Add a texture binding.
    pub fn with_texture(self, binding: u32, texture: Arc<Texture>) -> Self {
        self.with_entry(binding, BoundResource::texture(texture))
    }

    /// Add a depth texture that this group's pass both **samples** and binds as
    /// a **read-only depth attachment** (the co-use case). The descriptor
    /// records `DEPTH_STENCIL_READ_ONLY_OPTIMAL`, matching the layout the render
    /// graph transitions the image to; see [`SampledDepthLayout`]. The bound
    /// texture must have a depth format.
    pub fn with_depth_texture_co_attached(mut self, binding: u32, texture: Arc<Texture>) -> Self {
        self.entries.push(
            BindingEntry::new(binding, BoundResource::texture(texture))
                .with_sampled_depth_layout(SampledDepthLayout::DepthStencilReadOnly),
        );
        self
    }

    /// Add a sampler binding.
    pub fn with_sampler(self, binding: u32, sampler: Arc<Sampler>) -> Self {
        self.with_entry(binding, BoundResource::sampler(sampler))
    }

    /// Add a combined texture+sampler binding.
    pub fn with_combined(self, binding: u32, texture: Arc<Texture>, sampler: Arc<Sampler>) -> Self {
        self.with_entry(binding, BoundResource::combined(texture, sampler))
    }

    /// Add a top-level acceleration structure binding (#110). The layout
    /// entry at `binding` must be
    /// [`BindingType::AccelerationStructure`](crate::BindingType).
    pub fn with_acceleration_structure(self, binding: u32, tlas: Arc<Tlas>) -> Self {
        self.with_entry(binding, BoundResource::acceleration_structure(tlas))
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl Default for BindingGroupDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

// Ensure BindingGroupDescriptor is Send + Sync
static_assertions::assert_impl_all!(BindingGroupDescriptor: Send, Sync);

/// An immutable, device-created binding group — the *compiled* form of a
/// [`BindingGroupDescriptor`].
///
/// Created via
/// [`GraphicsDevice::create_binding_group`](crate::GraphicsDevice::create_binding_group),
/// which builds the backend GPU handle (a Vulkan descriptor set / a wgpu bind
/// group) **eagerly, once**, against the given [`BindingLayout`]. Encoding a
/// draw then performs zero descriptor allocations and zero descriptor writes —
/// it only binds the cached handle.
///
/// # Immutability
///
/// The group is sealed: all fields are private and there are no mutating
/// methods. This is a correctness requirement — the GPU handle was written once
/// from `descriptor` and must never disagree with it. To change bindings,
/// create a new group and swap it on the [`MaterialInstance`].
///
/// # Lifetime
///
/// The GPU handle lives exactly as long as the `Arc<BindingGroup>`: when the
/// last `Arc` drops, the descriptor set / bind group is freed. There is no
/// cache, no key, no eviction. In-flight safety is provided by the same
/// invariant as [`Buffer`]/[`Material`]: submitted render graphs retain their
/// `Arc<MaterialInstance>` (and thus `Arc<BindingGroup>`) until the frame slot
/// is recycled after its fence wait.
pub struct BindingGroup {
    descriptor: BindingGroupDescriptor,
    layout: Arc<BindingLayout>,
    gpu_handle: GpuBindingGroup,
    /// Declared after `gpu_handle` deliberately: fields drop in declaration
    /// order, and this keep-alive must outlive the handle's `Drop`, which
    /// needs the backend (owned by the device→instance chain) alive. See #50.
    device: Arc<GraphicsDevice>,
}

impl BindingGroup {
    /// Create a compiled binding group (called by
    /// [`GraphicsDevice::create_binding_group`]).
    pub(crate) fn new(
        device: Arc<GraphicsDevice>,
        layout: Arc<BindingLayout>,
        descriptor: BindingGroupDescriptor,
        gpu_handle: GpuBindingGroup,
    ) -> Self {
        Self {
            device,
            layout,
            descriptor,
            gpu_handle,
        }
    }

    /// The bound entries.
    pub fn entries(&self) -> &[BindingEntry] {
        &self.descriptor.entries
    }

    /// The debug label, if set.
    pub fn label(&self) -> Option<&str> {
        self.descriptor.label.as_deref()
    }

    /// The layout this group was created against.
    pub fn layout(&self) -> &Arc<BindingLayout> {
        &self.layout
    }

    /// The backend GPU handle (descriptor set / bind group), created eagerly.
    pub fn gpu_handle(&self) -> &GpuBindingGroup {
        &self.gpu_handle
    }

    /// The parent device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }
}

impl std::fmt::Debug for BindingGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindingGroup")
            .field("entry_count", &self.descriptor.entries.len())
            .field("label", &self.descriptor.label)
            .finish()
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
/// let binding_group = device.create_binding_group(
///     layout.clone(),
///     BindingGroupDescriptor::new()
///         .with_buffer(0, properties_buffer)
///         .with_combined(1, albedo_texture, linear_sampler),
/// )?;
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
    use crate::device::GraphicsDevice;
    use crate::instance::GraphicsInstance;
    use crate::materials::{BindingLayout, MaterialDescriptor, ShaderSource};
    use crate::types::{BufferDescriptor, BufferUsage};

    fn test_device() -> Arc<GraphicsDevice> {
        GraphicsInstance::new().unwrap().create_device().unwrap()
    }

    fn create_test_material() -> Arc<Material> {
        let device = test_device();
        let desc = MaterialDescriptor::new()
            .with_shader(ShaderSource::vertex(b"vs".to_vec(), "main"))
            .with_label("test_material");
        Arc::new(Material::new(
            device,
            desc,
            crate::backend::GpuPipeline::Dummy,
        ))
    }

    fn uniform_layout() -> Arc<BindingLayout> {
        Arc::new(BindingLayout::new().with_uniform_buffer(0))
    }

    /// A single-uniform-buffer binding group on `device` (dummy backend).
    fn create_test_group(device: &Arc<GraphicsDevice>) -> Arc<BindingGroup> {
        let buffer = device
            .create_buffer(&BufferDescriptor::new(256, BufferUsage::UNIFORM))
            .unwrap();
        device
            .create_binding_group(
                uniform_layout(),
                BindingGroupDescriptor::new().with_buffer(0, buffer),
            )
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
    fn test_binding_group_descriptor() {
        let device = test_device();
        let buffer = device
            .create_buffer(&BufferDescriptor::new(256, BufferUsage::UNIFORM))
            .unwrap();
        let desc = BindingGroupDescriptor::new()
            .with_buffer(0, buffer)
            .with_label("test_group");

        assert_eq!(desc.entries.len(), 1);
        assert_eq!(desc.label, Some("test_group".to_string()));
    }

    #[test]
    fn test_create_binding_group_exposes_getters() {
        let device = test_device();
        let group = device
            .create_binding_group(
                uniform_layout(),
                BindingGroupDescriptor::new()
                    .with_buffer(
                        0,
                        device
                            .create_buffer(&BufferDescriptor::new(256, BufferUsage::UNIFORM))
                            .unwrap(),
                    )
                    .with_label("g"),
            )
            .unwrap();
        assert_eq!(group.entries().len(), 1);
        assert_eq!(group.label(), Some("g"));
        assert_eq!(group.layout().entries.len(), 1);
        // `gpu_handle()` is present (the concrete backend variant depends on
        // which backend the test build selected).
        let _ = group.gpu_handle();
    }

    #[test]
    fn test_create_binding_group_undeclared_binding_errors() {
        let device = test_device();
        let buffer = device
            .create_buffer(&BufferDescriptor::new(256, BufferUsage::UNIFORM))
            .unwrap();
        // Layout declares only binding 0, but the group binds binding 1.
        let result = device.create_binding_group(
            uniform_layout(),
            BindingGroupDescriptor::new().with_buffer(1, buffer),
        );
        assert!(result.is_err(), "binding not in layout must error");
    }

    #[test]
    fn test_create_binding_group_type_mismatch_errors() {
        let device = test_device();
        let texture = device
            .create_texture(&crate::types::TextureDescriptor::new_2d(
                4,
                4,
                crate::types::TextureFormat::Rgba8Unorm,
                crate::types::TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();
        // Layout declares a uniform buffer at 0, but a texture is bound there.
        let result = device.create_binding_group(
            uniform_layout(),
            BindingGroupDescriptor::new().with_texture(0, texture),
        );
        assert!(result.is_err(), "resource-kind mismatch must error");
    }

    #[test]
    fn test_material_instance_with_bindings() {
        let device = test_device();
        let material = create_test_material();
        let group = create_test_group(&device);
        let instance = MaterialInstance::new(material).with_binding_group(group);

        assert_eq!(instance.binding_groups().len(), 1);
        assert!(instance.binding_group(0).is_some());
        assert!(instance.binding_group(1).is_none());
    }

    #[test]
    fn test_binding_group_sharing() {
        let device = test_device();
        let material = create_test_material();

        // A binding group shared between two instances.
        let shared_group = create_test_group(&device);

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
