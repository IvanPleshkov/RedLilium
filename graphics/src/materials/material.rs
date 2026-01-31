//! Material definition.
//!
//! A [`Material`] defines the shader and binding layouts used for rendering.
//! It is created by [`GraphicsDevice`] and can be shared across many [`MaterialInstance`]s.
//!
//! Binding layouts are stored as `Arc<BindingLayout>` to enable efficient batching -
//! the renderer can compare `Arc` pointers to group draw calls that share layouts.

use std::sync::Arc;

use crate::device::GraphicsDevice;

use super::bindings::BindingLayout;

/// Shader stage in the graphics pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Vertex shader.
    Vertex,
    /// Fragment shader.
    Fragment,
    /// Compute shader.
    Compute,
}

/// Shader source for a material.
#[derive(Debug, Clone)]
pub struct ShaderSource {
    /// The shader stage.
    pub stage: ShaderStage,

    /// Shader source code (WGSL, SPIR-V, etc. - backend dependent).
    pub source: Vec<u8>,

    /// Entry point function name.
    pub entry_point: String,
}

impl ShaderSource {
    /// Create a new shader source.
    pub fn new(
        stage: ShaderStage,
        source: impl Into<Vec<u8>>,
        entry_point: impl Into<String>,
    ) -> Self {
        Self {
            stage,
            source: source.into(),
            entry_point: entry_point.into(),
        }
    }

    /// Create a vertex shader source.
    pub fn vertex(source: impl Into<Vec<u8>>, entry_point: impl Into<String>) -> Self {
        Self::new(ShaderStage::Vertex, source, entry_point)
    }

    /// Create a fragment shader source.
    pub fn fragment(source: impl Into<Vec<u8>>, entry_point: impl Into<String>) -> Self {
        Self::new(ShaderStage::Fragment, source, entry_point)
    }
}

/// Descriptor for creating a material.
#[derive(Debug, Clone, Default)]
pub struct MaterialDescriptor {
    /// Shaders used by this material.
    pub shaders: Vec<ShaderSource>,

    /// Binding layouts for bind groups.
    /// Layouts are wrapped in `Arc` to enable sharing and efficient batching.
    pub binding_layouts: Vec<Arc<BindingLayout>>,

    /// Optional label for debugging.
    pub label: Option<String>,
}

impl MaterialDescriptor {
    /// Create a new material descriptor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a shader to the material.
    pub fn with_shader(mut self, shader: ShaderSource) -> Self {
        self.shaders.push(shader);
        self
    }

    /// Add a binding layout.
    pub fn with_binding_layout(mut self, layout: Arc<BindingLayout>) -> Self {
        self.binding_layouts.push(layout);
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A material defines the shader and binding layout for rendering.
///
/// Materials are created by [`GraphicsDevice::create_material`] and hold
/// a strong reference to their parent device.
///
/// # Example
///
/// ```ignore
/// let layout = Arc::new(BindingLayout::new()
///     .with_uniform_buffer(0)
///     .with_combined_texture_sampler(1));
///
/// let material = device.create_material(&MaterialDescriptor::new()
///     .with_shader(ShaderSource::vertex(vs_source, "vs_main"))
///     .with_shader(ShaderSource::fragment(fs_source, "fs_main"))
///     .with_binding_layout(layout)
///     .with_label("pbr_material"))?;
/// ```
pub struct Material {
    device: Arc<GraphicsDevice>,
    descriptor: MaterialDescriptor,
}

impl Material {
    /// Create a new material (called by GraphicsDevice).
    pub(crate) fn new(device: Arc<GraphicsDevice>, descriptor: MaterialDescriptor) -> Self {
        Self { device, descriptor }
    }

    /// Get the parent device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Get the material descriptor.
    pub fn descriptor(&self) -> &MaterialDescriptor {
        &self.descriptor
    }

    /// Get the material label, if set.
    pub fn label(&self) -> Option<&str> {
        self.descriptor.label.as_deref()
    }

    /// Get the binding layouts.
    pub fn binding_layouts(&self) -> &[Arc<BindingLayout>] {
        &self.descriptor.binding_layouts
    }

    /// Get the shaders.
    pub fn shaders(&self) -> &[ShaderSource] {
        &self.descriptor.shaders
    }
}

impl std::fmt::Debug for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Material")
            .field("label", &self.descriptor.label)
            .field("shader_count", &self.descriptor.shaders.len())
            .field(
                "binding_layout_count",
                &self.descriptor.binding_layouts.len(),
            )
            .finish()
    }
}

// Ensure Material is Send + Sync
static_assertions::assert_impl_all!(Material: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;

    fn create_test_device() -> Arc<GraphicsDevice> {
        let instance = GraphicsInstance::new().unwrap();
        instance.create_device().unwrap()
    }

    #[test]
    fn test_material_descriptor_builder() {
        let desc = MaterialDescriptor::new()
            .with_shader(ShaderSource::vertex(b"vs_code".to_vec(), "main"))
            .with_shader(ShaderSource::fragment(b"fs_code".to_vec(), "main"))
            .with_label("test_material");

        assert_eq!(desc.shaders.len(), 2);
        assert_eq!(desc.label, Some("test_material".to_string()));
    }

    #[test]
    fn test_material_creation() {
        let device = create_test_device();
        let desc = MaterialDescriptor::new()
            .with_shader(ShaderSource::vertex(b"vs".to_vec(), "main"))
            .with_label("test");

        let material = Material::new(device, desc);
        assert_eq!(material.label(), Some("test"));
        assert_eq!(material.shaders().len(), 1);
    }

    #[test]
    fn test_shader_source() {
        let vs = ShaderSource::vertex(b"code".to_vec(), "vs_main");
        assert_eq!(vs.stage, ShaderStage::Vertex);
        assert_eq!(vs.entry_point, "vs_main");

        let fs = ShaderSource::fragment(b"code".to_vec(), "fs_main");
        assert_eq!(fs.stage, ShaderStage::Fragment);
    }

    #[test]
    fn test_binding_layout_sharing() {
        let device = create_test_device();

        // Create a shared layout
        let shared_layout = Arc::new(BindingLayout::new().with_uniform_buffer(0));

        let desc1 = MaterialDescriptor::new()
            .with_shader(ShaderSource::vertex(b"vs".to_vec(), "main"))
            .with_binding_layout(shared_layout.clone());

        let desc2 = MaterialDescriptor::new()
            .with_shader(ShaderSource::vertex(b"vs".to_vec(), "main"))
            .with_binding_layout(shared_layout.clone());

        let material1 = Material::new(device.clone(), desc1);
        let material2 = Material::new(device, desc2);

        // Both materials share the same layout
        assert!(Arc::ptr_eq(
            &material1.binding_layouts()[0],
            &material2.binding_layouts()[0]
        ));
    }
}
