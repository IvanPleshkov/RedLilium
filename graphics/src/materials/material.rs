//! Material definition.
//!
//! A [`Material`] defines the shader and binding layouts used for rendering.
//! It is created by [`GraphicsDevice`] and can be shared across many [`MaterialInstance`]s.
//!
//! Binding layouts are stored as `Arc<BindingLayout>` to enable efficient batching -
//! the renderer can compare `Arc` pointers to group draw calls that share layouts.

use std::sync::Arc;

use crate::backend::GpuPipeline;
use crate::device::GraphicsDevice;
use crate::mesh::VertexLayout;
use crate::types::TextureFormat;
use redlilium_core::mesh::PrimitiveTopology;

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

    /// Create a compute shader source.
    pub fn compute(source: impl Into<Vec<u8>>, entry_point: impl Into<String>) -> Self {
        Self::new(ShaderStage::Compute, source, entry_point)
    }
}

/// Blend factor for blending operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendFactor {
    /// 0.0
    #[default]
    Zero,
    /// 1.0
    One,
    /// Source color
    Src,
    /// 1 - source color
    OneMinusSrc,
    /// Source alpha
    SrcAlpha,
    /// 1 - source alpha
    OneMinusSrcAlpha,
    /// Destination color
    Dst,
    /// 1 - destination color
    OneMinusDst,
    /// Destination alpha
    DstAlpha,
    /// 1 - destination alpha
    OneMinusDstAlpha,
    /// min(source alpha, 1 - destination alpha)
    SrcAlphaSaturated,
    /// Constant color
    Constant,
    /// 1 - constant color
    OneMinusConstant,
}

/// Blend operation for combining colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendOperation {
    /// source + destination
    #[default]
    Add,
    /// source - destination
    Subtract,
    /// destination - source
    ReverseSubtract,
    /// min(source, destination)
    Min,
    /// max(source, destination)
    Max,
}

/// Blend component configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    /// Source factor.
    pub src_factor: BlendFactor,
    /// Destination factor.
    pub dst_factor: BlendFactor,
    /// Blend operation.
    pub operation: BlendOperation,
}

impl Default for BlendComponent {
    fn default() -> Self {
        Self {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::Zero,
            operation: BlendOperation::Add,
        }
    }
}

impl BlendComponent {
    /// Create an over blending component (standard alpha blending).
    pub fn over() -> Self {
        Self {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        }
    }

    /// Create a premultiplied alpha blending component.
    pub fn premultiplied() -> Self {
        Self {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        }
    }
}

/// Blend state for color blending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BlendState {
    /// Color blend component.
    pub color: BlendComponent,
    /// Alpha blend component.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// Create a standard alpha blending state (src over dst).
    pub fn alpha_blending() -> Self {
        Self {
            color: BlendComponent::over(),
            alpha: BlendComponent::over(),
        }
    }

    /// Create a premultiplied alpha blending state.
    pub fn premultiplied_alpha() -> Self {
        Self {
            color: BlendComponent::premultiplied(),
            alpha: BlendComponent::premultiplied(),
        }
    }

    /// Create an additive blending state.
    pub fn additive() -> Self {
        Self {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        }
    }
}

/// Descriptor for creating a material.
#[derive(Debug, Clone)]
pub struct MaterialDescriptor {
    /// Shaders used by this material.
    pub shaders: Vec<ShaderSource>,

    /// Binding layouts for bind groups.
    /// Layouts are wrapped in `Arc` to enable sharing and efficient batching.
    pub binding_layouts: Vec<Arc<BindingLayout>>,

    /// Expected vertex layout for this material.
    /// Used for pipeline creation and mesh compatibility checking.
    /// Wrapped in `Arc` — same `Arc` pointer means same pipeline variant.
    pub vertex_layout: Arc<VertexLayout>,

    /// Blend state for color blending. If None, blending is disabled.
    pub blend_state: Option<BlendState>,

    /// Primitive topology (how vertices are assembled into primitives).
    pub topology: PrimitiveTopology,

    /// Color attachment formats for the render pass.
    pub color_formats: Vec<TextureFormat>,

    /// Depth attachment format, if any.
    pub depth_format: Option<TextureFormat>,

    /// Optional label for debugging.
    pub label: Option<String>,
}

impl Default for MaterialDescriptor {
    fn default() -> Self {
        Self {
            shaders: Vec::new(),
            binding_layouts: Vec::new(),
            vertex_layout: Arc::new(VertexLayout::new()),
            blend_state: None,
            topology: PrimitiveTopology::TriangleList,
            color_formats: Vec::new(),
            depth_format: None,
            label: None,
        }
    }
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

    /// Set the expected vertex layout for pipeline creation and mesh compatibility checking.
    pub fn with_vertex_layout(mut self, layout: Arc<VertexLayout>) -> Self {
        self.vertex_layout = layout;
        self
    }

    /// Set the blend state for color blending.
    pub fn with_blend_state(mut self, blend_state: BlendState) -> Self {
        self.blend_state = Some(blend_state);
        self
    }

    /// Set the primitive topology.
    pub fn with_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Add a color attachment format.
    pub fn with_color_format(mut self, format: TextureFormat) -> Self {
        self.color_formats.push(format);
        self
    }

    /// Set the depth attachment format.
    pub fn with_depth_format(mut self, format: TextureFormat) -> Self {
        self.depth_format = Some(format);
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
    gpu_handle: GpuPipeline,
}

impl Material {
    /// Create a new material (called by GraphicsDevice).
    pub(crate) fn new(
        device: Arc<GraphicsDevice>,
        descriptor: MaterialDescriptor,
        gpu_handle: GpuPipeline,
    ) -> Self {
        Self {
            device,
            descriptor,
            gpu_handle,
        }
    }

    /// Get the GPU pipeline handle.
    pub fn gpu_handle(&self) -> &GpuPipeline {
        &self.gpu_handle
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

    /// Get the expected vertex layout.
    pub fn vertex_layout(&self) -> &Arc<VertexLayout> {
        &self.descriptor.vertex_layout
    }

    /// Get the shaders.
    pub fn shaders(&self) -> &[ShaderSource] {
        &self.descriptor.shaders
    }

    /// Get the blend state.
    pub fn blend_state(&self) -> Option<&BlendState> {
        self.descriptor.blend_state.as_ref()
    }

    /// Get the primitive topology.
    pub fn topology(&self) -> PrimitiveTopology {
        self.descriptor.topology
    }

    /// Get the color attachment formats.
    pub fn color_formats(&self) -> &[TextureFormat] {
        &self.descriptor.color_formats
    }

    /// Get the depth attachment format.
    pub fn depth_format(&self) -> Option<TextureFormat> {
        self.descriptor.depth_format
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

        let material = Material::new(device, desc, GpuPipeline::Dummy);
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

        let material1 = Material::new(device.clone(), desc1, GpuPipeline::Dummy);
        let material2 = Material::new(device, desc2, GpuPipeline::Dummy);

        // Both materials share the same layout
        assert!(Arc::ptr_eq(
            &material1.binding_layouts()[0],
            &material2.binding_layouts()[0]
        ));
    }
}
