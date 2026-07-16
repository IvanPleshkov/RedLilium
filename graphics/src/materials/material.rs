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
pub use redlilium_core::material::PolygonMode;
pub use redlilium_core::math::DepthConvention;
use redlilium_core::mesh::PrimitiveTopology;
pub use redlilium_core::sampler::CompareFunction;

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
    /// Task (amplification) shader — optional meshlet-culling stage ahead of
    /// the mesh stage. Requires [`DeviceCapabilities::mesh_shading`] and Slang
    /// authoring (WGSL has no mesh-shading syntax).
    ///
    /// [`DeviceCapabilities::mesh_shading`]: crate::DeviceCapabilities::mesh_shading
    Task,
    /// Mesh shader — replaces the vertex stage (and vertex input entirely) in
    /// a meshlet pipeline. Requires [`DeviceCapabilities::mesh_shading`] and
    /// Slang authoring (WGSL has no mesh-shading syntax).
    ///
    /// [`DeviceCapabilities::mesh_shading`]: crate::DeviceCapabilities::mesh_shading
    Mesh,
}

/// Source language of the shader code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShaderSourceLanguage {
    /// WGSL source (native for wgpu, naga→SPIR-V for Vulkan).
    #[default]
    Wgsl,
    /// Slang source (compiled to WGSL for wgpu, SPIR-V for Vulkan via Slang compiler).
    Slang,
}

/// Shader source for a material.
#[derive(Debug, Clone)]
pub struct ShaderSource {
    /// The shader stage.
    pub stage: ShaderStage,

    /// Shader source code (UTF-8 text in the given language).
    pub source: Vec<u8>,

    /// Entry point function name.
    pub entry_point: String,

    /// Source language.
    pub language: ShaderSourceLanguage,

    /// Compile-time defines (only used for Slang sources).
    /// Each entry is (name, value). Empty value means `#define NAME` without a value.
    pub defines: Vec<(String, String)>,
}

impl ShaderSource {
    /// Create a new WGSL shader source (backward compatible).
    pub fn new(
        stage: ShaderStage,
        source: impl Into<Vec<u8>>,
        entry_point: impl Into<String>,
    ) -> Self {
        Self {
            stage,
            source: source.into(),
            entry_point: entry_point.into(),
            language: ShaderSourceLanguage::Wgsl,
            defines: Vec::new(),
        }
    }

    /// Create a Slang shader source with optional compile-time defines.
    pub fn slang(
        stage: ShaderStage,
        source: impl Into<Vec<u8>>,
        entry_point: impl Into<String>,
        defines: Vec<(String, String)>,
    ) -> Self {
        Self {
            stage,
            source: source.into(),
            entry_point: entry_point.into(),
            language: ShaderSourceLanguage::Slang,
            defines,
        }
    }

    /// Create a vertex shader source (WGSL).
    pub fn vertex(source: impl Into<Vec<u8>>, entry_point: impl Into<String>) -> Self {
        Self::new(ShaderStage::Vertex, source, entry_point)
    }

    /// Create a fragment shader source (WGSL).
    pub fn fragment(source: impl Into<Vec<u8>>, entry_point: impl Into<String>) -> Self {
        Self::new(ShaderStage::Fragment, source, entry_point)
    }

    /// Create a compute shader source (WGSL).
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

/// Which triangle faces are culled (not rasterized).
///
/// Faces are classified front/back by their winding relative to
/// [`FrontFace`]. `None` (the engine default) draws both — required for
/// double-sided materials and safe for content authored without a consistent
/// winding (#39).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CullMode {
    /// Draw both faces (no culling).
    #[default]
    None,
    /// Cull front-facing triangles.
    Front,
    /// Cull back-facing triangles.
    Back,
}

/// The winding order that defines a **front**-facing triangle, in the engine's
/// logical (app-facing) convention: counter-clockwise, matching glTF, OpenGL and
/// a right-handed math convention (#39).
///
/// This is normalized at the API boundary: each backend maps it to its physical
/// convention. The wgpu backend passes it through. The Vulkan backend renders
/// with a negative-viewport-height Y-flip (matching wgpu/OpenGL screen space);
/// on conformant native Vulkan that flip reverses screen-space winding, so the
/// backend **inverts** the winding to compensate — but on the portability
/// subset (MoltenVK / Metal) facing is determined independently of the flip
/// (like wgpu-on-Metal), so the logical winding is passed through there. See
/// `backend/vulkan/pipeline.rs::vk_front_face`. Defining winding once,
/// logically, is what lets culling be enabled without the backends diverging
/// (XB-M4, #39).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FrontFace {
    /// Counter-clockwise winding is front-facing (engine default).
    #[default]
    Ccw,
    /// Clockwise winding is front-facing.
    Cw,
}

/// Rasterizer state grouped for a material's pipeline (#39): winding-based
/// culling plus fill/wireframe mode. Defaults reproduce the engine's historical
/// hardcodes (no culling, CCW-front, filled polygons), so adding it changes
/// nothing until a material opts in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RasterState {
    /// Which faces to cull.
    pub cull_mode: CullMode,
    /// Winding that defines a front face (logical/app-facing).
    pub front_face: FrontFace,
    /// Fill or wireframe.
    pub polygon_mode: PolygonMode,
}

/// Depth-attachment state grouped for a material's pipeline (#39). Present
/// (`Some`) exactly when the material renders into a pass with a depth
/// attachment; the whole struct is `None` otherwise (mirroring wgpu's
/// `Option<DepthStencilState>`).
///
/// Shaped to grow: stencil and depth-bias are out of scope for #39 but will land
/// here (bias with shadows) without another migration.
///
/// The engine's depth convention is **reversed-Z** ([`DepthConvention`],
/// ADR-038): [`new`](Self::new) defaults `compare` to `GreaterEqual` and pairs
/// with a reversed projection and clear depth 0.0. Opting a target out to
/// classic depth ([`for_convention`](Self::for_convention) with
/// [`DepthConvention::Classic`]) must flip projection, clear, and compare
/// together — one convention per depth target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthState {
    /// Depth attachment format.
    pub format: TextureFormat,
    /// Depth comparison function (default `GreaterEqual` — reversed-Z;
    /// `LessEqual` for the classic opt-out).
    pub compare: CompareFunction,
    /// Whether the pipeline writes depth. `false` for testing against a
    /// **read-only** depth attachment (e.g. sampling the depth it tests
    /// against, #60) — writing to a read-only attachment is invalid.
    pub write: bool,
}

impl DepthState {
    /// A depth attachment with the engine's default test (reversed-Z:
    /// `GreaterEqual`, writes on).
    pub fn new(format: TextureFormat) -> Self {
        Self::for_convention(DepthConvention::default(), format)
    }

    /// A depth attachment whose compare derives from an explicit
    /// [`DepthConvention`] (writes on). The convention must match the
    /// projection matrix and clear depth used with the same target.
    pub fn for_convention(convention: DepthConvention, format: TextureFormat) -> Self {
        Self {
            format,
            compare: convention.compare(),
            write: true,
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

    /// Rasterizer state: winding-based culling, front-face convention, and
    /// fill/wireframe mode (#39). Defaults reproduce the historical hardcodes.
    pub raster: RasterState,

    /// Color attachment formats for the render pass.
    pub color_formats: Vec<TextureFormat>,

    /// Depth attachment state (format + compare + write), or `None` when the
    /// material renders into a pass with no depth attachment (#39). Must be
    /// `Some` iff the target pass has a depth attachment, and its `format` must
    /// match — validated at pass-encode time.
    pub depth: Option<DepthState>,

    /// MSAA sample count of the target attachments this pipeline renders into
    /// (default `1` = no MSAA). Must match the pass's attachment sample count —
    /// validated at pass-encode time (#39, XB-L6).
    pub sample_count: u32,

    /// `(group, binding)` pairs whose reflected `UniformBuffer` should be made a
    /// [`DynamicUniformBuffer`](crate::BindingType::DynamicUniformBuffer) (bound
    /// with a per-draw offset). Applied after reflection in `create_material`.
    pub dynamic_uniforms: Vec<(u32, u32)>,

    /// Per-set update-frequency classes reflected from rate-classified
    /// `ParameterBlock`s (parallel to `binding_layouts`; empty for legacy
    /// shaders). Filled by reflection in `create_material`, read by render
    /// systems to decide how each set is bound (Decision 7).
    pub set_update_rates: Vec<Option<super::UpdateRate>>,

    /// Shader variant selection (#6, Decision 5). Applied by `create_material`
    /// to **every Slang stage at once** — the stages of one material are one
    /// module and must compile with identical defines (per-stage `defines` on
    /// [`ShaderSource`] allow them to drift, which is always a bug). Build it
    /// with [`ShaderVariantSpace::select`](crate::shader::ShaderVariantSpace::select)
    /// so typos and missing system axes fail at key-build time.
    pub variant: Option<crate::shader::VariantKey>,

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
            raster: RasterState::default(),
            color_formats: Vec::new(),
            depth: None,
            sample_count: 1,
            dynamic_uniforms: Vec::new(),
            set_update_rates: Vec::new(),
            variant: None,
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

    /// Select the shader variant (#6, Decision 5) — the defines every Slang
    /// stage of this material compiles with. Order-independent with
    /// [`with_shader`](Self::with_shader): the key is applied to the stages
    /// inside `create_material`, not here.
    pub fn with_variant(mut self, variant: crate::shader::VariantKey) -> Self {
        self.variant = Some(variant);
        self
    }

    /// Whether this material uses the mesh-shading pipeline (has a
    /// [`ShaderStage::Mesh`] stage). Such materials have no vertex input:
    /// geometry is fetched by the mesh shader from storage buffers, and draws
    /// are issued with
    /// [`GraphicsPass::add_draw_mesh_tasks`](crate::GraphicsPass::add_draw_mesh_tasks).
    pub fn uses_mesh_shading(&self) -> bool {
        self.shaders.iter().any(|s| s.stage == ShaderStage::Mesh)
    }

    /// Validate the stage combination (#111). A graphics material is either
    /// vertex+fragment or task?+mesh+fragment; mixing the two vertex-input
    /// models, a task stage without a mesh stage, or a mesh material with a
    /// non-empty vertex layout are all descriptor bugs caught here rather
    /// than as driver errors at pipeline-creation time.
    pub fn validate_stage_combination(&self) -> Result<(), crate::error::GraphicsError> {
        let has = |stage: ShaderStage| self.shaders.iter().any(|s| s.stage == stage);
        let err = |msg: String| Err(crate::error::GraphicsError::InvalidParameter(msg));

        if has(ShaderStage::Task) && !has(ShaderStage::Mesh) {
            return err(format!(
                "material {:?}: a task stage requires a mesh stage (task shaders only \
                 amplify mesh-shader dispatches)",
                self.label
            ));
        }
        if has(ShaderStage::Mesh) {
            if has(ShaderStage::Vertex) {
                return err(format!(
                    "material {:?}: mesh and vertex stages are mutually exclusive (the mesh \
                     stage replaces vertex input entirely)",
                    self.label
                ));
            }
            if !self.vertex_layout.attributes.is_empty() || !self.vertex_layout.buffers.is_empty() {
                return err(format!(
                    "material {:?}: mesh pipelines have no vertex input; remove the vertex \
                     layout (mesh shaders fetch geometry from storage buffers)",
                    self.label
                ));
            }
        }
        Ok(())
    }

    /// Resolve [`variant`](Self::variant) into per-stage defines: a clone of
    /// this descriptor whose Slang [`ShaderSource::defines`] carry the
    /// variant's defines merged with any ad-hoc per-stage ones.
    ///
    /// Errors when the variant names a define a stage already sets ad-hoc
    /// (split ownership means no overlap — a silent override in either
    /// direction would hide a real conflict), and when a variant is set on a
    /// material with no Slang stages (WGSL has no preprocessor; the variant
    /// would be silently ignored).
    pub fn resolve_variant(&self) -> Result<Self, crate::error::GraphicsError> {
        let Some(variant) = &self.variant else {
            return Ok(self.clone());
        };
        let mut desc = self.clone();
        if variant.is_empty() {
            return Ok(desc);
        }

        let mut applied = false;
        for shader in &mut desc.shaders {
            if shader.language != ShaderSourceLanguage::Slang {
                continue;
            }
            for (name, value) in variant.defines() {
                if shader.defines.iter().any(|(n, _)| n == name) {
                    return Err(crate::error::GraphicsError::InvalidParameter(format!(
                        "material {:?}: variant define {name:?} is also set ad-hoc on the \
                         {:?} stage — a define is owned either by the variant key or by the \
                         stage, never both",
                        self.label, shader.stage,
                    )));
                }
                shader.defines.push((name.clone(), value.clone()));
            }
            shader.defines.sort();
            applied = true;
        }

        if !applied {
            return Err(crate::error::GraphicsError::InvalidParameter(format!(
                "material {:?}: a shader variant is set but there are no Slang stages to \
                 apply it to (WGSL has no preprocessor)",
                self.label,
            )));
        }
        Ok(desc)
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

    /// Set the polygon rasterization mode (fill or wireframe).
    pub fn with_polygon_mode(mut self, mode: PolygonMode) -> Self {
        self.raster.polygon_mode = mode;
        self
    }

    /// Set which faces are culled (default [`CullMode::None`]).
    pub fn with_cull_mode(mut self, cull_mode: CullMode) -> Self {
        self.raster.cull_mode = cull_mode;
        self
    }

    /// Set the winding that defines a front face (default [`FrontFace::Ccw`]).
    pub fn with_front_face(mut self, front_face: FrontFace) -> Self {
        self.raster.front_face = front_face;
        self
    }

    /// Replace the whole rasterizer state at once.
    pub fn with_raster(mut self, raster: RasterState) -> Self {
        self.raster = raster;
        self
    }

    /// Add a color attachment format.
    pub fn with_color_format(mut self, format: TextureFormat) -> Self {
        self.color_formats.push(format);
        self
    }

    /// Mark a reflected uniform binding `(group, binding)` as a dynamic uniform
    /// (bound once, offset supplied per draw). Applied after reflection.
    pub fn with_dynamic_uniform(mut self, group: u32, binding: u32) -> Self {
        self.dynamic_uniforms.push((group, binding));
        self
    }

    /// Set the depth attachment format (creating the [`DepthState`] with the
    /// default `LessEqual`/write-on test, or updating the format if already set).
    pub fn with_depth_format(mut self, format: TextureFormat) -> Self {
        match &mut self.depth {
            Some(d) => d.format = format,
            None => self.depth = Some(DepthState::new(format)),
        }
        self
    }

    /// Replace the whole depth state (format + compare + write) at once, or clear
    /// it with `None` (no depth attachment).
    pub fn with_depth(mut self, depth: Option<DepthState>) -> Self {
        self.depth = depth;
        self
    }

    /// Set the depth comparison function (e.g. `LessEqual` for a classic-depth
    /// opt-out target; the default is reversed-Z `GreaterEqual`). No effect
    /// unless a depth format is set (call [`with_depth_format`](Self::with_depth_format) first).
    pub fn with_depth_compare(mut self, compare: CompareFunction) -> Self {
        if let Some(d) = &mut self.depth {
            d.compare = compare;
        }
        self
    }

    /// Set whether the pipeline writes depth (default `true`). Use `false` for a
    /// pipeline that depth-tests against a read-only depth attachment (#60). No
    /// effect unless a depth format is set (call [`with_depth_format`](Self::with_depth_format) first).
    pub fn with_depth_write(mut self, write: bool) -> Self {
        if let Some(d) = &mut self.depth {
            d.write = write;
        }
        self
    }

    /// Set the MSAA sample count (default `1`). Must match the target pass's
    /// attachment sample count.
    pub fn with_sample_count(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
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
    descriptor: MaterialDescriptor,
    gpu_handle: GpuPipeline,
    /// Declared after `gpu_handle` deliberately: fields drop in declaration
    /// order, and this keep-alive must outlive the handle's `Drop`, which
    /// needs the backend (owned by the device→instance chain) alive. See #50.
    device: Arc<GraphicsDevice>,
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

    /// Per-set update-frequency classes (parallel to
    /// [`binding_layouts`](Self::binding_layouts); empty for legacy shaders
    /// without rate-classified `ParameterBlock`s).
    pub fn set_update_rates(&self) -> &[Option<super::UpdateRate>] {
        &self.descriptor.set_update_rates
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

    /// Get the polygon rasterization mode.
    pub fn polygon_mode(&self) -> PolygonMode {
        self.descriptor.raster.polygon_mode
    }

    /// Get the rasterizer state (culling, front-face, polygon mode).
    pub fn raster(&self) -> RasterState {
        self.descriptor.raster
    }

    /// Get the color attachment formats.
    pub fn color_formats(&self) -> &[TextureFormat] {
        &self.descriptor.color_formats
    }

    /// Get the depth attachment state, if any.
    pub fn depth(&self) -> Option<DepthState> {
        self.descriptor.depth
    }

    /// Get the depth attachment format, if any.
    pub fn depth_format(&self) -> Option<TextureFormat> {
        self.descriptor.depth.map(|d| d.format)
    }

    /// Get the MSAA sample count.
    pub fn sample_count(&self) -> u32 {
        self.descriptor.sample_count
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

    /// Mesh-shading stage rules (#111): a graphics material is either
    /// vertex(+fragment) or task?+mesh+fragment; a mesh material has no
    /// vertex layout.
    #[test]
    fn mesh_stage_combination_validation() {
        let slang =
            |stage: ShaderStage| ShaderSource::slang(stage, b"src".to_vec(), "main", vec![]);

        let full = MaterialDescriptor::new()
            .with_shader(slang(ShaderStage::Task))
            .with_shader(slang(ShaderStage::Mesh))
            .with_shader(slang(ShaderStage::Fragment));
        assert!(full.uses_mesh_shading());
        assert!(full.validate_stage_combination().is_ok());

        // The task stage is optional.
        assert!(
            MaterialDescriptor::new()
                .with_shader(slang(ShaderStage::Mesh))
                .with_shader(slang(ShaderStage::Fragment))
                .validate_stage_combination()
                .is_ok()
        );

        // A task stage without a mesh stage amplifies nothing.
        assert!(
            MaterialDescriptor::new()
                .with_shader(slang(ShaderStage::Task))
                .with_shader(slang(ShaderStage::Fragment))
                .validate_stage_combination()
                .is_err()
        );

        // Mesh and vertex stages are mutually exclusive.
        assert!(
            MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(b"vs".to_vec(), "main"))
                .with_shader(slang(ShaderStage::Mesh))
                .validate_stage_combination()
                .is_err()
        );

        // A mesh material must not declare a vertex layout.
        assert!(
            MaterialDescriptor::new()
                .with_shader(slang(ShaderStage::Mesh))
                .with_vertex_layout(VertexLayout::position_only())
                .validate_stage_combination()
                .is_err()
        );

        // Classic materials are unaffected.
        let classic = MaterialDescriptor::new()
            .with_shader(ShaderSource::vertex(b"vs".to_vec(), "main"))
            .with_vertex_layout(VertexLayout::position_only());
        assert!(!classic.uses_mesh_shading());
        assert!(classic.validate_stage_combination().is_ok());
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
