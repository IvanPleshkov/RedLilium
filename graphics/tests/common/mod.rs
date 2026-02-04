//! Common utilities for GPU integration tests.
//!
//! This module provides shared test infrastructure that can be reused
//! across different backend implementations.

use std::cell::RefCell;
use std::sync::Arc;

use redlilium_graphics::{
    BackendType, BindingGroup, Buffer, BufferDescriptor, BufferUsage, ColorAttachment,
    DepthStencilAttachment, FramePipeline, GraphicsDevice, GraphicsInstance, GraphicsPass,
    InstanceParameters, LoadOp, Material, MaterialDescriptor, MaterialInstance, Mesh,
    MeshDescriptor, RenderGraph, RenderTargetConfig, SamplerDescriptor, ShaderSource, StoreOp,
    Texture, TextureDescriptor, TextureFormat, TextureUsage, TransferConfig, TransferOperation,
    TransferPass, VertexAttribute, VertexBufferLayout, VertexLayout, WgpuBackendType,
};

/// Compute the aligned bytes per row for a texture (256-byte alignment for wgpu).
pub fn aligned_bytes_per_row(width: u32, bytes_per_pixel: u32) -> u32 {
    let unpadded = width * bytes_per_pixel;
    (unpadded + 255) & !255
}

/// Compute the required buffer size for reading back a texture with row alignment.
pub fn readback_buffer_size(width: u32, height: u32, bytes_per_pixel: u32) -> u64 {
    let aligned_row = aligned_bytes_per_row(width, bytes_per_pixel);
    (aligned_row * height) as u64
}

// ============================================================================
// Backend Enumeration
// ============================================================================

/// Available GPU backends for testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Dummy backend (no actual GPU operations).
    Dummy,
    /// Vulkan backend (native via ash).
    Vulkan,
    /// WebGPU backend (via wgpu with Vulkan).
    WebGpu,
}

impl Backend {
    /// Check if this backend is currently available.
    pub fn is_available(&self) -> bool {
        match self {
            // Dummy backend is always available
            Backend::Dummy => true,
            // Vulkan backend via ash is available when the feature is enabled
            #[cfg(feature = "vulkan-backend")]
            Backend::Vulkan => true,
            #[cfg(not(feature = "vulkan-backend"))]
            Backend::Vulkan => false,
            // WebGpu backend (wgpu) is available when the feature is enabled
            #[cfg(feature = "wgpu-backend")]
            Backend::WebGpu => true,
            #[cfg(not(feature = "wgpu-backend"))]
            Backend::WebGpu => false,
        }
    }

    /// Get the backend name for display.
    #[allow(dead_code)]
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Dummy => "dummy",
            Backend::Vulkan => "vulkan",
            Backend::WebGpu => "webgpu",
        }
    }

    /// Convert to InstanceParameters for creating a GraphicsInstance.
    pub fn to_instance_parameters(self) -> InstanceParameters {
        match self {
            Backend::Dummy => InstanceParameters::new().with_backend(BackendType::Dummy),
            Backend::Vulkan => InstanceParameters::new()
                .with_backend(BackendType::Wgpu)
                .with_wgpu_backend(WgpuBackendType::Auto),
            Backend::WebGpu => InstanceParameters::new()
                .with_backend(BackendType::Wgpu)
                .with_wgpu_backend(WgpuBackendType::Auto),
        }
    }
}

// ============================================================================
// Test Context
// ============================================================================

/// Test context providing access to graphics resources.
///
/// This struct manages the graphics instance, device, and frame pipeline for a test,
/// and provides helper methods for common operations.
pub struct TestContext {
    /// The backend being tested.
    #[allow(dead_code)]
    pub backend: Backend,
    /// Graphics instance (Arc-wrapped).
    #[allow(dead_code)]
    instance: Arc<GraphicsInstance>,
    /// Graphics device for creating resources.
    pub device: Arc<GraphicsDevice>,
    /// Frame pipeline for graph execution (uses RefCell for interior mutability).
    pipeline: RefCell<FramePipeline>,
}

impl TestContext {
    /// Create a new test context for the given backend.
    ///
    /// Returns `None` if the backend is not available.
    pub fn new(backend: Backend) -> Option<Self> {
        if !backend.is_available() {
            return None;
        }

        let params = backend.to_instance_parameters();
        let instance = GraphicsInstance::with_parameters(params).ok()?;
        let device = instance.create_device().ok()?;
        // Use 1 frame in flight for synchronous test execution
        let pipeline = device.create_pipeline(1);

        Some(Self {
            backend,
            instance,
            device,
            pipeline: RefCell::new(pipeline),
        })
    }

    /// Create a buffer with the given size and usage flags.
    pub fn create_buffer(&self, size: u64, usage: BufferUsage) -> Arc<Buffer> {
        self.device
            .create_buffer(&BufferDescriptor::new(size, usage))
            .expect("Failed to create buffer")
    }

    /// Create a staging buffer (CPU writable, can copy from).
    pub fn create_staging_buffer(&self, size: u64) -> Arc<Buffer> {
        self.create_buffer(size, BufferUsage::COPY_SRC | BufferUsage::MAP_WRITE)
    }

    /// Create a readback buffer (CPU readable, can copy to).
    pub fn create_readback_buffer(&self, size: u64) -> Arc<Buffer> {
        self.create_buffer(size, BufferUsage::COPY_DST | BufferUsage::MAP_READ)
    }

    /// Create a GPU buffer (for general GPU operations).
    pub fn create_gpu_buffer(&self, size: u64, usage: BufferUsage) -> Arc<Buffer> {
        self.create_buffer(size, usage | BufferUsage::COPY_SRC | BufferUsage::COPY_DST)
    }

    /// Create a 2D texture with the given dimensions and format.
    pub fn create_texture_2d(
        &self,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Arc<Texture> {
        self.device
            .create_texture(&TextureDescriptor::new_2d(width, height, format, usage))
            .expect("Failed to create texture")
    }

    /// Create a render target texture.
    pub fn create_render_target(&self, width: u32, height: u32) -> Arc<Texture> {
        self.create_texture_2d(
            width,
            height,
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        )
    }

    /// Create a depth texture.
    pub fn create_depth_texture(&self, width: u32, height: u32) -> Arc<Texture> {
        self.create_texture_2d(
            width,
            height,
            TextureFormat::Depth32Float,
            TextureUsage::RENDER_ATTACHMENT,
        )
    }

    /// Execute a render graph and wait for completion.
    ///
    /// Uses [`FramePipeline`] and [`FrameSchedule`] internally for proper
    /// graph execution through the frame scheduling system.
    pub fn execute_graph(&self, graph: &RenderGraph) {
        let mut pipeline = self.pipeline.borrow_mut();
        let mut schedule = pipeline.begin_frame();

        // Submit the graph - this actually executes on the GPU
        let handle = schedule.submit("test_graph", graph, &[]);

        // Finish the schedule (offscreen, no presentation)
        schedule.finish(&[handle]);

        // End the frame and wait for completion
        pipeline.end_frame(schedule);

        // Wait for all GPU work to complete before returning
        pipeline.wait_idle();
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        // Ensure GPU is idle before cleanup
        self.pipeline.borrow().wait_idle();
    }
}

// ============================================================================
// Mesh Helpers
// ============================================================================

/// Vertex data for a simple quad (two triangles).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QuadVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

impl QuadVertex {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// Create vertex layout for QuadVertex.
pub fn quad_vertex_layout() -> Arc<VertexLayout> {
    Arc::new(
        VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(QuadVertex::SIZE as u32))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::texcoord0(12)) // offset after position (3 * 4 bytes)
            .with_label("quad_vertex_layout"),
    )
}

/// Create a fullscreen quad mesh.
///
/// The quad covers the entire clip space from (-1, -1) to (1, 1).
#[allow(dead_code)]
pub fn create_fullscreen_quad(ctx: &TestContext) -> Arc<Mesh> {
    let layout = quad_vertex_layout();
    ctx.device
        .create_mesh(
            &MeshDescriptor::new(layout)
                .with_vertex_count(6)
                .with_label("fullscreen_quad"),
        )
        .expect("Failed to create quad mesh")
}

/// Quad vertex data for a fullscreen quad (two triangles, CCW winding).
#[allow(dead_code)]
pub const FULLSCREEN_QUAD_VERTICES: [QuadVertex; 6] = [
    // First triangle (bottom-left, bottom-right, top-right)
    QuadVertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        position: [1.0, -1.0, 0.0],
        uv: [1.0, 1.0],
    },
    QuadVertex {
        position: [1.0, 1.0, 0.0],
        uv: [1.0, 0.0],
    },
    // Second triangle (bottom-left, top-right, top-left)
    QuadVertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        position: [1.0, 1.0, 0.0],
        uv: [1.0, 0.0],
    },
    QuadVertex {
        position: [-1.0, 1.0, 0.0],
        uv: [0.0, 0.0],
    },
];

/// Create a small quad at the center of the screen.
///
/// The quad covers clip space from (-0.5, -0.5) to (0.5, 0.5).
#[allow(dead_code)]
pub fn create_centered_quad(ctx: &TestContext) -> Arc<Mesh> {
    let layout = quad_vertex_layout();
    ctx.device
        .create_mesh(
            &MeshDescriptor::new(layout)
                .with_vertex_count(6)
                .with_label("centered_quad"),
        )
        .expect("Failed to create quad mesh")
}

/// Quad vertex data for a centered quad (covers 50% of the screen).
#[allow(dead_code)]
pub const CENTERED_QUAD_VERTICES: [QuadVertex; 6] = [
    // First triangle
    QuadVertex {
        position: [-0.5, -0.5, 0.0],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        position: [0.5, -0.5, 0.0],
        uv: [1.0, 1.0],
    },
    QuadVertex {
        position: [0.5, 0.5, 0.0],
        uv: [1.0, 0.0],
    },
    // Second triangle
    QuadVertex {
        position: [-0.5, -0.5, 0.0],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        position: [0.5, 0.5, 0.0],
        uv: [1.0, 0.0],
    },
    QuadVertex {
        position: [-0.5, 0.5, 0.0],
        uv: [0.0, 0.0],
    },
];

// ============================================================================
// Render Graph Helpers
// ============================================================================

/// Create a simple transfer graph that copies between buffers.
#[allow(dead_code)]
pub fn create_buffer_copy_graph(src: Arc<Buffer>, dst: Arc<Buffer>) -> RenderGraph {
    let mut graph = RenderGraph::new();

    let mut transfer = TransferPass::new("buffer_copy".into());
    transfer.set_transfer_config(
        TransferConfig::new().with_operation(TransferOperation::copy_buffer_whole(src, dst)),
    );
    graph.add_transfer_pass(transfer);

    graph
}

/// Create a simple graphics pass with a single color attachment.
pub fn create_simple_render_pass(
    name: &str,
    target: Arc<Texture>,
    clear_color: [f32; 4],
) -> GraphicsPass {
    let mut pass = GraphicsPass::new(name.into());
    pass.set_render_targets(
        RenderTargetConfig::new().with_color(
            ColorAttachment::from_texture(target)
                .with_load_op(LoadOp::clear_color(
                    clear_color[0],
                    clear_color[1],
                    clear_color[2],
                    clear_color[3],
                ))
                .with_store_op(StoreOp::Store),
        ),
    );
    pass
}

/// Create a graphics pass with color and depth attachments.
pub fn create_render_pass_with_depth(
    name: &str,
    color_target: Arc<Texture>,
    depth_target: Arc<Texture>,
    clear_color: [f32; 4],
    clear_depth: f32,
) -> GraphicsPass {
    let mut pass = GraphicsPass::new(name.into());
    pass.set_render_targets(
        RenderTargetConfig::new()
            .with_color(
                ColorAttachment::from_texture(color_target)
                    .with_load_op(LoadOp::clear_color(
                        clear_color[0],
                        clear_color[1],
                        clear_color[2],
                        clear_color[3],
                    ))
                    .with_store_op(StoreOp::Store),
            )
            .with_depth_stencil(
                DepthStencilAttachment::from_texture(depth_target)
                    .with_depth_load_op(LoadOp::clear_depth(clear_depth))
                    .with_depth_store_op(StoreOp::Store),
            ),
    );
    pass
}

/// Create a graphics pass with multiple render targets (MRT).
pub fn create_mrt_pass(name: &str, targets: Vec<(Arc<Texture>, [f32; 4])>) -> GraphicsPass {
    let mut pass = GraphicsPass::new(name.into());
    let mut config = RenderTargetConfig::new();

    for (target, clear_color) in targets {
        config = config.with_color(
            ColorAttachment::from_texture(target)
                .with_load_op(LoadOp::clear_color(
                    clear_color[0],
                    clear_color[1],
                    clear_color[2],
                    clear_color[3],
                ))
                .with_store_op(StoreOp::Store),
        );
    }

    pass.set_render_targets(config);
    pass
}

// ============================================================================
// Verification Helpers
// ============================================================================

/// Expected pixel color for verification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpectedPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl ExpectedPixel {
    #[allow(dead_code)]
    pub const RED: Self = Self {
        r: 255,
        g: 0,
        b: 0,
        a: 255,
    };
    #[allow(dead_code)]
    pub const GREEN: Self = Self {
        r: 0,
        g: 255,
        b: 0,
        a: 255,
    };
    #[allow(dead_code)]
    pub const BLUE: Self = Self {
        r: 0,
        g: 0,
        b: 255,
        a: 255,
    };
    #[allow(dead_code)]
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
        a: 255,
    };
    #[allow(dead_code)]
    pub const BLACK: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
    };
    #[allow(dead_code)]
    pub const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        a: 0,
    };

    #[allow(dead_code)]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    #[allow(dead_code)]
    pub fn from_float(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r: (r * 255.0).clamp(0.0, 255.0) as u8,
            g: (g * 255.0).clamp(0.0, 255.0) as u8,
            b: (b * 255.0).clamp(0.0, 255.0) as u8,
            a: (a * 255.0).clamp(0.0, 255.0) as u8,
        }
    }

    /// Check if this pixel matches another within a tolerance.
    #[allow(dead_code)]
    pub fn matches(&self, other: &ExpectedPixel, tolerance: u8) -> bool {
        self.r.abs_diff(other.r) <= tolerance
            && self.g.abs_diff(other.g) <= tolerance
            && self.b.abs_diff(other.b) <= tolerance
            && self.a.abs_diff(other.a) <= tolerance
    }
}

/// Verify that a pixel in the readback data matches the expected value.
/// Uses 256-byte row alignment as required by wgpu texture readback.
#[allow(dead_code)]
pub fn verify_pixel(
    data: &[u8],
    width: u32,
    x: u32,
    y: u32,
    expected: ExpectedPixel,
    tolerance: u8,
) -> bool {
    // Account for row alignment (256 bytes)
    let bytes_per_pixel = 4u32;
    let aligned_row_bytes = aligned_bytes_per_row(width, bytes_per_pixel);
    let offset = (y * aligned_row_bytes + x * bytes_per_pixel) as usize;

    if offset + 4 > data.len() {
        return false;
    }

    let actual = ExpectedPixel {
        r: data[offset],
        g: data[offset + 1],
        b: data[offset + 2],
        a: data[offset + 3],
    };

    actual.matches(&expected, tolerance)
}

/// Get the actual pixel value for debugging.
#[allow(dead_code)]
pub fn get_pixel(data: &[u8], width: u32, x: u32, y: u32) -> ExpectedPixel {
    let bytes_per_pixel = 4u32;
    let aligned_row_bytes = aligned_bytes_per_row(width, bytes_per_pixel);
    let offset = (y * aligned_row_bytes + x * bytes_per_pixel) as usize;

    ExpectedPixel {
        r: data.get(offset).copied().unwrap_or(0),
        g: data.get(offset + 1).copied().unwrap_or(0),
        b: data.get(offset + 2).copied().unwrap_or(0),
        a: data.get(offset + 3).copied().unwrap_or(0),
    }
}

/// Verify that a region of pixels all match the expected value.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn verify_region(
    data: &[u8],
    width: u32,
    x_start: u32,
    y_start: u32,
    region_width: u32,
    region_height: u32,
    expected: ExpectedPixel,
    tolerance: u8,
) -> bool {
    for y in y_start..(y_start + region_height) {
        for x in x_start..(x_start + region_width) {
            if !verify_pixel(data, width, x, y, expected, tolerance) {
                return false;
            }
        }
    }
    true
}

// ============================================================================
// Test Data Generators
// ============================================================================

/// Generate test data pattern for buffer tests.
pub fn generate_test_pattern(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

/// Generate a solid color image data.
#[allow(dead_code)]
pub fn generate_solid_color(width: u32, height: u32, color: ExpectedPixel) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut data = Vec::with_capacity(pixel_count * 4);
    for _ in 0..pixel_count {
        data.push(color.r);
        data.push(color.g);
        data.push(color.b);
        data.push(color.a);
    }
    data
}

// ============================================================================
// Shader Helpers
// ============================================================================

/// Simple WGSL shader that samples a texture and outputs its color.
/// Used for pass-to-pass texture read tests (e.g., layout tracking integration test).
pub const TEXTURE_SAMPLE_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(input_texture, input_sampler, in.uv);
}
"#;

/// Simple WGSL shader that outputs a solid red color.
/// The vertex shader reads position from vertex buffer (location 0).
pub const SOLID_RED_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(3) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

/// Create a material with the given WGSL shader source.
#[allow(dead_code)]
pub fn create_solid_color_material(ctx: &TestContext) -> Arc<Material> {
    ctx.device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(
                    SOLID_RED_SHADER.as_bytes().to_vec(),
                    "vs_main",
                ))
                .with_shader(ShaderSource::fragment(
                    SOLID_RED_SHADER.as_bytes().to_vec(),
                    "fs_main",
                ))
                .with_vertex_layout(quad_vertex_layout())
                .with_label("solid_red_material"),
        )
        .expect("Failed to create material")
}

/// Create a material instance from a material.
#[allow(dead_code)]
pub fn create_material_instance(material: Arc<Material>) -> Arc<MaterialInstance> {
    Arc::new(MaterialInstance::new(material).with_label("test_instance"))
}

/// Quad vertex data for a left-half quad (covers x from -1 to 0).
#[allow(dead_code)]
pub const LEFT_HALF_QUAD_VERTICES: [QuadVertex; 6] = [
    // First triangle (bottom-left, bottom-center, top-center)
    QuadVertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        position: [0.0, -1.0, 0.0],
        uv: [0.5, 1.0],
    },
    QuadVertex {
        position: [0.0, 1.0, 0.0],
        uv: [0.5, 0.0],
    },
    // Second triangle (bottom-left, top-center, top-left)
    QuadVertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        position: [0.0, 1.0, 0.0],
        uv: [0.5, 0.0],
    },
    QuadVertex {
        position: [-1.0, 1.0, 0.0],
        uv: [0.0, 0.0],
    },
];

/// Create a left-half quad mesh.
#[allow(dead_code)]
pub fn create_left_half_quad(ctx: &TestContext) -> Arc<Mesh> {
    let layout = quad_vertex_layout();
    ctx.device
        .create_mesh(
            &MeshDescriptor::new(layout)
                .with_vertex_count(6)
                .with_label("left_half_quad"),
        )
        .expect("Failed to create quad mesh")
}

/// Write quad vertex data to a mesh's vertex buffer.
#[allow(dead_code)]
pub fn write_quad_vertices(ctx: &TestContext, mesh: &Mesh, vertices: &[QuadVertex]) {
    let vb = mesh
        .vertex_buffer(0)
        .expect("Mesh should have vertex buffer");
    // Convert vertices to bytes
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            vertices.as_ptr() as *const u8,
            vertices.len() * QuadVertex::SIZE,
        )
    };
    ctx.device
        .write_buffer(vb, 0, bytes)
        .expect("Failed to write quad vertex buffer");
}

/// Create a material that samples a texture and outputs its color.
/// Used for pass-to-pass texture read tests.
#[allow(dead_code)]
pub fn create_texture_sample_material(ctx: &TestContext) -> Arc<Material> {
    use redlilium_graphics::BindingLayout;

    // Create binding layout for group 0: texture at binding 0, sampler at binding 1
    let binding_layout = Arc::new(
        BindingLayout::new()
            .with_texture(0)
            .with_sampler(1)
            .with_label("texture_sample_bindings"),
    );

    ctx.device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(
                    TEXTURE_SAMPLE_SHADER.as_bytes().to_vec(),
                    "vs_main",
                ))
                .with_shader(ShaderSource::fragment(
                    TEXTURE_SAMPLE_SHADER.as_bytes().to_vec(),
                    "fs_main",
                ))
                .with_vertex_layout(quad_vertex_layout())
                .with_binding_layout(binding_layout)
                .with_label("texture_sample_material"),
        )
        .expect("Failed to create texture sample material")
}

/// Create a material instance that samples from a texture.
/// Binds the texture to binding 0 and sampler to binding 1 in a binding group.
#[allow(dead_code)]
pub fn create_texture_sample_instance(
    ctx: &TestContext,
    material: Arc<Material>,
    texture: Arc<Texture>,
) -> Arc<MaterialInstance> {
    // Create a sampler for the texture (nearest filtering with clamp to edge)
    let sampler = ctx
        .device
        .create_sampler(&SamplerDescriptor::nearest().with_label("test_sampler"))
        .expect("Failed to create sampler");

    // Create binding group with texture at binding 0 and sampler at binding 1
    let binding_group = Arc::new(
        BindingGroup::new()
            .with_texture(0, texture)
            .with_sampler(1, sampler)
            .with_label("texture_sample_bindings"),
    );

    Arc::new(
        MaterialInstance::new(material)
            .with_binding_group(binding_group)
            .with_label("texture_sample_instance"),
    )
}

/// Create a render target that can be both rendered to AND sampled from.
/// This is needed for pass-to-pass texture use (layout tracking tests).
#[allow(dead_code)]
pub fn create_sampleable_render_target(ctx: &TestContext, width: u32, height: u32) -> Arc<Texture> {
    ctx.create_texture_2d(
        width,
        height,
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_SRC,
    )
}
