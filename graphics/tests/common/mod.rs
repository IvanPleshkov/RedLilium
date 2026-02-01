//! Common utilities for GPU integration tests.
//!
//! This module provides shared test infrastructure that can be reused
//! across different backend implementations.

use std::sync::Arc;

use redlilium_graphics::{
    Buffer, BufferDescriptor, BufferUsage, ColorAttachment, DepthStencilAttachment, GraphicsDevice,
    GraphicsInstance, GraphicsPass, LoadOp, Mesh, MeshDescriptor, RenderGraph, RenderTargetConfig,
    StoreOp, Texture, TextureDescriptor, TextureFormat, TextureUsage, TransferConfig,
    TransferOperation, TransferPass, VertexAttribute, VertexBufferLayout, VertexLayout,
};

// ============================================================================
// Backend Enumeration
// ============================================================================

/// Available GPU backends for testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Dummy backend (no actual GPU operations).
    Dummy,
    /// Vulkan backend.
    Vulkan,
    /// WebGPU backend.
    WebGpu,
}

impl Backend {
    /// Check if this backend is currently available.
    pub fn is_available(&self) -> bool {
        match self {
            // Dummy backend is always available
            Backend::Dummy => true,
            // Other backends are not implemented yet
            Backend::Vulkan | Backend::WebGpu => false,
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
}

// ============================================================================
// Test Context
// ============================================================================

/// Test context providing access to graphics resources.
///
/// This struct manages the graphics instance and device for a test,
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
}

impl TestContext {
    /// Create a new test context for the given backend.
    ///
    /// Returns `None` if the backend is not available.
    pub fn new(backend: Backend) -> Option<Self> {
        if !backend.is_available() {
            return None;
        }

        let instance = GraphicsInstance::new().ok()?;
        let device = instance.create_device().ok()?;

        Some(Self {
            backend,
            instance,
            device,
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
    /// This is a simplified execution path for tests that don't need
    /// frame pipelining or complex scheduling.
    pub fn execute_graph(&self, graph: &RenderGraph) {
        let compiled = graph.compile().expect("Failed to compile render graph");
        let mut pipeline = self.device.create_pipeline(1);
        let mut schedule = pipeline.begin_frame();
        schedule.present("test", compiled, &[]);
        pipeline.end_frame(schedule);
        pipeline.wait_idle();
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
#[allow(dead_code)]
pub fn verify_pixel(
    data: &[u8],
    width: u32,
    x: u32,
    y: u32,
    expected: ExpectedPixel,
    tolerance: u8,
) -> bool {
    let offset = ((y * width + x) * 4) as usize;
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
