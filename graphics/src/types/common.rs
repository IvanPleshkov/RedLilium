//! Common types shared across the graphics system.

// ============================================================================
// Viewport
// ============================================================================

/// Viewport configuration for rendering.
///
/// Defines the rectangular region of the framebuffer that will be rendered to,
/// along with the depth range mapping.
///
/// # Coordinate System
///
/// This engine uses the **D3D/Metal/wgpu coordinate convention**:
///
/// - **Depth range**: `[0, 1]` (not OpenGL's `[-1, 1]`)
/// - **Y-axis**: +Y points down in NDC (Vulkan convention)
/// - **Origin**: Top-left corner
///
/// This means projection matrices should be built for `[0, 1]` depth range.
/// When using libraries like `glam` or `nalgebra`, use the "right-handed Z-up
/// with depth 0 to 1" projection functions.
///
/// # Example
///
/// ```ignore
/// // Full-screen viewport with standard depth range
/// let viewport = Viewport::new(0.0, 0.0, 1920.0, 1080.0);
///
/// // Custom depth range (e.g., for split-depth rendering)
/// let viewport = Viewport::new(0.0, 0.0, 1920.0, 1080.0)
///     .with_depth_range(0.0, 0.5);
/// ```
///
/// # Projection Matrix Guidance
///
/// When building projection matrices, use functions designed for `[0, 1]` depth:
///
/// ```ignore
/// // glam example:
/// let proj = glam::Mat4::perspective_rh(fov_y, aspect, near, far);
///
/// // Note: glam's perspective_rh uses [0, 1] depth range by default
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// X coordinate of the viewport's top-left corner.
    pub x: f32,
    /// Y coordinate of the viewport's top-left corner.
    pub y: f32,
    /// Width of the viewport.
    pub width: f32,
    /// Height of the viewport.
    pub height: f32,
    /// Minimum depth value (default: 0.0).
    ///
    /// This engine uses `[0, 1]` depth range by convention.
    pub min_depth: f32,
    /// Maximum depth value (default: 1.0).
    ///
    /// This engine uses `[0, 1]` depth range by convention.
    pub max_depth: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

impl Viewport {
    /// Create a new viewport with standard `[0, 1]` depth range.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate of the top-left corner
    /// * `y` - Y coordinate of the top-left corner
    /// * `width` - Viewport width
    /// * `height` - Viewport height
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Create a viewport from dimensions with origin at (0, 0).
    pub fn from_dimensions(width: u32, height: u32) -> Self {
        Self::new(0.0, 0.0, width as f32, height as f32)
    }

    /// Set the depth range.
    ///
    /// Both `min_depth` and `max_depth` should be in the range `[0, 1]`.
    ///
    /// # Note
    ///
    /// Unusual depth configurations (like `min > max` for reverse-Z) are valid
    /// and can be useful for improved depth precision.
    pub fn with_depth_range(mut self, min_depth: f32, max_depth: f32) -> Self {
        self.min_depth = min_depth;
        self.max_depth = max_depth;
        self
    }
}

// ============================================================================
// Scissor Rectangle
// ============================================================================

/// Scissor rectangle for clipping rendering.
///
/// Pixels outside the scissor rectangle are discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ScissorRect {
    /// X coordinate of the top-left corner.
    pub x: i32,
    /// Y coordinate of the top-left corner.
    pub y: i32,
    /// Width of the scissor rectangle.
    pub width: u32,
    /// Height of the scissor rectangle.
    pub height: u32,
}

impl ScissorRect {
    /// Create a new scissor rectangle.
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Create a scissor rectangle from dimensions with origin at (0, 0).
    pub fn from_dimensions(width: u32, height: u32) -> Self {
        Self::new(0, 0, width, height)
    }
}

// ============================================================================
// Extent3d
// ============================================================================

/// 3D extent for textures and buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Extent3d {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Depth in pixels (1 for 2D textures).
    pub depth: u32,
}

impl Extent3d {
    /// Create a new 2D extent.
    pub fn new_2d(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }

    /// Create a new 3D extent.
    pub fn new_3d(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }
}

/// Clear value for render targets.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ClearValue {
    /// No clear operation.
    #[default]
    None,
    /// Clear color attachment with RGBA values.
    Color { r: f32, g: f32, b: f32, a: f32 },
    /// Clear depth attachment.
    Depth(f32),
    /// Clear stencil attachment.
    Stencil(u32),
    /// Clear depth and stencil attachments.
    DepthStencil { depth: f32, stencil: u32 },
}

impl ClearValue {
    /// Create a color clear value.
    pub fn color(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Color { r, g, b, a }
    }

    /// Create a depth clear value.
    pub fn depth(value: f32) -> Self {
        Self::Depth(value)
    }

    /// No clear operation.
    pub fn none() -> Self {
        Self::None
    }
}
