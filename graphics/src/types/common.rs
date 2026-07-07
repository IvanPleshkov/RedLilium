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
/// When using `nalgebra`, use the "right-handed Z-up with depth 0 to 1"
/// projection functions.
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
/// // nalgebra example:
/// let proj = redlilium_core::math::perspective_rh(fov_y, aspect, near, far);
///
/// // Note: this uses [0, 1] depth range (wgpu/Vulkan convention)
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

    /// Clamp this rect to a `target_width` × `target_height` render target,
    /// returning non-negative, in-bounds integer bounds usable by either
    /// backend.
    ///
    /// Both APIs reject out-of-bounds scissors — Vulkan requires
    /// `offset >= 0` and the rect within the framebuffer
    /// (VUID-vkCmdSetScissor-x-00595), wgpu requires the rect ⊆ target and a
    /// negative origin cast to `u32` becomes ~4 billion — yet UI clip rects
    /// (egui) routinely go negative or past the edge during a resize. Clamp
    /// in this one shared place. A rect fully outside the target yields zero
    /// width/height (draws nothing), which is the correct result.
    pub fn clamped(&self, target_width: u32, target_height: u32) -> ClampedScissor {
        let tw = i64::from(target_width);
        let th = i64::from(target_height);
        // i64 math avoids overflow when x + width wraps i32/u32.
        let left = i64::from(self.x).clamp(0, tw);
        let top = i64::from(self.y).clamp(0, th);
        // `clamp` is monotonic, so right >= left and bottom >= top.
        let right = (i64::from(self.x) + i64::from(self.width)).clamp(0, tw);
        let bottom = (i64::from(self.y) + i64::from(self.height)).clamp(0, th);
        ClampedScissor {
            x: left as u32,
            y: top as u32,
            width: (right - left) as u32,
            height: (bottom - top) as u32,
        }
    }
}

/// A [`ScissorRect`] clamped to a render target: non-negative origin, extent
/// within bounds. Produced by [`ScissorRect::clamped`]; the `u32` fields are
/// safe to hand to both `vkCmdSetScissor` and `wgpu::RenderPass::set_scissor_rect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClampedScissor {
    /// Non-negative x origin, within `[0, target_width]`.
    pub x: u32,
    /// Non-negative y origin, within `[0, target_height]`.
    pub y: u32,
    /// Width, with `x + width <= target_width`.
    pub width: u32,
    /// Height, with `y + height <= target_height`.
    pub height: u32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scissor_within_bounds_unchanged() {
        let c = ScissorRect::new(10, 20, 100, 50).clamped(800, 600);
        assert_eq!((c.x, c.y, c.width, c.height), (10, 20, 100, 50));
    }

    #[test]
    fn scissor_negative_origin_shrinks_extent() {
        // Origin left/above the target: clamp to 0 and shrink the extent by
        // the clipped-off amount (egui clip rect during resize).
        let c = ScissorRect::new(-5, -8, 20, 30).clamped(800, 600);
        assert_eq!((c.x, c.y, c.width, c.height), (0, 0, 15, 22));
    }

    #[test]
    fn scissor_past_edge_clamps_extent() {
        // Extends past the right/bottom edge: extent shrinks to fit.
        let c = ScissorRect::new(790, 590, 100, 100).clamped(800, 600);
        assert_eq!((c.x, c.y, c.width, c.height), (790, 590, 10, 10));
    }

    #[test]
    fn scissor_fully_outside_is_empty() {
        // Entirely past the edge → zero extent (draws nothing).
        let c = ScissorRect::new(900, 700, 50, 50).clamped(800, 600);
        assert_eq!((c.x, c.y, c.width, c.height), (800, 600, 0, 0));
    }

    #[test]
    fn scissor_extreme_values_do_not_overflow() {
        // i32::MIN origin + u32::MAX width spans [-2.1e9, +2.1e9], which fully
        // contains the target — clamps to cover it, no panic or wrap.
        let c = ScissorRect::new(i32::MIN, i32::MIN, u32::MAX, u32::MAX).clamped(800, 600);
        assert_eq!((c.x, c.y, c.width, c.height), (0, 0, 800, 600));
        // i32::MAX origin is past the far edge → empty.
        let c = ScissorRect::new(i32::MAX, i32::MAX, u32::MAX, u32::MAX).clamped(800, 600);
        assert_eq!((c.x, c.y, c.width, c.height), (800, 600, 0, 0));
    }
}
