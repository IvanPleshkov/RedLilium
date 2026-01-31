//! Common types shared across the graphics system.

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
