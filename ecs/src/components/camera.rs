//! Camera components for defining viewpoints and render targets.
//!
//! This module provides camera components that define how the scene is rendered:
//! - [`Camera`]: Main camera component with projection and render target settings
//! - [`CameraProjection`]: Projection mode (perspective, orthographic, or custom)
//! - [`RenderTarget`]: Where the camera renders to (window surface or texture)
//! - [`CameraViewport`]: Viewport rectangle within the render target

use bevy_ecs::component::Component;
use glam::{Mat4, Vec2, Vec4};

use super::GlobalTransform;

/// Projection mode for cameras.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraProjection {
    /// Perspective projection for 3D scenes.
    Perspective {
        /// Vertical field of view in radians.
        fov_y: f32,
        /// Near clipping plane distance.
        near: f32,
        /// Far clipping plane distance.
        far: f32,
    },
    /// Orthographic projection for 2D or isometric views.
    Orthographic {
        /// Half-height of the view in world units.
        scale: f32,
        /// Near clipping plane distance.
        near: f32,
        /// Far clipping plane distance.
        far: f32,
    },
    /// Custom projection matrix for special effects.
    Custom(Mat4),
}

impl Default for CameraProjection {
    fn default() -> Self {
        Self::Perspective {
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl CameraProjection {
    /// Creates a perspective projection with the given FOV in degrees.
    #[inline]
    pub fn perspective(fov_y_degrees: f32, near: f32, far: f32) -> Self {
        Self::Perspective {
            fov_y: fov_y_degrees.to_radians(),
            near,
            far,
        }
    }

    /// Creates an orthographic projection.
    #[inline]
    pub fn orthographic(scale: f32, near: f32, far: f32) -> Self {
        Self::Orthographic { scale, near, far }
    }

    /// Creates a custom projection from a matrix.
    #[inline]
    pub fn custom(matrix: Mat4) -> Self {
        Self::Custom(matrix)
    }

    /// Computes the projection matrix for the given aspect ratio.
    pub fn compute_matrix(&self, aspect_ratio: f32) -> Mat4 {
        match self {
            Self::Perspective { fov_y, near, far } => {
                Mat4::perspective_rh(*fov_y, aspect_ratio, *near, *far)
            }
            Self::Orthographic { scale, near, far } => {
                let half_width = scale * aspect_ratio;
                Mat4::orthographic_rh(-half_width, half_width, -*scale, *scale, *near, *far)
            }
            Self::Custom(matrix) => *matrix,
        }
    }

    /// Returns the near clipping plane distance.
    pub fn near(&self) -> f32 {
        match self {
            Self::Perspective { near, .. } => *near,
            Self::Orthographic { near, .. } => *near,
            Self::Custom(_) => 0.1, // Default for custom
        }
    }

    /// Returns the far clipping plane distance.
    pub fn far(&self) -> f32 {
        match self {
            Self::Perspective { far, .. } => *far,
            Self::Orthographic { far, .. } => *far,
            Self::Custom(_) => 1000.0, // Default for custom
        }
    }
}

/// Render target specification for a camera.
#[derive(Debug, Clone, PartialEq)]
pub enum RenderTarget {
    /// Render to the primary window surface.
    Surface {
        /// Window identifier (0 for main window).
        window_id: u32,
    },
    /// Render to an offscreen texture.
    Texture {
        /// Handle to the target texture resource.
        texture_id: u64,
        /// Size of the render target.
        size: Vec2,
    },
}

impl Default for RenderTarget {
    fn default() -> Self {
        Self::Surface { window_id: 0 }
    }
}

impl RenderTarget {
    /// Creates a surface render target for the main window.
    #[inline]
    pub fn main_window() -> Self {
        Self::Surface { window_id: 0 }
    }

    /// Creates a surface render target for a specific window.
    #[inline]
    pub fn window(window_id: u32) -> Self {
        Self::Surface { window_id }
    }

    /// Creates a texture render target.
    #[inline]
    pub fn texture(texture_id: u64, width: u32, height: u32) -> Self {
        Self::Texture {
            texture_id,
            size: Vec2::new(width as f32, height as f32),
        }
    }

    /// Returns true if this renders to a texture.
    #[inline]
    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture { .. })
    }

    /// Returns true if this renders to a surface.
    #[inline]
    pub fn is_surface(&self) -> bool {
        matches!(self, Self::Surface { .. })
    }

    /// Returns the size of the render target (only available for textures).
    #[inline]
    pub fn size(&self) -> Option<Vec2> {
        match self {
            Self::Texture { size, .. } => Some(*size),
            Self::Surface { .. } => None, // Size determined by window
        }
    }

    /// Returns the texture ID if this is a texture target.
    #[inline]
    pub fn texture_id(&self) -> Option<u64> {
        match self {
            Self::Texture { texture_id, .. } => Some(*texture_id),
            Self::Surface { .. } => None,
        }
    }

    /// Returns the window ID if this is a surface target.
    #[inline]
    pub fn window_id(&self) -> Option<u32> {
        match self {
            Self::Surface { window_id } => Some(*window_id),
            Self::Texture { .. } => None,
        }
    }
}

/// Viewport specification for a camera (normalized 0-1 coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraViewport {
    /// X offset (0-1 normalized).
    pub x: f32,
    /// Y offset (0-1 normalized).
    pub y: f32,
    /// Width (0-1 normalized).
    pub width: f32,
    /// Height (0-1 normalized).
    pub height: f32,
}

impl Default for CameraViewport {
    fn default() -> Self {
        Self::FULL
    }
}

impl CameraViewport {
    /// Full viewport covering the entire render target.
    pub const FULL: Self = Self {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    };

    /// Creates a viewport with the given normalized coordinates.
    #[inline]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Computes the pixel coordinates for a given target size.
    #[inline]
    pub fn to_pixels(&self, target_width: u32, target_height: u32) -> (u32, u32, u32, u32) {
        (
            (self.x * target_width as f32) as u32,
            (self.y * target_height as f32) as u32,
            (self.width * target_width as f32) as u32,
            (self.height * target_height as f32) as u32,
        )
    }

    /// Returns the aspect ratio of this viewport.
    #[inline]
    pub fn aspect_ratio(&self, target_width: u32, target_height: u32) -> f32 {
        let (_, _, w, h) = self.to_pixels(target_width, target_height);
        if h == 0 { 1.0 } else { w as f32 / h as f32 }
    }
}

/// Camera component that defines a viewpoint for rendering.
///
/// Entities with `Camera` + `Transform` + `GlobalTransform` will be processed
/// by the camera system to render the scene from their perspective.
///
/// # Priority
///
/// The `priority` field determines render order:
/// - Lower values render first
/// - Texture-target cameras should use negative priorities to ensure they render
///   before surface cameras that might sample from them
/// - Surface-target cameras typically use 0 or positive priorities
///
/// # Example
///
/// ```
/// use redlilium_ecs::components::{Camera, CameraProjection, RenderTarget, Transform, GlobalTransform};
/// use glam::Vec3;
///
/// // Main game camera rendering to screen
/// let main_camera = (
///     Camera::new().with_priority(0),
///     Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
///     GlobalTransform::IDENTITY,
/// );
///
/// // Minimap camera rendering to texture (renders first due to negative priority)
/// let minimap_camera = (
///     Camera::new()
///         .with_priority(-10)
///         .with_target(RenderTarget::texture(1, 256, 256))
///         .with_projection(CameraProjection::orthographic(50.0, 0.1, 500.0)),
///     Transform::from_xyz(0.0, 100.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
///     GlobalTransform::IDENTITY,
/// );
/// ```
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Camera {
    /// Projection mode (perspective, orthographic, or custom).
    pub projection: CameraProjection,

    /// Render target (surface or texture).
    pub target: RenderTarget,

    /// Rendering priority. Lower values render first.
    pub priority: i32,

    /// Whether this camera is active and should render.
    pub is_active: bool,

    /// Clear color for this camera (None = don't clear).
    pub clear_color: Option<Vec4>,

    /// Render layers mask - only entities on matching layers are visible.
    pub render_layers: u32,

    /// Viewport rectangle (normalized 0-1). Default is full target.
    pub viewport: CameraViewport,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            projection: CameraProjection::default(),
            target: RenderTarget::default(),
            priority: 0,
            is_active: true,
            clear_color: Some(Vec4::new(0.1, 0.1, 0.1, 1.0)),
            render_layers: 1, // Default layer
            viewport: CameraViewport::default(),
        }
    }
}

impl Camera {
    /// Creates a new camera with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns this camera with a different projection.
    #[must_use]
    pub fn with_projection(mut self, projection: CameraProjection) -> Self {
        self.projection = projection;
        self
    }

    /// Returns this camera with a different render target.
    #[must_use]
    pub fn with_target(mut self, target: RenderTarget) -> Self {
        self.target = target;
        self
    }

    /// Returns this camera with a different priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Returns this camera with active state changed.
    #[must_use]
    pub fn with_active(mut self, active: bool) -> Self {
        self.is_active = active;
        self
    }

    /// Returns this camera with a different clear color.
    #[must_use]
    pub fn with_clear_color(mut self, color: Vec4) -> Self {
        self.clear_color = Some(color);
        self
    }

    /// Returns this camera with no clear operation.
    #[must_use]
    pub fn without_clear(mut self) -> Self {
        self.clear_color = None;
        self
    }

    /// Returns this camera with different render layers.
    #[must_use]
    pub fn with_render_layers(mut self, layers: u32) -> Self {
        self.render_layers = layers;
        self
    }

    /// Returns this camera with a different viewport.
    #[must_use]
    pub fn with_viewport(mut self, viewport: CameraViewport) -> Self {
        self.viewport = viewport;
        self
    }

    /// Computes the view matrix from a GlobalTransform.
    ///
    /// The view matrix is the inverse of the camera's world transform.
    #[inline]
    pub fn compute_view_matrix(&self, global_transform: &GlobalTransform) -> Mat4 {
        global_transform.to_matrix().inverse()
    }

    /// Computes the projection matrix for the given target size.
    #[inline]
    pub fn compute_projection_matrix(&self, target_width: u32, target_height: u32) -> Mat4 {
        let aspect = self.viewport.aspect_ratio(target_width, target_height);
        self.projection.compute_matrix(aspect)
    }

    /// Computes the combined view-projection matrix.
    #[inline]
    pub fn compute_view_projection(
        &self,
        global_transform: &GlobalTransform,
        target_width: u32,
        target_height: u32,
    ) -> Mat4 {
        let view = self.compute_view_matrix(global_transform);
        let projection = self.compute_projection_matrix(target_width, target_height);
        projection * view
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn camera_default() {
        let camera = Camera::new();
        assert!(camera.is_active);
        assert_eq!(camera.priority, 0);
        assert!(camera.target.is_surface());
        assert!(camera.clear_color.is_some());
    }

    #[test]
    fn camera_builder() {
        let camera = Camera::new()
            .with_priority(-10)
            .with_target(RenderTarget::texture(1, 512, 512))
            .with_projection(CameraProjection::orthographic(50.0, 0.1, 500.0))
            .without_clear();

        assert_eq!(camera.priority, -10);
        assert!(camera.target.is_texture());
        assert!(camera.clear_color.is_none());
        assert!(matches!(
            camera.projection,
            CameraProjection::Orthographic { .. }
        ));
    }

    #[test]
    fn projection_perspective() {
        let proj = CameraProjection::perspective(60.0, 0.1, 100.0);
        let matrix = proj.compute_matrix(16.0 / 9.0);
        // Perspective matrix: w_axis.w is 0 (uses z for division)
        // z_axis.w is -1 for right-handed perspective
        assert!((matrix.z_axis.w - (-1.0)).abs() < 1e-6);
        // Should have non-zero focal length in x and y
        assert!(matrix.x_axis.x != 0.0);
        assert!(matrix.y_axis.y != 0.0);
    }

    #[test]
    fn projection_orthographic() {
        let proj = CameraProjection::orthographic(10.0, 0.1, 100.0);
        let matrix = proj.compute_matrix(1.0);
        // Orthographic should have 1.0 in w.w
        assert!((matrix.w_axis.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn render_target_surface() {
        let target = RenderTarget::main_window();
        assert!(target.is_surface());
        assert!(!target.is_texture());
        assert_eq!(target.window_id(), Some(0));
        assert!(target.texture_id().is_none());
    }

    #[test]
    fn render_target_texture() {
        let target = RenderTarget::texture(42, 256, 256);
        assert!(!target.is_surface());
        assert!(target.is_texture());
        assert_eq!(target.texture_id(), Some(42));
        assert!(target.window_id().is_none());
        assert_eq!(target.size(), Some(Vec2::new(256.0, 256.0)));
    }

    #[test]
    fn viewport_to_pixels() {
        let viewport = CameraViewport::new(0.25, 0.25, 0.5, 0.5);
        let (x, y, w, h) = viewport.to_pixels(1920, 1080);
        assert_eq!(x, 480);
        assert_eq!(y, 270);
        assert_eq!(w, 960);
        assert_eq!(h, 540);
    }

    #[test]
    fn view_matrix() {
        let camera = Camera::new();
        let transform = GlobalTransform::from_translation(Vec3::new(0.0, 5.0, 10.0));
        let view = camera.compute_view_matrix(&transform);
        // View matrix should move the origin to where camera was
        let origin_in_view = view.transform_point3(Vec3::new(0.0, 5.0, 10.0));
        assert!((origin_in_view - Vec3::ZERO).length() < 1e-5);
    }
}
