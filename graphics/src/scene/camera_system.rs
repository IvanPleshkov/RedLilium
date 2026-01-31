//! Camera system for managing per-camera render graphs.
//!
//! The camera system coordinates rendering across multiple cameras:
//! - Extracts camera data from ECS entities
//! - Sorts cameras by priority (texture targets first, then surfaces)
//! - Filters visible items per camera based on render layers
//! - Creates and executes render graphs for each camera
//!
//! # Priority Ordering
//!
//! Cameras are rendered in priority order (lower first):
//! 1. Texture-target cameras (negative priorities recommended)
//! 2. Surface-target cameras (zero/positive priorities)
//!
//! This ensures render-to-texture cameras complete before surface cameras
//! that might sample from those textures.

use std::cmp::Ordering;

use crate::error::GraphicsError;
use crate::graph::{PassType, RenderGraph};

use super::render_world::RenderWorld;
use glam::{Mat4, Vec3};

/// Extracted camera data ready for rendering.
///
/// This is a snapshot of camera state extracted from ECS,
/// containing all data needed to render from this camera's perspective.
#[derive(Debug, Clone)]
pub struct ExtractedCamera {
    /// Entity ID of the camera.
    pub entity_id: u64,
    /// View matrix (inverse of camera's global transform).
    pub view_matrix: Mat4,
    /// Projection matrix.
    pub projection_matrix: Mat4,
    /// Combined view-projection matrix.
    pub view_projection: Mat4,
    /// Camera world position for sorting.
    pub position: Vec3,
    /// Rendering priority (lower = renders first).
    pub priority: i32,
    /// Whether this renders to a texture (vs surface).
    pub is_texture_target: bool,
    /// Render target identifier.
    pub target_id: u64,
    /// Target size in pixels (width, height).
    pub target_size: (u32, u32),
    /// Clear color as [r, g, b, a] (None = don't clear).
    pub clear_color: Option<[f32; 4]>,
    /// Render layers mask.
    pub render_layers: u32,
    /// Viewport in pixels (x, y, width, height).
    pub viewport: (u32, u32, u32, u32),
}

impl ExtractedCamera {
    /// Creates a new extracted camera with default values.
    pub fn new(entity_id: u64) -> Self {
        Self {
            entity_id,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_projection: Mat4::IDENTITY,
            position: Vec3::ZERO,
            priority: 0,
            is_texture_target: false,
            target_id: 0,
            target_size: (1920, 1080),
            clear_color: Some([0.1, 0.1, 0.1, 1.0]),
            render_layers: 1,
            viewport: (0, 0, 1920, 1080),
        }
    }
}

/// Per-camera render context holding the graph and filtered items.
#[derive(Debug)]
pub struct CameraRenderContext {
    /// The extracted camera data.
    pub camera: ExtractedCamera,
    /// Render graph for this camera.
    pub graph: RenderGraph,
    /// Indices into RenderWorld opaque items visible to this camera.
    pub visible_opaque: Vec<usize>,
    /// Indices into RenderWorld masked items visible to this camera.
    pub visible_masked: Vec<usize>,
    /// Indices into RenderWorld transparent items visible to this camera.
    pub visible_transparent: Vec<usize>,
}

impl CameraRenderContext {
    /// Creates a new camera render context.
    pub fn new(camera: ExtractedCamera) -> Self {
        Self {
            camera,
            graph: RenderGraph::new(),
            visible_opaque: Vec::new(),
            visible_masked: Vec::new(),
            visible_transparent: Vec::new(),
        }
    }

    /// Sets up a basic forward rendering graph for this camera.
    pub fn setup_forward_graph(&mut self) {
        // Clear pass (if camera has clear color)
        if self.camera.clear_color.is_some() {
            let _clear_pass = self.graph.add_pass(
                format!("camera_{}_clear", self.camera.entity_id),
                PassType::Graphics,
            );
        }

        // Geometry pass for opaque objects
        let _geometry_pass = self.graph.add_pass(
            format!("camera_{}_geometry", self.camera.entity_id),
            PassType::Graphics,
        );

        // Masked pass for alpha-masked objects
        if !self.visible_masked.is_empty() {
            let _masked_pass = self.graph.add_pass(
                format!("camera_{}_masked", self.camera.entity_id),
                PassType::Graphics,
            );
        }

        // Transparent pass
        if !self.visible_transparent.is_empty() {
            let _transparent_pass = self.graph.add_pass(
                format!("camera_{}_transparent", self.camera.entity_id),
                PassType::Graphics,
            );
        }
    }

    /// Returns the total number of visible items for this camera.
    #[inline]
    pub fn visible_count(&self) -> usize {
        self.visible_opaque.len() + self.visible_masked.len() + self.visible_transparent.len()
    }

    /// Clears the render graph and visible item lists.
    pub fn clear(&mut self) {
        self.graph.clear();
        self.visible_opaque.clear();
        self.visible_masked.clear();
        self.visible_transparent.clear();
    }
}

/// CameraSystem manages extraction, ordering, and rendering of all cameras.
///
/// # Lifecycle
///
/// Each frame follows this pattern:
///
/// ```ignore
/// camera_system.begin_frame();
///
/// // Extract cameras from ECS
/// for (entity, camera, transform) in camera_query.iter() {
///     camera_system.add_camera(extracted_camera);
/// }
///
/// // Prepare (sort cameras, filter items, setup graphs)
/// camera_system.prepare(&render_world);
///
/// // Render all cameras
/// camera_system.render(&backend)?;
///
/// camera_system.end_frame();
/// ```
#[derive(Debug, Default)]
pub struct CameraSystem {
    /// Extracted cameras sorted by render order.
    cameras: Vec<CameraRenderContext>,
    /// Current frame number.
    frame_count: u64,
}

impl CameraSystem {
    /// Creates a new camera system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears all cameras for a new frame.
    pub fn begin_frame(&mut self) {
        for ctx in &mut self.cameras {
            ctx.clear();
        }
        self.cameras.clear();
    }

    /// Adds an extracted camera to the system.
    pub fn add_camera(&mut self, camera: ExtractedCamera) {
        self.cameras.push(CameraRenderContext::new(camera));
    }

    /// Returns the number of active cameras.
    #[inline]
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// Returns true if there are no cameras.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cameras.is_empty()
    }

    /// Sorts cameras by priority and target type.
    ///
    /// Texture-target cameras first (sorted by priority), then surface cameras.
    pub fn sort_cameras(&mut self) {
        self.cameras.sort_by(|a, b| {
            // First, texture targets before surface targets
            match (a.camera.is_texture_target, b.camera.is_texture_target) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => {
                    // Within same target type, sort by priority (lower first)
                    a.camera.priority.cmp(&b.camera.priority)
                }
            }
        });
    }

    /// Filters render world items for each camera based on render layers.
    pub fn filter_visible_items(&mut self, render_world: &RenderWorld) {
        for context in &mut self.cameras {
            let camera_layers = context.camera.render_layers;

            // Filter opaque items
            context.visible_opaque.clear();
            for (idx, item) in render_world.opaque_items().iter().enumerate() {
                if (item.mesh.render_layers & camera_layers) != 0 {
                    context.visible_opaque.push(idx);
                }
            }

            // Filter masked items
            context.visible_masked.clear();
            for (idx, item) in render_world.masked_items().iter().enumerate() {
                if (item.mesh.render_layers & camera_layers) != 0 {
                    context.visible_masked.push(idx);
                }
            }

            // Filter transparent items
            context.visible_transparent.clear();
            for (idx, item) in render_world.transparent_items().iter().enumerate() {
                if (item.mesh.render_layers & camera_layers) != 0 {
                    context.visible_transparent.push(idx);
                }
            }
        }
    }

    /// Sorts visible items per camera (front-to-back for opaque, back-to-front for transparent).
    pub fn sort_items_per_camera(&mut self, render_world: &RenderWorld) {
        for context in &mut self.cameras {
            let camera_pos = context.camera.position;

            // Sort opaque front-to-back (for early depth rejection)
            context.visible_opaque.sort_by(|&a, &b| {
                let item_a = &render_world.opaque_items()[a];
                let item_b = &render_world.opaque_items()[b];
                let dist_a = item_a.transform.world_position.distance_squared(camera_pos);
                let dist_b = item_b.transform.world_position.distance_squared(camera_pos);
                dist_a.partial_cmp(&dist_b).unwrap_or(Ordering::Equal)
            });

            // Sort transparent back-to-front (for correct blending)
            context.visible_transparent.sort_by(|&a, &b| {
                let item_a = &render_world.transparent_items()[a];
                let item_b = &render_world.transparent_items()[b];
                let dist_a = item_a.transform.world_position.distance_squared(camera_pos);
                let dist_b = item_b.transform.world_position.distance_squared(camera_pos);
                dist_b.partial_cmp(&dist_a).unwrap_or(Ordering::Equal)
            });
        }
    }

    /// Sets up render graphs for all cameras.
    pub fn setup_graphs(&mut self) {
        for context in &mut self.cameras {
            context.setup_forward_graph();
        }
    }

    /// Prepares all cameras for rendering.
    ///
    /// This performs:
    /// 1. Camera sorting (texture targets first, then by priority)
    /// 2. Item filtering by render layers
    /// 3. Item sorting per camera
    /// 4. Render graph setup
    pub fn prepare(&mut self, render_world: &RenderWorld) {
        self.sort_cameras();
        self.filter_visible_items(render_world);
        self.sort_items_per_camera(render_world);
        self.setup_graphs();

        log::trace!(
            "Prepared {} cameras with {} total items",
            self.cameras.len(),
            self.cameras
                .iter()
                .map(|c| c.visible_count())
                .sum::<usize>()
        );
    }

    /// Compiles and validates all camera render graphs.
    ///
    /// This prepares the graphs for execution. In the future, this will
    /// also execute the graphs on the GPU.
    pub fn render(&mut self) -> Result<(), GraphicsError> {
        for context in &mut self.cameras {
            // Compile this camera's render graph
            let _compiled = context.graph.compile().map_err(|e| {
                log::error!(
                    "Failed to compile camera {} graph: {}",
                    context.camera.entity_id,
                    e
                );
                GraphicsError::Internal(format!("Graph compilation failed: {e}"))
            })?;

            // TODO: Execute the compiled graph on the GPU

            log::trace!(
                "Rendered camera {} ({}) with {} opaque, {} masked, {} transparent items",
                context.camera.entity_id,
                if context.camera.is_texture_target {
                    "texture"
                } else {
                    "surface"
                },
                context.visible_opaque.len(),
                context.visible_masked.len(),
                context.visible_transparent.len()
            );
        }

        Ok(())
    }

    /// Ends the current frame.
    pub fn end_frame(&mut self) {
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    /// Returns an iterator over camera contexts.
    pub fn cameras(&self) -> impl Iterator<Item = &CameraRenderContext> {
        self.cameras.iter()
    }

    /// Returns a mutable iterator over camera contexts.
    pub fn cameras_mut(&mut self) -> impl Iterator<Item = &mut CameraRenderContext> {
        self.cameras.iter_mut()
    }

    /// Returns a reference to a camera context by index.
    pub fn get(&self, index: usize) -> Option<&CameraRenderContext> {
        self.cameras.get(index)
    }

    /// Returns a mutable reference to a camera context by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut CameraRenderContext> {
        self.cameras.get_mut(index)
    }

    /// Returns the current frame count.
    #[inline]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::RenderWorld;
    use crate::scene::extracted::{ExtractedMaterial, ExtractedMesh, ExtractedTransform};
    use glam::Vec4;

    fn make_camera(entity_id: u64, priority: i32, is_texture: bool) -> ExtractedCamera {
        ExtractedCamera {
            entity_id,
            priority,
            is_texture_target: is_texture,
            ..ExtractedCamera::new(0)
        }
    }

    fn make_render_item(position: Vec3, render_layers: u32) -> crate::scene::extracted::RenderItem {
        crate::scene::extracted::RenderItem {
            entity_id: 1,
            transform: ExtractedTransform::from_matrix(Mat4::from_translation(position)),
            mesh: ExtractedMesh {
                mesh_id: 1,
                cast_shadows: true,
                receive_shadows: true,
                render_layers,
            },
            material: ExtractedMaterial {
                base_color: Vec4::ONE,
                alpha_mode: 0, // Opaque
                ..ExtractedMaterial::DEFAULT
            },
        }
    }

    #[test]
    fn camera_system_lifecycle() {
        let mut system = CameraSystem::new();

        system.begin_frame();
        system.add_camera(make_camera(1, 0, false));

        let render_world = RenderWorld::new();
        system.prepare(&render_world);

        assert!(system.render().is_ok());
        system.end_frame();

        assert_eq!(system.frame_count(), 1);
    }

    #[test]
    fn camera_sorting_texture_first() {
        let mut system = CameraSystem::new();

        system.begin_frame();
        system.add_camera(make_camera(1, 0, false)); // Surface, priority 0
        system.add_camera(make_camera(2, -10, true)); // Texture, priority -10
        system.add_camera(make_camera(3, 10, false)); // Surface, priority 10

        let render_world = RenderWorld::new();
        system.prepare(&render_world);

        // Texture cameras should come first, then surface cameras by priority
        let cameras: Vec<_> = system.cameras().collect();
        assert_eq!(cameras[0].camera.entity_id, 2); // Texture first
        assert_eq!(cameras[1].camera.entity_id, 1); // Surface priority 0
        assert_eq!(cameras[2].camera.entity_id, 3); // Surface priority 10
    }

    #[test]
    fn camera_sorting_by_priority() {
        let mut system = CameraSystem::new();

        system.begin_frame();
        system.add_camera(make_camera(1, 10, true)); // Texture, priority 10
        system.add_camera(make_camera(2, -5, true)); // Texture, priority -5
        system.add_camera(make_camera(3, 0, true)); // Texture, priority 0

        let render_world = RenderWorld::new();
        system.prepare(&render_world);

        let cameras: Vec<_> = system.cameras().collect();
        assert_eq!(cameras[0].camera.entity_id, 2); // Priority -5
        assert_eq!(cameras[1].camera.entity_id, 3); // Priority 0
        assert_eq!(cameras[2].camera.entity_id, 1); // Priority 10
    }

    #[test]
    fn render_layer_filtering() {
        let mut system = CameraSystem::new();
        let mut render_world = RenderWorld::new();

        // Camera sees layer 1 only
        let mut camera = make_camera(1, 0, false);
        camera.render_layers = 1;

        system.begin_frame();
        system.add_camera(camera);

        // Add items on different layers
        render_world.add_item(make_render_item(Vec3::ZERO, 1)); // Layer 1 - visible
        render_world.add_item(make_render_item(Vec3::ONE, 2)); // Layer 2 - not visible
        render_world.add_item(make_render_item(Vec3::X, 1 | 2)); // Both layers - visible

        system.prepare(&render_world);

        let ctx = system.get(0).unwrap();
        assert_eq!(ctx.visible_opaque.len(), 2); // Only items on layer 1
    }

    #[test]
    fn multiple_cameras_same_items() {
        let mut system = CameraSystem::new();
        let mut render_world = RenderWorld::new();

        // Two cameras, both see layer 1
        let mut cam1 = make_camera(1, 0, false);
        cam1.render_layers = 1;
        let mut cam2 = make_camera(2, 1, false);
        cam2.render_layers = 1;

        system.begin_frame();
        system.add_camera(cam1);
        system.add_camera(cam2);

        render_world.add_item(make_render_item(Vec3::ZERO, 1));

        system.prepare(&render_world);

        // Both cameras should see the same item
        assert_eq!(system.get(0).unwrap().visible_opaque.len(), 1);
        assert_eq!(system.get(1).unwrap().visible_opaque.len(), 1);
    }
}
