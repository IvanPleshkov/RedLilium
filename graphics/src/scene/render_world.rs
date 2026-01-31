//! RenderWorld holds extracted render data from an ECS world.
//!
//! The RenderWorld is a snapshot of render-relevant data extracted from the
//! main ECS world. It allows the main world to continue simulation while
//! rendering uses a consistent view of the previous frame's state.

use super::extracted::{ExtractedMaterial, ExtractedMesh, ExtractedTransform, RenderItem};

/// A snapshot of render-relevant data extracted from an ECS world.
///
/// The RenderWorld is populated during the extract phase and consumed during
/// the render phase. It provides a stable, read-only view of entity data
/// for rendering.
///
/// # Usage
///
/// ```ignore
/// let mut render_world = RenderWorld::new();
///
/// // Extract phase: populate from ECS
/// render_world.extract_from_ecs(&ecs_world);
///
/// // Render phase: use for rendering
/// for item in render_world.opaque_items() {
///     render_item(item);
/// }
///
/// // Clear for next frame
/// render_world.clear();
/// ```
#[derive(Debug, Default)]
pub struct RenderWorld {
    /// Opaque render items (alpha_mode == Opaque).
    opaque_items: Vec<RenderItem>,
    /// Transparent render items (alpha_mode == Blend), sorted back-to-front.
    transparent_items: Vec<RenderItem>,
    /// Alpha-masked render items (alpha_mode == Mask).
    masked_items: Vec<RenderItem>,
    /// Frame number for debugging.
    frame: u64,
}

impl RenderWorld {
    /// Creates a new empty render world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a render world with pre-allocated capacity.
    pub fn with_capacity(opaque: usize, transparent: usize, masked: usize) -> Self {
        Self {
            opaque_items: Vec::with_capacity(opaque),
            transparent_items: Vec::with_capacity(transparent),
            masked_items: Vec::with_capacity(masked),
            frame: 0,
        }
    }

    /// Clears all items for the next frame.
    pub fn clear(&mut self) {
        self.opaque_items.clear();
        self.transparent_items.clear();
        self.masked_items.clear();
        self.frame = self.frame.wrapping_add(1);
    }

    /// Returns the current frame number.
    #[inline]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Adds a render item to the appropriate queue based on alpha mode.
    pub fn add_item(&mut self, item: RenderItem) {
        match item.material.alpha_mode {
            0 => self.opaque_items.push(item),      // Opaque
            1 => self.masked_items.push(item),      // Mask
            2 => self.transparent_items.push(item), // Blend
            _ => self.opaque_items.push(item),      // Unknown -> Opaque
        }
    }

    /// Adds an item directly constructed from components.
    pub fn add(
        &mut self,
        entity_id: u64,
        transform: ExtractedTransform,
        mesh: ExtractedMesh,
        material: ExtractedMaterial,
    ) {
        self.add_item(RenderItem::new(entity_id, transform, mesh, material));
    }

    /// Returns all opaque items.
    #[inline]
    pub fn opaque_items(&self) -> &[RenderItem] {
        &self.opaque_items
    }

    /// Returns all transparent items.
    #[inline]
    pub fn transparent_items(&self) -> &[RenderItem] {
        &self.transparent_items
    }

    /// Returns all alpha-masked items.
    #[inline]
    pub fn masked_items(&self) -> &[RenderItem] {
        &self.masked_items
    }

    /// Returns the total number of items across all queues.
    #[inline]
    pub fn total_items(&self) -> usize {
        self.opaque_items.len() + self.transparent_items.len() + self.masked_items.len()
    }

    /// Returns true if there are no items to render.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_items() == 0
    }

    /// Sorts opaque items front-to-back for early depth rejection.
    ///
    /// Call this after extraction, before rendering.
    /// Requires a camera position for distance calculation.
    pub fn sort_opaque_front_to_back(&mut self, camera_position: glam::Vec3) {
        self.opaque_items.sort_by(|a, b| {
            let dist_a = a.transform.world_position.distance_squared(camera_position);
            let dist_b = b.transform.world_position.distance_squared(camera_position);
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sorts transparent items back-to-front for correct blending.
    ///
    /// Call this after extraction, before rendering.
    /// Requires a camera position for distance calculation.
    pub fn sort_transparent_back_to_front(&mut self, camera_position: glam::Vec3) {
        self.transparent_items.sort_by(|a, b| {
            let dist_a = a.transform.world_position.distance_squared(camera_position);
            let dist_b = b.transform.world_position.distance_squared(camera_position);
            dist_b
                .partial_cmp(&dist_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Iterates over all items (opaque first, then masked, then transparent).
    pub fn iter_all(&self) -> impl Iterator<Item = &RenderItem> {
        self.opaque_items
            .iter()
            .chain(self.masked_items.iter())
            .chain(self.transparent_items.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    fn make_item(entity_id: u64, alpha_mode: u8, position: Vec3) -> RenderItem {
        RenderItem {
            entity_id,
            transform: ExtractedTransform::from_matrix(Mat4::from_translation(position)),
            mesh: ExtractedMesh::new(1),
            material: ExtractedMaterial {
                alpha_mode,
                ..ExtractedMaterial::DEFAULT
            },
        }
    }

    #[test]
    fn render_world_sorting_by_alpha_mode() {
        let mut world = RenderWorld::new();

        world.add_item(make_item(1, 0, Vec3::ZERO)); // Opaque
        world.add_item(make_item(2, 2, Vec3::ZERO)); // Blend
        world.add_item(make_item(3, 1, Vec3::ZERO)); // Mask

        assert_eq!(world.opaque_items().len(), 1);
        assert_eq!(world.transparent_items().len(), 1);
        assert_eq!(world.masked_items().len(), 1);
        assert_eq!(world.total_items(), 3);
    }

    #[test]
    fn render_world_clear() {
        let mut world = RenderWorld::new();
        world.add_item(make_item(1, 0, Vec3::ZERO));

        assert_eq!(world.frame(), 0);
        world.clear();
        assert_eq!(world.frame(), 1);
        assert!(world.is_empty());
    }

    #[test]
    fn sort_opaque_front_to_back() {
        let mut world = RenderWorld::new();
        let camera = Vec3::ZERO;

        world.add_item(make_item(1, 0, Vec3::new(10.0, 0.0, 0.0))); // Far
        world.add_item(make_item(2, 0, Vec3::new(1.0, 0.0, 0.0))); // Near
        world.add_item(make_item(3, 0, Vec3::new(5.0, 0.0, 0.0))); // Middle

        world.sort_opaque_front_to_back(camera);

        assert_eq!(world.opaque_items()[0].entity_id, 2); // Near first
        assert_eq!(world.opaque_items()[1].entity_id, 3); // Middle
        assert_eq!(world.opaque_items()[2].entity_id, 1); // Far last
    }
}
