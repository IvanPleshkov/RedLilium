use redlilium_core::math::Aabb;
use redlilium_debug_drawer::DebugDrawer;

use crate::ui::Selection;

/// Color for selected entity AABB wireframe (orange).
const SELECTION_COLOR: [f32; 4] = [1.0, 0.6, 0.0, 1.0];

/// How to display AABBs for a selected entity with multiple components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionAabbMode {
    /// Union all component AABBs into a single bounding box (default).
    #[default]
    United,
    /// Draw each component's AABB separately.
    PerComponent,
}

/// System that draws axis-aligned bounding boxes around selected entities.
///
/// For each selected entity, queries AABBs from all its components
/// (via [`Component::aabb`](crate::Component::aabb)), transforms them
/// into world space, and draws them using the [`DebugDrawer`].
///
/// The [`mode`](Self::mode) field controls whether component AABBs are
/// unioned into one box or drawn individually.
///
/// # Access
///
/// - Reads: `Selection` (resource), `DebugDrawer` (resource), `GlobalTransform` per entity
pub struct DrawSelectionAabb {
    /// How to combine component AABBs. Defaults to [`SelectionAabbMode::United`].
    pub mode: SelectionAabbMode,
}

impl Default for DrawSelectionAabb {
    fn default() -> Self {
        Self {
            mode: SelectionAabbMode::United,
        }
    }
}

impl crate::System for DrawSelectionAabb {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(crate::Res<Selection>, crate::Res<DebugDrawer>)>()
            .execute(|(selection, drawer)| {
                if selection.is_empty() {
                    return;
                }

                let world = ctx.world();
                let mut draw_ctx = drawer.context();

                match self.mode {
                    SelectionAabbMode::United => {
                        // Accumulate world-space AABBs across all selected entities
                        let mut combined: Option<Aabb> = None;
                        for &entity in selection.entities() {
                            let Some(aabb) = world.entity_aabb(entity) else {
                                continue;
                            };
                            let Some(gt) = world.get::<crate::GlobalTransform>(entity) else {
                                continue;
                            };
                            let world_aabb = transform_aabb(&aabb, &gt.0);
                            combined = Some(match combined {
                                Some(c) => c.union(&world_aabb),
                                None => world_aabb,
                            });
                        }
                        if let Some(aabb) = combined {
                            draw_ctx.draw_aabb(aabb.min, aabb.max, SELECTION_COLOR);
                        }
                    }
                    SelectionAabbMode::PerComponent => {
                        for &entity in selection.entities() {
                            let Some(gt) = world.get::<crate::GlobalTransform>(entity) else {
                                continue;
                            };
                            for aabb in world.entity_aabbs(entity) {
                                let world_aabb = transform_aabb(&aabb, &gt.0);
                                draw_ctx.draw_aabb(world_aabb.min, world_aabb.max, SELECTION_COLOR);
                            }
                        }
                    }
                }
            });
        Ok(())
    }
}

/// Transform a local-space AABB by a 4x4 world matrix, returning a new world-space AABB.
fn transform_aabb(aabb: &Aabb, matrix: &redlilium_core::math::Mat4) -> Aabb {
    let corners = [
        [aabb.min[0], aabb.min[1], aabb.min[2]],
        [aabb.max[0], aabb.min[1], aabb.min[2]],
        [aabb.max[0], aabb.max[1], aabb.min[2]],
        [aabb.min[0], aabb.max[1], aabb.min[2]],
        [aabb.min[0], aabb.min[1], aabb.max[2]],
        [aabb.max[0], aabb.min[1], aabb.max[2]],
        [aabb.max[0], aabb.max[1], aabb.max[2]],
        [aabb.min[0], aabb.max[1], aabb.max[2]],
    ];

    let mut world_min = [f32::MAX; 3];
    let mut world_max = [f32::MIN; 3];

    for corner in &corners {
        let v = redlilium_core::math::Vec4::new(corner[0], corner[1], corner[2], 1.0);
        let transformed = matrix * v;
        world_min[0] = world_min[0].min(transformed.x);
        world_min[1] = world_min[1].min(transformed.y);
        world_min[2] = world_min[2].min(transformed.z);
        world_max[0] = world_max[0].max(transformed.x);
        world_max[1] = world_max[1].max(transformed.y);
        world_max[2] = world_max[2].max(transformed.z);
    }

    Aabb::new(world_min, world_max)
}
