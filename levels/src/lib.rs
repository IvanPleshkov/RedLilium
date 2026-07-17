//! # RedLilium Levels
//!
//! Semantic level authoring (docs/DESIGN_PROCEDURAL_LEVELS.md): scenes store
//! a *graph* — road cross-sections and the connections between them — and
//! geometry is derived from it. This crate owns the graph components and, for
//! now, an editing-view overlay that visualizes the graph through the
//! `DebugDrawer`; piece generation comes later.
//!
//! Boundary rule (design P2): no type in this crate leaks into `ecs` or
//! `editor`. The editor sees ordinary components via `register_types` and a
//! read-only preview system via `build_editing_view` — both generic
//! extension points.

pub mod bezier;
mod draw;

pub use draw::DrawLevelGraph;

use redlilium_ecs::{Component, Entity, Schedules, Update, World};

/// A road cross-section ("срез"): a straight segment along the entity's
/// **local X axis**, centered at the origin, `2 * half_width` long. The
/// entity's **local +Z is the chain direction** — the side roads continue
/// toward. Placing and orienting nodes is plain `Transform` editing; every
/// road attached to a node samples its boundary control points uniformly
/// along this segment.
#[derive(Clone, Component)]
pub struct RoadNode {
    /// Half of the cross-section length, meters.
    pub half_width: f32,
}

impl Default for RoadNode {
    fn default() -> Self {
        Self { half_width: 3.0 }
    }
}

/// A road connecting two [`RoadNode`] entities with a bicubic Bézier patch.
///
/// The patch's boundary rows are *derived* from the node segments (4 points
/// each, uniformly spaced — never stored); the two interior rows extend from
/// the boundaries along each node's +Z by `tangent_*` meters. Chain
/// convention: both nodes' +Z point along the road's travel direction, so
/// the patch leaves `a` along its +Z and arrives at `b` from its −Z side.
#[derive(Clone, Component)]
pub struct RoadSegment {
    /// Start cross-section entity (must carry [`RoadNode`]).
    pub a: Entity,
    /// End cross-section entity (must carry [`RoadNode`]).
    pub b: Entity,
    /// Tangent length at `a`, meters — how far the patch keeps `a`'s heading.
    pub tangent_a: f32,
    /// Tangent length at `b`, meters — how far the patch keeps `b`'s heading.
    pub tangent_b: f32,
}

impl Default for RoadSegment {
    fn default() -> Self {
        Self {
            a: Entity::DANGLING,
            b: Entity::DANGLING,
            tangent_a: 4.0,
            tangent_b: 4.0,
        }
    }
}

/// Registers the level-graph components and the editing-view overlay.
/// Compose from a game plugin: forward `register_types` and
/// `build_editing_view`; `build` is a no-op until runtime generation lands.
pub struct LevelsPlugin;

impl redlilium_runtime::Plugin for LevelsPlugin {
    fn register_types(&self, world: &mut World) {
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();
    }

    fn build(&self, _app: &mut redlilium_runtime::App) {}

    fn build_editing_view(&self, schedules: &mut Schedules) {
        schedules.get_mut::<Update>().add(DrawLevelGraph);
    }
}
