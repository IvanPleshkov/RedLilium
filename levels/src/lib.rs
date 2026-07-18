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

pub mod attachment;
pub mod bezier;
mod draw;
pub mod graph;
pub mod junction;
mod tool;

pub use attachment::{AttachToEdgeAction, EdgeAttachment};
pub use draw::DrawLevelGraph;
pub use junction::{CreateJunctionAction, Junction, StampJunctionAction};
pub use tool::{AddNodeAction, AddRoadAction, CONNECT_TOOL, ConnectRoadsTool};

use redlilium_ecs::{Component, Entity, Update, World};

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
    /// Tangent length at `a`, meters — how far the patch keeps `a`'s
    /// heading. **`≤ 0` means auto**: `|chord| / 3` for a lone segment,
    /// Catmull-Rom (`|far_next − far_prev| / 6`, equal on both sides — C1)
    /// at a two-segment chain joint. See [`graph::segment_tangents`].
    pub tangent_a: f32,
    /// Tangent length at `b`, meters — same rules as `tangent_a`.
    pub tangent_b: f32,
}

impl Default for RoadSegment {
    fn default() -> Self {
        Self {
            a: Entity::DANGLING,
            b: Entity::DANGLING,
            tangent_a: 0.0, // auto
            tangent_b: 0.0, // auto
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
        world.register_inspector_default::<Junction>();
        world.register_inspector_default::<EdgeAttachment>();
    }

    fn build(&self, _app: &mut redlilium_runtime::App) {}

    fn build_editing_view(&self, view: &mut redlilium_runtime::EditingView<'_>) {
        view.schedules.get_mut::<Update>().add(DrawLevelGraph);
        view.tools.add(Box::new(ConnectRoadsTool::default()));
        // "Add road" arms the connect tool: clicks then place/connect nodes
        // until Escape. The op itself edits nothing.
        view.ops.add(
            "Add road",
            |_ctx| true,
            |ctx| ctx.request_tool = Some(CONNECT_TOOL.to_owned()),
        );
        // "Create junction": close the selected road nodes into one
        // junction loop (order irrelevant — derived by angle).
        view.ops.add(
            "Create junction",
            |ctx| {
                ctx.selection
                    .iter()
                    .filter(|&&e| ctx.world.get::<RoadNode>(e).is_some())
                    .count()
                    >= 3
            },
            |ctx| {
                let connectors: Vec<Entity> = ctx
                    .selection
                    .iter()
                    .copied()
                    .filter(|&e| ctx.world.get::<RoadNode>(e).is_some())
                    .collect();
                ctx.actions
                    .push(Box::new(CreateJunctionAction::new(connectors)));
            },
        );
        // "Add 4-way junction": stamp a cross template at the click point;
        // drag the connectors into shape, attach roads with the connect tool.
        view.ops.add(
            "Add 4-way junction",
            |ctx| ctx.cursor_ray.as_ref().and_then(tool::ground_hit).is_some(),
            |ctx| {
                if let Some(point) = ctx.cursor_ray.as_ref().and_then(tool::ground_hit) {
                    ctx.actions.push(Box::new(StampJunctionAction::new(point)));
                }
            },
        );
    }
}
