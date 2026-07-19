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

pub mod anchor;
pub mod bezier;
pub mod building;
mod draw;
pub mod graph;
pub mod junction;
pub mod parcel;
mod tool;

pub use anchor::{AnchorNodeAction, DeriveEdgeAnchors, EdgeAnchor, settle_edge_anchors};
pub use building::{Building, PlaceBuildingAction};
pub use draw::DrawLevelGraph;
pub use junction::{CreateJunctionAction, Junction, StampJunctionAction};
pub use parcel::{AddParcelAction, Parcel, ParcelGate, ParcelVertex, parcel_loop};
pub use tool::{AddNodeAction, AddRoadAction, CONNECT_TOOL, ConnectRoadsTool};

use redlilium_ecs::{Component, Entity, PostUpdate, Update, UpdateGlobalTransforms, World};

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
    /// The road meets `b` on its **+Z (front) side** instead of the default
    /// −Z. Set when `b` is a socket (junction connector, edge-anchored
    /// node): sockets keep +Z toward the road network, so a road arriving
    /// *at* one must approach from the front — and a road between two
    /// sockets departs `a` along +Z and enters `b` from the front.
    pub b_from_front: bool,
}

impl Default for RoadSegment {
    fn default() -> Self {
        Self {
            a: Entity::DANGLING,
            b: Entity::DANGLING,
            tangent_a: 0.0, // auto
            tangent_b: 0.0, // auto
            b_from_front: false,
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
        world.register_inspector_default::<EdgeAnchor>();
        world.register_inspector_default::<Parcel>();
        world.register_inspector_default::<ParcelVertex>();
        world.register_inspector_default::<ParcelGate>();
        world.register_inspector_default::<Building>();
    }

    fn build(&self, _app: &mut redlilium_runtime::App) {}

    fn build_editing_view(&self, view: &mut redlilium_runtime::EditingView<'_>) {
        view.schedules.get_mut::<Update>().add(DrawLevelGraph);
        // Derived data (anchored-node placements) writes the world, so it
        // cannot live in `Update` — the editor marks that schedule read-only
        // to structurally enforce the EditAction invariant. It goes in
        // `PostUpdate` next to `UpdateGlobalTransforms`, ordered before it
        // so a re-derived node's children (future buildings) propagate the
        // same frame.
        let post = view.schedules.get_mut::<PostUpdate>();
        post.add(DeriveEdgeAnchors::default());
        post.add_edge::<DeriveEdgeAnchors, UpdateGlobalTransforms>()
            .expect("derive → propagate is acyclic");
        // Click-select fallback: graph entities are meshless, so the GPU
        // entity-index pass cannot see them — nodes pick by cross-section,
        // roads by their midpoint handle cube.
        view.pickers.add(Box::new(tool::selection_pick));
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
        // "Add parcel": stamp a default rectangular boundary at the ground
        // click point; drag the vertex handles into shape afterwards.
        view.ops.add(
            "Add parcel",
            |ctx| ctx.cursor_ray.as_ref().and_then(tool::ground_hit).is_some(),
            |ctx| {
                if let Some(point) = ctx.cursor_ray.as_ref().and_then(tool::ground_hit) {
                    ctx.actions.push(Box::new(AddParcelAction::at_point(point)));
                }
            },
        );
        // "Place building": drop the default box recipe at the center of
        // each selected parcel (a parcel holds any number of buildings) —
        // move it and tune the parameters in the inspector afterwards.
        view.ops.add(
            "Place building",
            |ctx| {
                ctx.selection
                    .iter()
                    .any(|&e| ctx.world.get::<Parcel>(e).is_some())
            },
            |ctx| {
                let parcels: Vec<Entity> = ctx
                    .selection
                    .iter()
                    .copied()
                    .filter(|&e| ctx.world.get::<Parcel>(e).is_some())
                    .collect();
                for entity in parcels {
                    // Local drop point: centroid of the boundary vertices'
                    // local positions (the parcel origin when degenerate).
                    let centroid = ctx
                        .world
                        .get::<Parcel>(entity)
                        .map(|p| {
                            let locals: Vec<_> = p
                                .boundary
                                .iter()
                                .filter_map(|&v| {
                                    ctx.world
                                        .get::<redlilium_ecs::Transform>(v)
                                        .map(|t| t.translation)
                                })
                                .collect();
                            if locals.is_empty() {
                                redlilium_core::math::Vec3::zeros()
                            } else {
                                locals
                                    .iter()
                                    .fold(redlilium_core::math::Vec3::zeros(), |a, b| a + b)
                                    / locals.len() as f32
                            }
                        })
                        .unwrap_or_else(redlilium_core::math::Vec3::zeros);
                    let local = redlilium_ecs::Transform::new(
                        centroid,
                        redlilium_core::math::quat_from_rotation_y(0.0),
                        redlilium_core::math::Vec3::new(1.0, 1.0, 1.0),
                    );
                    ctx.actions.push(Box::new(PlaceBuildingAction::new(
                        entity,
                        local,
                        Building::default(),
                    )));
                }
            },
        );
        // Junction stamps: an N-armed template at the click point; drag the
        // connectors into shape, attach roads with the connect tool.
        for (label, arms) in [("Add 3-way junction", 3), ("Add 4-way junction", 4)] {
            view.ops.add(
                label,
                |ctx| ctx.cursor_ray.as_ref().and_then(tool::ground_hit).is_some(),
                move |ctx| {
                    if let Some(point) = ctx.cursor_ray.as_ref().and_then(tool::ground_hit) {
                        ctx.actions
                            .push(Box::new(StampJunctionAction::new(point, arms)));
                        // Straight into wiring: stamp, then click a connector
                        // and chain roads out of it.
                        ctx.request_tool = Some(CONNECT_TOOL.to_owned());
                    }
                },
            );
        }
    }
}
