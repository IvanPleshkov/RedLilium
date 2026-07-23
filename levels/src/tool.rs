//! The "Connect roads" viewport tool: click-driven road-graph authoring.
//!
//! Click a node to anchor, click another node to connect them, click empty
//! ground to spawn a new node there and connect in one stroke — the freshly
//! created node becomes the next anchor, so a whole chain lays down as a
//! series of clicks. Escape drops the anchor, a second Escape leaves the
//! tool. Node picking is CPU-side ray-vs-segment math: graph nodes carry no
//! mesh, so the editor's GPU entity-index pass cannot see them.

use std::sync::{Arc, Mutex};

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::{Vec3, quat_from_rotation_y};
use redlilium_debug_drawer::DebugDrawer;
use redlilium_ecs::ui::{ToolFlow, ViewportRay, ViewportTool, ViewportToolCtx};
use redlilium_ecs::{Entity, GlobalTransform, Transform, World};

use crate::anchor::{self, AnchorNodeAction, EdgeAnchor, EdgeHit};
use crate::{RoadNode, RoadSegment, bezier};

/// Stable tool label — menu ops request activation by this name.
pub const CONNECT_TOOL: &str = "Connect roads";

/// Cursor-to-node pick distance, world units.
const PICK_RADIUS: f32 = 1.5;
/// Cursor-to-road-edge pick distance, world units.
const EDGE_PICK_RADIUS: f32 = 1.0;
/// Highlight/preview colors.
const HOVER_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const ANCHOR_COLOR: [f32; 4] = [1.0, 0.4, 0.1, 1.0];
const PREVIEW_COLOR: [f32; 4] = [0.6, 1.0, 0.6, 1.0];

#[derive(Default)]
pub struct ConnectRoadsTool {
    anchor: Option<Entity>,
    /// Ground-spawned nodes materialize a frame later (the action drains
    /// through the queue); the report hands the new entity back so it can
    /// become the anchor for chaining.
    pending_anchor: Option<Arc<Mutex<Option<Entity>>>>,
}

impl ViewportTool for ConnectRoadsTool {
    fn label(&self) -> &str {
        CONNECT_TOOL
    }

    fn status_hint(&self) -> &str {
        "click a node or road edge to anchor · click ground to place · Esc: drop anchor / exit"
    }

    fn update(&mut self, ctx: &mut ViewportToolCtx<'_>) -> ToolFlow {
        // Adopt a node created by last click's action, once it exists.
        let reported = self
            .pending_anchor
            .as_ref()
            .and_then(|report| *report.lock().expect("anchor report"));
        if let Some(node) = reported {
            self.anchor = Some(node);
            self.pending_anchor = None;
        }
        if let Some(anchor) = self.anchor
            && !ctx.world.is_alive(anchor)
        {
            self.anchor = None;
        }

        if ctx.input.escape {
            if self.anchor.take().is_some() {
                return ToolFlow::Continue;
            }
            return ToolFlow::Finished;
        }

        let Some(ray) = ctx.input.ray else {
            return ToolFlow::Continue;
        };
        let hovered = node_under_cursor(ctx.world, &ray);
        // Edge hover matters only when no node is hit (nodes win: they sit
        // at road ends where edges also pass close).
        let edge = hovered
            .is_none()
            .then(|| anchor::edge_under_cursor(ctx.world, &ray, EDGE_PICK_RADIUS))
            .flatten()
            // The anchor's own roads' edges are not landing targets.
            .filter(|hit| !road_touches(ctx.world, hit.road, self.anchor));

        let ground = ground_hit(&ray);

        self.draw_feedback(ctx, hovered, edge, ground);

        if ctx.input.clicked {
            match (self.anchor, hovered, edge) {
                // First click on a node: anchor there.
                (None, Some(node), _) => self.anchor = Some(node),
                // Click a second node: connect, chain from it. A socket
                // target is met from its front (+Z) side — except a gate,
                // which is two-sided and accepts a road from whichever
                // side it comes from.
                (Some(anchor), Some(node), _) => {
                    if node != anchor {
                        let from_front = socket_meets_front(ctx.world, node, anchor);
                        ctx.actions
                            .push(Box::new(AddRoadAction::new(anchor, node, from_front)));
                        self.anchor = Some(node);
                    }
                }
                // Click a road edge: spawn a node glued to it. With an
                // anchor, a road arrives at it from outside and the gesture
                // terminates (a driveway INTO the edge). Without one, the
                // fresh anchored node becomes the anchor — a road chain
                // grows OUT of the edge.
                (anchor, None, Some(hit)) => {
                    let half_width = anchor
                        .and_then(|a| ctx.world.get::<RoadNode>(a).map(|n| n.half_width))
                        .unwrap_or_else(|| RoadNode::default().half_width);
                    let (u_min, u_max) = anchor::interval_around(ctx.world, &hit, half_width);
                    let action = AnchorNodeAction::new(
                        EdgeAnchor {
                            parent_road: hit.road,
                            right_edge: hit.right_edge,
                            u_min,
                            u_max,
                        },
                        anchor,
                    );
                    if anchor.is_none() {
                        self.pending_anchor = Some(action.report.clone());
                    } else {
                        self.anchor = None;
                    }
                    ctx.actions.push(Box::new(action));
                }
                // Click empty ground: spawn a node there (and a road from
                // the anchor, when one is set). The report chains the anchor.
                (anchor, None, None) => {
                    if let Some(point) = ground {
                        let action =
                            AddNodeAction::new(anchor, node_transform(anchor, point, ctx.world));
                        self.pending_anchor = Some(action.report.clone());
                        ctx.actions.push(Box::new(action));
                    }
                }
            }
        }
        ToolFlow::Continue
    }

    fn deactivate(&mut self) {
        self.anchor = None;
        self.pending_anchor = None;
    }
}

impl ConnectRoadsTool {
    fn draw_feedback(
        &self,
        ctx: &ViewportToolCtx<'_>,
        hovered: Option<Entity>,
        edge: Option<EdgeHit>,
        ground: Option<Vec3>,
    ) {
        if !ctx.world.has_resource::<DebugDrawer>() {
            return;
        }
        let drawer = ctx.world.resource::<DebugDrawer>();
        let mut draw = drawer.context();
        let pt = |v: &Vec3| [v.x, v.y, v.z];

        for (entity, color) in [(hovered, HOVER_COLOR), (self.anchor, ANCHOR_COLOR)] {
            let Some(entity) = entity else { continue };
            let Some((world_mat, half)) = node_shape(ctx.world, entity) else {
                continue;
            };
            let section = bezier::cross_section(&world_mat, half);
            let center = (section[0] + section[3]) * 0.5;
            draw.draw_circle(
                pt(&center),
                half + 0.4,
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                color,
            );
        }

        // Preview the would-be edge-anchored node on the hovered edge: its
        // chord cross-section, plus the road arriving from the anchor when
        // one is set.
        if let Some(hit) = edge {
            draw.draw_circle(
                pt(&hit.point),
                0.6,
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                HOVER_COLOR,
            );
            let half_width = self
                .anchor
                .and_then(|a| ctx.world.get::<RoadNode>(a).map(|n| n.half_width))
                .unwrap_or_else(|| RoadNode::default().half_width);
            let (u_min, u_max) = anchor::interval_around(ctx.world, &hit, half_width);
            let preview = EdgeAnchor {
                parent_road: hit.road,
                right_edge: hit.right_edge,
                u_min,
                u_max,
            };
            if let Some((t, hw)) = anchor::derive_anchor_state(ctx.world, &preview) {
                let mat = t.to_matrix();
                let section = bezier::cross_section(&mat, hw);
                draw.draw_line(pt(&section[0]), pt(&section[3]), HOVER_COLOR);
                let center = (section[0] + section[3]) * 0.5;
                let out = bezier::heading(&mat);
                draw.draw_line(pt(&center), pt(&(center + out * 1.5)), HOVER_COLOR);
                if let Some((a_mat, a_half)) = self.anchor.and_then(|a| node_shape(ctx.world, a)) {
                    let a_c = Vec3::new(a_mat[(0, 3)], a_mat[(1, 3)], a_mat[(2, 3)]);
                    let chord = ((t.translation - a_c).norm() / 3.0).max(0.1);
                    let patch =
                        bezier::patch_from_nodes(&a_mat, a_half, chord, &mat, hw, chord, true);
                    draw_patch_outline(&mut draw, &patch, PREVIEW_COLOR);
                }
            }
            return;
        }

        let Some(anchor) = self.anchor else { return };

        // Preview patch from the anchor to the hovered node / ground cursor.
        let Some((a_mat, a_half)) = node_shape(ctx.world, anchor) else {
            return;
        };
        let end = hovered
            .and_then(|node| node_shape(ctx.world, node))
            .or_else(|| {
                ground.map(|point| {
                    let t = node_transform(Some(anchor), point, ctx.world);
                    (t.to_matrix(), RoadNode::default().half_width)
                })
            });
        let Some((b_mat, b_half)) = end else { return };
        // Auto-tangent preview: chord/3, like a lone segment resolves.
        let chord = {
            let a = Vec3::new(a_mat[(0, 3)], a_mat[(1, 3)], a_mat[(2, 3)]);
            let b = Vec3::new(b_mat[(0, 3)], b_mat[(1, 3)], b_mat[(2, 3)]);
            ((b - a).norm() / 3.0).max(0.1)
        };
        let from_front = hovered.is_some_and(|node| socket_meets_front(ctx.world, node, anchor));
        let patch =
            bezier::patch_from_nodes(&a_mat, a_half, chord, &b_mat, b_half, chord, from_front);
        draw_patch_outline(&mut draw, &patch, PREVIEW_COLOR);
    }
}

/// Coarse patch preview: three longitudinal curves (edges + centerline).
fn draw_patch_outline(
    draw: &mut redlilium_debug_drawer::DebugDrawerContext<'_>,
    patch: &crate::bezier::Patch,
    color: [f32; 4],
) {
    let pt = |v: &Vec3| [v.x, v.y, v.z];
    for v in [0.0, 0.5, 1.0] {
        let mut prev = bezier::eval(patch, 0.0, v);
        for step in 1..=12 {
            let next = bezier::eval(patch, step as f32 / 12.0, v);
            draw.draw_line(pt(&prev), pt(&next), color);
            prev = next;
        }
    }
}

/// Whether `road`'s segment starts or ends at `node` — the anchor's own
/// roads are not attachment targets (a driveway onto itself).
fn road_touches(world: &World, road: Entity, node: Option<Entity>) -> bool {
    let Some(node) = node else { return false };
    world
        .get::<RoadSegment>(road)
        .is_some_and(|seg| seg.a == node || seg.b == node)
}

/// Which side a road drawn from `from` meets socket `node` on. Junction
/// connectors and edge-anchored nodes only ever accept their front (+Z) —
/// the back side is junction fill / the parent road. A **gate is
/// two-sided**: it sits on a stroke, which has no interior — a road
/// arriving from the gate's front meets the front, a road from behind
/// connects from behind. Non-sockets always take the default −Z arrival.
fn socket_meets_front(world: &World, node: Entity, from: Entity) -> bool {
    if !is_socket(world, node) {
        return false;
    }
    if world.get::<crate::stroke::Gate>(node).is_some()
        && let (Some((gate_m, _)), Some((from_m, _))) =
            (node_shape(world, node), node_shape(world, from))
    {
        let center = |m: &redlilium_core::math::Mat4| {
            let c = m * redlilium_core::math::Vec4::new(0.0, 0.0, 0.0, 1.0);
            Vec3::new(c.x, c.y, c.z)
        };
        let out = bezier::heading(&gate_m);
        return (center(&from_m) - center(&gate_m)).dot(&out) >= 0.0;
    }
    true
}

/// Whether `node` is a socket — a junction connector, an edge-anchored
/// node, or a stroke gate. Sockets keep +Z toward the road network, so a
/// road arriving AT one must meet it from the front
/// (`RoadSegment::b_from_front`).
fn is_socket(world: &World, node: Entity) -> bool {
    if world.get::<EdgeAnchor>(node).is_some() || world.get::<crate::stroke::Gate>(node).is_some() {
        return true;
    }
    if let Ok(junctions) = world.read_all::<crate::Junction>()
        && junctions.iter().any(|(_, j)| j.connectors.contains(&node))
    {
        return true;
    }
    false
}

/// A node's world matrix + half width, when it is a live road node.
fn node_shape(world: &World, entity: Entity) -> Option<(redlilium_core::math::Mat4, f32)> {
    let node = world.get::<RoadNode>(entity)?;
    let gt = world.get::<GlobalTransform>(entity)?;
    Some((gt.0, node.half_width))
}

/// CPU pick: the closest road node whose cross-section segment passes within
/// [`PICK_RADIUS`] of the cursor ray.
fn node_under_cursor(world: &World, ray: &ViewportRay) -> Option<Entity> {
    let nodes = world.read_all::<RoadNode>().ok()?;
    let mut best: Option<(f32, Entity)> = None;
    for (index, node) in nodes.iter() {
        let Some(entity) = world.entity_at_index(index) else {
            continue;
        };
        let Some(gt) = world.get::<GlobalTransform>(entity) else {
            continue;
        };
        let section = bezier::cross_section(&gt.0, node.half_width);
        let dist = ray_segment_distance(ray, section[0], section[3]);
        if dist <= PICK_RADIUS && best.is_none_or(|(d, _)| dist < d) {
            best = Some((dist, entity));
        }
    }
    best.map(|(_, e)| e)
}

/// Pick radius of a road's selection-handle cube (matches its drawn size,
/// with a little slack).
const ROAD_HANDLE_PICK_RADIUS: f32 = crate::draw::HANDLE_HALF * 2.0;

/// CPU selection pick for the editor's click-select: graph entities carry
/// no mesh, so the GPU entity-index pass cannot see them. The candidate is
/// chosen by cursor proximity between road nodes (their cross-section
/// segments) and road handle cubes (patch midpoints — the proxy for the
/// road entity); the claim is then depth-checked against the editor's own
/// scene hit — a mesh in front of the control keeps the editor's pick.
pub(crate) fn selection_pick(
    world: &World,
    query: &redlilium_ecs::ui::ViewportPickQuery,
) -> Option<redlilium_ecs::ui::ViewportPickHit> {
    let ray = &query.ray;
    // Two pick tiers: point handles (tier 1 — stroke vertex cubes, road
    // midpoint cubes) beat extended geometry (tier 0 — cross-sections,
    // stroke paths, footprint rings) whenever they are within their own
    // radius. Without the tiers a handle sitting ON a line could never
    // win: the line through it is always at least as close to the ray.
    // (priority, perpendicular distance, t along ray, entity).
    let mut best: Option<(u8, f32, f32, Entity)> = None;
    let mut consider = |priority: u8, dist: f32, limit: f32, t: f32, entity: Entity| {
        if dist <= limit
            && best.is_none_or(|(p, d, _, _)| priority > p || (priority == p && dist < d))
        {
            best = Some((priority, dist, t, entity));
        }
    };
    if let Ok(nodes) = world.read_all::<RoadNode>() {
        for (index, node) in nodes.iter() {
            let Some(entity) = world.entity_at_index(index) else {
                continue;
            };
            let Some(gt) = world.get::<GlobalTransform>(entity) else {
                continue;
            };
            let section = bezier::cross_section(&gt.0, node.half_width);
            let (dist, _, t) = ray_segment_closest(ray, section[0], section[3]);
            consider(0, dist, PICK_RADIUS, t, entity);
        }
    }
    if let Ok(segments) = world.read_all::<RoadSegment>() {
        for (index, _) in segments.iter() {
            let Some(road) = world.entity_at_index(index) else {
                continue;
            };
            let Some(patch) = crate::graph::road_patch(world, road) else {
                continue;
            };
            let mid = bezier::eval(&patch, 0.5, 0.5);
            let (dist, _, t) = ray_segment_closest(ray, mid, mid);
            consider(1, dist, ROAD_HANDLE_PICK_RADIUS, t, road);
        }
    }
    if let Ok(strokes) = world.read_all::<crate::stroke::Stroke>() {
        for (index, stroke) in strokes.iter() {
            let Some(entity) = world.entity_at_index(index) else {
                continue;
            };
            let Some(path) = crate::stroke::stroke_path(world, stroke) else {
                continue;
            };
            // The stroke picks by its (tessellated) open path; the vertex
            // handle cubes pick the vertex entities themselves and, as
            // point handles, take the higher tier.
            for i in 0..path.len() - 1 {
                let (dist, _, t) = ray_segment_closest(ray, path[i], path[i + 1]);
                consider(0, dist, PICK_RADIUS, t, entity);
            }
            if let Some(corners) = crate::stroke::stroke_corners(world, stroke) {
                for (vertex, p, _, _) in corners {
                    let (dist, _, t) = ray_segment_closest(ray, p, p);
                    consider(1, dist, ROAD_HANDLE_PICK_RADIUS, t, vertex);
                }
            }
        }
    }
    if let Ok(cuts) = world.read_all::<crate::cut::Cut>() {
        for (index, cut) in cuts.iter() {
            let Some(entity) = world.entity_at_index(index) else {
                continue;
            };
            // A cut picks by either lip (the derived lower one included —
            // it is the same entity); its vertex cubes pick the vertices.
            let Some((upper, lower)) = crate::cut::cut_paths(world, cut) else {
                continue;
            };
            for lip in [&upper, &lower] {
                for i in 0..lip.len() - 1 {
                    let (dist, _, t) = ray_segment_closest(ray, lip[i], lip[i + 1]);
                    consider(0, dist, PICK_RADIUS, t, entity);
                }
            }
            if let Some(corners) = crate::cut::cut_corners(world, cut) {
                for (vertex, p, _, _) in corners {
                    let (dist, _, t) = ray_segment_closest(ray, p, p);
                    consider(1, dist, ROAD_HANDLE_PICK_RADIUS, t, vertex);
                }
            }
        }
    }
    if let Ok(buildings) = world.read_all::<crate::building::Building>() {
        for (index, building) in buildings.iter() {
            let Some(entity) = world.entity_at_index(index) else {
                continue;
            };
            let Some(gt) = world.get::<GlobalTransform>(entity) else {
                continue;
            };
            // Buildings pick by their ground-floor footprint perimeter.
            let ring = crate::building::footprint_corners(building);
            let corner = |[x, z]: [f32; 2]| {
                let p = gt.0 * redlilium_core::math::Vec4::new(x, 0.0, z, 1.0);
                Vec3::new(p.x, p.y, p.z)
            };
            for i in 0..4 {
                let (dist, _, t) =
                    ray_segment_closest(ray, corner(ring[i]), corner(ring[(i + 1) % 4]));
                consider(0, dist, PICK_RADIUS, t, entity);
            }
        }
    }
    let (_, _, t, entity) = best?;
    if let Some(point) = query.scene_point {
        let t_scene = (point - ray.origin).dot(&ray.dir) / ray.dir.dot(&ray.dir).max(1e-8);
        if t_scene < t {
            return None; // the scene mesh is in front — editor's pick stands
        }
    }
    Some(redlilium_ecs::ui::ViewportPickHit { entity, t })
}

/// Intersection of the cursor ray with the ground plane (y = 0), when the
/// ray points at it. Also serves the menu ops that place things at the
/// click point.
pub(crate) fn ground_hit(ray: &ViewportRay) -> Option<Vec3> {
    if ray.dir.y.abs() < 1e-6 {
        return None;
    }
    let t = -ray.origin.y / ray.dir.y;
    (t > 0.0).then(|| ray.origin + ray.dir * t)
}

/// Closest distance between a ray and a segment.
fn ray_segment_distance(ray: &ViewportRay, p0: Vec3, p1: Vec3) -> f32 {
    ray_segment_closest(ray, p0, p1).0
}

/// Closest approach between a ray and a segment: `(distance, s, t)` where
/// `s` is the segment parameter of the closest point and `t` the ray
/// parameter (depth along the ray). Shared with edge and selection picking.
pub(crate) fn ray_segment_closest(ray: &ViewportRay, p0: Vec3, p1: Vec3) -> (f32, f32, f32) {
    let u = ray.dir;
    let v = p1 - p0;
    let w = ray.origin - p0;
    let a = u.dot(&u);
    let b = u.dot(&v);
    let c = v.dot(&v).max(1e-8);
    let d = u.dot(&w);
    let e = v.dot(&w);
    let denom = a * c - b * b;
    let s = if denom.abs() < 1e-6 {
        (e / c).clamp(0.0, 1.0)
    } else {
        ((a * e - b * d) / denom).clamp(0.0, 1.0)
    };
    let on_segment = p0 + v * s;
    let t = ((on_segment - ray.origin).dot(&u) / a).max(0.0);
    let on_ray = ray.origin + u * t;
    ((on_ray - on_segment).norm(), s, t)
}

/// Transform for a ground-spawned node at `point`: oriented so its +Z (the
/// chain direction) continues away from the anchor; identity yaw when there
/// is no anchor.
fn node_transform(anchor: Option<Entity>, point: Vec3, world: &World) -> Transform {
    let yaw = anchor
        .and_then(|a| node_shape(world, a))
        .map(|(mat, _)| {
            let center4 = mat * redlilium_core::math::Vec4::new(0.0, 0.0, 0.0, 1.0);
            let dir = point - Vec3::new(center4.x, center4.y, center4.z);
            dir.x.atan2(dir.z)
        })
        .unwrap_or(0.0);
    Transform::new(point, quat_from_rotation_y(yaw), Vec3::new(1.0, 1.0, 1.0))
}

// ---------------------------------------------------------------------------
// Edit actions
// ---------------------------------------------------------------------------

/// Undoable "connect two existing nodes with a road".
#[derive(Debug)]
pub struct AddRoadAction {
    a: Entity,
    b: Entity,
    b_from_front: bool,
    road: Option<Entity>,
}

impl AddRoadAction {
    pub fn new(a: Entity, b: Entity, b_from_front: bool) -> Self {
        Self {
            a,
            b,
            b_from_front,
            road: None,
        }
    }
}

impl EditAction<World> for AddRoadAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.a) || !world.is_alive(self.b) {
            return Err(EditActionError::TargetNotFound(
                "road node despawned".into(),
            ));
        }
        let road = world.spawn();
        world
            .insert(
                road,
                RoadSegment {
                    a: self.a,
                    b: self.b,
                    b_from_front: self.b_from_front,
                    ..RoadSegment::default()
                },
            )
            .map_err(|e| EditActionError::Custom(e.to_string()))?;
        self.road = Some(road);
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if let Some(road) = self.road.take() {
            world.despawn(road);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Add road"
    }
}

/// Undoable "spawn a road node at a point" — with an optional road from an
/// existing anchor node. The `report` hands the created node back to the
/// tool (actions apply a frame after they are pushed).
///
/// When the anchor becomes an *interior chain node* (exactly two roads, not
/// a junction connector or attachment socket), the same click re-aims its
/// +Z at the Catmull-Rom direction `prev → new` — without this, a chained
/// node keeps the first chord's heading forever and laid chains "snake"
/// (each span departs along the previous chord, then has to bend back).
pub struct AddNodeAction {
    anchor: Option<Entity>,
    transform: Transform,
    created_node: Option<Entity>,
    created_road: Option<Entity>,
    /// The anchor's pre-reorientation transform, for undo.
    reoriented: Option<(Entity, Transform)>,
    pub report: Arc<Mutex<Option<Entity>>>,
}

impl AddNodeAction {
    pub fn new(anchor: Option<Entity>, transform: Transform) -> Self {
        Self {
            anchor,
            transform,
            created_node: None,
            created_road: None,
            reoriented: None,
            report: Arc::new(Mutex::new(None)),
        }
    }
}

/// Whether `node` may be auto-reoriented by chain authoring: a plain
/// interior chain node — exactly two roads, no junction or attachment
/// claims it (sockets keep their authored heading). Returns the far end of
/// its *other* road (the chain-previous node), skipping `exclude`.
fn reorientable(world: &World, node: Entity, exclude: Entity) -> Option<Entity> {
    let mut prev = None;
    let mut roads = 0;
    let segments = world.read_all::<RoadSegment>().ok()?;
    for (_, seg) in segments.iter() {
        let far = if seg.a == node {
            Some(seg.b)
        } else if seg.b == node {
            Some(seg.a)
        } else {
            None
        };
        if let Some(far) = far {
            roads += 1;
            if far != exclude {
                prev = Some(far);
            }
        }
    }
    drop(segments);
    if roads != 2 {
        return None;
    }
    if world.get::<EdgeAnchor>(node).is_some() || world.get::<crate::stroke::Gate>(node).is_some() {
        return None;
    }
    if let Ok(junctions) = world.read_all::<crate::Junction>()
        && junctions.iter().any(|(_, j)| j.connectors.contains(&node))
    {
        return None;
    }
    prev
}

impl std::fmt::Debug for AddNodeAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddNodeAction")
            .field("anchor", &self.anchor)
            .finish()
    }
}

impl EditAction<World> for AddNodeAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let node = world.spawn();
        let insert = |world: &mut World, e, r: Result<(), redlilium_ecs::WorldError>| {
            r.map_err(|err| {
                world.despawn(e);
                EditActionError::Custom(err.to_string())
            })
        };
        let t = world.insert(node, self.transform);
        insert(world, node, t)?;
        let g = world.insert(node, GlobalTransform(self.transform.to_matrix()));
        insert(world, node, g)?;
        let n = world.insert(node, RoadNode::default());
        insert(world, node, n)?;

        self.created_road = None;
        if let Some(anchor) = self.anchor.filter(|a| world.is_alive(*a)) {
            let road = world.spawn();
            let r = world.insert(
                road,
                RoadSegment {
                    a: anchor,
                    b: node,
                    ..RoadSegment::default()
                },
            );
            if let Err(err) = r {
                world.despawn(road);
                world.despawn(node);
                return Err(EditActionError::Custom(err.to_string()));
            }
            self.created_road = Some(road);
        }

        // Chain smoothing: re-aim an interior anchor at prev → new.
        self.reoriented = None;
        if let (Some(anchor), Some(_)) = (self.anchor, self.created_road)
            && let Some(prev) = reorientable(world, anchor, node)
            && let (Some(prev_c), Some(new_c)) = (
                crate::graph::node_center(world, prev),
                crate::graph::node_center(world, node),
            )
        {
            let dir = new_c - prev_c;
            if dir.norm() > 1e-4
                && let Some(old_t) = world.get::<Transform>(anchor).copied()
            {
                let new_t = Transform::new(
                    old_t.translation,
                    quat_from_rotation_y(dir.x.atan2(dir.z)),
                    old_t.scale,
                );
                let _ = world.insert(anchor, new_t);
                let _ = world.insert(anchor, GlobalTransform(new_t.to_matrix()));
                self.reoriented = Some((anchor, old_t));
            }
        }

        self.created_node = Some(node);
        *self.report.lock().expect("anchor report") = Some(node);
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if let Some(road) = self.created_road.take() {
            world.despawn(road);
        }
        if let Some(node) = self.created_node.take() {
            world.despawn(node);
        }
        if let Some((anchor, old_t)) = self.reoriented.take()
            && world.is_alive(anchor)
        {
            let _ = world.insert(anchor, old_t);
            let _ = world.insert(anchor, GlobalTransform(old_t.to_matrix()));
        }
        *self.report.lock().expect("anchor report") = None;
        Ok(())
    }

    fn description(&self) -> &str {
        "Add road node"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray(origin: [f32; 3], dir: [f32; 3]) -> ViewportRay {
        ViewportRay {
            origin: Vec3::new(origin[0], origin[1], origin[2]),
            dir: Vec3::new(dir[0], dir[1], dir[2]).normalize(),
        }
    }

    #[test]
    fn ray_segment_distance_hits_and_misses() {
        // Ray straight down over the middle of an X-axis segment at origin.
        let r = ray([0.0, 10.0, 0.0], [0.0, -1.0, 0.0]);
        let d = ray_segment_distance(&r, Vec3::new(-3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0));
        assert!(d < 1e-4);
        // Ray passing 2 units to the side of the segment's end.
        let r = ray([5.0, 10.0, 0.0], [0.0, -1.0, 0.0]);
        let d = ray_segment_distance(&r, Vec3::new(-3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0));
        assert!((d - 2.0).abs() < 1e-4);
    }

    #[test]
    fn ground_hit_requires_downward_ray() {
        assert!(ground_hit(&ray([0.0, 5.0, 0.0], [0.0, -1.0, 0.0])).is_some());
        assert!(ground_hit(&ray([0.0, 5.0, 0.0], [0.0, 1.0, 0.0])).is_none());
        assert!(ground_hit(&ray([0.0, 5.0, 0.0], [1.0, 0.0, 0.0])).is_none());
    }

    #[test]
    fn selection_pick_defers_to_a_nearer_scene_hit() {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();

        let node = world.spawn();
        let t = Transform::default();
        world.insert(node, t).unwrap();
        world.insert(node, GlobalTransform(t.to_matrix())).unwrap();
        world.insert(node, RoadNode::default()).unwrap();

        // Ray straight down onto the node (t = 10 at the hit).
        let ray = ViewportRay {
            origin: Vec3::new(0.0, 10.0, 0.0),
            dir: Vec3::new(0.0, -1.0, 0.0),
        };
        let query = |scene_point| redlilium_ecs::ui::ViewportPickQuery {
            ray,
            scene_entity: None,
            scene_point,
        };

        // Empty space under the click: the node is claimed.
        let hit = selection_pick(&world, &query(None)).expect("claim");
        assert_eq!(hit.entity, node);
        assert!((hit.t - 10.0).abs() < 1e-3);
        // A scene mesh in FRONT of the control (t = 5): defer to the editor.
        assert!(selection_pick(&world, &query(Some(Vec3::new(0.0, 5.0, 0.0)))).is_none());
        // A scene mesh BEHIND the control (t = 15): the control wins.
        let hit = selection_pick(&world, &query(Some(Vec3::new(0.0, -5.0, 0.0)))).expect("wins");
        assert_eq!(hit.entity, node);
    }

    #[test]
    fn vertex_handles_win_over_their_stroke_line() {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();
        world.register_inspector_default::<crate::stroke::Stroke>();
        world.register_inspector_default::<crate::stroke::StrokeVertex>();

        use redlilium_core::abstract_editor::EditAction;
        crate::stroke::AddStrokeAction::at_point(Vec3::zeros())
            .apply(&mut world)
            .unwrap();
        let (stroke, component) = world
            .read_all::<crate::stroke::Stroke>()
            .unwrap()
            .iter()
            .filter_map(|(index, s)| Some((world.entity_at_index(index)?, s.clone())))
            .next()
            .unwrap();

        let query = |x: f32, z: f32| redlilium_ecs::ui::ViewportPickQuery {
            ray: ViewportRay {
                origin: Vec3::new(x, 10.0, z),
                dir: Vec3::new(0.0, -1.0, 0.0),
            },
            scene_entity: None,
            scene_point: None,
        };
        // Click ON a vertex: the path segment through it is exactly as
        // close as the handle cube — the handle must still win (a line
        // pick here would make vertices unselectable).
        let hit = selection_pick(&world, &query(-4.0, 0.0)).expect("hit");
        assert_eq!(hit.entity, component.points[0], "vertex, not the stroke");
        // Click mid-segment, away from any handle: the stroke itself.
        let hit = selection_pick(&world, &query(0.0, 0.0)).expect("hit");
        assert_eq!(hit.entity, stroke);
    }

    #[test]
    fn cuts_pick_by_either_lip_and_vertices_win_on_top() {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();
        world.register_inspector_default::<crate::cut::Cut>();
        world.register_inspector_default::<crate::cut::CutVertex>();
        world.register_inspector_default::<crate::stroke::StrokeVertex>();

        use redlilium_core::abstract_editor::EditAction;
        // Stamp at height 5 so the LOWER lip (y = 3) is reachable by a
        // horizontal ray that never comes near the upper one.
        crate::cut::AddCutAction::at_point(Vec3::new(0.0, 5.0, 0.0))
            .apply(&mut world)
            .unwrap();
        let (cut, component) = world
            .read_all::<crate::cut::Cut>()
            .unwrap()
            .iter()
            .filter_map(|(index, c)| Some((world.entity_at_index(index)?, c.clone())))
            .next()
            .unwrap();

        let down = |x: f32, z: f32| redlilium_ecs::ui::ViewportPickQuery {
            ray: ViewportRay {
                origin: Vec3::new(x, 10.0, z),
                dir: Vec3::new(0.0, -1.0, 0.0),
            },
            scene_entity: None,
            scene_point: None,
        };
        // Click ON a vertex: the point handle beats the lip through it.
        let hit = selection_pick(&world, &down(-6.0, 0.0)).expect("hit");
        assert_eq!(hit.entity, component.points[0], "vertex, not the cut");
        // Mid-segment: the cut root.
        let hit = selection_pick(&world, &down(-3.0, 0.0)).expect("hit");
        assert_eq!(hit.entity, cut);
        // A horizontal ray at the DERIVED lip's height (y = 3): the lower
        // lip is pickable too, and it selects the same cut entity.
        let lower = redlilium_ecs::ui::ViewportPickQuery {
            ray: ViewportRay {
                origin: Vec3::new(-3.0, 3.0, 10.0),
                dir: Vec3::new(0.0, 0.0, -1.0),
            },
            scene_entity: None,
            scene_point: None,
        };
        let hit = selection_pick(&world, &lower).expect("hit");
        assert_eq!(hit.entity, cut);
    }

    #[test]
    fn gates_are_met_from_the_side_the_road_comes_from() {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<crate::Junction>();
        world.register_inspector_default::<EdgeAnchor>();
        world.register_inspector_default::<crate::stroke::Gate>();

        let spawn = |world: &mut World, x: f32, yaw: f32| {
            let e = world.spawn();
            let t = Transform::new(
                Vec3::new(x, 0.0, 0.0),
                quat_from_rotation_y(yaw),
                Vec3::new(1.0, 1.0, 1.0),
            );
            world.insert(e, t).unwrap();
            world.insert(e, GlobalTransform(t.to_matrix())).unwrap();
            world.insert(e, RoadNode::default()).unwrap();
            e
        };
        // Gate at the origin facing +X (its stroke runs north-south).
        let gate = spawn(&mut world, 0.0, std::f32::consts::FRAC_PI_2);
        world.insert(gate, crate::stroke::Gate::default()).unwrap();
        let outside = spawn(&mut world, 5.0, 0.0);
        let inside = spawn(&mut world, -5.0, 0.0);

        // A road from the gate's front side meets the front; a road from
        // behind the line connects from behind.
        assert!(socket_meets_front(&world, gate, outside));
        assert!(!socket_meets_front(&world, gate, inside));
        // Other sockets stay front-only regardless of approach side.
        let anchored = spawn(&mut world, 3.0, 0.0);
        world.insert(anchored, EdgeAnchor::default()).unwrap();
        assert!(socket_meets_front(&world, anchored, inside));
    }

    #[test]
    fn gate_is_a_socket() {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<crate::Junction>();
        world.register_inspector_default::<EdgeAnchor>();
        world.register_inspector_default::<crate::stroke::Gate>();

        let gate = world.spawn();
        world.insert(gate, RoadNode::default()).unwrap();
        world.insert(gate, crate::stroke::Gate::default()).unwrap();
        assert!(is_socket(&world, gate));
        // A plain node is not.
        let plain = world.spawn();
        world.insert(plain, RoadNode::default()).unwrap();
        assert!(!is_socket(&world, plain));
    }

    #[test]
    fn add_road_action_roundtrip() {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();

        let make_node = |world: &mut World| {
            let e = world.spawn();
            world.insert(e, Transform::default()).unwrap();
            world.insert(e, RoadNode::default()).unwrap();
            e
        };
        let a = make_node(&mut world);
        let b = make_node(&mut world);

        let mut action = AddRoadAction::new(a, b, false);
        action.apply(&mut world).unwrap();
        let roads = world.read_all::<RoadSegment>().unwrap().iter().count();
        assert_eq!(roads, 1);
        action.undo(&mut world).unwrap();
        let roads = world.read_all::<RoadSegment>().unwrap().iter().count();
        assert_eq!(roads, 0);
    }

    #[test]
    fn chained_ground_clicks_reorient_interior_node() {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();
        world.register_inspector_default::<crate::Junction>();
        world.register_inspector_default::<EdgeAnchor>();

        // Click 1: lone node at origin. Click 2: node at (0, 20) — chord +Z.
        let mut a1 =
            AddNodeAction::new(None, node_transform(None, Vec3::new(0.0, 0.0, 0.0), &world));
        a1.apply(&mut world).unwrap();
        let n1 = a1.report.lock().unwrap().unwrap();
        let t2 = node_transform(Some(n1), Vec3::new(0.0, 0.0, 20.0), &world);
        let mut a2 = AddNodeAction::new(Some(n1), t2);
        a2.apply(&mut world).unwrap();
        let n2 = a2.report.lock().unwrap().unwrap();
        let heading_before = bezier::heading(&world.get::<GlobalTransform>(n2).unwrap().0);
        assert!((heading_before - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-4);

        // Click 3: sharp right turn to (20, 20). n2 must re-aim at the
        // Catmull-Rom direction n1 → n3 (diagonal), not keep the old chord.
        let t3 = node_transform(Some(n2), Vec3::new(20.0, 0.0, 20.0), &world);
        let mut a3 = AddNodeAction::new(Some(n2), t3);
        a3.apply(&mut world).unwrap();
        let heading_after = bezier::heading(&world.get::<GlobalTransform>(n2).unwrap().0);
        let expected = Vec3::new(20.0, 0.0, 20.0).normalize();
        assert!(
            (heading_after - expected).norm() < 1e-3,
            "interior node re-aims along prev→new, got {heading_after:?}"
        );

        // Undo click 3: n2's heading reverts to the old chord.
        a3.undo(&mut world).unwrap();
        let reverted = bezier::heading(&world.get::<GlobalTransform>(n2).unwrap().0);
        assert!((reverted - heading_before).norm() < 1e-4);
    }

    #[test]
    fn add_node_action_chains_via_report() {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();

        let mut action = AddNodeAction::new(None, Transform::default());
        let report = action.report.clone();
        assert!(report.lock().unwrap().is_none());
        action.apply(&mut world).unwrap();
        let node = report.lock().unwrap().expect("node reported");
        assert!(world.get::<RoadNode>(node).is_some());
        action.undo(&mut world).unwrap();
        assert!(report.lock().unwrap().is_none());
        assert!(!world.is_alive(node));
    }
}
