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

use crate::attachment::{self, AttachToEdgeAction, EdgeAttachment, EdgeHit};
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
        "click a node to anchor · click ground to place · click a road edge to attach · Esc: drop anchor / exit"
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
        // Edge hover only matters with an anchor and when no node is hit
        // (nodes win: they sit at road ends where edges also pass close).
        let edge = (self.anchor.is_some() && hovered.is_none())
            .then(|| attachment::edge_under_cursor(ctx.world, &ray, EDGE_PICK_RADIUS))
            .flatten()
            // The anchor's own roads' edges are not attachment targets.
            .filter(|hit| !road_touches(ctx.world, hit.road, self.anchor));

        let ground = ground_hit(&ray);

        self.draw_feedback(ctx, hovered, edge, ground);

        if ctx.input.clicked {
            match (self.anchor, hovered, edge) {
                // First click on a node: anchor there.
                (None, Some(node), _) => self.anchor = Some(node),
                // Click a second node: connect, chain from it.
                (Some(anchor), Some(node), _) => {
                    if node != anchor {
                        ctx.actions.push(Box::new(AddRoadAction::new(anchor, node)));
                        self.anchor = Some(node);
                    }
                }
                // Click a road edge: hang an attachment from the anchor
                // onto it. A driveway terminates — the anchor drops.
                (Some(anchor), None, Some(hit)) => {
                    let half_width = ctx
                        .world
                        .get::<RoadNode>(anchor)
                        .map(|n| n.half_width)
                        .unwrap_or(3.0);
                    let (u_min, u_max) = attachment::interval_around(ctx.world, &hit, half_width);
                    ctx.actions
                        .push(Box::new(AttachToEdgeAction::new(EdgeAttachment {
                            road: hit.road,
                            node: anchor,
                            right_edge: hit.right_edge,
                            u_min,
                            u_max,
                            ..EdgeAttachment::default()
                        })));
                    self.anchor = None;
                }
                // Click empty ground: spawn a node there (and a road from
                // the anchor, when one is set). The report chains the anchor.
                // (`edge` is only computed with an anchor, so the remaining
                // pattern is anchor-less-with-edge — treat it as ground.)
                (anchor, None, _) => {
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

        let Some(anchor) = self.anchor else { return };

        // Preview an attachment onto the hovered road edge.
        if let Some(hit) = edge {
            draw.draw_circle(
                pt(&hit.point),
                0.6,
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                HOVER_COLOR,
            );
            let half_width = ctx
                .world
                .get::<RoadNode>(anchor)
                .map(|n| n.half_width)
                .unwrap_or(3.0);
            let (u_min, u_max) = attachment::interval_around(ctx.world, &hit, half_width);
            let preview = EdgeAttachment {
                road: hit.road,
                node: anchor,
                right_edge: hit.right_edge,
                u_min,
                u_max,
                ..EdgeAttachment::default()
            };
            if let Some(patch) = attachment::attachment_patch(ctx.world, &preview) {
                draw_patch_outline(&mut draw, &patch, PREVIEW_COLOR);
            }
            return;
        }

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
        let patch = bezier::patch_from_nodes(&a_mat, a_half, chord, &b_mat, b_half, chord);
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

/// Closest approach between a ray and a segment: `(distance, s)` where `s`
/// is the segment parameter of the closest point. Shared with edge picking.
pub(crate) fn ray_segment_closest(ray: &ViewportRay, p0: Vec3, p1: Vec3) -> (f32, f32) {
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
    ((on_ray - on_segment).norm(), s)
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
    road: Option<Entity>,
}

impl AddRoadAction {
    pub fn new(a: Entity, b: Entity) -> Self {
        Self { a, b, road: None }
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
    if let Ok(junctions) = world.read_all::<crate::Junction>()
        && junctions.iter().any(|(_, j)| j.connectors.contains(&node))
    {
        return None;
    }
    if let Ok(attachments) = world.read_all::<EdgeAttachment>()
        && attachments.iter().any(|(_, a)| a.node == node)
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

        let mut action = AddRoadAction::new(a, b);
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
        world.register_inspector_default::<EdgeAttachment>();

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
