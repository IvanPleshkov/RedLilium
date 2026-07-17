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

use crate::{RoadNode, RoadSegment, bezier};

/// Stable tool label — menu ops request activation by this name.
pub const CONNECT_TOOL: &str = "Connect roads";

/// Cursor-to-node pick distance, world units.
const PICK_RADIUS: f32 = 1.5;
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
        "click a node to anchor · click ground to place · Esc: drop anchor / exit"
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
        let ground = ground_hit(&ray);

        self.draw_feedback(ctx, hovered, ground);

        if ctx.input.clicked {
            match (self.anchor, hovered) {
                // First click on a node: anchor there.
                (None, Some(node)) => self.anchor = Some(node),
                // Click a second node: connect, chain from it.
                (Some(anchor), Some(node)) => {
                    if node != anchor {
                        ctx.actions.push(Box::new(AddRoadAction::new(anchor, node)));
                        self.anchor = Some(node);
                    }
                }
                // Click empty ground: spawn a node there (and a road from
                // the anchor, when one is set). The report chains the anchor.
                (anchor, None) => {
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

        // Preview patch from the anchor to the hovered node / ground cursor.
        let Some(anchor) = self.anchor else { return };
        let Some((a_mat, a_half)) = node_shape(ctx.world, anchor) else {
            return;
        };
        let seg = RoadSegment::default();
        let end = hovered
            .and_then(|node| node_shape(ctx.world, node))
            .or_else(|| {
                ground.map(|point| {
                    let t = node_transform(Some(anchor), point, ctx.world);
                    (t.to_matrix(), RoadNode::default().half_width)
                })
            });
        let Some((b_mat, b_half)) = end else { return };
        let patch =
            bezier::patch_from_nodes(&a_mat, a_half, seg.tangent_a, &b_mat, b_half, seg.tangent_b);
        for v in [0.0, 0.5, 1.0] {
            let mut prev = bezier::eval(&patch, 0.0, v);
            for step in 1..=12 {
                let next = bezier::eval(&patch, step as f32 / 12.0, v);
                draw.draw_line(pt(&prev), pt(&next), PREVIEW_COLOR);
                prev = next;
            }
        }
    }
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
/// ray points at it.
fn ground_hit(ray: &ViewportRay) -> Option<Vec3> {
    if ray.dir.y.abs() < 1e-6 {
        return None;
    }
    let t = -ray.origin.y / ray.dir.y;
    (t > 0.0).then(|| ray.origin + ray.dir * t)
}

/// Closest distance between a ray and a segment.
fn ray_segment_distance(ray: &ViewportRay, p0: Vec3, p1: Vec3) -> f32 {
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
    (on_ray - on_segment).norm()
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
pub struct AddNodeAction {
    anchor: Option<Entity>,
    transform: Transform,
    created_node: Option<Entity>,
    created_road: Option<Entity>,
    pub report: Arc<Mutex<Option<Entity>>>,
}

impl AddNodeAction {
    pub fn new(anchor: Option<Entity>, transform: Transform) -> Self {
        Self {
            anchor,
            transform,
            created_node: None,
            created_road: None,
            report: Arc::new(Mutex::new(None)),
        }
    }
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
