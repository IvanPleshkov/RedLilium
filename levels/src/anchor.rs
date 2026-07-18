//! Edge anchors (design P5): a [`RoadNode`] **glued to part of a road's
//! boundary curve** — the driveway/building-exit case.
//!
//! The anchored node's `Transform` and `half_width` are *derived data*, like
//! `GlobalTransform`: position is the midpoint of the chord between the edge
//! points at `u_min`/`u_max`, local X runs along the chord, +Z points
//! outward (away from the parent road), and `half_width` is half the chord
//! length. The seam with the parent road is the **straight** chord, not the
//! curved edge — a road piece's geometry is always the span between two
//! straight segments, and closing the sliver between chord and true edge
//! curve is the mesh generator's job (it receives the parametric interval).
//!
//! Because the anchor is parametric, the node follows every parent-road
//! edit. The driveway itself is an ordinary [`RoadSegment`] out of the
//! anchored node, so its own edges are pickable and anchorable like any
//! road's — chains (road → driveway → parking lot) need no special support.

use std::sync::{Arc, Mutex};

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::{Vec3, quat_from_rotation_y};
use redlilium_ecs::{
    Component, Entity, GlobalTransform, System, SystemContext, SystemError, Transform, World,
    WriteAll,
};

use crate::graph::road_patch;
use crate::{RoadNode, RoadSegment, bezier};

/// Glues the carrying [`RoadNode`] to a side edge of a road. The node's
/// `Transform` and `half_width` are derived from the parent edge every frame
/// by [`DeriveEdgeAnchors`] — author the interval, not the placement.
#[derive(Debug, Clone, Component)]
pub struct EdgeAnchor {
    /// The road entity (carries [`RoadSegment`]) whose edge this node sits
    /// on — possibly itself a driveway, chains are legitimate.
    pub parent_road: Entity,
    /// Which side edge: `true` = the `v = 1` edge (the cross-sections'
    /// local +X ends), `false` = the `v = 0` edge.
    pub right_edge: bool,
    /// Landing interval on the parent's `u` axis, `0 ≤ u_min < u_max ≤ 1`.
    pub u_min: f32,
    pub u_max: f32,
}

impl Default for EdgeAnchor {
    fn default() -> Self {
        Self {
            parent_road: Entity::DANGLING,
            right_edge: true,
            u_min: 0.4,
            u_max: 0.6,
        }
    }
}

/// Derive the anchored node's placement from its parent edge: `(transform,
/// half_width)`. `None` when the parent road is gone or the interval is
/// degenerate — broken topology yields no placement, the node keeps its
/// last one.
pub fn derive_anchor_state(world: &World, anchor: &EdgeAnchor) -> Option<(Transform, f32)> {
    let patch = road_patch(world, anchor.parent_road)?;
    let v_edge = if anchor.right_edge { 1.0 } else { 0.0 };
    let (u_min, u_max) = (
        anchor.u_min.min(anchor.u_max),
        anchor.u_min.max(anchor.u_max),
    );
    let p0 = bezier::eval(&patch, u_min, v_edge);
    let p1 = bezier::eval(&patch, u_max, v_edge);
    let chord = p1 - p0;
    let length = chord.norm();
    if length < 1e-3 {
        return None;
    }

    // Local X exactly along the chord (yaw-only: a chord tilted in Y keeps
    // its midpoint height but the cross-section stays horizontal).
    let cx = Vec3::new(chord.x, 0.0, chord.z).normalize();
    let mut yaw = (-cx.z).atan2(cx.x);
    // +Z outward: flip 180° if the perpendicular faces the parent road.
    let outward = {
        let dv = bezier::eval_dv(&patch, (u_min + u_max) * 0.5, v_edge);
        dv * if anchor.right_edge { 1.0 } else { -1.0 }
    };
    let z_axis = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    if z_axis.dot(&outward) < 0.0 {
        yaw += std::f32::consts::PI;
    }

    let mid = (p0 + p1) * 0.5;
    let transform = Transform::new(mid, quat_from_rotation_y(yaw), Vec3::new(1.0, 1.0, 1.0));
    Some((transform, (length * 0.5).max(0.1)))
}

/// One derivation pass: every anchored node whose derived placement differs
/// from its current one, with the new values. Empty means settled.
fn anchor_updates(world: &World) -> Vec<(Entity, Transform, f32)> {
    let Ok(anchors) = world.read_all::<EdgeAnchor>() else {
        return Vec::new();
    };
    let mut updates = Vec::new();
    for (index, anchor) in anchors.iter() {
        let Some(entity) = world.entity_at_index(index) else {
            continue;
        };
        let Some((transform, half_width)) = derive_anchor_state(world, anchor) else {
            continue;
        };
        let current_t = world.get::<Transform>(entity);
        let current_n = world.get::<RoadNode>(entity);
        let changed = match (current_t, current_n) {
            (Some(t), Some(n)) => {
                (t.to_matrix() - transform.to_matrix()).norm() > 1e-4
                    || (n.half_width - half_width).abs() > 1e-4
            }
            _ => true,
        };
        if changed {
            updates.push((entity, transform, half_width));
        }
    }
    updates
}

/// Anchor chains settle one dependency level per pass; cycles (possible via
/// the inspector) simply stop at the cap instead of recursing forever.
const MAX_PASSES: usize = 8;

/// Settle every edge anchor in place — the `&mut World` variant used by
/// scene baking and tests. The frame-loop variant is [`DeriveEdgeAnchors`].
pub fn settle_edge_anchors(world: &mut World) {
    for _ in 0..MAX_PASSES {
        let updates = anchor_updates(world);
        if updates.is_empty() {
            return;
        }
        for (entity, transform, half_width) in updates {
            let _ = world.insert(entity, transform);
            let _ = world.insert(entity, GlobalTransform(transform.to_matrix()));
            if let Some(mut node) = world.get::<RoadNode>(entity).cloned() {
                node.half_width = half_width;
                let _ = world.insert(entity, node);
            }
        }
    }
}

/// Editing-view system: re-derives anchored nodes' placements from their
/// parent edges. Writes only when a placement actually changed, so a
/// settled graph stays untouched (no change-tick churn).
pub struct DeriveEdgeAnchors;

impl System for DeriveEdgeAnchors {
    type Result = ();

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        for _ in 0..MAX_PASSES {
            let updates = anchor_updates(ctx.raw_world());
            if updates.is_empty() {
                break;
            }
            ctx.lock::<(
                WriteAll<Transform>,
                WriteAll<GlobalTransform>,
                WriteAll<RoadNode>,
            )>()
            .execute(|(mut transforms, mut globals, mut nodes)| {
                for (entity, transform, half_width) in &updates {
                    if let Some(mut slot) = transforms.get_mut(entity.index()) {
                        *slot = *transform;
                    }
                    if let Some(mut global) = globals.get_mut(entity.index()) {
                        global.0 = transform.to_matrix();
                    }
                    if let Some(mut node) = nodes.get_mut(entity.index()) {
                        node.half_width = *half_width;
                    }
                }
            });
        }
        Ok(())
    }
}

/// An edge point under the cursor: which road, which side, where along it.
#[derive(Debug, Clone, Copy)]
pub struct EdgeHit {
    pub road: Entity,
    pub right_edge: bool,
    pub u: f32,
    pub point: Vec3,
}

/// Tessellation used for edge picking.
const PICK_STEPS: usize = 24;

/// CPU pick against every road's two side edges: the closest edge point
/// within `radius` of the ray. Driveways are ordinary roads, so their edges
/// participate too.
pub fn edge_under_cursor(
    world: &World,
    ray: &redlilium_ecs::ui::ViewportRay,
    radius: f32,
) -> Option<EdgeHit> {
    let roads = world.read_all::<RoadSegment>().ok()?;
    let mut best: Option<(f32, EdgeHit)> = None;
    for (index, _) in roads.iter() {
        let Some(road) = world.entity_at_index(index) else {
            continue;
        };
        let Some(patch) = road_patch(world, road) else {
            continue;
        };
        for (right_edge, v_edge) in [(false, 0.0f32), (true, 1.0f32)] {
            let mut prev = bezier::eval(&patch, 0.0, v_edge);
            for step in 1..=PICK_STEPS {
                let u1 = step as f32 / PICK_STEPS as f32;
                let next = bezier::eval(&patch, u1, v_edge);
                let (dist, s) = crate::tool::ray_segment_closest(ray, prev, next);
                if dist <= radius && best.is_none_or(|(d, _)| dist < d) {
                    let u0 = (step - 1) as f32 / PICK_STEPS as f32;
                    let u = u0 + (u1 - u0) * s;
                    best = Some((
                        dist,
                        EdgeHit {
                            road,
                            right_edge,
                            u,
                            point: prev + (next - prev) * s,
                        },
                    ));
                }
                prev = next;
            }
        }
    }
    best.map(|(_, hit)| hit)
}

/// The `u` half-interval a node's width covers around a hit point: the
/// node's full width mapped through the local edge-length density.
pub fn interval_around(world: &World, hit: &EdgeHit, node_half_width: f32) -> (f32, f32) {
    let Some(patch) = road_patch(world, hit.road) else {
        return (hit.u - 0.1, hit.u + 0.1);
    };
    let v_edge = if hit.right_edge { 1.0 } else { 0.0 };
    // Local edge speed |dP/du| via finite difference.
    let h = 1e-3;
    let u = hit.u.clamp(h, 1.0 - h);
    let speed = (bezier::eval(&patch, u + h, v_edge) - bezier::eval(&patch, u - h, v_edge)).norm()
        / (2.0 * h);
    let half_u = if speed > 1e-4 {
        (node_half_width / speed).min(0.45)
    } else {
        0.1
    };
    ((hit.u - half_u).max(0.0), (hit.u + half_u).min(1.0))
}

// ---------------------------------------------------------------------------
// Edit action
// ---------------------------------------------------------------------------

/// Undoable "spawn a node glued to a road edge" — with an optional road
/// arriving at it from an existing anchor node (the driveway-INTO-the-edge
/// gesture; the anchored node's +Z faces outward, so the road meets it from
/// the front). The `report` hands the created node back to the tool.
pub struct AnchorNodeAction {
    anchor: EdgeAnchor,
    from: Option<Entity>,
    created_node: Option<Entity>,
    created_road: Option<Entity>,
    pub report: Arc<Mutex<Option<Entity>>>,
}

impl AnchorNodeAction {
    pub fn new(anchor: EdgeAnchor, from: Option<Entity>) -> Self {
        Self {
            anchor,
            from,
            created_node: None,
            created_road: None,
            report: Arc::new(Mutex::new(None)),
        }
    }
}

impl std::fmt::Debug for AnchorNodeAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnchorNodeAction")
            .field("anchor", &self.anchor)
            .field("from", &self.from)
            .finish()
    }
}

impl EditAction<World> for AnchorNodeAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let Some((transform, half_width)) = derive_anchor_state(world, &self.anchor) else {
            return Err(EditActionError::TargetNotFound(
                "anchor parent road missing or degenerate".into(),
            ));
        };
        let node = world.spawn();
        let insert = |world: &mut World, e, r: Result<(), redlilium_ecs::WorldError>| {
            r.map_err(|err| {
                world.despawn(e);
                EditActionError::Custom(err.to_string())
            })
        };
        let t = world.insert(node, transform);
        insert(world, node, t)?;
        let g = world.insert(node, GlobalTransform(transform.to_matrix()));
        insert(world, node, g)?;
        let n = world.insert(node, RoadNode { half_width });
        insert(world, node, n)?;
        let a = world.insert(node, self.anchor.clone());
        insert(world, node, a)?;

        self.created_road = None;
        if let Some(from) = self.from.filter(|f| world.is_alive(*f)) {
            let road = world.spawn();
            let r = world.insert(
                road,
                RoadSegment {
                    a: from,
                    b: node,
                    b_from_front: true,
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
        "Anchor node to road edge"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (World, Entity) {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();
        world.register_inspector_default::<EdgeAnchor>();

        let node = |world: &mut World, x: f32, z: f32, yaw: f32| {
            let e = world.spawn();
            let t = Transform::new(
                Vec3::new(x, 0.0, z),
                quat_from_rotation_y(yaw),
                Vec3::new(1.0, 1.0, 1.0),
            );
            world.insert(e, t).unwrap();
            world.insert(e, GlobalTransform(t.to_matrix())).unwrap();
            world.insert(e, RoadNode::default()).unwrap();
            e
        };
        // Straight road along +Z from origin to (0, 0, 20).
        let a = node(&mut world, 0.0, 0.0, 0.0);
        let b = node(&mut world, 0.0, 20.0, 0.0);
        let road = world.spawn();
        world
            .insert(
                road,
                RoadSegment {
                    a,
                    b,
                    ..RoadSegment::default()
                },
            )
            .unwrap();
        (world, road)
    }

    #[test]
    fn derived_state_sits_on_edge_facing_outward() {
        let (world, road) = setup();
        let anchor = EdgeAnchor {
            parent_road: road,
            right_edge: true,
            u_min: 0.4,
            u_max: 0.6,
        };
        let (t, half_width) = derive_anchor_state(&world, &anchor).expect("state");
        // Right edge of a straight +Z road sits at x = +3; interval midpoint
        // is z = 10; +Z must face outward (+X).
        assert!((t.translation - Vec3::new(3.0, 0.0, 10.0)).norm() < 1e-3);
        let heading = bezier::heading(&t.to_matrix());
        assert!((heading - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-3);
        // Chord of a fifth of a 20 m straight edge → half width ≈ 2.
        assert!((half_width - 2.0).abs() < 1e-2);
    }

    #[test]
    fn settle_follows_parent_road_edits() {
        let (mut world, road) = setup();
        let anchored = world.spawn();
        world.insert(anchored, Transform::default()).unwrap();
        world
            .insert(anchored, GlobalTransform(Transform::default().to_matrix()))
            .unwrap();
        world.insert(anchored, RoadNode::default()).unwrap();
        world
            .insert(
                anchored,
                EdgeAnchor {
                    parent_road: road,
                    right_edge: true,
                    u_min: 0.4,
                    u_max: 0.6,
                },
            )
            .unwrap();

        settle_edge_anchors(&mut world);
        let before = world.get::<Transform>(anchored).unwrap().translation;
        assert!((before - Vec3::new(3.0, 0.0, 10.0)).norm() < 1e-3);

        // Move the road's far node: the anchored node follows the edge.
        let seg_b = world.get::<RoadSegment>(road).unwrap().b;
        let t = Transform::new(
            Vec3::new(8.0, 0.0, 20.0),
            quat_from_rotation_y(0.4),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(seg_b, t).unwrap();
        world.insert(seg_b, GlobalTransform(t.to_matrix())).unwrap();
        settle_edge_anchors(&mut world);
        let after = world.get::<Transform>(anchored).unwrap().translation;
        assert!((after - before).norm() > 0.1, "anchor follows the edge");

        // Settled: a second pass writes nothing.
        assert!(anchor_updates(&world).is_empty());
    }

    #[test]
    fn anchor_chain_settles_through_a_driveway() {
        let (mut world, road) = setup();
        // Driveway: anchored node on the main road + an outer node, joined
        // by an ordinary segment.
        let first = world.spawn();
        world.insert(first, Transform::default()).unwrap();
        world
            .insert(first, GlobalTransform(Transform::default().to_matrix()))
            .unwrap();
        world.insert(first, RoadNode::default()).unwrap();
        world
            .insert(
                first,
                EdgeAnchor {
                    parent_road: road,
                    right_edge: true,
                    u_min: 0.3,
                    u_max: 0.5,
                },
            )
            .unwrap();
        let outer = world.spawn();
        let t = Transform::new(
            Vec3::new(12.0, 0.0, 8.0),
            quat_from_rotation_y(-std::f32::consts::FRAC_PI_2),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(outer, t).unwrap();
        world.insert(outer, GlobalTransform(t.to_matrix())).unwrap();
        world.insert(outer, RoadNode::default()).unwrap();
        let driveway = world.spawn();
        world
            .insert(
                driveway,
                RoadSegment {
                    a: first,
                    b: outer,
                    b_from_front: true,
                    ..RoadSegment::default()
                },
            )
            .unwrap();
        // Second-level anchor: a node glued to the DRIVEWAY's edge.
        let second = world.spawn();
        world.insert(second, Transform::default()).unwrap();
        world
            .insert(second, GlobalTransform(Transform::default().to_matrix()))
            .unwrap();
        world.insert(second, RoadNode::default()).unwrap();
        world
            .insert(
                second,
                EdgeAnchor {
                    parent_road: driveway,
                    right_edge: false,
                    u_min: 0.3,
                    u_max: 0.7,
                },
            )
            .unwrap();

        settle_edge_anchors(&mut world);
        // Both levels settled: no pending updates, and the second-level node
        // actually left the origin (it derived through the first level).
        assert!(anchor_updates(&world).is_empty());
        let second_pos = world.get::<Transform>(second).unwrap().translation;
        assert!(second_pos.norm() > 1.0, "chained anchor derived");
    }

    #[test]
    fn anchor_action_roundtrip_with_road() {
        let (mut world, road) = setup();
        let from = world.spawn();
        let t = Transform::new(
            Vec3::new(12.0, 0.0, 10.0),
            quat_from_rotation_y(-std::f32::consts::FRAC_PI_2),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(from, t).unwrap();
        world.insert(from, GlobalTransform(t.to_matrix())).unwrap();
        world.insert(from, RoadNode::default()).unwrap();

        let mut action = AnchorNodeAction::new(
            EdgeAnchor {
                parent_road: road,
                right_edge: true,
                u_min: 0.4,
                u_max: 0.6,
            },
            Some(from),
        );
        let report = action.report.clone();
        action.apply(&mut world).unwrap();
        let node = report.lock().unwrap().expect("node reported");
        assert!(world.get::<EdgeAnchor>(node).is_some());
        // The road arrives at the anchored node from its front.
        let seg = world
            .read_all::<RoadSegment>()
            .unwrap()
            .iter()
            .map(|(_, s)| s.clone())
            .find(|s| s.b == node)
            .expect("driveway road");
        assert_eq!(seg.a, from);
        assert!(seg.b_from_front);

        action.undo(&mut world).unwrap();
        assert!(!world.is_alive(node));
        assert_eq!(world.read_all::<RoadSegment>().unwrap().iter().count(), 1);
        assert!(report.lock().unwrap().is_none());
    }
}
