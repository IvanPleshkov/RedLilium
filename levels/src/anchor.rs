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
/// `Transform` and `half_width` are derived from the parent edge every
/// frame by [`DeriveEdgeAnchors`] — author the interval, not the placement.
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

/// Project a world position onto the parent edge: the interval (of fixed
/// `width`) whose chord midpoint is closest to `pos`. Coarse scan + ternary
/// refinement — deterministic, ~1e-6 in `u`.
fn project_onto_edge(
    patch: &crate::bezier::Patch,
    v_edge: f32,
    width: f32,
    pos: Vec3,
) -> (f32, f32) {
    let half = (width * 0.5).min(0.5);
    let (lo, hi) = (half, 1.0 - half);
    if hi <= lo {
        return (0.0, 1.0);
    }
    let mid = |c: f32| {
        (bezier::eval(patch, c - half, v_edge) + bezier::eval(patch, c + half, v_edge)) * 0.5
    };
    const COARSE: usize = 64;
    let mut best = lo;
    let mut best_d = f32::MAX;
    for i in 0..=COARSE {
        let c = lo + (hi - lo) * (i as f32 / COARSE as f32);
        let d = (mid(c) - pos).norm_squared();
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    let step = (hi - lo) / COARSE as f32;
    let (mut a, mut b) = ((best - step).max(lo), (best + step).min(hi));
    for _ in 0..40 {
        let m1 = a + (b - a) / 3.0;
        let m2 = b - (b - a) / 3.0;
        if (mid(m1) - pos).norm_squared() < (mid(m2) - pos).norm_squared() {
            b = m2;
        } else {
            a = m1;
        }
    }
    let c = (a + b) * 0.5;
    (c - half, c + half)
}

/// Derive an anchored **stroke**'s placement — the inverted derivation:
/// the rigid stroke dictates its frontage length, and the edge interval's
/// *width* derives from it (mapped through the local edge-length density);
/// only the interval's center is authored (it slides). The transform faces
/// the opposite way from an anchored node: **+Z points into the parent
/// road** the stroke fronts, the tail extends outward behind.
/// Returns `(transform, derived interval)`.
pub(crate) fn derive_stroke_anchor(
    world: &World,
    entity: Entity,
    anchor: &EdgeAnchor,
) -> Option<(Transform, (f32, f32))> {
    let stroke = world.get::<crate::stroke::Stroke>(entity)?;
    let frontage = crate::stroke::stroke_frontage(world, stroke)?;
    let patch = road_patch(world, anchor.parent_road)?;
    let v_edge = if anchor.right_edge { 1.0 } else { 0.0 };

    let center = ((anchor.u_min + anchor.u_max) * 0.5).clamp(0.0, 1.0);
    // Frontage length → u half-width via |dP/du| at the center.
    let h = 1e-3;
    let u = center.clamp(h, 1.0 - h);
    let speed = (bezier::eval(&patch, u + h, v_edge) - bezier::eval(&patch, u - h, v_edge)).norm()
        / (2.0 * h);
    let half_u = if speed > 1e-4 {
        ((frontage * 0.5) / speed).min(0.45)
    } else {
        return None;
    };
    let center = center.clamp(half_u, 1.0 - half_u);
    let interval = (center - half_u, center + half_u);

    let scaffold = EdgeAnchor {
        u_min: interval.0,
        u_max: interval.1,
        ..anchor.clone()
    };
    let (node_t, _) = derive_anchor_state(world, &scaffold)?;
    // Flip 180°: a node's +Z points away from the road (its network is the
    // driveway growing outward); the stroke's frontage FACES the road.
    let transform = Transform::new(
        node_t.translation,
        node_t.rotation * quat_from_rotation_y(std::f32::consts::PI),
        node_t.scale,
    );
    Some((transform, interval))
}

/// One planned write: derived placement, plus the shifted interval when the
/// node is *sliding* (its transform was authored away from the derived one).
type AnchorUpdate = (Entity, Transform, f32, Option<(f32, f32)>);

/// One derivation pass. `cache` remembers each anchored node's last settled
/// transform: a node that differs from BOTH the cache and the derived state
/// was moved by the author (gizmo/inspector) → project it onto the parent
/// edge and shift the interval (sliding). A node equal to the cache whose
/// derived state moved away follows the edge (parametric follow). Settled
/// nodes are recorded into the cache. Empty result means settled.
pub(crate) fn anchor_updates(
    world: &World,
    cache: &mut std::collections::HashMap<Entity, Transform>,
    slide: bool,
) -> Vec<AnchorUpdate> {
    let Ok(anchors) = world.read_all::<EdgeAnchor>() else {
        return Vec::new();
    };
    let mut updates = Vec::new();
    for (index, anchor) in anchors.iter() {
        let Some(entity) = world.entity_at_index(index) else {
            continue;
        };
        // Two carrier kinds: a stroke derives its transform AND the edge
        // interval (inverted derivation — see `derive_stroke_anchor`); a
        // node/bare entity derives its transform (+ half_width when a
        // RoadNode is present) from the authored interval.
        let is_stroke = world.get::<crate::stroke::Stroke>(entity).is_some();
        let derived = if is_stroke {
            derive_stroke_anchor(world, entity, anchor)
                .map(|(t, interval)| (t, 0.0, Some(interval)))
        } else {
            derive_anchor_state(world, anchor).map(|(t, hw)| (t, hw, None))
        };
        let Some((derived_t, derived_hw, derived_interval)) = derived else {
            continue;
        };
        let Some(t) = world.get::<Transform>(entity) else {
            updates.push((entity, derived_t, derived_hw, derived_interval));
            continue;
        };
        // Width receiver: the road node's half_width when present; a bare
        // anchored entity derives only its transform (never a reason to
        // stay unsettled).
        let current_w = world.get::<RoadNode>(entity).map(|n| n.half_width);
        let interval_ok = derived_interval.is_none_or(|(lo, hi)| {
            (anchor.u_min - lo).abs() <= 1e-4 && (anchor.u_max - hi).abs() <= 1e-4
        });
        let settled = (t.to_matrix() - derived_t.to_matrix()).norm() <= 1e-4
            && current_w.is_none_or(|w| (w - derived_hw).abs() <= 1e-4)
            && interval_ok;
        if settled {
            cache.insert(entity, *t);
            continue;
        }
        let moved = slide
            && cache
                .get(&entity)
                .is_some_and(|prev| (prev.to_matrix() - t.to_matrix()).norm() > 1e-4);
        if moved {
            // Sliding: keep the interval width, move its center to the
            // closest edge point to where the author dragged the entity.
            let slid = (|| {
                let patch = road_patch(world, anchor.parent_road)?;
                let v_edge = if anchor.right_edge { 1.0 } else { 0.0 };
                let (u_min, u_max) = if let Some((lo, hi)) = derived_interval {
                    (lo, hi)
                } else {
                    (
                        anchor.u_min.min(anchor.u_max),
                        anchor.u_min.max(anchor.u_max),
                    )
                };
                let (new_min, new_max) =
                    project_onto_edge(&patch, v_edge, u_max - u_min, t.translation);
                if (new_min - u_min).abs() < 1e-5 && (new_max - u_max).abs() < 1e-5 {
                    return None; // landed where it already was — just snap
                }
                let slid = EdgeAnchor {
                    u_min: new_min,
                    u_max: new_max,
                    ..anchor.clone()
                };
                if is_stroke {
                    let (slid_t, interval) = derive_stroke_anchor(world, entity, &slid)?;
                    Some((entity, slid_t, 0.0, Some(interval)))
                } else {
                    let (slid_t, slid_hw) = derive_anchor_state(world, &slid)?;
                    Some((entity, slid_t, slid_hw, Some((new_min, new_max))))
                }
            })();
            if let Some(update) = slid {
                updates.push(update);
                continue;
            }
        }
        updates.push((entity, derived_t, derived_hw, derived_interval));
    }
    updates
}

/// Anchor chains settle one dependency level per pass; cycles (possible via
/// the inspector) simply stop at the cap instead of recursing forever.
const MAX_PASSES: usize = 8;

/// Settle every edge anchor in place — the `&mut World` variant used by
/// scene baking and tests. Follow-only (no sliding: baking has no notion
/// of "the author just moved this node"). The frame-loop variant is
/// [`DeriveEdgeAnchors`].
pub fn settle_edge_anchors(world: &mut World) {
    let mut cache = std::collections::HashMap::new();
    for _ in 0..MAX_PASSES {
        let updates = anchor_updates(world, &mut cache, false);
        if updates.is_empty() {
            return;
        }
        apply_updates(world, &updates);
    }
}

/// Apply planned writes through `&mut World` (baking/tests path).
fn apply_updates(world: &mut World, updates: &[AnchorUpdate]) {
    for (entity, transform, half_width, interval) in updates {
        let _ = world.insert(*entity, *transform);
        let _ = world.insert(*entity, GlobalTransform(transform.to_matrix()));
        if let Some(mut node) = world.get::<RoadNode>(*entity).cloned() {
            node.half_width = *half_width;
            let _ = world.insert(*entity, node);
        }
        if let Some((u_min, u_max)) = interval
            && let Some(mut anchor) = world.get::<EdgeAnchor>(*entity).cloned()
        {
            anchor.u_min = *u_min;
            anchor.u_max = *u_max;
            let _ = world.insert(*entity, anchor);
        }
    }
}

/// Editing-view system: re-derives anchored nodes' placements from their
/// parent edges, and converts authored moves of an anchored node (gizmo or
/// inspector Transform edits) into interval shifts — the node *slides*
/// along the parent edge. Writes only when a placement actually changed,
/// so a settled graph stays untouched (no change-tick churn).
///
/// The interval write is derived-data maintenance, not an edit action of
/// its own: undo of the authoring Transform action restores the node's
/// on-edge position, and the projection recovers the old interval from it.
#[derive(Default)]
pub struct DeriveEdgeAnchors {
    /// Last settled transform per anchored node — how an authored move
    /// (differs from cache AND derived) is told apart from a parent-edge
    /// move (matches cache, derived went away).
    cache: std::sync::Mutex<std::collections::HashMap<Entity, Transform>>,
}

impl System for DeriveEdgeAnchors {
    type Result = ();

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        let mut cache = self.cache.lock().expect("anchor cache");
        cache.retain(|entity, _| ctx.raw_world().is_alive(*entity));
        for _ in 0..MAX_PASSES {
            let updates = anchor_updates(ctx.raw_world(), &mut cache, true);
            if updates.is_empty() {
                break;
            }
            ctx.lock::<(
                WriteAll<Transform>,
                WriteAll<GlobalTransform>,
                WriteAll<RoadNode>,
                WriteAll<EdgeAnchor>,
            )>()
            .execute(|(mut transforms, mut globals, mut nodes, mut anchors)| {
                for (entity, transform, half_width, interval) in &updates {
                    if let Some(mut slot) = transforms.get_mut(entity.index()) {
                        *slot = *transform;
                    }
                    if let Some(mut global) = globals.get_mut(entity.index()) {
                        global.0 = transform.to_matrix();
                    }
                    if let Some(mut node) = nodes.get_mut(entity.index()) {
                        node.half_width = *half_width;
                    }
                    if let Some((u_min, u_max)) = interval
                        && let Some(mut anchor) = anchors.get_mut(entity.index())
                    {
                        anchor.u_min = *u_min;
                        anchor.u_max = *u_max;
                    }
                }
            });
            for (entity, transform, _, _) in &updates {
                cache.insert(*entity, *transform);
            }
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
                let (dist, s, _) = crate::tool::ray_segment_closest(ray, prev, next);
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
        assert!(anchor_updates(&world, &mut Default::default(), false).is_empty());
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
        assert!(anchor_updates(&world, &mut Default::default(), false).is_empty());
        let second_pos = world.get::<Transform>(second).unwrap().translation;
        assert!(second_pos.norm() > 1.0, "chained anchor derived");
    }

    #[test]
    fn authored_move_slides_the_interval_and_undo_recovers_it() {
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

        // Settle with sliding enabled, priming the cache (system behavior).
        let mut cache = std::collections::HashMap::new();
        let settle = |world: &mut World, cache: &mut std::collections::HashMap<_, _>| {
            for _ in 0..MAX_PASSES {
                let updates = anchor_updates(world, cache, true);
                if updates.is_empty() {
                    break;
                }
                apply_updates(world, &updates);
                for (entity, transform, _, _) in &updates {
                    cache.insert(*entity, *transform);
                }
            }
        };
        settle(&mut world, &mut cache);
        let home_t = *world.get::<Transform>(anchored).unwrap();
        assert!((home_t.translation - Vec3::new(3.0, 0.0, 10.0)).norm() < 1e-3);

        // Author drags the node up the road (gizmo writes Transform): the
        // interval slides to the projection, the node snaps to the edge.
        let dragged = Transform::new(
            Vec3::new(4.5, 0.0, 16.0),
            home_t.rotation,
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(anchored, dragged).unwrap();
        world
            .insert(anchored, GlobalTransform(dragged.to_matrix()))
            .unwrap();
        settle(&mut world, &mut cache);
        let slid = world.get::<EdgeAnchor>(anchored).unwrap().clone();
        let slid_t = *world.get::<Transform>(anchored).unwrap();
        let center = (slid.u_min + slid.u_max) * 0.5;
        assert!((center - 0.8).abs() < 1e-3, "slid to u≈0.8, got {center}");
        assert!(
            ((slid.u_max - slid.u_min) - 0.2).abs() < 1e-4,
            "interval width preserved"
        );
        assert!(
            (slid_t.translation - Vec3::new(3.0, 0.0, 16.0)).norm() < 1e-3,
            "snapped back onto the edge, got {:?}",
            slid_t.translation
        );

        // Parent-road edits must NOT slide: move the road's far node — the
        // anchored node follows parametrically, interval unchanged.
        let seg_b = world.get::<RoadSegment>(road).unwrap().b;
        let t = Transform::new(
            Vec3::new(6.0, 0.0, 20.0),
            quat_from_rotation_y(0.3),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(seg_b, t).unwrap();
        world.insert(seg_b, GlobalTransform(t.to_matrix())).unwrap();
        settle(&mut world, &mut cache);
        let after_follow = world.get::<EdgeAnchor>(anchored).unwrap().clone();
        assert!((after_follow.u_min - slid.u_min).abs() < 1e-6, "no drift");
        assert!((after_follow.u_max - slid.u_max).abs() < 1e-6, "no drift");
        // Restore the road for the undo half of the test.
        let back = Transform::new(
            Vec3::new(0.0, 0.0, 20.0),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(seg_b, back).unwrap();
        world
            .insert(seg_b, GlobalTransform(back.to_matrix()))
            .unwrap();
        settle(&mut world, &mut cache);

        // Undo of the drag restores the node's old on-edge transform; the
        // projection recovers the original interval from it.
        world.insert(anchored, home_t).unwrap();
        world
            .insert(anchored, GlobalTransform(home_t.to_matrix()))
            .unwrap();
        settle(&mut world, &mut cache);
        let reverted = world.get::<EdgeAnchor>(anchored).unwrap().clone();
        assert!(
            (reverted.u_min - 0.4).abs() < 1e-3 && (reverted.u_max - 0.6).abs() < 1e-3,
            "projection recovered the original interval, got [{}, {}]",
            reverted.u_min,
            reverted.u_max
        );
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
