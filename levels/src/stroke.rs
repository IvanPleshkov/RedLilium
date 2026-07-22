//! Strokes: open pen-model polylines on the landscape — the boundary
//! primitive of the architecture chapter.
//!
//! A stroke is **bare geometry**: an ordered open polyline with optional
//! curved segments, drawn point-wise onto the world — a fence line, a
//! scarp, a curb, a plot edge. What a stroke *means* is deliberately not
//! encoded here; stroke geometry will be handed to the generator alongside
//! road geometry through a single semantic mechanism designed later.
//! Because strokes are open lines rather than closed contours, plots never
//! need stitching: a border shared by two plots is *one* stroke, and
//! terrain flows continuously everywhere it isn't told otherwise. Closed
//! contours (with their own interior fill) are a future level assembled
//! from stroke pieces — a stroke itself is never closed.
//!
//! - **Path**: ordered child [`StrokeVertex`] entities referenced by
//!   [`Stroke::points`]. Each segment is a cubic Bézier steered by the
//!   **pen model**: every vertex carries two local handle vectors
//!   (`handle_out` toward the next vertex, `handle_in` toward the
//!   previous). Both adjacent handles zero → a straight segment; mirrored
//!   collinear handles → a **C1** joint; arbitrary handles → curves
//!   meeting at a corner. Handles live in the vertex's local space, so
//!   rotating a vertex with the gizmo steers its curve. Vertex local
//!   translations carry heights — a stroke rides the world in full 3D.
//! - **Gates** ([`Gate`]): connection sockets droppable onto a stroke —
//!   child `RoadNode`s, +Z toward the side the drop click came from.
//!   Two-sided per the socket rule: a road is met from whichever side it
//!   comes from.
//! - **Grouping is plain hierarchy**: there is no container component. A
//!   root entity holding strokes, buildings and roads as its subtree IS
//!   the prefab ("villa" = fences + buildings + a driveway under one
//!   root) — see [`DuplicateSubtreeAction`].

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::{Vec3, quat_from_rotation_y};
use redlilium_ecs::{
    Component, Entity, GlobalTransform, Transform, World, remove_parent, set_parent,
};

/// An open polyline: `points` lists the [`StrokeVertex`] children in path
/// order (explicit — never re-derived). Dead or non-vertex references are
/// skipped at evaluation; fewer than 2 live vertices means no path.
#[derive(Debug, Clone, Default, Component)]
pub struct Stroke {
    pub points: Vec<Entity>,
}

/// A path vertex (a child of the stroke; its local translation is the
/// vertex position, heights included).
///
/// The handles make the adjacent segments curve (pen model): the segment
/// leaving this vertex uses `handle_out` as its first Bézier control
/// offset, the segment arriving uses `handle_in` as its last. Zero handle
/// = straight approach on that side. Handles are **vertex-local** vectors:
/// rotating the vertex rotates its curve. C1 is authored by mirroring
/// (`handle_out = -handle_in`); anything else is a corner.
#[derive(Debug, Clone, Default, Component)]
pub struct StrokeVertex {
    /// Bézier control offset of the departing segment, vertex-local.
    pub handle_out: Vec3,
    /// Bézier control offset of the arriving segment, vertex-local.
    pub handle_in: Vec3,
}

/// Marker on a connection socket dropped onto a stroke: a child `RoadNode`
/// on the line, +Z toward the side it was dropped from. Two-sided — roads
/// are met from whichever side they come from (the standard socket rule
/// generalized; see `tool::socket_meets_front`).
#[derive(Debug, Clone, Default, Component)]
pub struct Gate;

/// Tessellation of one curved segment.
const CURVE_STEPS: usize = 12;

/// Handles below this length (meters) count as absent — the approach on
/// that side is straight.
const HANDLE_EPS: f32 = 1e-3;

/// The live path vertices in world space: `(vertex entity, position,
/// world handle_out, world handle_in)`, path order. `None` with fewer
/// than 2 live vertices.
pub(crate) fn stroke_corners(
    world: &World,
    stroke: &Stroke,
) -> Option<Vec<(Entity, Vec3, Vec3, Vec3)>> {
    let corners: Vec<(Entity, Vec3, Vec3, Vec3)> = stroke
        .points
        .iter()
        .filter_map(|&v| {
            let vertex = world.get::<StrokeVertex>(v)?;
            let gt = world.get::<GlobalTransform>(v)?;
            let point = |local: Vec3| {
                let p = gt.0 * redlilium_core::math::Vec4::new(local.x, local.y, local.z, 1.0);
                Vec3::new(p.x, p.y, p.z)
            };
            Some((
                v,
                point(Vec3::zeros()),
                point(vertex.handle_out),
                point(vertex.handle_in),
            ))
        })
        .collect();
    (corners.len() >= 2).then_some(corners)
}

/// The stroke's path in world space, tessellated: straight segments
/// contribute their start vertex, curved segments a cubic-Bézier fan of
/// [`CURVE_STEPS`] samples. **Open** — the last vertex ends the path, it
/// never connects back. `None` with fewer than 2 live vertices.
pub fn stroke_path(world: &World, stroke: &Stroke) -> Option<Vec<Vec3>> {
    let corners = stroke_corners(world, stroke)?;
    let n = corners.len();
    let mut points = Vec::with_capacity(n * 2);
    for i in 0..n - 1 {
        let (_, p0, h_out, _) = corners[i];
        let (_, p3, _, h_in) = corners[i + 1];
        points.push(p0);
        let straight = (h_out - p0).norm() < HANDLE_EPS && (h_in - p3).norm() < HANDLE_EPS;
        if straight {
            continue;
        }
        // Cubic p0 → h_out → h_in → p3; interior samples only (endpoints
        // are the vertices themselves).
        for step in 1..CURVE_STEPS {
            let t = step as f32 / CURVE_STEPS as f32;
            let s = 1.0 - t;
            points.push(
                p0 * (s * s * s)
                    + h_out * (3.0 * t * s * s)
                    + h_in * (3.0 * t * t * s)
                    + p3 * (t * t * t),
            );
        }
    }
    points.push(corners[n - 1].1);
    Some(points)
}

/// Default path for a freshly stamped stroke: an L in local space — 8 m
/// along +X (the frontage when edge-anchored), then 8 m back along −Z.
const DEFAULT_STROKE: [[f32; 2]; 3] = [[-4.0, 0.0], [4.0, 0.0], [4.0, -8.0]];

/// The stroke's frontage length: the **local** distance between its first
/// two vertices — the span that glues to a road when the stroke is
/// edge-anchored. The rigid stroke dictates this length; the road-edge
/// interval width derives from it (see `anchor::derive_stroke_anchor`).
pub(crate) fn stroke_frontage(world: &World, stroke: &Stroke) -> Option<f32> {
    let mut locals = stroke.points.iter().filter_map(|&v| {
        world.get::<StrokeVertex>(v)?;
        world.get::<Transform>(v).map(|t| t.translation)
    });
    let a = locals.next()?;
    let b = locals.next()?;
    let d = (b - a).norm();
    (d > 1e-3).then_some(d)
}

/// Cursor reach for dropping a gate onto a stroke, world units.
const GATE_DROP_RADIUS: f32 = 2.0;

/// Where an "Add gate" click lands on a stroke: the stroke-local transform
/// for a gate at the closest path point — positioned on the (tessellated)
/// line, +Z toward the side the cursor is on (an open line has no interior
/// to point away from — the author picks the facing by which side they
/// click from; clicks dead-on default to the left-hand normal). `None`
/// when the cursor is too far from the path or the stroke has no usable
/// transform.
pub(crate) fn gate_spot(
    world: &World,
    stroke_entity: Entity,
    ray: &redlilium_ecs::ui::ViewportRay,
) -> Option<Transform> {
    let stroke = world.get::<Stroke>(stroke_entity)?;
    let path = stroke_path(world, stroke)?;
    let mut best: Option<(f32, Vec3, Vec3, Vec3)> = None;
    for i in 0..path.len() - 1 {
        let (a, b) = (path[i], path[i + 1]);
        let (dist, s, t) = crate::tool::ray_segment_closest(ray, a, b);
        if best.is_none_or(|(d, _, _, _)| dist < d) {
            let on_segment = a + (b - a) * s;
            let on_ray = ray.origin + ray.dir * t;
            best = Some((dist, on_segment, b - a, on_ray - on_segment));
        }
    }
    let (dist, point, along, side) = best?;
    if dist > GATE_DROP_RADIUS {
        return None;
    }
    let along = Vec3::new(along.x, 0.0, along.z);
    if along.norm() < 1e-4 {
        return None;
    }
    let along = along.normalize();
    let n = Vec3::new(-along.z, 0.0, along.x);
    let side = Vec3::new(side.x, 0.0, side.z);
    let outward = if side.dot(&n) < 0.0 { -n } else { n };

    // World gate pose → stroke-local (gates are children).
    let parent = world.get::<GlobalTransform>(stroke_entity)?.0;
    let inv = parent.try_inverse()?;
    let local_point = {
        let p = inv * redlilium_core::math::Vec4::new(point.x, point.y, point.z, 1.0);
        Vec3::new(p.x, p.y, p.z)
    };
    let local_out = {
        let d = inv * redlilium_core::math::Vec4::new(outward.x, outward.y, outward.z, 0.0);
        Vec3::new(d.x, 0.0, d.z)
    };
    if local_out.norm() < 1e-4 {
        return None;
    }
    let local_out = local_out.normalize();
    Some(Transform::new(
        local_point,
        quat_from_rotation_y(local_out.x.atan2(local_out.z)),
        Vec3::new(1.0, 1.0, 1.0),
    ))
}

// ---------------------------------------------------------------------------
// Edit actions
// ---------------------------------------------------------------------------

/// Undoable "stamp a stroke": the root entity plus a default L of vertex
/// children — drag the vertices into shape afterwards. Free-standing at a
/// point, or glued to a road edge (the anchored variant derives its
/// placement from the edge and dictates the interval width through its
/// frontage; slide it along the road with the gizmo afterwards).
#[derive(Debug)]
pub struct AddStrokeAction {
    transform: Transform,
    anchor: Option<crate::anchor::EdgeAnchor>,
    created: Vec<Entity>,
}

impl AddStrokeAction {
    pub fn at_point(point: Vec3) -> Self {
        Self {
            transform: Transform::new(point, quat_from_rotation_y(0.0), Vec3::new(1.0, 1.0, 1.0)),
            anchor: None,
            created: Vec::new(),
        }
    }

    /// Glue to a road edge; only the interval's center matters — the width
    /// derives from the stroke's frontage at apply time.
    pub fn on_edge(anchor: crate::anchor::EdgeAnchor) -> Self {
        Self {
            transform: Transform::default(),
            anchor: Some(anchor),
            created: Vec::new(),
        }
    }
}

impl EditAction<World> for AddStrokeAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let undo_partial = |world: &mut World, created: &mut Vec<Entity>| {
            for e in created.drain(..).rev() {
                remove_parent(world, e);
                world.despawn(e);
            }
        };
        let stroke = world.spawn();
        self.created.push(stroke);
        let inserted = world
            .insert(stroke, self.transform)
            .and_then(|_| world.insert(stroke, GlobalTransform(self.transform.to_matrix())))
            .and_then(|_| world.insert(stroke, redlilium_ecs::Name("Stroke".to_owned())));
        if let Err(e) = inserted {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }

        let mut points = Vec::with_capacity(DEFAULT_STROKE.len());
        for (n, [x, z]) in DEFAULT_STROKE.into_iter().enumerate() {
            let local = Transform::new(
                Vec3::new(x, 0.0, z),
                quat_from_rotation_y(0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            let vertex = world.spawn();
            self.created.push(vertex);
            let inserted = world
                .insert(vertex, local)
                .and_then(|_| {
                    world.insert(
                        vertex,
                        GlobalTransform(self.transform.to_matrix() * local.to_matrix()),
                    )
                })
                .and_then(|_| world.insert(vertex, StrokeVertex::default()))
                .and_then(|_| {
                    world.insert(vertex, redlilium_ecs::Name(format!("Point {}", n + 1)))
                });
            if let Err(e) = inserted {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::Custom(e.to_string()));
            }
            set_parent(world, vertex, stroke);
            points.push(vertex);
        }
        if let Err(e) = world.insert(stroke, Stroke { points }) {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }

        // Anchored variant: derive the on-edge placement now that the
        // path (and with it the frontage) exists, and store the derived
        // interval so the graph ships settled.
        if let Some(anchor) = &self.anchor {
            let Some((t, (u_min, u_max))) =
                crate::anchor::derive_stroke_anchor(world, stroke, anchor)
            else {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::TargetNotFound(
                    "stroke anchor road missing or degenerate".into(),
                ));
            };
            let settled = crate::anchor::EdgeAnchor {
                u_min,
                u_max,
                ..anchor.clone()
            };
            let inserted = world
                .insert(stroke, t)
                .and_then(|_| world.insert(stroke, GlobalTransform(t.to_matrix())))
                .and_then(|_| world.insert(stroke, settled));
            if let Err(e) = inserted {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::Custom(e.to_string()));
            }
            // Children were placed against the pre-derive transform —
            // refresh their world matrices under the new one.
            let vertices: Vec<Entity> = world.get::<Stroke>(stroke).unwrap().points.clone();
            for v in vertices {
                if let Some(local) = world.get::<Transform>(v).copied() {
                    let _ = world.insert(v, GlobalTransform(t.to_matrix() * local.to_matrix()));
                }
            }
        }
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        // Children first (reverse creation order), then the stroke itself.
        for e in self.created.drain(..).rev() {
            remove_parent(world, e);
            world.despawn(e);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Add stroke"
    }
}

/// Undoable "add a gate to a stroke": spawns a child `RoadNode` + [`Gate`]
/// at a stroke-local transform (typically produced by [`gate_spot`] — on
/// the line, +Z toward the click side).
#[derive(Debug)]
pub struct AddGateAction {
    stroke: Entity,
    local: Transform,
    created: Option<Entity>,
}

impl AddGateAction {
    pub fn new(stroke: Entity, local: Transform) -> Self {
        Self {
            stroke,
            local,
            created: None,
        }
    }
}

impl EditAction<World> for AddGateAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if world.get::<Stroke>(self.stroke).is_none() {
            return Err(EditActionError::TargetNotFound(
                "gate target is not a stroke".into(),
            ));
        }
        let parent_m = world
            .get::<GlobalTransform>(self.stroke)
            .map(|gt| gt.0)
            .ok_or_else(|| EditActionError::TargetNotFound("stroke has no transform".into()))?;
        let gate = world.spawn();
        let inserted = world
            .insert(gate, self.local)
            .and_then(|_| world.insert(gate, GlobalTransform(parent_m * self.local.to_matrix())))
            .and_then(|_| world.insert(gate, crate::RoadNode { half_width: 1.5 }))
            .and_then(|_| world.insert(gate, Gate))
            .and_then(|_| world.insert(gate, redlilium_ecs::Name("Gate".to_owned())));
        if let Err(e) = inserted {
            world.despawn(gate);
            return Err(EditActionError::Custom(e.to_string()));
        }
        set_parent(world, gate, self.stroke);
        self.created = Some(gate);
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if let Some(gate) = self.created.take() {
            remove_parent(world, gate);
            world.despawn(gate);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Add gate"
    }
}

/// Undoable "duplicate a subtree at a point": extracts the selected root's
/// subtree as a [`Prefab`](redlilium_ecs::Prefab) — strokes with their
/// vertices, gates, buildings, roads, everything — and instantiates the
/// copy with its root at `point`. This is the prefab payoff: group content
/// under one root entity ("villa" = fences + buildings + a driveway) and
/// stamp it anywhere, "one recipe, ten placements" without an asset file
/// yet.
#[derive(Debug)]
pub struct DuplicateSubtreeAction {
    source: Entity,
    point: Vec3,
    created: Vec<Entity>,
}

impl DuplicateSubtreeAction {
    pub fn new(source: Entity, point: Vec3) -> Self {
        Self {
            source,
            point,
            created: Vec::new(),
        }
    }
}

impl EditAction<World> for DuplicateSubtreeAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.source) {
            return Err(EditActionError::TargetNotFound(
                "duplicate source despawned".into(),
            ));
        }
        let prefab = world.extract_prefab(self.source);
        if prefab.is_empty() {
            return Err(EditActionError::TargetNotFound(
                "subtree extraction came back empty".into(),
            ));
        }
        self.created = prefab.instantiate(world);
        let root = self.created[0];

        // Place the copy: keep the source's rotation, move to the click
        // point. A copied edge anchor would fight the authored placement —
        // the duplicate starts free (re-glue it explicitly if wanted).
        let _ = world.remove::<crate::anchor::EdgeAnchor>(root);
        let rotation = world
            .get::<Transform>(root)
            .map(|t| t.rotation)
            .unwrap_or_else(|| quat_from_rotation_y(0.0));
        let t = Transform::new(self.point, rotation, Vec3::new(1.0, 1.0, 1.0));
        let _ = world.insert(root, t);
        // Refresh world matrices down the copied subtree (extraction is
        // BFS, so parents precede children in `created`).
        let _ = world.insert(root, GlobalTransform(t.to_matrix()));
        let members: std::collections::HashSet<Entity> = self.created.iter().copied().collect();
        for &e in self.created.iter().skip(1) {
            let parent_m = world
                .get::<redlilium_ecs::Parent>(e)
                .filter(|p| members.contains(&p.0))
                .and_then(|p| world.get::<GlobalTransform>(p.0).map(|gt| gt.0));
            if let (Some(parent_m), Some(local)) = (parent_m, world.get::<Transform>(e).copied()) {
                let _ = world.insert(e, GlobalTransform(parent_m * local.to_matrix()));
            }
        }
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        for e in self.created.drain(..).rev() {
            remove_parent(world, e);
            world.despawn(e);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Duplicate subtree"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> World {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<Stroke>();
        world.register_inspector_default::<StrokeVertex>();
        world.register_inspector_default::<Gate>();
        world
    }

    fn spawned_stroke(world: &World) -> (Entity, Stroke) {
        world
            .read_all::<Stroke>()
            .unwrap()
            .iter()
            .filter_map(|(index, s)| Some((world.entity_at_index(index)?, s.clone())))
            .next()
            .unwrap()
    }

    #[test]
    fn add_stroke_stamps_an_open_path_and_undo_reverts() {
        let mut world = world();
        let mut action = AddStrokeAction::at_point(Vec3::new(10.0, 2.0, 5.0));
        action.apply(&mut world).unwrap();

        let (stroke, component) = spawned_stroke(&world);
        assert_eq!(component.points.len(), 3);

        let path = stroke_path(&world, &component).expect("path");
        // Open: one point per vertex, no closing segment.
        assert_eq!(path.len(), 3);
        // The whole path inherits the stroke's height (full 3D, no ground
        // plane); the L default starts at local (−4, 0, 0).
        assert!((path[0] - Vec3::new(6.0, 2.0, 5.0)).norm() < 1e-4);
        assert!((path[2] - Vec3::new(14.0, 2.0, -3.0)).norm() < 1e-4);
        for v in &component.points {
            assert_eq!(
                world.get::<redlilium_ecs::Parent>(*v).unwrap().0,
                stroke,
                "vertices are children of the stroke"
            );
        }

        // Dragging a vertex reshapes the path (order stays authored).
        let v2 = component.points[2];
        let t = Transform::new(
            Vec3::new(1.0, 0.0, -2.0),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(v2, t).unwrap();
        world
            .insert(
                v2,
                GlobalTransform(world.get::<GlobalTransform>(stroke).unwrap().0 * t.to_matrix()),
            )
            .unwrap();
        let reshaped = stroke_path(&world, &component).expect("path");
        assert!((reshaped[2] - Vec3::new(11.0, 2.0, 3.0)).norm() < 1e-4);

        action.undo(&mut world).unwrap();
        assert!(world.read_all::<Stroke>().unwrap().iter().next().is_none());
        assert!(
            world
                .read_all::<StrokeVertex>()
                .unwrap()
                .iter()
                .next()
                .is_none()
        );
    }

    #[test]
    fn handles_curve_segments_straight_by_default() {
        let mut world = world();
        let mut action = AddStrokeAction::at_point(Vec3::zeros());
        action.apply(&mut world).unwrap();
        let (_, component) = spawned_stroke(&world);

        // All handles zero → pure polyline: exactly one point per vertex.
        assert_eq!(stroke_path(&world, &component).unwrap().len(), 3);

        // Give the first segment (v0 → v1, from (−4,0,0) to (4,0,0)) a
        // bulge: mirrored-style handles pushing toward +Z.
        let v0 = component.points[0];
        let v1 = component.points[1];
        world
            .insert(
                v0,
                StrokeVertex {
                    handle_out: Vec3::new(2.0, 0.0, 2.0),
                    handle_in: Vec3::zeros(),
                },
            )
            .unwrap();
        world
            .insert(
                v1,
                StrokeVertex {
                    handle_in: Vec3::new(-2.0, 0.0, 2.0),
                    handle_out: Vec3::zeros(),
                },
            )
            .unwrap();

        let path = stroke_path(&world, &component).unwrap();
        // One curved segment → CURVE_STEPS−1 extra samples.
        assert_eq!(path.len(), 3 + (CURVE_STEPS - 1));
        // The curve bulges toward +Z at its middle (cubic midpoint z =
        // 3/4 · 2 = 1.5), while the straight segment stays put.
        let mid = path[CURVE_STEPS / 2];
        assert!((mid.z - 1.5).abs() < 1e-3, "bulged to z=1.5, got {}", mid.z);
        assert!(mid.x.abs() < 1e-3);
    }

    #[test]
    fn mirrored_handles_make_a_c1_joint_and_rotation_steers_the_curve() {
        let mut world = world();
        let mut action = AddStrokeAction::at_point(Vec3::zeros());
        action.apply(&mut world).unwrap();
        let (_, component) = spawned_stroke(&world);

        // Curve both segments around v1 with mirrored handles: C1 — the
        // arriving and departing tangents at v1 are collinear.
        let v1 = component.points[1];
        let h = Vec3::new(0.0, 0.0, -2.5);
        world
            .insert(
                v1,
                StrokeVertex {
                    handle_in: -h,
                    handle_out: h,
                },
            )
            .unwrap();

        let path = stroke_path(&world, &component).unwrap();
        // Both segments around v1 curved → two fans; v1 itself is a sample.
        let idx_v1 = CURVE_STEPS; // v0 + (CURVE_STEPS−1) samples, then v1
        assert!((path[idx_v1] - Vec3::new(4.0, 0.0, 0.0)).norm() < 1e-4);
        // Tangent continuity at the vertex, measured analytically: the
        // arriving Bézier tangent is p − handle_in, the departing one is
        // handle_out − p — mirrored handles make them identical.
        let corners = stroke_corners(&world, &component).unwrap();
        let (_, p, h_out, h_in) = corners[1];
        let arrive = (p - h_in).normalize();
        let depart = (h_out - p).normalize();
        assert!(
            (arrive - depart).norm() < 1e-5,
            "C1 at the mirrored vertex: {arrive:?} vs {depart:?}"
        );

        // Rotating the vertex steers the curve: yaw v1 by 90° and the
        // world-space handles rotate with it.
        let yaw = Transform::new(
            Vec3::new(4.0, 0.0, 0.0),
            quat_from_rotation_y(std::f32::consts::FRAC_PI_2),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(v1, yaw).unwrap();
        world.insert(v1, GlobalTransform(yaw.to_matrix())).unwrap();
        let corners = stroke_corners(&world, &component).unwrap();
        let (_, p, h_out, _) = corners[1];
        let dir = (h_out - p).normalize();
        // Local (0,0,−2.5) under yaw +90° → world −X.
        assert!((dir - Vec3::new(-1.0, 0.0, 0.0)).norm() < 1e-3);
    }

    fn world_with_road() -> (World, Entity) {
        let mut world = world();
        world.register_inspector_default::<crate::RoadNode>();
        world.register_inspector_default::<crate::RoadSegment>();
        world.register_inspector_default::<crate::anchor::EdgeAnchor>();
        let node = |world: &mut World, x: f32, z: f32| {
            let e = world.spawn();
            let t = Transform::new(
                Vec3::new(x, 0.0, z),
                quat_from_rotation_y(0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            world.insert(e, t).unwrap();
            world.insert(e, GlobalTransform(t.to_matrix())).unwrap();
            world.insert(e, crate::RoadNode::default()).unwrap();
            e
        };
        // Straight road along +Z from origin to (0, 0, 20).
        let a = node(&mut world, 0.0, 0.0);
        let b = node(&mut world, 0.0, 20.0);
        let road = world.spawn();
        world
            .insert(
                road,
                crate::RoadSegment {
                    a,
                    b,
                    ..crate::RoadSegment::default()
                },
            )
            .unwrap();
        (world, road)
    }

    #[test]
    fn anchored_stroke_dictates_the_interval_and_follows_frontage_edits() {
        let (mut world, road) = world_with_road();
        AddStrokeAction::on_edge(crate::anchor::EdgeAnchor {
            parent_road: road,
            right_edge: true,
            u_min: 0.5,
            u_max: 0.5,
        })
        .apply(&mut world)
        .unwrap();
        let (stroke, component) = spawned_stroke(&world);

        // The default frontage is 8 m; on a 20 m straight edge that is a
        // derived u half-width of 0.2 around the authored center 0.5.
        let anchor = world
            .get::<crate::anchor::EdgeAnchor>(stroke)
            .unwrap()
            .clone();
        assert!((anchor.u_min - 0.3).abs() < 1e-3, "got {}", anchor.u_min);
        assert!((anchor.u_max - 0.7).abs() < 1e-3, "got {}", anchor.u_max);
        // Sits on the right edge (x = +3), frontage FACING the road: +Z
        // points into it (−X), the tail extends outward (+X).
        let t = *world.get::<Transform>(stroke).unwrap();
        assert!((t.translation - Vec3::new(3.0, 0.0, 10.0)).norm() < 1e-3);
        let heading = crate::bezier::heading(&t.to_matrix());
        assert!((heading - Vec3::new(-1.0, 0.0, 0.0)).norm() < 1e-3);
        // Ships settled.
        assert!(crate::anchor::anchor_updates(&world, &mut Default::default(), false).is_empty());

        // The stroke dictates: widen the frontage (move the second vertex
        // out) and the edge interval follows, center preserved.
        let v1 = component.points[1];
        let wider = Transform::new(
            Vec3::new(6.0, 0.0, 0.0),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(v1, wider).unwrap();
        crate::settle_edge_anchors(&mut world);
        let anchor = world
            .get::<crate::anchor::EdgeAnchor>(stroke)
            .unwrap()
            .clone();
        // Frontage 10 m → half-width 0.25.
        assert!((anchor.u_min - 0.25).abs() < 1e-3, "got {}", anchor.u_min);
        assert!((anchor.u_max - 0.75).abs() < 1e-3, "got {}", anchor.u_max);
    }

    #[test]
    fn anchored_stroke_slides_keeping_its_derived_width() {
        let (mut world, road) = world_with_road();
        AddStrokeAction::on_edge(crate::anchor::EdgeAnchor {
            parent_road: road,
            right_edge: true,
            u_min: 0.5,
            u_max: 0.5,
        })
        .apply(&mut world)
        .unwrap();
        let (stroke, _) = spawned_stroke(&world);

        // Prime the settled cache, then drag the stroke up the road.
        let mut cache = std::collections::HashMap::new();
        let settle = |world: &mut World, cache: &mut std::collections::HashMap<_, _>| {
            for _ in 0..8 {
                let updates = crate::anchor::anchor_updates(world, cache, true);
                if updates.is_empty() {
                    break;
                }
                for (entity, transform, _, interval) in &updates {
                    let _ = world.insert(*entity, *transform);
                    let _ = world.insert(*entity, GlobalTransform(transform.to_matrix()));
                    if let Some((u_min, u_max)) = interval
                        && let Some(mut a) =
                            world.get::<crate::anchor::EdgeAnchor>(*entity).cloned()
                    {
                        a.u_min = *u_min;
                        a.u_max = *u_max;
                        let _ = world.insert(*entity, a);
                    }
                    cache.insert(*entity, *transform);
                }
            }
        };
        settle(&mut world, &mut cache);
        let home = *world.get::<Transform>(stroke).unwrap();

        let dragged = Transform::new(
            home.translation + Vec3::new(1.0, 0.0, 4.0),
            home.rotation,
            home.scale,
        );
        world.insert(stroke, dragged).unwrap();
        world
            .insert(stroke, GlobalTransform(dragged.to_matrix()))
            .unwrap();
        settle(&mut world, &mut cache);

        let anchor = world
            .get::<crate::anchor::EdgeAnchor>(stroke)
            .unwrap()
            .clone();
        let center = (anchor.u_min + anchor.u_max) * 0.5;
        assert!((center - 0.7).abs() < 1e-2, "slid to u≈0.7, got {center}");
        assert!(
            ((anchor.u_max - anchor.u_min) - 0.4).abs() < 1e-3,
            "width stays frontage-derived"
        );
        // Snapped back onto the edge.
        let t = *world.get::<Transform>(stroke).unwrap();
        assert!((t.translation.x - 3.0).abs() < 1e-3);
    }

    #[test]
    fn gate_spot_faces_the_click_side_and_add_gate_roundtrips() {
        let mut world = world();
        AddStrokeAction::at_point(Vec3::new(10.0, 0.0, 5.0))
            .apply(&mut world)
            .unwrap();
        let (stroke, _) = spawned_stroke(&world);
        world.register_inspector_default::<crate::RoadNode>();

        // Ray straight down just NORTH of the first segment's middle (the
        // segment spans x 6..14 at z = 5; the cursor is at z = 5.5): the
        // gate lands on the line, +Z toward the click side (+Z north).
        let ray = redlilium_ecs::ui::ViewportRay {
            origin: Vec3::new(10.0, 10.0, 5.5),
            dir: Vec3::new(0.0, -1.0, 0.0),
        };
        let local = gate_spot(&world, stroke, &ray).expect("spot on the path");
        assert!(local.translation.norm() < 1e-3, "segment midpoint");
        let out = crate::bezier::heading(&local.to_matrix());
        assert!((out - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-3);

        // The mirror click from the south flips the facing.
        let ray_south = redlilium_ecs::ui::ViewportRay {
            origin: Vec3::new(10.0, 10.0, 4.5),
            dir: Vec3::new(0.0, -1.0, 0.0),
        };
        let flipped = gate_spot(&world, stroke, &ray_south).expect("spot");
        let out = crate::bezier::heading(&flipped.to_matrix());
        assert!((out - Vec3::new(0.0, 0.0, -1.0)).norm() < 1e-3);

        let mut action = AddGateAction::new(stroke, local);
        action.apply(&mut world).unwrap();
        let gate = world
            .read_all::<Gate>()
            .unwrap()
            .iter()
            .filter_map(|(index, _)| world.entity_at_index(index))
            .next()
            .expect("gate spawned");
        assert!(world.get::<crate::RoadNode>(gate).is_some());
        assert_eq!(world.get::<redlilium_ecs::Parent>(gate).unwrap().0, stroke);
        let gt = world.get::<GlobalTransform>(gate).unwrap().0;
        let pos = Vec3::new(gt[(0, 3)], gt[(1, 3)], gt[(2, 3)]);
        assert!((pos - Vec3::new(10.0, 0.0, 5.0)).norm() < 1e-3);

        action.undo(&mut world).unwrap();
        assert!(!world.is_alive(gate));
    }

    #[test]
    fn subtree_duplicates_with_all_content_remapped() {
        let mut world = world();
        world.register_inspector_default::<crate::RoadNode>();
        world.register_inspector_default::<crate::RoadSegment>();
        world.register_inspector_default::<crate::building::Building>();

        // A group in the making: the stroke root carries a gate, a
        // building and an internal road from an inner node to the gate —
        // any entity's subtree is a prefab, no container component needed.
        AddStrokeAction::at_point(Vec3::new(0.0, 0.0, 0.0))
            .apply(&mut world)
            .unwrap();
        let (root, _) = spawned_stroke(&world);
        let gate_local = Transform::new(
            Vec3::zeros(),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        let mut add_gate = AddGateAction::new(root, gate_local);
        add_gate.apply(&mut world).unwrap();
        let gate = world
            .read_all::<Gate>()
            .unwrap()
            .iter()
            .filter_map(|(index, _)| world.entity_at_index(index))
            .next()
            .unwrap();
        crate::building::PlaceBuildingAction::new(
            Some(root),
            Transform::new(
                Vec3::new(0.0, 0.0, -5.0),
                quat_from_rotation_y(0.0),
                Vec3::new(1.0, 1.0, 1.0),
            ),
            crate::building::Building::default(),
        )
        .apply(&mut world)
        .unwrap();
        let inner = world.spawn();
        let inner_t = Transform::new(
            Vec3::new(0.0, 0.0, -6.0),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(inner, inner_t).unwrap();
        world
            .insert(inner, GlobalTransform(inner_t.to_matrix()))
            .unwrap();
        world.insert(inner, crate::RoadNode::default()).unwrap();
        set_parent(&mut world, inner, root);
        let internal_road = world.spawn();
        world
            .insert(
                internal_road,
                crate::RoadSegment {
                    a: inner,
                    b: gate,
                    ..crate::RoadSegment::default()
                },
            )
            .unwrap();
        set_parent(&mut world, internal_road, root);

        // Duplicate the whole thing 30 m to the east.
        let mut dup = DuplicateSubtreeAction::new(root, Vec3::new(30.0, 0.0, 0.0));
        dup.apply(&mut world).unwrap();

        let strokes: Vec<(Entity, Stroke)> = world
            .read_all::<Stroke>()
            .unwrap()
            .iter()
            .filter_map(|(index, s)| Some((world.entity_at_index(index)?, s.clone())))
            .collect();
        assert_eq!(strokes.len(), 2);
        let (copy, copy_stroke) = strokes
            .iter()
            .find(|(e, _)| *e != root)
            .expect("the duplicate");

        // Point references remapped: the copy points at its own fresh
        // vertices, none shared with the source.
        let source_points: std::collections::HashSet<Entity> = strokes
            .iter()
            .find(|(e, _)| *e == root)
            .unwrap()
            .1
            .points
            .iter()
            .copied()
            .collect();
        assert_eq!(copy_stroke.points.len(), 3);
        for v in &copy_stroke.points {
            assert!(!source_points.contains(v), "vertex remapped, not shared");
            assert_eq!(world.get::<redlilium_ecs::Parent>(*v).unwrap().0, *copy);
        }
        // The copied path is the source path shifted by (+30, 0, 0).
        let src_path = stroke_path(&world, &strokes[0].1).unwrap();
        let copy_path = stroke_path(&world, copy_stroke).unwrap();
        assert_eq!(src_path.len(), copy_path.len());
        for (a, b) in src_path.iter().zip(&copy_path) {
            assert!((*a + Vec3::new(30.0, 0.0, 0.0) - *b).norm() < 1e-3);
        }
        // The copied internal road references the COPY's inner node and
        // gate — entity refs inside the subtree remapped.
        let roads: Vec<crate::RoadSegment> = world
            .read_all::<crate::RoadSegment>()
            .unwrap()
            .iter()
            .map(|(_, s)| s.clone())
            .collect();
        assert_eq!(roads.len(), 2);
        let copy_road = roads
            .iter()
            .find(|s| s.a != inner)
            .expect("copied internal road");
        assert_ne!(copy_road.b, gate, "gate reference remapped");
        assert_eq!(
            world.get::<redlilium_ecs::Parent>(copy_road.b).unwrap().0,
            *copy,
            "the copied road ends at the copy's own gate"
        );
        // Two buildings, two gates total.
        assert_eq!(
            world
                .read_all::<crate::building::Building>()
                .unwrap()
                .iter()
                .count(),
            2
        );

        // Undo removes the whole copied subtree; the source is intact.
        dup.undo(&mut world).unwrap();
        assert_eq!(world.read_all::<Stroke>().unwrap().iter().count(), 1);
        assert!(world.is_alive(root) && world.is_alive(gate) && world.is_alive(inner));
        assert_eq!(
            world
                .read_all::<crate::RoadSegment>()
                .unwrap()
                .iter()
                .count(),
            1
        );
    }

    #[test]
    fn path_needs_two_live_vertices() {
        let mut world = world();
        let mut action = AddStrokeAction::at_point(Vec3::zeros());
        action.apply(&mut world).unwrap();
        let (_, component) = spawned_stroke(&world);
        world.despawn(component.points[0]);
        assert_eq!(stroke_path(&world, &component).unwrap().len(), 2);
        world.despawn(component.points[1]);
        assert!(stroke_path(&world, &component).is_none());
    }
}
