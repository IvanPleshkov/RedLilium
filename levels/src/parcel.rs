//! Parcels: prefab-shaped containers of the architecture chapter.
//!
//! A parcel is a piece of the world bounded by a **closed polyline** and
//! owning everything inside: buildings, internal roads, props — all child
//! entities in the parcel's local space, which is exactly what makes the
//! subtree a natural **prefab** ("parcel with a villa", "parcel with a
//! whole factory"). Terrain never enters the boundary; the perimeter (a
//! curve with heights) is the terrain's boundary condition, possibly with
//! a sharp cut/fill transition.
//!
//! - **Boundary**: ordered child [`ParcelVertex`] entities referenced by
//!   [`Parcel::boundary`]. Order is explicit — boundaries may be concave,
//!   so no re-derivation by angle (unlike junction loops). Each segment is
//!   a cubic Bézier steered by the **pen model**: every vertex carries two
//!   local handle vectors (`handle_out` toward the next vertex,
//!   `handle_in` toward the previous). Both adjacent handles zero → a
//!   straight segment; mirrored collinear handles → a **C1** joint;
//!   arbitrary handles → curves meeting at a corner. Handles live in the
//!   vertex's local space, so rotating a vertex with the gizmo steers its
//!   curve.
//! - **Gates** ([`ParcelGate`]): parcel-owned connection sockets on the
//!   boundary — child `RoadNode`s, +Z facing outward. Two-sided: a network
//!   road arrives at a gate from the front (`b_from_front`), the parcel's
//!   internal roads connect to the same node from behind. A parcel may own
//!   any number of gates.
//! - **Content**: buildings are child entities with their own footprint
//!   (see [`crate::building`]); internal roads are ordinary
//!   `RoadNode`/`RoadSegment` children — the road math reads
//!   `GlobalTransform` and does not care about hierarchy.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::{Vec3, quat_from_rotation_y};
use redlilium_ecs::{
    Component, Entity, GlobalTransform, Transform, World, remove_parent, set_parent,
};

/// A parcel: the container entity. `boundary` lists the [`ParcelVertex`]
/// children in perimeter order (explicit — concave boundaries are legal).
/// Dead or non-vertex references are skipped at evaluation; fewer than 3
/// live vertices means no boundary.
#[derive(Debug, Clone, Default, Component)]
pub struct Parcel {
    pub boundary: Vec<Entity>,
}

/// A boundary vertex (a child of the parcel; its local translation is the
/// vertex position, heights included — parcels are not flat in general).
///
/// The handles make the adjacent segments curve (pen model): the segment
/// leaving this vertex uses `handle_out` as its first Bézier control
/// offset, the segment arriving uses `handle_in` as its last. Zero handle
/// = straight approach on that side. Handles are **vertex-local** vectors:
/// rotating the vertex rotates its curve. C1 is authored by mirroring
/// (`handle_out = -handle_in`); anything else is a corner.
#[derive(Debug, Clone, Default, Component)]
pub struct ParcelVertex {
    /// Bézier control offset of the departing segment, vertex-local.
    pub handle_out: Vec3,
    /// Bézier control offset of the arriving segment, vertex-local.
    pub handle_in: Vec3,
}

/// Marker on a parcel-owned connection socket: a child `RoadNode` sitting
/// on the boundary, +Z outward. Network roads meet it from the front,
/// internal roads from behind — the standard socket rule.
#[derive(Debug, Clone, Default, Component)]
pub struct ParcelGate;

/// Tessellation of one curved boundary segment.
const CURVE_STEPS: usize = 12;

/// Handles below this length (meters) count as absent — the approach on
/// that side is straight.
const HANDLE_EPS: f32 = 1e-3;

/// The live boundary corners in world space: `(vertex entity, position,
/// world handle_out, world handle_in)`, perimeter order. `None` with fewer
/// than 3 live vertices.
pub(crate) fn parcel_corners(
    world: &World,
    parcel: &Parcel,
) -> Option<Vec<(Entity, Vec3, Vec3, Vec3)>> {
    let corners: Vec<(Entity, Vec3, Vec3, Vec3)> = parcel
        .boundary
        .iter()
        .filter_map(|&v| {
            let vertex = world.get::<ParcelVertex>(v)?;
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
    (corners.len() >= 3).then_some(corners)
}

/// The parcel's boundary in world space, tessellated: straight segments
/// contribute their start vertex, curved segments a cubic-Bézier fan of
/// [`CURVE_STEPS`] samples. Closed — the last point connects back to the
/// first. `None` with fewer than 3 live vertices.
pub fn parcel_loop(world: &World, parcel: &Parcel) -> Option<Vec<Vec3>> {
    let corners = parcel_corners(world, parcel)?;
    let n = corners.len();
    let mut points = Vec::with_capacity(n * 2);
    for i in 0..n {
        let (_, p0, h_out, _) = corners[i];
        let (_, p3, _, h_in) = corners[(i + 1) % n];
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
    Some(points)
}

/// Default boundary for a freshly stamped parcel: an 8×8 rectangle in
/// local space, front edge along +X at z = 0, interior extending to −Z.
const DEFAULT_BOUNDARY: [[f32; 2]; 4] = [[-4.0, 0.0], [4.0, 0.0], [4.0, -8.0], [-4.0, -8.0]];

/// The parcel's frontage length: the **local** distance between its first
/// two boundary vertices — the edge that glues to a road when the parcel
/// is edge-anchored. The rigid prefab dictates this length; the road-edge
/// interval width derives from it (see `anchor::derive_parcel_anchor`).
pub(crate) fn parcel_frontage(world: &World, parcel: &Parcel) -> Option<f32> {
    let mut locals = parcel.boundary.iter().filter_map(|&v| {
        world.get::<ParcelVertex>(v)?;
        world.get::<Transform>(v).map(|t| t.translation)
    });
    let a = locals.next()?;
    let b = locals.next()?;
    let d = (b - a).norm();
    (d > 1e-3).then_some(d)
}

/// Cursor reach for dropping a gate onto a parcel boundary, world units.
const GATE_DROP_RADIUS: f32 = 2.0;

/// Where an "Add gate" click lands on a parcel's boundary: the parcel-local
/// transform for a gate at the closest boundary point — positioned on the
/// (tessellated) loop, +Z along the outward normal (away from the loop
/// centroid). `None` when the cursor is too far from the boundary or the
/// parcel has no usable transform.
pub(crate) fn gate_spot(
    world: &World,
    parcel_entity: Entity,
    ray: &redlilium_ecs::ui::ViewportRay,
) -> Option<Transform> {
    let parcel = world.get::<Parcel>(parcel_entity)?;
    let lp = parcel_loop(world, parcel)?;
    let mut best: Option<(f32, Vec3, Vec3)> = None;
    for i in 0..lp.len() {
        let (a, b) = (lp[i], lp[(i + 1) % lp.len()]);
        let (dist, s, _) = crate::tool::ray_segment_closest(ray, a, b);
        if best.is_none_or(|(d, _, _)| dist < d) {
            best = Some((dist, a + (b - a) * s, b - a));
        }
    }
    let (dist, point, along) = best?;
    if dist > GATE_DROP_RADIUS {
        return None;
    }
    let along = Vec3::new(along.x, 0.0, along.z);
    if along.norm() < 1e-4 {
        return None;
    }
    let along = along.normalize();
    let centroid = lp.iter().fold(Vec3::zeros(), |a, p| a + p) / lp.len() as f32;
    let n = Vec3::new(-along.z, 0.0, along.x);
    let outward = if (point - centroid).dot(&n) >= 0.0 {
        n
    } else {
        -n
    };

    // World gate pose → parcel-local (gates are children).
    let parent = world.get::<GlobalTransform>(parcel_entity)?.0;
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
// Edit action
// ---------------------------------------------------------------------------

/// Undoable "stamp a parcel": the container entity plus a default
/// rectangular boundary of vertex children — drag the vertices into shape
/// afterwards. Free-standing at a point, or glued to a road edge (the
/// anchored variant derives its placement from the edge and dictates the
/// interval width through its frontage; slide it along the road with the
/// gizmo afterwards).
#[derive(Debug)]
pub struct AddParcelAction {
    transform: Transform,
    anchor: Option<crate::anchor::EdgeAnchor>,
    created: Vec<Entity>,
}

impl AddParcelAction {
    pub fn at_point(point: Vec3) -> Self {
        Self {
            transform: Transform::new(point, quat_from_rotation_y(0.0), Vec3::new(1.0, 1.0, 1.0)),
            anchor: None,
            created: Vec::new(),
        }
    }

    /// Glue to a road edge; only the interval's center matters — the width
    /// derives from the parcel's frontage at apply time.
    pub fn on_edge(anchor: crate::anchor::EdgeAnchor) -> Self {
        Self {
            transform: Transform::default(),
            anchor: Some(anchor),
            created: Vec::new(),
        }
    }
}

impl EditAction<World> for AddParcelAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let undo_partial = |world: &mut World, created: &mut Vec<Entity>| {
            for e in created.drain(..).rev() {
                remove_parent(world, e);
                world.despawn(e);
            }
        };
        let parcel = world.spawn();
        self.created.push(parcel);
        let inserted = world
            .insert(parcel, self.transform)
            .and_then(|_| world.insert(parcel, GlobalTransform(self.transform.to_matrix())));
        if let Err(e) = inserted {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }

        let mut boundary = Vec::with_capacity(DEFAULT_BOUNDARY.len());
        for [x, z] in DEFAULT_BOUNDARY {
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
                .and_then(|_| world.insert(vertex, ParcelVertex::default()));
            if let Err(e) = inserted {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::Custom(e.to_string()));
            }
            set_parent(world, vertex, parcel);
            boundary.push(vertex);
        }
        if let Err(e) = world.insert(parcel, Parcel { boundary }) {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }

        // Anchored variant: derive the on-edge placement now that the
        // boundary (and with it the frontage) exists, and store the derived
        // interval so the graph ships settled.
        if let Some(anchor) = &self.anchor {
            let Some((t, (u_min, u_max))) =
                crate::anchor::derive_parcel_anchor(world, parcel, anchor)
            else {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::TargetNotFound(
                    "parcel anchor road missing or degenerate".into(),
                ));
            };
            let settled = crate::anchor::EdgeAnchor {
                u_min,
                u_max,
                ..anchor.clone()
            };
            let inserted = world
                .insert(parcel, t)
                .and_then(|_| world.insert(parcel, GlobalTransform(t.to_matrix())))
                .and_then(|_| world.insert(parcel, settled));
            if let Err(e) = inserted {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::Custom(e.to_string()));
            }
            // Children were placed against the pre-derive transform —
            // refresh their world matrices under the new one.
            let vertices: Vec<Entity> = world.get::<Parcel>(parcel).unwrap().boundary.clone();
            for v in vertices {
                if let Some(local) = world.get::<Transform>(v).copied() {
                    let _ = world.insert(v, GlobalTransform(t.to_matrix() * local.to_matrix()));
                }
            }
        }
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        // Children first (reverse creation order), then the parcel itself.
        for e in self.created.drain(..).rev() {
            remove_parent(world, e);
            world.despawn(e);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Add parcel"
    }
}

/// Undoable "add a gate to a parcel": spawns a child `RoadNode` +
/// [`ParcelGate`] at a parcel-local transform (typically produced by
/// [`gate_spot`] — on the boundary, +Z outward).
#[derive(Debug)]
pub struct AddGateAction {
    parcel: Entity,
    local: Transform,
    created: Option<Entity>,
}

impl AddGateAction {
    pub fn new(parcel: Entity, local: Transform) -> Self {
        Self {
            parcel,
            local,
            created: None,
        }
    }
}

impl EditAction<World> for AddGateAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if world.get::<Parcel>(self.parcel).is_none() {
            return Err(EditActionError::TargetNotFound(
                "gate target is not a parcel".into(),
            ));
        }
        let parent_m = world
            .get::<GlobalTransform>(self.parcel)
            .map(|gt| gt.0)
            .ok_or_else(|| EditActionError::TargetNotFound("parcel has no transform".into()))?;
        let gate = world.spawn();
        let inserted = world
            .insert(gate, self.local)
            .and_then(|_| world.insert(gate, GlobalTransform(parent_m * self.local.to_matrix())))
            .and_then(|_| world.insert(gate, crate::RoadNode { half_width: 1.5 }))
            .and_then(|_| world.insert(gate, ParcelGate));
        if let Err(e) = inserted {
            world.despawn(gate);
            return Err(EditActionError::Custom(e.to_string()));
        }
        set_parent(world, gate, self.parcel);
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

/// Undoable "duplicate a parcel at a point": extracts the parcel's subtree
/// as a [`Prefab`](redlilium_ecs::Prefab) — boundary vertices, gates,
/// buildings, internal roads, everything — and instantiates the copy with
/// its root at `point`. This is the parcel-as-prefab payoff: "one villa
/// recipe, ten placements" without an asset file yet.
#[derive(Debug)]
pub struct DuplicateParcelAction {
    source: Entity,
    point: Vec3,
    created: Vec<Entity>,
}

impl DuplicateParcelAction {
    pub fn new(source: Entity, point: Vec3) -> Self {
        Self {
            source,
            point,
            created: Vec::new(),
        }
    }
}

impl EditAction<World> for DuplicateParcelAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if world.get::<Parcel>(self.source).is_none() {
            return Err(EditActionError::TargetNotFound(
                "duplicate source is not a parcel".into(),
            ));
        }
        let prefab = world.extract_prefab(self.source);
        if prefab.is_empty() {
            return Err(EditActionError::TargetNotFound(
                "parcel subtree extraction came back empty".into(),
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
        "Duplicate parcel"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> World {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<Parcel>();
        world.register_inspector_default::<ParcelVertex>();
        world.register_inspector_default::<ParcelGate>();
        world
    }

    #[test]
    fn add_parcel_stamps_a_loop_and_undo_reverts() {
        let mut world = world();
        let mut action = AddParcelAction::at_point(Vec3::new(10.0, 2.0, 5.0));
        action.apply(&mut world).unwrap();

        let parcels: Vec<(Entity, Parcel)> = world
            .read_all::<Parcel>()
            .unwrap()
            .iter()
            .filter_map(|(index, p)| Some((world.entity_at_index(index)?, p.clone())))
            .collect();
        assert_eq!(parcels.len(), 1);
        let (parcel, component) = &parcels[0];
        assert_eq!(component.boundary.len(), 4);

        let lp = parcel_loop(&world, component).expect("loop");
        assert_eq!(lp.len(), 4);
        // Front-left default vertex lands at parcel origin + (−4, 0, 0);
        // the whole loop inherits the parcel's height (full 3D, no ground
        // plane).
        assert!((lp[0] - Vec3::new(6.0, 2.0, 5.0)).norm() < 1e-4);
        for v in &component.boundary {
            assert_eq!(
                world.get::<redlilium_ecs::Parent>(*v).unwrap().0,
                *parcel,
                "vertices are children of the parcel"
            );
        }

        // Dragging a vertex reshapes the loop (order stays authored, no
        // re-sorting — concave shapes must survive).
        let v2 = component.boundary[2];
        let t = Transform::new(
            Vec3::new(1.0, 0.0, -2.0),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(v2, t).unwrap();
        world
            .insert(
                v2,
                GlobalTransform(world.get::<GlobalTransform>(*parcel).unwrap().0 * t.to_matrix()),
            )
            .unwrap();
        let reshaped = parcel_loop(&world, component).expect("loop");
        assert!((reshaped[2] - Vec3::new(11.0, 2.0, 3.0)).norm() < 1e-4);

        action.undo(&mut world).unwrap();
        assert!(world.read_all::<Parcel>().unwrap().iter().next().is_none());
        assert!(
            world
                .read_all::<ParcelVertex>()
                .unwrap()
                .iter()
                .next()
                .is_none()
        );
    }

    #[test]
    fn handles_curve_segments_straight_by_default() {
        let mut world = world();
        let mut action = AddParcelAction::at_point(Vec3::zeros());
        action.apply(&mut world).unwrap();
        let component: Parcel = world
            .read_all::<Parcel>()
            .unwrap()
            .iter()
            .map(|(_, p)| p.clone())
            .next()
            .unwrap();

        // All handles zero → pure polyline: exactly one point per vertex.
        assert_eq!(parcel_loop(&world, &component).unwrap().len(), 4);

        // Give the front edge (v0 → v1, from (−4,0,0) to (4,0,0)) a bulge:
        // mirrored-style handles pushing toward +Z.
        let v0 = component.boundary[0];
        let v1 = component.boundary[1];
        world
            .insert(
                v0,
                ParcelVertex {
                    handle_out: Vec3::new(2.0, 0.0, 2.0),
                    handle_in: Vec3::zeros(),
                },
            )
            .unwrap();
        world
            .insert(
                v1,
                ParcelVertex {
                    handle_in: Vec3::new(-2.0, 0.0, 2.0),
                    handle_out: Vec3::zeros(),
                },
            )
            .unwrap();

        let lp = parcel_loop(&world, &component).unwrap();
        // One curved segment → CURVE_STEPS−1 extra samples.
        assert_eq!(lp.len(), 4 + (CURVE_STEPS - 1));
        // The curve bulges toward +Z at its middle (cubic midpoint z =
        // 3/4 · 2 = 1.5), while the straight segments stay put.
        let mid = lp[CURVE_STEPS / 2];
        assert!((mid.z - 1.5).abs() < 1e-3, "bulged to z=1.5, got {}", mid.z);
        assert!(mid.x.abs() < 1e-3);
    }

    #[test]
    fn mirrored_handles_make_a_c1_joint_and_rotation_steers_the_curve() {
        let mut world = world();
        let mut action = AddParcelAction::at_point(Vec3::zeros());
        action.apply(&mut world).unwrap();
        let component: Parcel = world
            .read_all::<Parcel>()
            .unwrap()
            .iter()
            .map(|(_, p)| p.clone())
            .next()
            .unwrap();

        // Curve both segments around v1 with mirrored handles: C1 — the
        // arriving and departing tangents at v1 are collinear.
        let v0 = component.boundary[0];
        let v1 = component.boundary[1];
        let v2 = component.boundary[2];
        let h = Vec3::new(0.0, 0.0, -2.5);
        world
            .insert(
                v1,
                ParcelVertex {
                    handle_in: -h,
                    handle_out: h,
                },
            )
            .unwrap();
        // Far ends stay straight (zero handles on v0.out / v2.in sides are
        // already the default).
        let _ = (v0, v2);

        let lp = parcel_loop(&world, &component).unwrap();
        // Both segments around v1 curved → two fans; v1 itself is a sample.
        let idx_v1 = CURVE_STEPS; // v0 + (CURVE_STEPS−1) samples, then v1
        assert!((lp[idx_v1] - Vec3::new(4.0, 0.0, 0.0)).norm() < 1e-4);
        // Tangent continuity at the vertex, measured analytically: the
        // arriving Bézier tangent is p − handle_in, the departing one is
        // handle_out − p — mirrored handles make them identical.
        let corners = parcel_corners(&world, &component).unwrap();
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
        let corners = parcel_corners(&world, &component).unwrap();
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

    fn spawned_parcel(world: &World) -> (Entity, Parcel) {
        world
            .read_all::<Parcel>()
            .unwrap()
            .iter()
            .filter_map(|(index, p)| Some((world.entity_at_index(index)?, p.clone())))
            .next()
            .unwrap()
    }

    #[test]
    fn anchored_parcel_dictates_the_interval_and_follows_frontage_edits() {
        let (mut world, road) = world_with_road();
        AddParcelAction::on_edge(crate::anchor::EdgeAnchor {
            parent_road: road,
            right_edge: true,
            u_min: 0.5,
            u_max: 0.5,
        })
        .apply(&mut world)
        .unwrap();
        let (parcel, component) = spawned_parcel(&world);

        // The default frontage is 8 m; on a 20 m straight edge that is a
        // derived u half-width of 0.2 around the authored center 0.5.
        let anchor = world
            .get::<crate::anchor::EdgeAnchor>(parcel)
            .unwrap()
            .clone();
        assert!((anchor.u_min - 0.3).abs() < 1e-3, "got {}", anchor.u_min);
        assert!((anchor.u_max - 0.7).abs() < 1e-3, "got {}", anchor.u_max);
        // Sits on the right edge (x = +3), frontage FACING the road: +Z
        // points into it (−X), interior extends outward (+X).
        let t = *world.get::<Transform>(parcel).unwrap();
        assert!((t.translation - Vec3::new(3.0, 0.0, 10.0)).norm() < 1e-3);
        let heading = crate::bezier::heading(&t.to_matrix());
        assert!((heading - Vec3::new(-1.0, 0.0, 0.0)).norm() < 1e-3);
        // Ships settled.
        assert!(crate::anchor::anchor_updates(&world, &mut Default::default(), false).is_empty());

        // The parcel dictates: widen the frontage (move the second vertex
        // out) and the edge interval follows, center preserved.
        let v1 = component.boundary[1];
        let wider = Transform::new(
            Vec3::new(6.0, 0.0, 0.0),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(v1, wider).unwrap();
        crate::settle_edge_anchors(&mut world);
        let anchor = world
            .get::<crate::anchor::EdgeAnchor>(parcel)
            .unwrap()
            .clone();
        // Frontage 10 m → half-width 0.25.
        assert!((anchor.u_min - 0.25).abs() < 1e-3, "got {}", anchor.u_min);
        assert!((anchor.u_max - 0.75).abs() < 1e-3, "got {}", anchor.u_max);
    }

    #[test]
    fn anchored_parcel_slides_keeping_its_derived_width() {
        let (mut world, road) = world_with_road();
        AddParcelAction::on_edge(crate::anchor::EdgeAnchor {
            parent_road: road,
            right_edge: true,
            u_min: 0.5,
            u_max: 0.5,
        })
        .apply(&mut world)
        .unwrap();
        let (parcel, _) = spawned_parcel(&world);

        // Prime the settled cache, then drag the parcel up the road.
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
        let home = *world.get::<Transform>(parcel).unwrap();

        let dragged = Transform::new(
            home.translation + Vec3::new(1.0, 0.0, 4.0),
            home.rotation,
            home.scale,
        );
        world.insert(parcel, dragged).unwrap();
        world
            .insert(parcel, GlobalTransform(dragged.to_matrix()))
            .unwrap();
        settle(&mut world, &mut cache);

        let anchor = world
            .get::<crate::anchor::EdgeAnchor>(parcel)
            .unwrap()
            .clone();
        let center = (anchor.u_min + anchor.u_max) * 0.5;
        assert!((center - 0.7).abs() < 1e-2, "slid to u≈0.7, got {center}");
        assert!(
            ((anchor.u_max - anchor.u_min) - 0.4).abs() < 1e-3,
            "width stays frontage-derived"
        );
        // Snapped back onto the edge.
        let t = *world.get::<Transform>(parcel).unwrap();
        assert!((t.translation.x - 3.0).abs() < 1e-3);
    }

    #[test]
    fn gate_spot_and_add_gate_roundtrip() {
        let mut world = world();
        AddParcelAction::at_point(Vec3::new(10.0, 0.0, 5.0))
            .apply(&mut world)
            .unwrap();
        let (parcel, _) = spawned_parcel(&world);
        world.register_inspector_default::<crate::RoadNode>();

        // Ray straight down at the middle of the front edge (world
        // (10, 0, 5); front edge spans x 6..14 at z = 5).
        let ray = redlilium_ecs::ui::ViewportRay {
            origin: Vec3::new(10.0, 10.0, 5.0),
            dir: Vec3::new(0.0, -1.0, 0.0),
        };
        let local = gate_spot(&world, parcel, &ray).expect("spot on the boundary");
        assert!(local.translation.norm() < 1e-3, "front-edge midpoint");
        // Outward of the front edge is +Z (interior extends to −Z).
        let out = crate::bezier::heading(&local.to_matrix());
        assert!((out - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-3);

        let mut action = AddGateAction::new(parcel, local);
        action.apply(&mut world).unwrap();
        let gate = world
            .read_all::<ParcelGate>()
            .unwrap()
            .iter()
            .filter_map(|(index, _)| world.entity_at_index(index))
            .next()
            .expect("gate spawned");
        assert!(world.get::<crate::RoadNode>(gate).is_some());
        assert_eq!(world.get::<redlilium_ecs::Parent>(gate).unwrap().0, parcel);
        let gt = world.get::<GlobalTransform>(gate).unwrap().0;
        let pos = Vec3::new(gt[(0, 3)], gt[(1, 3)], gt[(2, 3)]);
        assert!((pos - Vec3::new(10.0, 0.0, 5.0)).norm() < 1e-3);

        action.undo(&mut world).unwrap();
        assert!(!world.is_alive(gate));
    }

    #[test]
    fn parcel_prefab_duplicates_with_all_content_remapped() {
        let mut world = world();
        world.register_inspector_default::<crate::RoadNode>();
        world.register_inspector_default::<crate::RoadSegment>();
        world.register_inspector_default::<crate::building::Building>();

        // Parcel with a gate, a building, and an internal road from an
        // inner node to the gate (meeting it from behind).
        AddParcelAction::at_point(Vec3::new(0.0, 0.0, 0.0))
            .apply(&mut world)
            .unwrap();
        let (parcel, _) = spawned_parcel(&world);
        let gate_local = Transform::new(
            Vec3::zeros(),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        let mut add_gate = AddGateAction::new(parcel, gate_local);
        add_gate.apply(&mut world).unwrap();
        let gate = world
            .read_all::<ParcelGate>()
            .unwrap()
            .iter()
            .filter_map(|(index, _)| world.entity_at_index(index))
            .next()
            .unwrap();
        crate::building::PlaceBuildingAction::new(
            parcel,
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
        set_parent(&mut world, inner, parcel);
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
        set_parent(&mut world, internal_road, parcel);

        // Duplicate the whole thing 30 m to the east.
        let mut dup = DuplicateParcelAction::new(parcel, Vec3::new(30.0, 0.0, 0.0));
        dup.apply(&mut world).unwrap();

        let parcels: Vec<(Entity, Parcel)> = world
            .read_all::<Parcel>()
            .unwrap()
            .iter()
            .filter_map(|(index, p)| Some((world.entity_at_index(index)?, p.clone())))
            .collect();
        assert_eq!(parcels.len(), 2);
        let (copy, copy_parcel) = parcels
            .iter()
            .find(|(e, _)| *e != parcel)
            .expect("the duplicate");

        // Boundary references remapped: the copy points at its own fresh
        // vertices, none shared with the source.
        let source_boundary: std::collections::HashSet<Entity> = parcels
            .iter()
            .find(|(e, _)| *e == parcel)
            .unwrap()
            .1
            .boundary
            .iter()
            .copied()
            .collect();
        assert_eq!(copy_parcel.boundary.len(), 4);
        for v in &copy_parcel.boundary {
            assert!(!source_boundary.contains(v), "vertex remapped, not shared");
            assert_eq!(world.get::<redlilium_ecs::Parent>(*v).unwrap().0, *copy);
        }
        // The copied loop is the source loop shifted by (+30, 0, 0).
        let src_loop = parcel_loop(&world, &parcels[0].1).unwrap();
        let copy_loop = parcel_loop(&world, copy_parcel).unwrap();
        assert_eq!(src_loop.len(), copy_loop.len());
        for (a, b) in src_loop.iter().zip(&copy_loop) {
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
        assert_eq!(world.read_all::<Parcel>().unwrap().iter().count(), 1);
        assert!(world.is_alive(parcel) && world.is_alive(gate) && world.is_alive(inner));
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
    fn loop_needs_three_live_vertices() {
        let mut world = world();
        let mut action = AddParcelAction::at_point(Vec3::zeros());
        action.apply(&mut world).unwrap();
        let component: Parcel = world
            .read_all::<Parcel>()
            .unwrap()
            .iter()
            .map(|(_, p)| p.clone())
            .next()
            .unwrap();
        world.despawn(component.boundary[0]);
        assert_eq!(parcel_loop(&world, &component).unwrap().len(), 3);
        world.despawn(component.boundary[1]);
        assert!(parcel_loop(&world, &component).is_none());
    }
}
