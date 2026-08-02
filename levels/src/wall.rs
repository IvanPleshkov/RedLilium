//! Interior walls and derived floor faces (building slice 2).
//!
//! Walls are **centerlines**, rooms are **never authored**: the planar
//! faces of the wall graph (bounded by the building envelope) are
//! derived, the same move as terrain regions between roads. A border
//! shared by two rooms is ONE wall — the plots lesson. A wall's vertical
//! span attaches to [`Datum`](crate::building::Datum) entities
//! (`None` = unbounded on that side), so the same floor plan can hold a
//! full-height hall next to storey-high partitions; the face pass at an
//! elevation only sees the walls that span it. Cells, portals and the
//! merge of unseparated volumes arrive in later slices — this one is
//! flat: faces at a single elevation.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::{Vec3, Vec4, quat_from_rotation_y};
use redlilium_ecs::{
    Component, Entity, GlobalTransform, Transform, World, remove_parent, set_parent,
};

use crate::building::{Building, Datum};
use crate::stroke::StrokeVertex;

/// An interior wall centerline: an open polyline of child
/// [`StrokeVertex`] entities (pen model — curved walls come free), in
/// the wall entity's local space; the wall itself is a child of the
/// building root and is listed in [`Building::walls`].
#[derive(Debug, Clone, Default, Component)]
pub struct Wall {
    /// Path vertices in order (explicit — never re-derived). Dead or
    /// non-vertex references are skipped; fewer than 2 live vertices
    /// means no wall.
    pub points: Vec<Entity>,
    /// Datum the wall's foot attaches to; `None` reaches the building's
    /// base — the wall exists on every level below `top`.
    pub bottom: Option<Entity>,
    /// Datum the wall's head attaches to; `None` reaches the envelope
    /// top.
    pub top: Option<Entity>,
}

/// The wall's corners in **building-local** space: the wall's own
/// transform composed with each vertex's (local transforms only — like
/// gate derivation, moving the building root must not reshape the plan).
pub(crate) fn wall_corners_local(
    world: &World,
    wall_entity: Entity,
) -> Option<crate::stroke::Corners> {
    let wall = world.get::<Wall>(wall_entity)?;
    let wall_m = world.get::<Transform>(wall_entity)?.to_matrix();
    let mut corners = crate::stroke::Corners::new();
    for &v in &wall.points {
        let Some(vertex) = world.get::<StrokeVertex>(v) else {
            continue;
        };
        let Some(t) = world.get::<Transform>(v) else {
            continue;
        };
        let m = wall_m * t.to_matrix();
        let point = |local: Vec3| {
            let p = m * Vec4::new(local.x, local.y, local.z, 1.0);
            Vec3::new(p.x, p.y, p.z)
        };
        corners.push((
            point(Vec3::zeros()),
            point(vertex.handle_out),
            point(vertex.handle_in),
        ));
    }
    (corners.len() >= 2).then_some(corners)
}

/// The wall's vertical span as elevations, unbounded sides resolved to
/// infinities. Used by the face pass; drawing clamps to the ladder.
pub(crate) fn wall_span(world: &World, wall: &Wall) -> (f32, f32) {
    let resolve = |r: Option<Entity>| r.and_then(|d| world.get::<Datum>(d).map(|d| d.elevation));
    (
        resolve(wall.bottom).unwrap_or(f32::NEG_INFINITY),
        resolve(wall.top).unwrap_or(f32::INFINITY),
    )
}

// ---------------------------------------------------------------------------
// Derived floor faces
// ---------------------------------------------------------------------------

/// Weld/snap tolerance of the planar arrangement, meters (1 mm).
const WELD_EPS: f64 = 1e-3;
/// Faces below this area are numerical slivers, m².
const AREA_EPS: f64 = 1e-4;

/// The derived rooms of one building floor: the planar faces of the wall
/// graph at `elevation`, bounded by the envelope. Each face is a CCW
/// loop of building-local points at that elevation (tessellated — curved
/// walls contribute their fans). Walls participate when their span
/// covers the elevation; dangling wall ends split nothing (the face walk
/// traverses them twice); a wall overhanging the envelope splits nothing
/// outside (the outer face is unbounded and dropped). Collinear
/// overlapping walls are unsupported (an authoring degenerate).
pub fn floor_faces(world: &World, building: &Building, elevation: f32) -> Vec<Vec<Vec3>> {
    let Some(ring) = crate::building::envelope_ring_local(world, building) else {
        return Vec::new();
    };
    let mut polylines: Vec<Vec<[f64; 2]>> = Vec::new();
    polylines.push(ring.iter().map(|p| [p.x as f64, p.z as f64]).collect());
    for &w in &building.walls {
        let Some(wall) = world.get::<Wall>(w) else {
            continue;
        };
        let (bottom, top) = wall_span(world, wall);
        if bottom > elevation + 0.01 || top <= elevation + 0.01 {
            continue;
        }
        let Some(corners) = wall_corners_local(world, w) else {
            continue;
        };
        polylines.push(
            crate::stroke::tessellate(&corners)
                .into_iter()
                .map(|(_, _, p)| [p.x as f64, p.z as f64])
                .collect(),
        );
    }
    planar_faces(&polylines)
        .into_iter()
        .map(|face| {
            face.into_iter()
                .map(|[x, z]| Vec3::new(x as f32, elevation, z as f32))
                .collect()
        })
        .collect()
}

/// The bounded faces of a set of polylines as CCW loops — a planar
/// arrangement: segments split at crossings and T-junctions, endpoints
/// welded within [`WELD_EPS`], faces walked by the clockwise-most-turn
/// rule. The unbounded outer face (negative area) is dropped.
fn planar_faces(polylines: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    let sub = |a: [f64; 2], b: [f64; 2]| [a[0] - b[0], a[1] - b[1]];
    let cross = |a: [f64; 2], b: [f64; 2]| a[0] * b[1] - a[1] * b[0];
    let len = |a: [f64; 2]| (a[0] * a[0] + a[1] * a[1]).sqrt();

    // Segment soup.
    let mut segments: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for line in polylines {
        for pair in line.windows(2) {
            if len(sub(pair[1], pair[0])) > WELD_EPS {
                segments.push((pair[0], pair[1]));
            }
        }
    }

    // Split every segment at crossings and at other segments' endpoints
    // landing on it (T-junctions).
    let mut pieces: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for (i, &(a, b)) in segments.iter().enumerate() {
        let d = sub(b, a);
        let seg_len = len(d);
        let t_eps = WELD_EPS / seg_len;
        let mut ts: Vec<f64> = vec![0.0, 1.0];
        for (j, &(c, e)) in segments.iter().enumerate() {
            if i == j {
                continue;
            }
            let f = sub(e, c);
            let denom = cross(d, f);
            if denom.abs() > 1e-12 {
                let t = cross(sub(c, a), f) / denom;
                let u = cross(sub(c, a), d) / denom;
                let u_eps = WELD_EPS / len(f);
                if t > t_eps && t < 1.0 - t_eps && u > -u_eps && u < 1.0 + u_eps {
                    ts.push(t);
                }
            }
            // Endpoints of j projected onto i (covers T-junctions on
            // near-parallel meets the crossing test misses).
            for p in [c, e] {
                let t = (sub(p, a)[0] * d[0] + sub(p, a)[1] * d[1]) / (seg_len * seg_len);
                if t > t_eps && t < 1.0 - t_eps {
                    let on = [a[0] + d[0] * t, a[1] + d[1] * t];
                    if len(sub(p, on)) < WELD_EPS {
                        ts.push(t);
                    }
                }
            }
        }
        ts.sort_by(f64::total_cmp);
        let at = |t: f64| [a[0] + d[0] * t, a[1] + d[1] * t];
        for pair in ts.windows(2) {
            if pair[1] - pair[0] > t_eps {
                pieces.push((at(pair[0]), at(pair[1])));
            }
        }
    }

    // Weld endpoints into nodes (first-fit clustering — plans are small).
    let mut nodes: Vec<[f64; 2]> = Vec::new();
    let node_of = |p: [f64; 2], nodes: &mut Vec<[f64; 2]>| -> usize {
        for (i, n) in nodes.iter().enumerate() {
            if len(sub(p, *n)) < WELD_EPS {
                return i;
            }
        }
        nodes.push(p);
        nodes.len() - 1
    };
    let mut edges: std::collections::BTreeSet<(usize, usize)> = Default::default();
    for &(a, b) in &pieces {
        let (na, nb) = (node_of(a, &mut nodes), node_of(b, &mut nodes));
        if na != nb {
            edges.insert((na.min(nb), na.max(nb)));
        }
    }

    // Angular adjacency.
    let mut adjacency: Vec<Vec<(f64, usize)>> = vec![Vec::new(); nodes.len()];
    for &(a, b) in &edges {
        let d = sub(nodes[b], nodes[a]);
        adjacency[a].push((d[1].atan2(d[0]), b));
        adjacency[b].push(((-d[1]).atan2(-d[0]), a));
    }
    for list in &mut adjacency {
        list.sort_by(|x, y| x.0.total_cmp(&y.0));
    }

    // Face walk: from (u -> v), continue with v's clockwise-most edge
    // from the reverse direction — bounded faces come out CCW; a dead
    // end walks straight back (dangling edges split nothing).
    let mut visited: std::collections::HashSet<(usize, usize)> = Default::default();
    let mut faces = Vec::new();
    let starts: Vec<(usize, usize)> = edges.iter().flat_map(|&(a, b)| [(a, b), (b, a)]).collect();
    for start in starts {
        if visited.contains(&start) {
            continue;
        }
        let mut face_nodes = Vec::new();
        let mut cur = start;
        loop {
            visited.insert(cur);
            face_nodes.push(cur.0);
            let (u, v) = cur;
            let back = sub(nodes[u], nodes[v]);
            let back_angle = back[1].atan2(back[0]);
            let list = &adjacency[v];
            // Cyclically previous neighbor before the reverse edge.
            let next = list
                .iter()
                .rev()
                .find(|(angle, _)| *angle < back_angle - 1e-12)
                .or_else(|| list.last())
                .expect("a walked-into node has at least the reverse edge");
            cur = (v, next.1);
            if cur == start {
                break;
            }
        }
        let area = face_nodes
            .windows(2)
            .map(|w| cross(nodes[w[0]], nodes[w[1]]))
            .sum::<f64>()
            + cross(nodes[*face_nodes.last().unwrap()], nodes[face_nodes[0]]);
        if area / 2.0 > AREA_EPS {
            faces.push(face_nodes.iter().map(|&n| nodes[n]).collect());
        }
    }
    faces
}

// ---------------------------------------------------------------------------
// Edit action
// ---------------------------------------------------------------------------

/// Default wall: 4 m straight run along local X — short of the default
/// envelope on purpose (a dangling wall splits nothing until its ends
/// are dragged onto the envelope or another wall).
const DEFAULT_WALL: [[f32; 2]; 2] = [[-2.0, 0.0], [2.0, 0.0]];

/// Undoable "add a wall to a building": spawns the wall entity with its
/// vertex children at a building-local point, registers it in
/// [`Building::walls`]. Full height by default (both attachments
/// `None`); narrow the span in the inspector.
#[derive(Debug)]
pub struct AddWallAction {
    building: Entity,
    local: Vec3,
    created: Vec<Entity>,
    registered: bool,
}

impl AddWallAction {
    pub fn at_point(building: Entity, local: Vec3) -> Self {
        Self {
            building,
            local,
            created: Vec::new(),
            registered: false,
        }
    }
}

impl EditAction<World> for AddWallAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let Some(mut component) = world.get::<Building>(self.building).cloned() else {
            return Err(EditActionError::TargetNotFound(
                "wall target is not a building".into(),
            ));
        };
        let root_m = world
            .get::<GlobalTransform>(self.building)
            .map(|gt| gt.0)
            .ok_or_else(|| EditActionError::TargetNotFound("building has no transform".into()))?;
        let undo_partial = |world: &mut World, created: &mut Vec<Entity>| {
            for e in created.drain(..).rev() {
                remove_parent(world, e);
                world.despawn(e);
            }
        };

        let wall_t = Transform::new(
            self.local,
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        let wall = world.spawn();
        self.created.push(wall);
        let inserted = world
            .insert(wall, wall_t)
            .and_then(|_| world.insert(wall, GlobalTransform(root_m * wall_t.to_matrix())))
            .and_then(|_| world.insert(wall, redlilium_ecs::Name("Wall".to_owned())));
        if let Err(e) = inserted {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }
        set_parent(world, wall, self.building);

        let mut points = Vec::with_capacity(DEFAULT_WALL.len());
        for (n, [x, z]) in DEFAULT_WALL.into_iter().enumerate() {
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
                        GlobalTransform(root_m * wall_t.to_matrix() * local.to_matrix()),
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
            set_parent(world, vertex, wall);
            points.push(vertex);
        }

        if let Err(e) = world.insert(
            wall,
            Wall {
                points,
                bottom: None,
                top: None,
            },
        ) {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }
        component.walls.push(wall);
        if let Err(e) = world.insert(self.building, component) {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }
        self.registered = true;
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if self.registered
            && let Some(mut component) = world.get::<Building>(self.building).cloned()
        {
            let wall = self.created.first().copied();
            component.walls.retain(|&w| Some(w) != wall);
            let _ = world.insert(self.building, component);
            self.registered = false;
        }
        for e in self.created.drain(..).rev() {
            remove_parent(world, e);
            world.despawn(e);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Add wall"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::PlaceBuildingAction;

    fn world_with_building() -> (World, Entity, Building) {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<Building>();
        world.register_inspector_default::<Datum>();
        world.register_inspector_default::<StrokeVertex>();
        world.register_inspector_default::<Wall>();
        PlaceBuildingAction::new(None, Transform::default())
            .apply(&mut world)
            .unwrap();
        let (building, component) = world
            .read_all::<Building>()
            .unwrap()
            .iter()
            .filter_map(|(index, b)| Some((world.entity_at_index(index)?, b.clone())))
            .next()
            .unwrap();
        (world, building, component)
    }

    fn area(face: &[Vec3]) -> f32 {
        let mut sum = 0.0;
        for i in 0..face.len() {
            let a = face[i];
            let b = face[(i + 1) % face.len()];
            sum += a.x * b.z - b.x * a.z;
        }
        // Faces are CCW in XZ where the shoelace in (x, z) runs negative
        // (the XZ plane is left-handed seen from +Y); report magnitude.
        (sum / 2.0).abs()
    }

    fn stretch_to(world: &mut World, wall_points: &[Entity], ends: [[f32; 3]; 2]) {
        for (&v, p) in wall_points.iter().zip(ends) {
            world
                .insert(
                    v,
                    Transform::new(
                        Vec3::new(p[0], p[1], p[2]),
                        redlilium_core::math::quat_from_rotation_y(0.0),
                        Vec3::new(1.0, 1.0, 1.0),
                    ),
                )
                .unwrap();
        }
    }

    #[test]
    fn envelope_alone_is_one_room_and_dangling_walls_split_nothing() {
        let (mut world, building, component) = world_with_building();
        let faces = floor_faces(&world, &component, 0.0);
        assert_eq!(faces.len(), 1, "the envelope bounds a single face");
        assert!(
            (area(&faces[0]) - 36.0).abs() < 0.05,
            "6×6 default envelope"
        );

        // The default wall is 4 m in a 6 m envelope: both ends dangle,
        // nothing splits.
        AddWallAction::at_point(building, Vec3::zeros())
            .apply(&mut world)
            .unwrap();
        let component = world.get::<Building>(building).unwrap().clone();
        assert_eq!(component.walls.len(), 1);
        let faces = floor_faces(&world, &component, 0.0);
        assert_eq!(faces.len(), 1, "a dangling wall splits nothing");
        assert!((area(&faces[0]) - 36.0).abs() < 0.05);
    }

    #[test]
    fn a_crossing_wall_splits_the_floor_and_a_t_makes_three() {
        let (mut world, building, _) = world_with_building();
        AddWallAction::at_point(building, Vec3::zeros())
            .apply(&mut world)
            .unwrap();
        let component = world.get::<Building>(building).unwrap().clone();
        let wall = component.walls[0];
        let points = world.get::<Wall>(wall).unwrap().points.clone();
        // Stretch the wall onto the envelope: a full west-east crossing.
        stretch_to(&mut world, &points, [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        let faces = floor_faces(&world, &component, 0.0);
        assert_eq!(faces.len(), 2, "a crossing wall makes two rooms");
        let mut areas: Vec<f32> = faces.iter().map(|f| area(f)).collect();
        areas.sort_by(f32::total_cmp);
        assert!((areas[0] - 18.0).abs() < 0.05 && (areas[1] - 18.0).abs() < 0.05);

        // A second wall T-ing into the first from the south edge.
        AddWallAction::at_point(building, Vec3::zeros())
            .apply(&mut world)
            .unwrap();
        let component = world.get::<Building>(building).unwrap().clone();
        let second = component.walls[1];
        let points = world.get::<Wall>(second).unwrap().points.clone();
        stretch_to(&mut world, &points, [[0.0, 0.0, 0.0], [0.0, 0.0, -3.0]]);
        let faces = floor_faces(&world, &component, 0.0);
        assert_eq!(faces.len(), 3, "the T splits the south half");
        let mut areas: Vec<f32> = faces.iter().map(|f| area(f)).collect();
        areas.sort_by(f32::total_cmp);
        assert!((areas[0] - 9.0).abs() < 0.05, "got {areas:?}");
        assert!((areas[1] - 9.0).abs() < 0.05);
        assert!((areas[2] - 18.0).abs() < 0.05);
    }

    #[test]
    fn wall_spans_gate_the_face_pass_by_elevation() {
        let (mut world, building, _) = world_with_building();
        AddWallAction::at_point(building, Vec3::zeros())
            .apply(&mut world)
            .unwrap();
        let component = world.get::<Building>(building).unwrap().clone();
        let wall_e = component.walls[0];
        let points = world.get::<Wall>(wall_e).unwrap().points.clone();
        stretch_to(&mut world, &points, [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);

        // Attach the wall's foot to the +3.000 datum: a gallery partition
        // that exists upstairs only.
        let upstairs = component.datums[1];
        let mut wall = world.get::<Wall>(wall_e).unwrap().clone();
        wall.bottom = Some(upstairs);
        world.insert(wall_e, wall).unwrap();

        assert_eq!(
            floor_faces(&world, &component, 0.0).len(),
            1,
            "the ground floor stays open"
        );
        assert_eq!(
            floor_faces(&world, &component, 3.0).len(),
            2,
            "upstairs the partition splits"
        );
    }

    #[test]
    fn curved_walls_split_with_their_tessellation() {
        let (mut world, building, _) = world_with_building();
        AddWallAction::at_point(building, Vec3::zeros())
            .apply(&mut world)
            .unwrap();
        let component = world.get::<Building>(building).unwrap().clone();
        let points = world
            .get::<Wall>(component.walls[0])
            .unwrap()
            .points
            .clone();
        stretch_to(&mut world, &points, [[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        // Bow the wall north with mirrored C1 handles.
        for (&v, out) in points.iter().zip([[1.5f32, 0.0, 1.0], [1.5, 0.0, -1.0]]) {
            world
                .insert(
                    v,
                    StrokeVertex {
                        handle_out: Vec3::new(out[0], out[1], out[2]),
                        handle_in: Vec3::new(-out[0], -out[1], -out[2]),
                    },
                )
                .unwrap();
        }
        let faces = floor_faces(&world, &component, 0.0);
        assert_eq!(faces.len(), 2);
        let total: f32 = faces.iter().map(|f| area(f)).sum();
        assert!(
            (total - 36.0).abs() < 0.1,
            "areas partition the floor, got {total}"
        );
        let mut areas: Vec<f32> = faces.iter().map(|f| area(f)).collect();
        areas.sort_by(f32::total_cmp);
        assert!(
            areas[0] < 17.0 && areas[1] > 19.0,
            "the bow moves area across the split, got {areas:?}"
        );
    }

    #[test]
    fn add_wall_undo_restores_the_registry_and_subtree() {
        let (mut world, building, _) = world_with_building();
        let mut action = AddWallAction::at_point(building, Vec3::new(1.0, 0.0, 1.0));
        action.apply(&mut world).unwrap();
        let component = world.get::<Building>(building).unwrap().clone();
        assert_eq!(component.walls.len(), 1);
        let wall = component.walls[0];
        assert_eq!(
            world.get::<redlilium_ecs::Parent>(wall).unwrap().0,
            building
        );
        let points = world.get::<Wall>(wall).unwrap().points.clone();
        assert_eq!(points.len(), 2);
        for &v in &points {
            assert_eq!(world.get::<redlilium_ecs::Parent>(v).unwrap().0, wall);
        }

        action.undo(&mut world).unwrap();
        assert!(!world.is_alive(wall));
        assert!(points.iter().all(|&v| !world.is_alive(v)));
        assert!(
            world.get::<Building>(building).unwrap().walls.is_empty(),
            "the registry entry undoes with the wall"
        );
    }
}
