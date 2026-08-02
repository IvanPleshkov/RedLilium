//! Buildings: walls + plates + datum marks; spaces are derived.
//!
//! A building is **not a stack of storeys** — multi-storey voids (a
//! factory hall, an atrium) are first-class, so the floor-by-floor
//! partition is a derived special case, never the structure (design
//! 2026-07-24, see `docs/DESIGN_PROCEDURAL_LEVELS.md` §3). The authored
//! model, slice by slice:
//!
//! - **Envelope** (this slice): a closed contour of child vertices on the
//!   same pen-model machinery as strokes/cuts — curved walls come free.
//!   With the interior opted out (no walls, no plates) the building is
//!   just envelope × height: the successor of the old box massing.
//! - **Datum ladder** (this slice): [`Datum`] child entities — named
//!   elevations that are *guides*, not structure. Later slices attach
//!   plates and wall spans to datum entities so dragging one elevation
//!   reflows everything attached; nothing is required to sit on a datum.
//!   The topmost datum is the envelope height.
//! - **Walls, plates, portals, facades, basements** (later slices): wall
//!   centerlines whose planar faces derive rooms; plates that cut a
//!   face's vertical column into cells (cells separated by neither wall
//!   nor plate merge — the hall stays ONE tall space); Gate-style
//!   portals on walls/plates/envelope; facade strips with rhythm from
//!   the assembly-graph asset; excavation contracts through `Cut`.
//!
//! A building is an ordinary entity — nothing owns it structurally;
//! grouping is plain hierarchy (a villa = fences + buildings + a driveway
//! under one root; see `stroke::DuplicateSubtreeAction`). Connections to
//! the road network are gates on strokes or edge anchors — never
//! building fields.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::{Vec3, Vec4, quat_from_rotation_y};
use redlilium_ecs::{
    Component, Entity, GlobalTransform, Transform, World, remove_parent, set_parent,
};

use crate::stroke::StrokeVertex;

/// The building root: a closed envelope contour plus the datum ladder.
#[derive(Debug, Clone, Default, Component)]
pub struct Building {
    /// Closed envelope contour: child vertex entities ([`StrokeVertex`])
    /// in perimeter order, local XZ; the loop closes last → first
    /// implicitly. Dead or non-vertex references are skipped at
    /// evaluation; fewer than 3 live vertices means no envelope.
    pub points: Vec<Entity>,
    /// The datum ladder: child [`Datum`] entities. Order carries no
    /// meaning — elevations do.
    pub datums: Vec<Entity>,
    /// Interior wall children ([`Wall`](crate::wall::Wall)) — the floor
    /// plan whose planar faces derive the rooms (slice 2).
    pub walls: Vec<Entity>,
    /// Generator seed — same seed + same params ⇒ same building (P3).
    pub seed: u32,
}

/// One mark of a building's datum ladder: a named elevation in meters
/// above the building's origin plane (its `Name` is the label). A guide,
/// not structure — later slices attach plates and wall spans to a datum
/// entity, so editing one elevation reflows everything attached. A child
/// of the building root, deliberately without a transform: the elevation
/// field is the single source of truth.
#[derive(Debug, Clone, Default, Component)]
pub struct Datum {
    /// Height above the building origin plane, meters. Negative marks
    /// (basements) arrive in a later slice.
    pub elevation: f32,
}

/// Default envelope: a 6×6 m rectangle around the origin, perimeter
/// order (matches the old box's `half_width = half_depth = 3`).
const DEFAULT_ENVELOPE: [[f32; 2]; 4] = [[-3.0, 3.0], [3.0, 3.0], [3.0, -3.0], [-3.0, -3.0]];

/// Default datum ladder: ground plus two storey marks (the old default
/// box was 2 × 3 m tall).
const DEFAULT_LADDER: [f32; 3] = [0.0, 3.0, 6.0];

/// The envelope corners in **building-local** space (from the vertices'
/// local transforms — like a stroke's gate derivation, moving the root
/// must not reshape anything). `None` below 3 live vertices: a closed
/// contour needs a polygon, not a segment.
fn envelope_corners_local(world: &World, building: &Building) -> Option<crate::stroke::Corners> {
    let mut corners = crate::stroke::Corners::new();
    for &v in &building.points {
        let Some(vertex) = world.get::<StrokeVertex>(v) else {
            continue;
        };
        let Some(t) = world.get::<Transform>(v) else {
            continue;
        };
        let m = t.to_matrix();
        let point = |local: Vec3| {
            let p = m * Vec4::new(local.x, local.y, local.z, 1.0);
            Vec3::new(p.x, p.y, p.z)
        };
        corners.push((
            t.translation,
            point(vertex.handle_out),
            point(vertex.handle_in),
        ));
    }
    (corners.len() >= 3).then_some(corners)
}

/// The closed envelope ring in building-local space at the origin plane,
/// tessellated (curved segments fan out; the last sample returns to the
/// first vertex). `None` below 3 live vertices.
pub(crate) fn envelope_ring_local(world: &World, building: &Building) -> Option<Vec<Vec3>> {
    let mut corners = envelope_corners_local(world, building)?;
    corners.push(corners[0]);
    Some(
        crate::stroke::tessellate(&corners)
            .into_iter()
            .map(|(_, _, p)| p)
            .collect(),
    )
}

/// The live elevations of the building's datum ladder, ascending. Dead
/// references are skipped; an empty ladder yields just the origin plane.
pub(crate) fn ladder_elevations(world: &World, building: &Building) -> Vec<f32> {
    let mut levels: Vec<f32> = building
        .datums
        .iter()
        .filter_map(|&d| world.get::<Datum>(d).map(|datum| datum.elevation))
        .collect();
    if levels.is_empty() {
        levels.push(0.0);
    }
    levels.sort_by(f32::total_cmp);
    levels
}

/// The envelope height: the topmost datum elevation (never below the
/// origin plane — negative marks are basements, not roofs).
pub(crate) fn envelope_top(world: &World, building: &Building) -> f32 {
    ladder_elevations(world, building)
        .last()
        .copied()
        .unwrap_or(0.0)
        .max(0.0)
}

// ---------------------------------------------------------------------------
// Edit action
// ---------------------------------------------------------------------------

/// Undoable "place a building": spawns the root with its envelope vertex
/// children and datum ladder — as a child of `parent` (`transform` is
/// then parent-local; this is how group prefabs accrete content) or
/// free-standing in the world (`transform` is world-space).
#[derive(Debug)]
pub struct PlaceBuildingAction {
    parent: Option<Entity>,
    transform: Transform,
    contour: Vec<[f32; 2]>,
    ladder: Vec<f32>,
    created: Vec<Entity>,
}

impl PlaceBuildingAction {
    pub fn new(parent: Option<Entity>, transform: Transform) -> Self {
        Self::shaped(
            parent,
            transform,
            DEFAULT_ENVELOPE.to_vec(),
            DEFAULT_LADDER.to_vec(),
        )
    }

    /// Place with an explicit envelope contour (local XZ, perimeter
    /// order) and datum ladder (elevations in meters).
    pub fn shaped(
        parent: Option<Entity>,
        transform: Transform,
        contour: Vec<[f32; 2]>,
        ladder: Vec<f32>,
    ) -> Self {
        Self {
            parent,
            transform,
            contour,
            ladder,
            created: Vec::new(),
        }
    }
}

impl EditAction<World> for PlaceBuildingAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let parent_m = match self.parent {
            Some(parent) => Some(
                world
                    .get::<GlobalTransform>(parent)
                    .map(|gt| gt.0)
                    .or_else(|| world.get::<Transform>(parent).map(|t| t.to_matrix()))
                    .ok_or_else(|| {
                        EditActionError::TargetNotFound("building parent has no transform".into())
                    })?,
            ),
            None => None,
        };
        let world_m = match parent_m {
            Some(m) => m * self.transform.to_matrix(),
            None => self.transform.to_matrix(),
        };
        let undo_partial = |world: &mut World, created: &mut Vec<Entity>| {
            for e in created.drain(..).rev() {
                remove_parent(world, e);
                world.despawn(e);
            }
        };

        let root = world.spawn();
        self.created.push(root);
        let inserted = world
            .insert(root, self.transform)
            .and_then(|_| world.insert(root, GlobalTransform(world_m)))
            .and_then(|_| world.insert(root, redlilium_ecs::Name("Building".to_owned())));
        if let Err(e) = inserted {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }
        if let Some(parent) = self.parent {
            set_parent(world, root, parent);
        }

        let mut points = Vec::with_capacity(self.contour.len());
        for (n, [x, z]) in self.contour.iter().copied().enumerate() {
            let local = Transform::new(
                Vec3::new(x, 0.0, z),
                quat_from_rotation_y(0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            let vertex = world.spawn();
            self.created.push(vertex);
            let inserted = world
                .insert(vertex, local)
                .and_then(|_| world.insert(vertex, GlobalTransform(world_m * local.to_matrix())))
                .and_then(|_| world.insert(vertex, StrokeVertex::default()))
                .and_then(|_| {
                    world.insert(vertex, redlilium_ecs::Name(format!("Point {}", n + 1)))
                });
            if let Err(e) = inserted {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::Custom(e.to_string()));
            }
            set_parent(world, vertex, root);
            points.push(vertex);
        }

        let mut datums = Vec::with_capacity(self.ladder.len());
        for elevation in self.ladder.iter().copied() {
            let datum = world.spawn();
            self.created.push(datum);
            let inserted = world.insert(datum, Datum { elevation }).and_then(|_| {
                world.insert(datum, redlilium_ecs::Name(format!("Datum {elevation:+.3}")))
            });
            if let Err(e) = inserted {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::Custom(e.to_string()));
            }
            set_parent(world, datum, root);
            datums.push(datum);
        }

        if let Err(e) = world.insert(
            root,
            Building {
                points,
                datums,
                walls: Vec::new(),
                seed: 0,
            },
        ) {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
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
        "Place building"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_ecs::{Children, Parent};

    fn world_with_group() -> (World, Entity) {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<Building>();
        world.register_inspector_default::<Datum>();
        world.register_inspector_default::<StrokeVertex>();

        // A plain entity is a group — no container component exists.
        let group = world.spawn();
        let t = Transform::new(
            Vec3::new(10.0, 0.0, 5.0),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(group, t).unwrap();
        world.insert(group, GlobalTransform(t.to_matrix())).unwrap();
        (world, group)
    }

    fn spawned_building(world: &World) -> (Entity, Building) {
        world
            .read_all::<Building>()
            .unwrap()
            .iter()
            .filter_map(|(index, b)| Some((world.entity_at_index(index)?, b.clone())))
            .next()
            .expect("a building was spawned")
    }

    #[test]
    fn buildings_parent_into_a_group_and_a_group_holds_many() {
        let (mut world, group) = world_with_group();
        let local = |x: f32, z: f32| {
            Transform::new(
                Vec3::new(x, 0.0, z),
                quat_from_rotation_y(0.3),
                Vec3::new(1.0, 1.0, 1.0),
            )
        };
        let mut first = PlaceBuildingAction::new(Some(group), local(-2.0, -3.0));
        let mut second = PlaceBuildingAction::shaped(
            Some(group),
            local(2.0, -5.0),
            vec![[-2.0, 2.0], [2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]],
            vec![0.0, 3.0],
        );
        first.apply(&mut world).unwrap();
        second.apply(&mut world).unwrap();

        let children = world.get::<Children>(group).unwrap().0.clone();
        let buildings: Vec<Entity> = children
            .iter()
            .copied()
            .filter(|&e| world.get::<Building>(e).is_some())
            .collect();
        assert_eq!(buildings.len(), 2, "a group holds many buildings");
        for b in &buildings {
            assert_eq!(world.get::<Parent>(*b).unwrap().0, group);
        }
        // Child transform composes with the group's.
        let gt = world.get::<GlobalTransform>(buildings[0]).unwrap().0;
        let pos = Vec3::new(gt[(0, 3)], gt[(1, 3)], gt[(2, 3)]);
        assert!((pos - Vec3::new(8.0, 0.0, 2.0)).norm() < 1e-4);
        // Each building carries its own contour vertices and ladder.
        for b in &buildings {
            let component = world.get::<Building>(*b).unwrap().clone();
            assert!(component.points.len() >= 4);
            for &v in component.points.iter().chain(&component.datums) {
                assert_eq!(world.get::<Parent>(v).unwrap().0, *b);
            }
        }

        second.undo(&mut world).unwrap();
        first.undo(&mut world).unwrap();
        assert!(
            world
                .read_all::<Building>()
                .unwrap()
                .iter()
                .next()
                .is_none()
        );
        assert!(world.read_all::<Datum>().unwrap().iter().next().is_none());
        let children = world.get::<Children>(group).unwrap().0.clone();
        assert!(children.is_empty(), "the whole subtree undoes");
    }

    #[test]
    fn free_standing_building_and_dead_parent_rejection() {
        let (mut world, _) = world_with_group();
        // No parent: the transform is world-space.
        let t = Transform::new(
            Vec3::new(-3.0, 0.0, 7.0),
            quat_from_rotation_y(0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        let mut free = PlaceBuildingAction::new(None, t);
        free.apply(&mut world).unwrap();
        let (building, _) = spawned_building(&world);
        assert!(world.get::<Parent>(building).is_none());
        let gt = world.get::<GlobalTransform>(building).unwrap().0;
        let pos = Vec3::new(gt[(0, 3)], gt[(1, 3)], gt[(2, 3)]);
        assert!((pos - Vec3::new(-3.0, 0.0, 7.0)).norm() < 1e-4);
        free.undo(&mut world).unwrap();

        // A parent without any transform is a broken target.
        let bare = world.spawn();
        let err = PlaceBuildingAction::new(Some(bare), Transform::default()).apply(&mut world);
        assert!(err.is_err());
    }

    #[test]
    fn envelope_ring_closes_and_height_is_the_top_datum() {
        let (mut world, _) = world_with_group();
        PlaceBuildingAction::new(None, Transform::default())
            .apply(&mut world)
            .unwrap();
        let (_, component) = spawned_building(&world);

        let ring = envelope_ring_local(&world, &component).expect("closed ring");
        assert!(
            (ring[0] - *ring.last().unwrap()).norm() < 1e-4,
            "the ring returns to its first vertex"
        );
        // Straight rectangle: 4 corners + the closing repeat.
        assert_eq!(ring.len(), 5);
        assert!((envelope_top(&world, &component) - 6.0).abs() < 1e-4);
        assert_eq!(ladder_elevations(&world, &component), vec![0.0, 3.0, 6.0]);

        // Raising a datum raises the envelope — the ladder is live data.
        let top = component.datums[2];
        world.insert(top, Datum { elevation: 9.5 }).unwrap();
        assert!((envelope_top(&world, &component) - 9.5).abs() < 1e-4);

        // A curved wall: give one vertex mirrored handles and the ring
        // tessellates its two adjacent segments.
        let v = component.points[1];
        world
            .insert(
                v,
                StrokeVertex {
                    handle_in: Vec3::new(-1.0, 0.0, 1.0),
                    handle_out: Vec3::new(1.0, 0.0, -1.0),
                },
            )
            .unwrap();
        let curved = envelope_ring_local(&world, &component).unwrap();
        assert!(curved.len() > 5, "curved segments fan out");

        // Below 3 live vertices there is no envelope.
        world.despawn(component.points[0]);
        world.despawn(component.points[1]);
        assert!(envelope_ring_local(&world, &component).is_none());
    }
}
