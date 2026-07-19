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
//!   so no re-derivation by angle (unlike junction loops). Segments are
//!   straight for now; curved segments with optional C1 joints are the
//!   next slice.
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

/// Marker on a boundary-vertex entity (a child of the parcel; its local
/// translation is the vertex position, heights included — parcels are not
/// flat in general).
#[derive(Debug, Clone, Default, Component)]
pub struct ParcelVertex;

/// Marker on a parcel-owned connection socket: a child `RoadNode` sitting
/// on the boundary, +Z outward. Network roads meet it from the front,
/// internal roads from behind — the standard socket rule.
#[derive(Debug, Clone, Default, Component)]
pub struct ParcelGate;

/// The parcel's boundary polyline in world space (live vertices, perimeter
/// order). `None` with fewer than 3 live vertices.
pub fn parcel_loop(world: &World, parcel: &Parcel) -> Option<Vec<Vec3>> {
    let points: Vec<Vec3> = parcel
        .boundary
        .iter()
        .filter_map(|&v| {
            world.get::<ParcelVertex>(v)?;
            let gt = world.get::<GlobalTransform>(v)?;
            Some(Vec3::new(gt.0[(0, 3)], gt.0[(1, 3)], gt.0[(2, 3)]))
        })
        .collect();
    (points.len() >= 3).then_some(points)
}

/// Default boundary for a freshly stamped parcel: an 8×8 rectangle in
/// local space, front edge along +X at z = 0, interior extending to −Z.
const DEFAULT_BOUNDARY: [[f32; 2]; 4] = [[-4.0, 0.0], [4.0, 0.0], [4.0, -8.0], [-4.0, -8.0]];

// ---------------------------------------------------------------------------
// Edit action
// ---------------------------------------------------------------------------

/// Undoable "stamp a parcel at a point": the container entity plus a
/// default rectangular boundary of vertex children — drag the vertices
/// into shape afterwards.
#[derive(Debug)]
pub struct AddParcelAction {
    transform: Transform,
    created: Vec<Entity>,
}

impl AddParcelAction {
    pub fn at_point(point: Vec3) -> Self {
        Self {
            transform: Transform::new(point, quat_from_rotation_y(0.0), Vec3::new(1.0, 1.0, 1.0)),
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
                .and_then(|_| world.insert(vertex, ParcelVertex));
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
