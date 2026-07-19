//! Lots (parcels): the authoring primitive of the architecture chapter.
//!
//! A lot is a rectangle behind a straight **frontage segment**, following
//! the socket convention everywhere else in the graph: local X runs along
//! the frontage, **+Z faces the road network, the parcel occupies −Z** ("a
//! socket's structure owns −Z"). Two states, one component:
//!
//! - **Edge-anchored**: the entity also carries an [`EdgeAnchor`] — its
//!   `Transform` and `frontage` are derived from the chord of the parent
//!   road's edge interval, exactly like an anchored road node (sliding along
//!   the edge, following parent edits, undo — all shared machinery).
//! - **Free**: no anchor; the `Transform` is authored by hand, in full 3D —
//!   there is no ground-plane assumption, terrain later conforms to roads
//!   and lots, not the other way around.
//!
//! A building on the lot is the next slice; the lot itself only reserves
//! and orients the parcel.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::{Mat4, Vec3, Vec4, quat_from_rotation_y};
use redlilium_ecs::ui::ViewportRay;
use redlilium_ecs::{Component, Entity, GlobalTransform, Transform, World};

use crate::anchor::{self, EdgeAnchor, EdgeHit};

/// A parcel: `2 * frontage` wide along local X, extending `depth` meters
/// along local −Z behind the frontage segment at the origin. When the
/// entity is edge-anchored, `frontage` is derived data (half the interval
/// chord); `depth` is always authored.
#[derive(Debug, Clone, Component)]
pub struct Lot {
    /// Half of the frontage length, meters.
    pub frontage: f32,
    /// How far the parcel extends behind the frontage, meters.
    pub depth: f32,
}

impl Default for Lot {
    fn default() -> Self {
        Self {
            frontage: 4.0,
            depth: 8.0,
        }
    }
}

/// The lot's rectangle in world space: front-left, front-right, back-right,
/// back-left (perimeter order). Front edge is the frontage at local z = 0.
pub(crate) fn lot_corners(world_mat: &Mat4, lot: &Lot) -> [Vec3; 4] {
    let corner = |x: f32, z: f32| {
        let p = world_mat * Vec4::new(x, 0.0, z, 1.0);
        Vec3::new(p.x, p.y, p.z)
    };
    [
        corner(-lot.frontage, 0.0),
        corner(lot.frontage, 0.0),
        corner(lot.frontage, -lot.depth),
        corner(-lot.frontage, -lot.depth),
    ]
}

/// Edge pick distance for the "Add lot" op, world units.
const EDGE_PICK_RADIUS: f32 = 1.0;

/// Where an "Add lot" click lands: a road edge (anchored lot) or open
/// space (free lot). Edges win — they are the more specific target.
pub(crate) enum LotTarget {
    Edge(EdgeHit),
    Free(Vec3),
}

pub(crate) fn lot_target(world: &World, ray: &ViewportRay) -> Option<LotTarget> {
    if let Some(hit) = anchor::edge_under_cursor(world, ray, EDGE_PICK_RADIUS) {
        return Some(LotTarget::Edge(hit));
    }
    crate::tool::ground_hit(ray).map(LotTarget::Free)
}

// ---------------------------------------------------------------------------
// Edit action
// ---------------------------------------------------------------------------

/// Undoable "place a lot": free-standing at a point, or glued to a road
/// edge (the anchored variant derives its placement from the edge chord).
#[derive(Debug)]
pub struct AddLotAction {
    anchor: Option<EdgeAnchor>,
    transform: Transform,
    created: Option<Entity>,
}

impl AddLotAction {
    /// Free lot at `point`, identity yaw.
    pub fn at_point(point: Vec3) -> Self {
        Self {
            anchor: None,
            transform: Transform::new(point, quat_from_rotation_y(0.0), Vec3::new(1.0, 1.0, 1.0)),
            created: None,
        }
    }

    /// Lot glued to a road edge interval.
    pub fn on_edge(anchor: EdgeAnchor) -> Self {
        Self {
            anchor: Some(anchor),
            transform: Transform::default(),
            created: None,
        }
    }
}

impl EditAction<World> for AddLotAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let (transform, frontage) = match &self.anchor {
            Some(anchor) => anchor::derive_anchor_state(world, anchor).ok_or_else(|| {
                EditActionError::TargetNotFound("lot parent road missing or degenerate".into())
            })?,
            None => (self.transform, Lot::default().frontage),
        };
        let lot = world.spawn();
        let insert = |world: &mut World, e, r: Result<(), redlilium_ecs::WorldError>| {
            r.map_err(|err| {
                world.despawn(e);
                EditActionError::Custom(err.to_string())
            })
        };
        let t = world.insert(lot, transform);
        insert(world, lot, t)?;
        let g = world.insert(lot, GlobalTransform(transform.to_matrix()));
        insert(world, lot, g)?;
        let l = world.insert(
            lot,
            Lot {
                frontage,
                ..Lot::default()
            },
        );
        insert(world, lot, l)?;
        if let Some(anchor) = &self.anchor {
            let a = world.insert(lot, anchor.clone());
            insert(world, lot, a)?;
        }
        self.created = Some(lot);
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if let Some(lot) = self.created.take() {
            world.despawn(lot);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Add lot"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RoadNode, RoadSegment, bezier};

    fn setup() -> (World, Entity) {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();
        world.register_inspector_default::<EdgeAnchor>();
        world.register_inspector_default::<Lot>();

        let node = |world: &mut World, x: f32, z: f32| {
            let e = world.spawn();
            let t = Transform::new(
                Vec3::new(x, 0.0, z),
                quat_from_rotation_y(0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            world.insert(e, t).unwrap();
            world.insert(e, GlobalTransform(t.to_matrix())).unwrap();
            world.insert(e, RoadNode::default()).unwrap();
            e
        };
        // Straight road along +Z from origin to (0, 0, 20).
        let a = node(&mut world, 0.0, 0.0);
        let b = node(&mut world, 0.0, 20.0);
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
    fn anchored_lot_settles_on_edge_and_follows() {
        let (mut world, road) = setup();
        let lot = world.spawn();
        world.insert(lot, Transform::default()).unwrap();
        world
            .insert(lot, GlobalTransform(Transform::default().to_matrix()))
            .unwrap();
        world.insert(lot, Lot::default()).unwrap();
        world
            .insert(
                lot,
                EdgeAnchor {
                    parent_road: road,
                    right_edge: true,
                    u_min: 0.4,
                    u_max: 0.6,
                },
            )
            .unwrap();

        crate::settle_edge_anchors(&mut world);
        let t = *world.get::<Transform>(lot).unwrap();
        // Right edge of a straight +Z road sits at x = +3; interval midpoint
        // z = 10; +Z faces outward (+X), so the parcel extends toward −X.
        assert!((t.translation - Vec3::new(3.0, 0.0, 10.0)).norm() < 1e-3);
        let heading = bezier::heading(&t.to_matrix());
        assert!((heading - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-3);
        // Frontage derived from the chord: a fifth of a 20 m edge → half 2.
        let frontage = world.get::<Lot>(lot).unwrap().frontage;
        assert!((frontage - 2.0).abs() < 1e-2);
        // Depth stays authored.
        assert!((world.get::<Lot>(lot).unwrap().depth - Lot::default().depth).abs() < 1e-6);
        // Settled: no pending updates (the width receiver is the Lot).
        assert!(anchor::anchor_updates(&world, &mut Default::default(), false).is_empty());

        // Parent-road edit: the lot follows the edge.
        let seg_b = world.get::<RoadSegment>(road).unwrap().b;
        let moved = Transform::new(
            Vec3::new(8.0, 0.0, 20.0),
            quat_from_rotation_y(0.4),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(seg_b, moved).unwrap();
        world
            .insert(seg_b, GlobalTransform(moved.to_matrix()))
            .unwrap();
        crate::settle_edge_anchors(&mut world);
        let after = world.get::<Transform>(lot).unwrap().translation;
        assert!((after - t.translation).norm() > 0.1, "lot follows the edge");
    }

    #[test]
    fn add_lot_action_roundtrip_free_and_anchored() {
        let (mut world, road) = setup();

        // Free lot: lands exactly at the click point (any height — no
        // ground-plane assumption in the data model).
        let mut free = AddLotAction::at_point(Vec3::new(30.0, 2.0, 7.0));
        free.apply(&mut world).unwrap();
        let lots: Vec<Entity> = world
            .read_all::<Lot>()
            .unwrap()
            .iter()
            .filter_map(|(index, _)| world.entity_at_index(index))
            .collect();
        assert_eq!(lots.len(), 1);
        let t = world.get::<Transform>(lots[0]).unwrap();
        assert!((t.translation - Vec3::new(30.0, 2.0, 7.0)).norm() < 1e-6);
        assert!(world.get::<EdgeAnchor>(lots[0]).is_none());

        // Anchored lot: derives onto the edge chord at apply time.
        let mut anchored = AddLotAction::on_edge(EdgeAnchor {
            parent_road: road,
            right_edge: true,
            u_min: 0.3,
            u_max: 0.5,
        });
        anchored.apply(&mut world).unwrap();
        let anchored_lot = world
            .read_all::<Lot>()
            .unwrap()
            .iter()
            .filter_map(|(index, _)| world.entity_at_index(index))
            .find(|&e| world.get::<EdgeAnchor>(e).is_some())
            .expect("anchored lot");
        let t = world.get::<Transform>(anchored_lot).unwrap();
        assert!((t.translation.x - 3.0).abs() < 1e-3, "sits on the edge");
        assert!((world.get::<Lot>(anchored_lot).unwrap().frontage - 2.0).abs() < 1e-2);

        anchored.undo(&mut world).unwrap();
        free.undo(&mut world).unwrap();
        assert!(world.read_all::<Lot>().unwrap().iter().next().is_none());
    }
}
