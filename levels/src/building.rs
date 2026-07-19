//! Buildings on lots: recipe parameters + exit-socket materialization.
//!
//! A building lives as a [`Building`] component **on the lot entity** — the
//! lot reserves the parcel, the building fills it. The recipe is a flat set
//! of box-massing parameters for now (P4 stub; promoting them into a
//! reusable assembly-graph *asset* is a mechanical move once AssetRef
//! inspector editing lands — the fields are already the asset's fields).
//!
//! **Exit sockets are materialized at placement time by the edit action**,
//! not derived per frame: each [`ExitSpec`] spawns an ordinary child entity
//! with a [`RoadNode`] + [`BuildingExit`] marker, parented to the lot.
//! Children follow the lot through plain `GlobalTransform` propagation —
//! zero new machinery — and, being ordinary serialized entities, driveway
//! `RoadSegment`s can reference them stably across scene reloads (a
//! transient derived-spawn would break those references). Socket
//! convention as everywhere: the exit's **+Z faces outward into the road
//! network**; a road arriving at it meets it from the front.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_ecs::{
    Component, Entity, GlobalTransform, Transform, World, remove_parent, set_parent,
};

use crate::RoadNode;
use crate::lot::Lot;

/// One exit socket declared by a building: where a driveway leaves the lot.
/// Sits on the frontage line (local z = 0), facing the lot's +Z.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ExitSpec {
    /// Offset along the frontage from the lot center, meters. Clamped into
    /// the frontage at materialization.
    pub offset: f32,
    /// Socket half width, meters (the driveway's cross-section).
    pub half_width: f32,
}

/// Box-massing recipe parameters + placement state, on the lot entity.
#[derive(Debug, Clone, Component)]
pub struct Building {
    /// Storeys stacked on the footprint.
    pub floors: u32,
    /// Height of one storey, meters.
    pub floor_height: f32,
    /// Footprint inset from the lot bounds, meters (all four sides).
    pub inset: f32,
    /// Generator seed — same seed + same params ⇒ same building (P3).
    pub seed: u32,
    /// Exit sockets the recipe declares. Edited exits re-materialize via a
    /// fresh placement, not per-frame derivation.
    pub exits: Vec<ExitSpec>,
}

impl Default for Building {
    fn default() -> Self {
        Self {
            floors: 2,
            floor_height: 3.0,
            inset: 1.0,
            seed: 0,
            exits: vec![ExitSpec {
                offset: 0.0,
                half_width: 1.5,
            }],
        }
    }
}

/// Marker on a materialized exit-socket node. Roads arriving at one meet it
/// from the front, like any socket (`RoadSegment::b_from_front`).
#[derive(Debug, Clone, Default, Component)]
pub struct BuildingExit;

/// The building's footprint rectangle in lot-local space:
/// `(±half_x, z_front, z_back)`. `None` when the inset eats the lot.
pub(crate) fn footprint(lot: &Lot, building: &Building) -> Option<(f32, f32, f32)> {
    let half_x = lot.frontage - building.inset;
    let z_front = -building.inset;
    let z_back = -(lot.depth - building.inset);
    (half_x > 0.05 && z_back < z_front - 0.05).then_some((half_x, z_front, z_back))
}

// ---------------------------------------------------------------------------
// Edit action
// ---------------------------------------------------------------------------

/// Undoable "place a building on a lot": inserts the [`Building`] component
/// and materializes its exit sockets as child road nodes.
#[derive(Debug)]
pub struct PlaceBuildingAction {
    lot: Entity,
    building: Building,
    placed: bool,
    created_exits: Vec<Entity>,
}

impl PlaceBuildingAction {
    pub fn new(lot: Entity, building: Building) -> Self {
        Self {
            lot,
            building,
            placed: false,
            created_exits: Vec::new(),
        }
    }
}

impl EditAction<World> for PlaceBuildingAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let Some(lot) = world.get::<Lot>(self.lot).cloned() else {
            return Err(EditActionError::TargetNotFound(
                "building target is not a lot".into(),
            ));
        };
        if world.get::<Building>(self.lot).is_some() {
            return Err(EditActionError::Custom(
                "the lot already has a building".into(),
            ));
        }
        let lot_matrix = world
            .get::<GlobalTransform>(self.lot)
            .map(|gt| gt.0)
            .or_else(|| world.get::<Transform>(self.lot).map(|t| t.to_matrix()))
            .ok_or_else(|| EditActionError::TargetNotFound("lot has no transform".into()))?;

        let lot_entity = self.lot;
        let undo_partial = move |world: &mut World, exits: &mut Vec<Entity>| {
            for e in exits.drain(..) {
                remove_parent(world, e);
                world.despawn(e);
            }
            let _ = world.remove::<Building>(lot_entity);
        };

        world
            .insert(self.lot, self.building.clone())
            .map_err(|e| EditActionError::Custom(e.to_string()))?;

        for exit in &self.building.exits {
            let limit = (lot.frontage - exit.half_width).max(0.0);
            let local = Transform::new(
                redlilium_core::math::Vec3::new(exit.offset.clamp(-limit, limit), 0.0, 0.0),
                redlilium_core::math::quat_from_rotation_y(0.0),
                redlilium_core::math::Vec3::new(1.0, 1.0, 1.0),
            );
            let socket = world.spawn();
            self.created_exits.push(socket);
            let inserted = world
                .insert(socket, local)
                .and_then(|_| world.insert(socket, GlobalTransform(lot_matrix * local.to_matrix())))
                .and_then(|_| {
                    world.insert(
                        socket,
                        RoadNode {
                            half_width: exit.half_width,
                        },
                    )
                })
                .and_then(|_| world.insert(socket, BuildingExit));
            if let Err(e) = inserted {
                undo_partial(world, &mut self.created_exits);
                return Err(EditActionError::Custom(e.to_string()));
            }
            set_parent(world, socket, self.lot);
        }

        self.placed = true;
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        for socket in self.created_exits.drain(..) {
            remove_parent(world, socket);
            world.despawn(socket);
        }
        if self.placed {
            let _ = world.remove::<Building>(self.lot);
            self.placed = false;
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
    use redlilium_core::math::{Vec3, quat_from_rotation_y};
    use redlilium_ecs::{Children, Parent};

    fn setup_lot(x: f32, z: f32, yaw: f32) -> (World, Entity) {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<Lot>();
        world.register_inspector_default::<Building>();
        world.register_inspector_default::<BuildingExit>();

        let lot = world.spawn();
        let t = Transform::new(
            Vec3::new(x, 0.0, z),
            quat_from_rotation_y(yaw),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(lot, t).unwrap();
        world.insert(lot, GlobalTransform(t.to_matrix())).unwrap();
        world.insert(lot, Lot::default()).unwrap();
        (world, lot)
    }

    #[test]
    fn place_building_materializes_exits_and_undo_reverts() {
        let (mut world, lot) = setup_lot(10.0, 5.0, 0.0);
        let building = Building {
            exits: vec![ExitSpec {
                offset: 2.0,
                half_width: 1.5,
            }],
            ..Building::default()
        };
        let mut action = PlaceBuildingAction::new(lot, building);
        action.apply(&mut world).unwrap();

        assert!(world.get::<Building>(lot).is_some());
        let children = world.get::<Children>(lot).expect("children").0.clone();
        assert_eq!(children.len(), 1);
        let socket = children[0];
        assert_eq!(world.get::<Parent>(socket).unwrap().0, lot);
        assert!(world.get::<BuildingExit>(socket).is_some());
        assert!((world.get::<RoadNode>(socket).unwrap().half_width - 1.5).abs() < 1e-6);
        // Lot yaw 0 → exit at lot position + offset along world X, facing +Z.
        let gt = world.get::<GlobalTransform>(socket).unwrap().0;
        let pos = Vec3::new(gt[(0, 3)], gt[(1, 3)], gt[(2, 3)]);
        assert!((pos - Vec3::new(12.0, 0.0, 5.0)).norm() < 1e-4);
        let heading = crate::bezier::heading(&gt);
        assert!((heading - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-4);

        action.undo(&mut world).unwrap();
        assert!(world.get::<Building>(lot).is_none());
        assert!(!world.is_alive(socket));
        assert!(world.get::<Children>(lot).is_none_or(|c| c.is_empty()));
    }

    #[test]
    fn exit_offset_clamps_into_the_frontage() {
        let (mut world, lot) = setup_lot(0.0, 0.0, 0.0);
        let building = Building {
            exits: vec![ExitSpec {
                offset: 100.0,
                half_width: 1.5,
            }],
            ..Building::default()
        };
        let mut action = PlaceBuildingAction::new(lot, building);
        action.apply(&mut world).unwrap();
        let socket = world.get::<Children>(lot).unwrap().0[0];
        // frontage 4 − half_width 1.5 → clamped to x = 2.5.
        let x = world.get::<Transform>(socket).unwrap().translation.x;
        assert!((x - 2.5).abs() < 1e-4);
    }

    #[test]
    fn second_building_on_the_same_lot_is_rejected() {
        let (mut world, lot) = setup_lot(0.0, 0.0, 0.0);
        PlaceBuildingAction::new(lot, Building::default())
            .apply(&mut world)
            .unwrap();
        let err = PlaceBuildingAction::new(lot, Building::default()).apply(&mut world);
        assert!(err.is_err());
        // The failed apply must not have leaked extra exits.
        assert_eq!(world.get::<Children>(lot).unwrap().len(), 1);
    }

    #[test]
    fn footprint_degenerates_when_inset_eats_the_lot() {
        let lot = Lot {
            frontage: 4.0,
            depth: 8.0,
        };
        let b = |inset: f32| Building {
            inset,
            ..Building::default()
        };
        let (half_x, z_front, z_back) = footprint(&lot, &b(1.0)).expect("fits");
        assert!((half_x - 3.0).abs() < 1e-6);
        assert!((z_front + 1.0).abs() < 1e-6);
        assert!((z_back + 7.0).abs() < 1e-6);
        assert!(footprint(&lot, &b(4.0)).is_none());
    }
}
