//! Buildings: parcel content with a footprint of their own.
//!
//! A building is a **child entity of a parcel** — one parcel may hold any
//! number of them (a villa, or a whole factory of structures). The
//! component carries flat box-massing recipe parameters (P4 stub of the
//! eventual assembly-graph *asset*; the fields are already the asset's
//! fields, promotion is mechanical once AssetRef inspector editing lands).
//! The footprint is the building's own local rectangle — the parcel
//! constrains content spatially (validators later), it no longer defines
//! building shape. Connections to the road network belong to the parcel
//! (its gates), not to buildings.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_ecs::{
    Component, Entity, GlobalTransform, Transform, World, remove_parent, set_parent,
};

use crate::parcel::Parcel;

/// Box-massing recipe parameters. The footprint is
/// `2·half_width × 2·half_depth` in the entity's local XZ, centered at its
/// origin, extruded `floors × floor_height` up.
#[derive(Debug, Clone, Component)]
pub struct Building {
    /// Storeys stacked on the footprint.
    pub floors: u32,
    /// Height of one storey, meters.
    pub floor_height: f32,
    /// Footprint half-extent along local X, meters.
    pub half_width: f32,
    /// Footprint half-extent along local Z, meters.
    pub half_depth: f32,
    /// Generator seed — same seed + same params ⇒ same building (P3).
    pub seed: u32,
}

impl Default for Building {
    fn default() -> Self {
        Self {
            floors: 2,
            floor_height: 3.0,
            half_width: 3.0,
            half_depth: 3.0,
            seed: 0,
        }
    }
}

/// The building's footprint corners in its local space (perimeter order).
pub(crate) fn footprint_corners(building: &Building) -> [[f32; 2]; 4] {
    let (w, d) = (building.half_width, building.half_depth);
    [[-w, d], [w, d], [w, -d], [-w, -d]]
}

// ---------------------------------------------------------------------------
// Edit action
// ---------------------------------------------------------------------------

/// Undoable "place a building inside a parcel": spawns a child entity with
/// the recipe at a parcel-local transform.
#[derive(Debug)]
pub struct PlaceBuildingAction {
    parcel: Entity,
    local: Transform,
    building: Building,
    created: Option<Entity>,
}

impl PlaceBuildingAction {
    pub fn new(parcel: Entity, local: Transform, building: Building) -> Self {
        Self {
            parcel,
            local,
            building,
            created: None,
        }
    }
}

impl EditAction<World> for PlaceBuildingAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if world.get::<Parcel>(self.parcel).is_none() {
            return Err(EditActionError::TargetNotFound(
                "building target is not a parcel".into(),
            ));
        }
        let parcel_matrix = world
            .get::<GlobalTransform>(self.parcel)
            .map(|gt| gt.0)
            .or_else(|| world.get::<Transform>(self.parcel).map(|t| t.to_matrix()))
            .ok_or_else(|| EditActionError::TargetNotFound("parcel has no transform".into()))?;

        let entity = world.spawn();
        let inserted = world
            .insert(entity, self.local)
            .and_then(|_| {
                world.insert(
                    entity,
                    GlobalTransform(parcel_matrix * self.local.to_matrix()),
                )
            })
            .and_then(|_| world.insert(entity, self.building.clone()));
        if let Err(e) = inserted {
            world.despawn(entity);
            return Err(EditActionError::Custom(e.to_string()));
        }
        set_parent(world, entity, self.parcel);
        self.created = Some(entity);
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if let Some(entity) = self.created.take() {
            remove_parent(world, entity);
            world.despawn(entity);
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
    use crate::parcel::{AddParcelAction, ParcelVertex};
    use redlilium_core::math::{Vec3, quat_from_rotation_y};
    use redlilium_ecs::{Children, Parent};

    fn world_with_parcel() -> (World, Entity) {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<Parcel>();
        world.register_inspector_default::<ParcelVertex>();
        world.register_inspector_default::<Building>();

        AddParcelAction::at_point(Vec3::new(10.0, 0.0, 5.0))
            .apply(&mut world)
            .unwrap();
        let parcel = world
            .read_all::<Parcel>()
            .unwrap()
            .iter()
            .filter_map(|(index, _)| world.entity_at_index(index))
            .next()
            .unwrap();
        (world, parcel)
    }

    #[test]
    fn buildings_are_children_and_a_parcel_holds_many() {
        let (mut world, parcel) = world_with_parcel();
        let local = |x: f32, z: f32| {
            Transform::new(
                Vec3::new(x, 0.0, z),
                quat_from_rotation_y(0.3),
                Vec3::new(1.0, 1.0, 1.0),
            )
        };
        let mut first = PlaceBuildingAction::new(parcel, local(-2.0, -3.0), Building::default());
        let mut second = PlaceBuildingAction::new(
            parcel,
            local(2.0, -5.0),
            Building {
                floors: 5,
                ..Building::default()
            },
        );
        first.apply(&mut world).unwrap();
        second.apply(&mut world).unwrap();

        let children = world.get::<Children>(parcel).unwrap().0.clone();
        let buildings: Vec<Entity> = children
            .iter()
            .copied()
            .filter(|&e| world.get::<Building>(e).is_some())
            .collect();
        assert_eq!(buildings.len(), 2, "a parcel holds many buildings");
        for b in &buildings {
            assert_eq!(world.get::<Parent>(*b).unwrap().0, parcel);
        }
        // Child transform composes with the parcel's.
        let gt = world.get::<GlobalTransform>(buildings[0]).unwrap().0;
        let pos = Vec3::new(gt[(0, 3)], gt[(1, 3)], gt[(2, 3)]);
        assert!((pos - Vec3::new(8.0, 0.0, 2.0)).norm() < 1e-4);

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
        let children = world.get::<Children>(parcel).unwrap().0.clone();
        assert!(children.iter().all(|&e| world.get::<Building>(e).is_none()));
    }

    #[test]
    fn placing_outside_a_parcel_is_rejected() {
        let (mut world, _) = world_with_parcel();
        let stray = world.spawn();
        let err = PlaceBuildingAction::new(stray, Transform::default(), Building::default())
            .apply(&mut world);
        assert!(err.is_err());
    }
}
