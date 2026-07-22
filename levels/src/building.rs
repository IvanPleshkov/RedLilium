//! Buildings: free-standing architecture content with a footprint of its
//! own.
//!
//! A building is an ordinary entity — nothing owns it structurally.
//! Grouping is plain hierarchy: parent buildings, strokes and roads under
//! one root entity and the subtree duplicates as a prefab ("villa" =
//! fences + buildings + a driveway under one root; see
//! `stroke::DuplicateSubtreeAction`). The component carries flat
//! box-massing recipe parameters (P4 stub of the eventual assembly-graph
//! *asset*; the fields are already the asset's fields, promotion is
//! mechanical once AssetRef inspector editing lands). Connections to the
//! road network are gates on strokes or edge anchors — never building
//! fields.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_ecs::{
    Component, Entity, GlobalTransform, Transform, World, remove_parent, set_parent,
};

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

/// Undoable "place a building": spawns an entity with the recipe — as a
/// child of `parent` (`transform` is then parent-local; this is how group
/// prefabs accrete content) or free-standing in the world (`transform` is
/// world-space).
#[derive(Debug)]
pub struct PlaceBuildingAction {
    parent: Option<Entity>,
    transform: Transform,
    building: Building,
    created: Option<Entity>,
}

impl PlaceBuildingAction {
    pub fn new(parent: Option<Entity>, transform: Transform, building: Building) -> Self {
        Self {
            parent,
            transform,
            building,
            created: None,
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

        let entity = world.spawn();
        let world_m = match parent_m {
            Some(m) => m * self.transform.to_matrix(),
            None => self.transform.to_matrix(),
        };
        let inserted = world
            .insert(entity, self.transform)
            .and_then(|_| world.insert(entity, GlobalTransform(world_m)))
            .and_then(|_| world.insert(entity, self.building.clone()));
        if let Err(e) = inserted {
            world.despawn(entity);
            return Err(EditActionError::Custom(e.to_string()));
        }
        if let Some(parent) = self.parent {
            set_parent(world, entity, parent);
        }
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
    use redlilium_core::math::{Vec3, quat_from_rotation_y};
    use redlilium_ecs::{Children, Parent};

    fn world_with_group() -> (World, Entity) {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<Building>();

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
        let mut first =
            PlaceBuildingAction::new(Some(group), local(-2.0, -3.0), Building::default());
        let mut second = PlaceBuildingAction::new(
            Some(group),
            local(2.0, -5.0),
            Building {
                floors: 5,
                ..Building::default()
            },
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
        let children = world.get::<Children>(group).unwrap().0.clone();
        assert!(children.iter().all(|&e| world.get::<Building>(e).is_none()));
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
        let mut free = PlaceBuildingAction::new(None, t, Building::default());
        free.apply(&mut world).unwrap();
        let building = world
            .read_all::<Building>()
            .unwrap()
            .iter()
            .filter_map(|(index, _)| world.entity_at_index(index))
            .next()
            .unwrap();
        assert!(world.get::<Parent>(building).is_none());
        let gt = world.get::<GlobalTransform>(building).unwrap().0;
        let pos = Vec3::new(gt[(0, 3)], gt[(1, 3)], gt[(2, 3)]);
        assert!((pos - Vec3::new(-3.0, 0.0, 7.0)).norm() < 1e-4);
        free.undo(&mut world).unwrap();

        // A parent without any transform is a broken target.
        let bare = world.spawn();
        let err = PlaceBuildingAction::new(Some(bare), Transform::default(), Building::default())
            .apply(&mut world);
        assert!(err.is_err());
    }
}
