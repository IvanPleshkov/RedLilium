use crate::entity::Entity;
use crate::world::World;

/// A group of components that can be inserted together on an entity.
///
/// Implemented for tuples of components up to 8 elements.
/// Each element must be `Send + Sync + 'static` (the component bounds).
///
/// # Example
///
/// ```ignore
/// // Insert a bundle of components at once
/// world.insert_bundle(entity, (
///     Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
///     GlobalTransform::default(),
///     Visibility::VISIBLE,
///     Name::new("Player"),
/// )).unwrap();
///
/// // Spawn an entity with a bundle
/// let entity = world.spawn_with((
///     Transform::IDENTITY,
///     Visibility::VISIBLE,
/// ));
/// ```
pub trait Bundle: Send + 'static {
    /// Inserts all components in this bundle onto `entity`.
    ///
    /// # Errors
    ///
    /// Returns an error if any component type has not been registered.
    fn insert_into(
        self,
        world: &mut World,
        entity: Entity,
    ) -> Result<(), crate::world::ComponentNotRegistered>;
}

macro_rules! impl_bundle {
    ($($T:ident),+) => {
        impl<$($T: Send + Sync + 'static),+> Bundle for ($($T,)+) {
            fn insert_into(
                self,
                world: &mut World,
                entity: Entity,
            ) -> Result<(), crate::world::ComponentNotRegistered> {
                #[allow(non_snake_case)]
                let ($($T,)+) = self;
                $(world.insert(entity, $T)?;)+
                Ok(())
            }
        }
    };
}

impl_bundle!(A);
impl_bundle!(A, B);
impl_bundle!(A, B, C);
impl_bundle!(A, B, C, D);
impl_bundle!(A, B, C, D, E);
impl_bundle!(A, B, C, D, E, F);
impl_bundle!(A, B, C, D, E, F, G);
impl_bundle!(A, B, C, D, E, F, G, H);

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Velocity {
        x: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    #[test]
    fn single_element_bundle() {
        let mut world = World::new();
        world.register_component::<Health>();
        let entity = world.spawn();

        (Health(100),).insert_into(&mut world, entity).unwrap();

        assert_eq!(world.get::<Health>(entity), Some(&Health(100)));
    }

    #[test]
    fn two_element_bundle() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();
        let entity = world.spawn();

        (Position { x: 1.0, y: 2.0 }, Health(50))
            .insert_into(&mut world, entity)
            .unwrap();

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 1.0, y: 2.0 })
        );
        assert_eq!(world.get::<Health>(entity), Some(&Health(50)));
    }

    #[test]
    fn three_element_bundle() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        world.register_component::<Health>();
        let entity = world.spawn();

        (Position { x: 0.0, y: 0.0 }, Velocity { x: 1.0 }, Health(75))
            .insert_into(&mut world, entity)
            .unwrap();

        assert_eq!(world.get::<Velocity>(entity), Some(&Velocity { x: 1.0 }));
        assert_eq!(world.get::<Health>(entity), Some(&Health(75)));
    }

    #[test]
    fn unregistered_component_returns_err() {
        let mut world = World::new();
        world.register_component::<Position>();
        // Health is NOT registered
        let entity = world.spawn();

        let result = (Position { x: 0.0, y: 0.0 }, Health(100)).insert_into(&mut world, entity);
        assert!(result.is_err());
    }

    // --- derive(Bundle) tests ---

    #[derive(crate::Bundle)]
    struct PlayerBundle {
        position: Position,
        health: Health,
    }

    #[test]
    fn derive_bundle_struct() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();
        let entity = world.spawn();

        let bundle = PlayerBundle {
            position: Position { x: 5.0, y: 10.0 },
            health: Health(100),
        };
        bundle.insert_into(&mut world, entity).unwrap();

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 5.0, y: 10.0 })
        );
        assert_eq!(world.get::<Health>(entity), Some(&Health(100)));
    }

    #[derive(crate::Bundle)]
    struct SpatialBundle {
        position: Position,
        velocity: Velocity,
    }

    #[derive(crate::Bundle)]
    struct FullBundle {
        health: Health,
        #[bundle]
        spatial: SpatialBundle,
    }

    #[test]
    fn derive_bundle_nested() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        world.register_component::<Health>();
        let entity = world.spawn();

        let bundle = FullBundle {
            health: Health(200),
            spatial: SpatialBundle {
                position: Position { x: 1.0, y: 2.0 },
                velocity: Velocity { x: 3.0 },
            },
        };
        bundle.insert_into(&mut world, entity).unwrap();

        assert_eq!(world.get::<Health>(entity), Some(&Health(200)));
        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 1.0, y: 2.0 })
        );
        assert_eq!(world.get::<Velocity>(entity), Some(&Velocity { x: 3.0 }));
    }

    #[test]
    fn derive_bundle_with_spawn_with() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();

        let entity = world.spawn_with(PlayerBundle {
            position: Position { x: 7.0, y: 8.0 },
            health: Health(50),
        });

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 7.0, y: 8.0 })
        );
        assert_eq!(world.get::<Health>(entity), Some(&Health(50)));
    }

    #[test]
    fn derive_bundle_unregistered_returns_err() {
        let mut world = World::new();
        world.register_component::<Position>();
        // Health is NOT registered
        let entity = world.spawn();

        let bundle = PlayerBundle {
            position: Position { x: 0.0, y: 0.0 },
            health: Health(100),
        };
        let result = bundle.insert_into(&mut world, entity);
        assert!(result.is_err());
    }
}
