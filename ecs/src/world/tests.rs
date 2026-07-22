use super::*;

#[derive(Debug, Clone, PartialEq)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct Velocity {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct Health(u32);

struct Frozen;

#[test]
fn spawn_and_check_alive() {
    let mut world = World::new();
    let entity = world.spawn();
    assert!(world.is_alive(entity));
    assert_eq!(world.entity_count(), 1);
}

#[test]
fn despawn_removes_entity() {
    let mut world = World::new();
    let entity = world.spawn();
    assert!(world.despawn(entity));
    assert!(!world.is_alive(entity));
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn despawn_dead_entity_returns_false() {
    let mut world = World::new();
    let entity = world.spawn();
    world.despawn(entity);
    assert!(!world.despawn(entity));
}

#[test]
fn insert_and_get_component() {
    let mut world = World::new();
    world.register_component::<Position>();
    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();

    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 1.0, y: 2.0 })
    );
}

#[test]
fn insert_unregistered_returns_err() {
    let mut world = World::new();
    let entity = world.spawn();
    let result = world.insert(entity, Position { x: 0.0, y: 0.0 });
    assert!(matches!(
        result.unwrap_err(),
        WorldError::ComponentNotRegistered { type_name } if type_name.contains("Position")
    ));
}

#[test]
fn read_unregistered_returns_err() {
    let world = World::new();
    assert!(world.read::<Position>().is_err());
}

#[test]
fn write_unregistered_returns_err() {
    let mut world = World::new();
    assert!(world.write::<Position>().is_err());
}

#[test]
fn insert_on_dead_entity_returns_err() {
    let mut world = World::new();
    world.register_component::<Position>();
    let entity = world.spawn();
    world.despawn(entity);
    let result = world.insert(entity, Position { x: 0.0, y: 0.0 });
    assert!(matches!(result, Err(WorldError::EntityNotAlive { .. })));
}

#[test]
fn remove_component() {
    let mut world = World::new();
    world.register_component::<Health>();
    let entity = world.spawn();
    world.insert(entity, Health(100)).unwrap();

    assert_eq!(world.remove::<Health>(entity).unwrap(), Some(Health(100)));
    assert!(world.get::<Health>(entity).is_none());
}

#[test]
fn despawn_removes_all_components() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();
    let entity = world.spawn();
    world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
    world.insert(entity, Health(100)).unwrap();

    world.despawn(entity);

    // Spawn a new entity that reuses the same index
    let new_entity = world.spawn();
    assert_eq!(new_entity.index(), entity.index());

    // New entity should not have old components
    assert!(world.get::<Position>(new_entity).is_none());
    assert!(world.get::<Health>(new_entity).is_none());
}

#[test]
fn read_query_iterates_all() {
    let mut world = World::new();
    world.register_component::<Position>();
    for i in 0..3 {
        let e = world.spawn();
        world
            .insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                },
            )
            .unwrap();
    }

    let positions = world.read::<Position>().unwrap();
    assert_eq!(positions.len(), 3);

    let xs: Vec<f32> = positions.iter().map(|(_, p)| p.x).collect();
    assert!(xs.contains(&0.0));
    assert!(xs.contains(&1.0));
    assert!(xs.contains(&2.0));
}

#[test]
fn write_query_allows_mutation() {
    let mut world = World::new();
    world.register_component::<Position>();
    let e = world.spawn();
    world.insert(e, Position { x: 1.0, y: 2.0 }).unwrap();

    {
        let mut positions = world.write::<Position>().unwrap();
        for (_, mut pos) in positions.iter_mut() {
            pos.x += 10.0;
        }
    }

    assert_eq!(
        world.get::<Position>(e),
        Some(&Position { x: 11.0, y: 2.0 })
    );
}

#[test]
fn double_read_succeeds() {
    let mut world = World::new();
    world.register_component::<Position>();
    let e = world.spawn();
    world.insert(e, Position { x: 0.0, y: 0.0 }).unwrap();

    let _a = world.read::<Position>().unwrap();
    let _b = world.read::<Position>().unwrap();
}

#[test]
#[should_panic(expected = "already borrowed")]
fn read_write_conflict_panics() {
    let mut world = World::new();
    world.register_component::<Position>();
    let e = world.spawn();
    world.insert(e, Position { x: 0.0, y: 0.0 }).unwrap();

    let _r = world.read::<Position>().unwrap();
    let _w = world.write_storage::<Position>().unwrap();
}

#[test]
fn world_query_multi_write() {
    // World::query is the owner-side path for holding several storages at
    // once, now that single-storage write accessors take &mut self (issue #10).
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();
    let e = world.spawn();
    world.insert(e, Position { x: 1.0, y: 2.0 }).unwrap();
    world.insert(e, Health(50)).unwrap();

    {
        let mut q = world.query::<(crate::Write<Position>, crate::Write<Health>)>();
        let (positions, healths) = q.items_mut();
        for (_, mut pos) in positions.iter_mut() {
            pos.x += 10.0;
        }
        for (_, mut hp) in healths.iter_mut() {
            hp.0 += 1;
        }
    }

    assert_eq!(
        world.get::<Position>(e),
        Some(&Position { x: 11.0, y: 2.0 })
    );
    assert_eq!(world.get::<Health>(e), Some(&Health(51)));
}

#[test]
#[should_panic(expected = "more than once")]
fn world_query_aliasing_set_panics() {
    let mut world = World::new();
    world.register_component::<Position>();
    let _ = world.query::<(crate::Write<Position>, crate::Read<Position>)>();
}

#[test]
fn resource_insert_and_get() {
    let mut world = World::new();
    world.insert_resource(42u32);

    let val = world.resource::<u32>();
    assert_eq!(*val, 42);
}

#[test]
fn resource_mut_modify() {
    let mut world = World::new();
    world.insert_resource(42u32);

    {
        let mut val = world.resource_mut::<u32>();
        *val = 99;
    }

    let val = world.resource::<u32>();
    assert_eq!(*val, 99);
}

#[test]
fn parallel_resources_opposite_lock_order_no_deadlock() {
    // Two threads acquire the same two resources in OPPOSITE declaration order.
    // Because `acquire_sorted` takes locks in a global TypeId order, this cannot
    // deadlock. (Without sorted acquisition this test would hang.)
    use crate::query::access::AccessSet;
    use crate::{Res, ResMut};

    struct ResA;
    struct ResB;

    let mut world = World::new();
    world.insert_resource(ResA);
    world.insert_resource(ResB);
    let world = std::sync::Arc::new(world);

    let ab = <(ResMut<ResA>, ResMut<ResB>)>::access_infos();
    let ba = <(ResMut<ResB>, ResMut<ResA>)>::access_infos();
    // A read/write mix in opposite order too.
    let ab_mixed = <(Res<ResA>, ResMut<ResB>)>::access_infos();

    std::thread::scope(|s| {
        let w = world.clone();
        s.spawn(move || {
            for _ in 0..3000 {
                let g = w.acquire_sorted(&ab);
                drop(g);
            }
        });
        let w = world.clone();
        s.spawn(move || {
            for _ in 0..3000 {
                let g = w.acquire_sorted(&ba);
                drop(g);
            }
        });
        let w = world.clone();
        s.spawn(move || {
            for _ in 0..3000 {
                let g = w.acquire_sorted(&ab_mixed);
                drop(g);
            }
        });
    });
    // Reaching here means no deadlock and no panic-on-contention.
}

#[test]
fn entity_recycling_invalidates_components() {
    let mut world = World::new();
    world.register_component::<Position>();
    let old = world.spawn();
    world.insert(old, Position { x: 1.0, y: 2.0 }).unwrap();

    world.despawn(old);
    // No advance_tick(): the per-slot generation alone must make the recycled
    // handle distinct from the old one (ABA-safe within the same tick).
    let new = world.spawn();

    // Same index, different generation.
    assert_eq!(new.index(), old.index());
    assert_ne!(new.spawn_tick(), old.spawn_tick());
    assert_ne!(new, old);

    // The stale handle must report dead and must not see the new entity.
    assert!(!world.is_alive(old));
    assert!(world.is_alive(new));

    // New entity should not have old entity's components
    assert!(world.get::<Position>(new).is_none());
}

#[test]
fn with_filter_in_query() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();

    let e1 = world.spawn();
    world.insert(e1, Position { x: 1.0, y: 0.0 }).unwrap();
    world.insert(e1, Health(100)).unwrap();

    let e2 = world.spawn();
    world.insert(e2, Position { x: 2.0, y: 0.0 }).unwrap();

    let positions = world.read::<Position>().unwrap();
    let has_health = world.with::<Health>();

    let healthy_positions: Vec<f32> = positions
        .iter()
        .filter(|(idx, _)| has_health.matches(*idx))
        .map(|(_, p)| p.x)
        .collect();

    assert_eq!(healthy_positions, vec![1.0]);
}

#[test]
fn without_filter_in_query() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Frozen>();

    let e1 = world.spawn();
    world.insert(e1, Position { x: 1.0, y: 0.0 }).unwrap();
    world.insert(e1, Frozen).unwrap();

    let e2 = world.spawn();
    world.insert(e2, Position { x: 2.0, y: 0.0 }).unwrap();

    let positions = world.read::<Position>().unwrap();
    let not_frozen = world.without::<Frozen>();

    let unfrozen_positions: Vec<f32> = positions
        .iter()
        .filter(|(idx, _)| not_frozen.matches(*idx))
        .map(|(_, p)| p.x)
        .collect();

    assert_eq!(unfrozen_positions, vec![2.0]);
}

#[test]
fn combined_read_iteration() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Velocity>();

    let e1 = world.spawn();
    world.insert(e1, Position { x: 0.0, y: 0.0 }).unwrap();
    world.insert(e1, Velocity { x: 1.0, y: 0.0 }).unwrap();

    let e2 = world.spawn();
    world.insert(e2, Position { x: 5.0, y: 5.0 }).unwrap();
    // e2 has no Velocity

    let positions = world.read::<Position>().unwrap();
    let velocities = world.read::<Velocity>().unwrap();

    let mut count = 0;
    for (idx, _pos) in positions.iter() {
        if velocities.get(idx).is_some() {
            count += 1;
        }
    }
    assert_eq!(count, 1); // Only e1 has both
}

#[test]
fn removed_filter_after_remove() {
    let mut world = World::new();
    world.register_component::<Health>();

    let entity = world.spawn();
    world.insert(entity, Health(100)).unwrap();

    world.advance_tick(); // tick = 1
    let before_remove = world.current_tick();

    world.advance_tick(); // tick = 2
    let _ = world.remove::<Health>(entity);

    let removed = world.removed::<Health>(before_remove);
    assert!(removed.matches(entity.index()));
}

#[test]
fn removed_filter_not_matching_before_tick() {
    let mut world = World::new();
    world.register_component::<Health>();

    let entity = world.spawn();
    world.insert(entity, Health(100)).unwrap();

    world.advance_tick(); // tick = 2 (the world starts at tick 1)
    let _ = world.remove::<Health>(entity); // removed at tick 2

    // Query with since_tick = 2: removal at tick 2 is NOT strictly after 2
    let removed = world.removed::<Health>(2);
    assert!(!removed.matches(entity.index()));

    // Query with since_tick = 1: removal at tick 2 IS strictly after 1
    let removed = world.removed::<Health>(1);
    assert!(removed.matches(entity.index()));
}

#[test]
fn removed_filter_after_despawn() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();
    world.insert(entity, Health(100)).unwrap();

    world.advance_tick(); // tick = 1
    world.despawn(entity);

    // Both components should be tracked as removed
    let removed_pos = world.removed::<Position>(0);
    let removed_health = world.removed::<Health>(0);
    assert!(removed_pos.matches(entity.index()));
    assert!(removed_health.matches(entity.index()));
}

#[test]
fn removed_filter_iter() {
    let mut world = World::new();
    world.register_component::<Health>();

    let e1 = world.spawn();
    let e2 = world.spawn();
    let e3 = world.spawn();
    world.insert(e1, Health(100)).unwrap();
    world.insert(e2, Health(200)).unwrap();
    world.insert(e3, Health(300)).unwrap();

    world.advance_tick(); // tick = 1
    let _ = world.remove::<Health>(e1);
    let _ = world.remove::<Health>(e3);

    let removed = world.removed::<Health>(0);
    let mut entities: Vec<u32> = removed.iter().collect();
    entities.sort();
    assert_eq!(entities, vec![e1.index(), e3.index()]);
}

#[test]
fn clear_removed_tracking_works() {
    let mut world = World::new();
    world.register_component::<Health>();

    let entity = world.spawn();
    world.insert(entity, Health(100)).unwrap();

    world.advance_tick(); // tick = 1
    let _ = world.remove::<Health>(entity);

    assert!(world.removed::<Health>(0).matches(entity.index()));

    world.clear_removed_tracking();

    assert!(!world.removed::<Health>(0).matches(entity.index()));
}

#[test]
fn removed_filter_unregistered_matches_nothing() {
    let world = World::new();
    let removed = world.removed::<Health>(0);
    assert!(!removed.matches(0));
    assert_eq!(removed.iter().count(), 0);
}

#[test]
fn remove_nonexistent_component_not_tracked() {
    let mut world = World::new();
    world.register_component::<Health>();

    let entity = world.spawn();
    // Don't insert Health, just try to remove it
    world.advance_tick();
    let _ = world.remove::<Health>(entity);

    let removed = world.removed::<Health>(0);
    assert!(!removed.matches(entity.index()));
}

// ---- Batch operation tests ----

#[test]
fn spawn_batch_creates_entities() {
    let mut world = World::new();
    let entities = world.spawn_batch(5);

    assert_eq!(entities.len(), 5);
    assert_eq!(world.entity_count(), 5);
    for e in &entities {
        assert!(world.is_alive(*e));
    }
}

#[test]
fn spawn_batch_zero() {
    let mut world = World::new();
    let entities = world.spawn_batch(0);
    assert!(entities.is_empty());
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn spawn_batch_with_inserts_components() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();

    let entities = world
        .spawn_batch_with(3, (Position { x: 1.0, y: 2.0 }, Health(100)))
        .unwrap();

    assert_eq!(entities.len(), 3);
    for e in &entities {
        assert_eq!(
            world.get::<Position>(*e),
            Some(&Position { x: 1.0, y: 2.0 })
        );
        assert_eq!(world.get::<Health>(*e), Some(&Health(100)));
    }
}

#[test]
fn spawn_batch_with_fn_unique_data() {
    let mut world = World::new();
    world.register_component::<Position>();

    let entities = world
        .spawn_batch_with_fn(4, |i| {
            (Position {
                x: i as f32,
                y: (i * 10) as f32,
            },)
        })
        .unwrap();

    assert_eq!(entities.len(), 4);
    for (i, e) in entities.iter().enumerate() {
        assert_eq!(
            world.get::<Position>(*e),
            Some(&Position {
                x: i as f32,
                y: (i * 10) as f32
            })
        );
    }
}

#[test]
fn despawn_batch_removes_all() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();

    let entities = world.spawn_batch(4);
    for e in &entities {
        world.insert(*e, Position { x: 0.0, y: 0.0 }).unwrap();
        world.insert(*e, Health(50)).unwrap();
    }

    world.advance_tick(); // tick = 1
    world.despawn_batch(&entities);

    assert_eq!(world.entity_count(), 0);
    for e in &entities {
        assert!(!world.is_alive(*e));
    }

    // Removal tracking should work
    for e in &entities {
        assert!(world.removed::<Position>(0).matches(e.index()));
        assert!(world.removed::<Health>(0).matches(e.index()));
    }
}

#[test]
fn despawn_batch_skips_dead() {
    let mut world = World::new();
    let entities = world.spawn_batch(3);
    world.despawn(entities[1]); // pre-despawn one

    world.despawn_batch(&entities); // should not panic on already dead entity[1]
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn insert_batch_adds_components() {
    let mut world = World::new();
    world.register_component::<Health>();

    let entities = world.spawn_batch(3);
    let healths = vec![Health(10), Health(20), Health(30)];

    world
        .insert_batch(entities.iter().copied().zip(healths))
        .unwrap();

    assert_eq!(world.get::<Health>(entities[0]), Some(&Health(10)));
    assert_eq!(world.get::<Health>(entities[1]), Some(&Health(20)));
    assert_eq!(world.get::<Health>(entities[2]), Some(&Health(30)));
}

#[test]
fn insert_batch_records_tick() {
    let mut world = World::new();
    world.register_component::<Health>();
    world.advance_tick(); // tick = 1

    let entities = world.spawn_batch(2);
    let healths = vec![Health(10), Health(20)];

    world
        .insert_batch(entities.iter().copied().zip(healths))
        .unwrap();

    assert!(world.added::<Health>(0).matches(entities[0].index()));
    assert!(world.added::<Health>(0).matches(entities[1].index()));
}

#[test]
fn insert_batch_unregistered_returns_err() {
    let mut world = World::new();
    let entities = world.spawn_batch(1);
    let result = world.insert_batch(entities.iter().copied().zip(vec![Health(10)]));
    assert!(result.is_err());
}

#[test]
fn remove_batch_removes_components() {
    let mut world = World::new();
    world.register_component::<Health>();

    let entities = world.spawn_batch(3);
    for (i, e) in entities.iter().enumerate() {
        world.insert(*e, Health(i as u32 * 10)).unwrap();
    }

    world.advance_tick();
    world.remove_batch::<Health>(&entities[0..2]);

    assert!(world.get::<Health>(entities[0]).is_none());
    assert!(world.get::<Health>(entities[1]).is_none());
    assert_eq!(world.get::<Health>(entities[2]), Some(&Health(20)));

    // Removal tracking
    assert!(world.removed::<Health>(0).matches(entities[0].index()));
    assert!(world.removed::<Health>(0).matches(entities[1].index()));
    assert!(!world.removed::<Health>(0).matches(entities[2].index()));
}

#[test]
fn remove_batch_unregistered_no_panic() {
    let mut world = World::new();
    let entities = world.spawn_batch(2);
    // Should not panic when component type is not registered
    world.remove_batch::<Health>(&entities);
}

// ---- Lifecycle hook tests ----

#[derive(Debug, Clone, PartialEq)]
struct Marker(u32);

#[test]
fn on_add_fires_on_first_insert() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();
    world.on_add::<Position>(|world, entity| {
        let _ = world.insert(entity, Marker(1));
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();

    // Marker should have been added by on_add hook
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));
}

#[test]
fn on_add_does_not_fire_on_replace() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();
    world.on_add::<Position>(|world, entity| {
        let _ = world.insert(entity, Marker(1));
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));

    // Remove marker, then replace position — on_add should NOT fire
    let _ = world.remove::<Marker>(entity);
    world.insert(entity, Position { x: 3.0, y: 4.0 }).unwrap();
    assert!(world.get::<Marker>(entity).is_none());
}

#[test]
fn on_insert_fires_on_every_insert() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();
    world.on_insert::<Position>(|world, entity| {
        let count = world.get::<Marker>(entity).map(|m| m.0).unwrap_or(0);
        let _ = world.insert(entity, Marker(count + 1));
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 0.0 }).unwrap();
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));

    world.insert(entity, Position { x: 2.0, y: 0.0 }).unwrap();
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(2)));
}

#[test]
fn on_replace_fires_before_overwrite() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();
    world.on_replace::<Position>(|world, entity| {
        // Read old value and store it in Marker
        if let Some(pos) = world.get::<Position>(entity) {
            let _ = world.insert(entity, Marker(pos.x as u32));
        }
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 10.0, y: 0.0 }).unwrap();
    // on_replace should NOT fire on first insert
    assert!(world.get::<Marker>(entity).is_none());

    world.insert(entity, Position { x: 20.0, y: 0.0 }).unwrap();
    // Hook read old value x=10
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(10)));

    world.insert(entity, Position { x: 30.0, y: 0.0 }).unwrap();
    // Hook read old value x=20
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(20)));
}

#[test]
fn on_remove_fires_before_removal() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();
    world.on_remove::<Position>(|world, entity| {
        // Read component before it's removed
        if let Some(pos) = world.get::<Position>(entity) {
            let _ = world.insert(entity, Marker(pos.x as u32));
        }
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 42.0, y: 0.0 }).unwrap();
    let _ = world.remove::<Position>(entity);

    // Hook stored Position.x in Marker before removal
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(42)));
    assert!(world.get::<Position>(entity).is_none());
}

#[test]
fn on_remove_fires_during_despawn() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.insert_resource(0u32);

    world.on_remove::<Position>(|world, entity| {
        if let Some(pos) = world.get::<Position>(entity) {
            let mut counter = world.resource_mut::<u32>();
            *counter = pos.x as u32;
        }
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 77.0, y: 0.0 }).unwrap();
    world.despawn(entity);

    let val = world.resource::<u32>();
    assert_eq!(*val, 77);
}

#[test]
fn on_remove_fires_during_despawn_batch() {
    let mut world = World::new();
    world.register_component::<Health>();
    world.insert_resource(0u32);

    world.on_remove::<Health>(|world, _entity| {
        let mut counter = world.resource_mut::<u32>();
        *counter += 1;
    });

    let entities = world.spawn_batch(3);
    for e in &entities {
        world.insert(*e, Health(10)).unwrap();
    }
    world.despawn_batch(&entities);

    let count = world.resource::<u32>();
    assert_eq!(*count, 3);
}

#[test]
fn despawn_batch_fires_hooks_in_registration_order() {
    // #43: per entity, `despawn_batch` fires cross-type `on_remove` hooks in
    // registration order, not `HashMap` iteration order. Entities keep their
    // input order.
    let mut world = World::new();
    world.register_component::<Velocity>(); // seq 0
    world.register_component::<Position>(); // seq 1
    world.insert_resource::<Vec<&'static str>>(Vec::new());

    world.on_remove::<Velocity>(|world, _| {
        world.resource_mut::<Vec<&'static str>>().push("velocity");
    });
    world.on_remove::<Position>(|world, _| {
        world.resource_mut::<Vec<&'static str>>().push("position");
    });

    let entities = world.spawn_batch(2);
    for e in &entities {
        // Insert Position before Velocity to prove firing tracks registration.
        world.insert(*e, Position { x: 0.0, y: 0.0 }).unwrap();
        world.insert(*e, Velocity { x: 0.0, y: 0.0 }).unwrap();
    }
    world.despawn_batch(&entities);

    // Each entity fires velocity (seq 0) then position (seq 1), in input order.
    assert_eq!(
        *world.resource::<Vec<&'static str>>(),
        vec!["velocity", "position", "velocity", "position"],
    );
}

#[test]
fn on_remove_entity_still_alive_during_despawn() {
    let mut world = World::new();
    world.register_component::<Health>();
    world.insert_resource(false);

    world.on_remove::<Health>(|world, entity| {
        let mut was_alive = world.resource_mut::<bool>();
        *was_alive = world.is_alive(entity);
    });

    let entity = world.spawn();
    world.insert(entity, Health(1)).unwrap();
    world.despawn(entity);

    let was_alive = world.resource::<bool>();
    assert!(*was_alive);
}

#[test]
fn hooks_fire_during_insert_batch() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();

    world.on_add::<Position>(|world, entity| {
        let _ = world.insert(entity, Marker(1));
    });

    let entities = world.spawn_batch(3);
    let positions = vec![
        Position { x: 1.0, y: 0.0 },
        Position { x: 2.0, y: 0.0 },
        Position { x: 3.0, y: 0.0 },
    ];
    world
        .insert_batch(entities.iter().copied().zip(positions))
        .unwrap();

    for e in &entities {
        assert_eq!(world.get::<Marker>(*e), Some(&Marker(1)));
    }
}

#[test]
fn hooks_fire_during_insert_batch_with_tick_tracking() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();
    world.advance_tick(); // tick = 1

    world.on_add::<Position>(|world, entity| {
        let _ = world.insert(entity, Marker(1));
    });

    let entities = world.spawn_batch(2);
    let positions = vec![Position { x: 1.0, y: 0.0 }, Position { x: 2.0, y: 0.0 }];
    world
        .insert_batch(entities.iter().copied().zip(positions))
        .unwrap();

    for e in &entities {
        assert_eq!(world.get::<Marker>(*e), Some(&Marker(1)));
    }
    // Verify tick tracking works (insert_batch always tracks)
    assert!(world.added::<Position>(0).matches(entities[0].index()));
}

#[test]
fn hooks_fire_during_remove_batch() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();

    world.on_remove::<Position>(|world, entity| {
        if let Some(pos) = world.get::<Position>(entity) {
            let _ = world.insert(entity, Marker(pos.x as u32));
        }
    });

    let entities = world.spawn_batch(3);
    for (i, e) in entities.iter().enumerate() {
        world
            .insert(
                *e,
                Position {
                    x: (i + 1) as f32,
                    y: 0.0,
                },
            )
            .unwrap();
    }
    world.remove_batch::<Position>(&entities);

    assert_eq!(world.get::<Marker>(entities[0]), Some(&Marker(1)));
    assert_eq!(world.get::<Marker>(entities[1]), Some(&Marker(2)));
    assert_eq!(world.get::<Marker>(entities[2]), Some(&Marker(3)));
}

#[test]
fn on_add_required_component_pattern() {
    // Classic use case: inserting A automatically inserts B
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Velocity>();

    world.on_add::<Position>(|world, entity| {
        if world.get::<Velocity>(entity).is_none() {
            let _ = world.insert(entity, Velocity { x: 0.0, y: 0.0 });
        }
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();

    assert_eq!(
        world.get::<Velocity>(entity),
        Some(&Velocity { x: 0.0, y: 0.0 })
    );
}

#[test]
fn multiple_hooks_on_same_component() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();
    world.insert_resource(0u32);

    world.on_add::<Position>(|world, entity| {
        let _ = world.insert(entity, Marker(1));
    });
    world.on_insert::<Position>(|world, _entity| {
        let mut counter = world.resource_mut::<u32>();
        *counter += 1;
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 0.0 }).unwrap();

    // Both hooks should have fired
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));
    assert_eq!(*world.resource::<u32>(), 1);

    // Replace — only on_insert fires, not on_add
    let _ = world.remove::<Marker>(entity);
    world.insert(entity, Position { x: 2.0, y: 0.0 }).unwrap();

    assert!(world.get::<Marker>(entity).is_none());
    assert_eq!(*world.resource::<u32>(), 2);
}

#[test]
fn hooks_via_commands() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Marker>();
    world.on_add::<Position>(|world, entity| {
        let _ = world.insert(entity, Marker(99));
    });

    // Queue insertion via command buffer
    let entity = world.spawn();
    {
        let cmds = world.resource::<crate::commands::CommandBuffer>();
        cmds.push(move |world: &mut World| {
            let _ = world.insert(entity, Position { x: 1.0, y: 0.0 });
        });
    }
    world.apply_commands();

    // Hook should have fired when command was applied
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(99)));
}

#[test]
fn multiple_on_add_hooks_fire_in_order() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.insert_resource::<Vec<u32>>(Vec::new());

    world.on_add::<Position>(|world, _entity| {
        world.resource_mut::<Vec<u32>>().push(1);
    });
    world.on_add::<Position>(|world, _entity| {
        world.resource_mut::<Vec<u32>>().push(2);
    });
    world.on_add::<Position>(|world, _entity| {
        world.resource_mut::<Vec<u32>>().push(3);
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();

    assert_eq!(*world.resource::<Vec<u32>>(), vec![1, 2, 3]);
}

#[test]
fn multiple_on_remove_hooks_fire_in_order() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.insert_resource::<Vec<u32>>(Vec::new());

    world.on_remove::<Position>(|world, _entity| {
        world.resource_mut::<Vec<u32>>().push(10);
    });
    world.on_remove::<Position>(|world, _entity| {
        world.resource_mut::<Vec<u32>>().push(20);
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
    let _ = world.remove::<Position>(entity);

    assert_eq!(*world.resource::<Vec<u32>>(), vec![10, 20]);
}

#[test]
fn no_hooks_batch_fast_path() {
    // Ensure batch operations still work efficiently without hooks
    let mut world = World::new();
    world.register_component::<Health>();

    let entities = world.spawn_batch(100);
    let healths: Vec<Health> = (0..100).map(Health).collect();
    world
        .insert_batch(entities.iter().copied().zip(healths))
        .unwrap();

    for (i, e) in entities.iter().enumerate() {
        assert_eq!(world.get::<Health>(*e), Some(&Health(i as u32)));
    }
}

#[test]
fn despawn_multiple_components_hooks() {
    // Despawn fires on_remove for each component type
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();
    world.insert_resource(0u32);

    world.on_remove::<Position>(|world, _entity| {
        let mut c = world.resource_mut::<u32>();
        *c += 10;
    });
    world.on_remove::<Health>(|world, _entity| {
        let mut c = world.resource_mut::<u32>();
        *c += 1;
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
    world.insert(entity, Health(100)).unwrap();
    world.despawn(entity);

    // Both hooks should have fired
    assert_eq!(*world.resource::<u32>(), 11);
}

// --- Hook cascade / liveness tests (issue #22) ---

#[test]
fn on_replace_despawn_leaves_no_phantom_data() {
    let mut world = World::new();
    world.register_component::<Position>();

    world.on_replace::<Position>(|world, entity| {
        world.despawn(entity);
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 1.0 }).unwrap();

    // The replace fires the hook, which despawns the entity — the write
    // must be aborted instead of landing in the freed slot.
    let result = world.insert(entity, Position { x: 2.0, y: 2.0 });
    assert!(matches!(result, Err(WorldError::EntityNotAlive { .. })));
    assert!(!world.is_alive(entity));

    // A respawn reusing the slot must not inherit phantom data.
    let reused = world.spawn();
    assert_eq!(reused.index(), entity.index());
    assert_eq!(world.get::<Position>(reused), None);
}

#[test]
fn on_remove_despawn_spares_the_slots_new_owner() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.insert_resource::<Option<Entity>>(None);
    world.insert_resource(false); // hook already ran (guards reentrancy)

    // The hook despawns the entity and immediately respawns the slot with
    // its own Position.
    world.on_remove::<Position>(|world, entity| {
        if *world.resource::<bool>() {
            return; // reentrant call from the despawn below
        }
        *world.resource_mut::<bool>() = true;
        world.despawn(entity);
        let new = world.spawn();
        assert_eq!(new.index(), entity.index());
        world.insert(new, Position { x: 9.0, y: 9.0 }).unwrap();
        *world.resource_mut::<Option<Entity>>() = Some(new);
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 1.0 }).unwrap();

    // The hook's cleanup wins; the stale handle must not remove the new
    // owner's component by raw index.
    let removed = world.remove::<Position>(entity).unwrap();
    assert!(removed.is_none());

    let new_owner = world.resource::<Option<Entity>>().unwrap();
    assert_eq!(
        world.get::<Position>(new_owner),
        Some(&Position { x: 9.0, y: 9.0 })
    );
}

#[test]
fn despawn_cascading_removal_fires_hooks_once() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();
    world.insert_resource(0u32); // Health on_remove fire count

    // Position's hook cascades: removes Health explicitly.
    world.on_remove::<Position>(|world, entity| {
        let _ = world.remove::<Health>(entity);
    });
    world.on_remove::<Health>(|world, _entity| {
        *world.resource_mut::<u32>() += 1;
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
    world.insert(entity, Health(100)).unwrap();
    world.despawn(entity);

    // Health's on_remove fired exactly once (inside the cascade), not a
    // second time from despawn's own snapshot.
    assert_eq!(*world.resource::<u32>(), 1);
}

#[test]
fn despawn_fires_hooks_in_registration_order() {
    // #43: the cross-type `on_remove` firing order must be deterministic
    // (component registration order), not `HashMap` iteration order. Insert
    // order deliberately differs from registration order to prove the firing
    // order tracks registration, not insertion or hashing.
    let mut world = World::new();
    world.register_component::<Velocity>(); // seq 0
    world.register_component::<Position>(); // seq 1
    world.register_component::<Health>(); // seq 2
    world.insert_resource::<Vec<&'static str>>(Vec::new());

    world.on_remove::<Velocity>(|world, _| {
        world.resource_mut::<Vec<&'static str>>().push("velocity");
    });
    world.on_remove::<Position>(|world, _| {
        world.resource_mut::<Vec<&'static str>>().push("position");
    });
    world.on_remove::<Health>(|world, _| {
        world.resource_mut::<Vec<&'static str>>().push("health");
    });

    let entity = world.spawn();
    world.insert(entity, Health(1)).unwrap();
    world.insert(entity, Velocity { x: 0.0, y: 0.0 }).unwrap();
    world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
    world.despawn(entity);

    assert_eq!(
        *world.resource::<Vec<&'static str>>(),
        vec!["velocity", "position", "health"],
    );
}

#[test]
fn despawn_hook_sibling_visibility_follows_registration_order() {
    // #43: with deterministic registration-order firing plus immediate
    // removal after each type's hooks, an `on_remove` hook sees sibling
    // hooked components registered *after* it still present, and those
    // registered *before* it already removed.
    let mut world = World::new();
    world.register_component::<Velocity>(); // seq 0, fires first
    world.register_component::<Health>(); // seq 1, fires second
    world.insert_resource::<Vec<(&'static str, bool)>>(Vec::new());

    // Velocity (earlier) checks whether Health (later) is still present.
    world.on_remove::<Velocity>(|world, entity| {
        let saw = world.get::<Health>(entity).is_some();
        world
            .resource_mut::<Vec<(&'static str, bool)>>()
            .push(("velocity_saw_health", saw));
    });
    // Health (later) checks whether Velocity (earlier) is still present.
    world.on_remove::<Health>(|world, entity| {
        let saw = world.get::<Velocity>(entity).is_some();
        world
            .resource_mut::<Vec<(&'static str, bool)>>()
            .push(("health_saw_velocity", saw));
    });

    let entity = world.spawn();
    world.insert(entity, Velocity { x: 0.0, y: 0.0 }).unwrap();
    world.insert(entity, Health(1)).unwrap();
    world.despawn(entity);

    assert_eq!(
        *world.resource::<Vec<(&'static str, bool)>>(),
        vec![
            ("velocity_saw_health", true),
            ("health_saw_velocity", false),
        ],
    );
}

#[test]
fn remove_batch_skips_stale_handles() {
    let mut world = World::new();
    world.register_component::<Position>();

    let old = world.spawn();
    world.insert(old, Position { x: 1.0, y: 1.0 }).unwrap();
    world.despawn(old);

    // The slot now belongs to someone else.
    let new = world.spawn();
    assert_eq!(new.index(), old.index());
    world.insert(new, Position { x: 5.0, y: 5.0 }).unwrap();

    // A stale handle in the batch must not strip the new owner.
    world.remove_batch::<Position>(&[old]);
    assert_eq!(
        world.get::<Position>(new),
        Some(&Position { x: 5.0, y: 5.0 })
    );
}

#[test]
fn on_replace_removing_component_makes_insert_a_fresh_add() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.insert_resource(0u32); // on_add fire count

    world.on_add::<Position>(|world, _entity| {
        *world.resource_mut::<u32>() += 1;
    });
    world.on_replace::<Position>(|world, entity| {
        let _ = world.remove::<Position>(entity);
    });

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 1.0 }).unwrap();
    assert_eq!(*world.resource::<u32>(), 1);

    // The replace hook removed the old value, so this counts as a fresh
    // add again (on_add fires), not a replacement.
    world.insert(entity, Position { x: 2.0, y: 2.0 }).unwrap();
    assert_eq!(*world.resource::<u32>(), 2);
    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 2.0, y: 2.0 })
    );
}

#[test]
fn deferred_insert_after_despawn_is_skipped_not_a_panic() {
    let mut world = World::new();
    world.register_component::<Health>();

    let entity = world.spawn();

    // Two systems race within one frame: one queues a despawn, another
    // queues an insert on the same entity.
    {
        let commands = world.resource::<crate::commands::CommandBuffer>();
        commands.despawn(entity);
        commands.insert(entity, Health(1));
        commands.insert_batch(vec![(entity, Health(2))]);
    }
    world.apply_commands(); // must not panic

    assert!(!world.is_alive(entity));
}

// --- Required components tests ---

#[derive(Debug, Clone, PartialEq, Default)]
struct ReqA(u32);
#[derive(Debug, Clone, PartialEq, Default)]
struct ReqB(u32);
#[derive(Debug, Clone, PartialEq, Default)]
struct ReqC(u32);

#[test]
fn required_component_inserted_automatically() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_required::<ReqA, ReqB>();

    let entity = world.spawn();
    world.insert(entity, ReqA(1)).unwrap();

    assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
}

#[test]
fn required_component_not_overwritten() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_component::<ReqB>();
    world.register_required::<ReqA, ReqB>();

    let entity = world.spawn();
    world.insert(entity, ReqB(42)).unwrap();
    world.insert(entity, ReqA(1)).unwrap();

    // Existing ReqB should NOT be overwritten
    assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(42)));
}

#[test]
fn required_component_not_applied_on_replace() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_required::<ReqA, ReqB>();

    let entity = world.spawn();
    world.insert(entity, ReqA(1)).unwrap();
    assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));

    // Remove ReqB, then replace ReqA — requirements should NOT fire again
    let _ = world.remove::<ReqB>(entity);
    world.insert(entity, ReqA(2)).unwrap();
    assert!(world.get::<ReqB>(entity).is_none());
}

#[test]
fn transitive_requirements() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_required::<ReqA, ReqB>();
    world.register_required::<ReqB, ReqC>();

    let entity = world.spawn();
    world.insert(entity, ReqA(1)).unwrap();

    assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
    assert_eq!(world.get::<ReqC>(entity), Some(&ReqC(0)));
}

#[test]
fn required_components_coexist_with_on_add_hook() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_component::<Marker>();
    world.register_required::<ReqA, ReqB>();

    world.on_add::<ReqA>(|world, entity| {
        let _ = world.insert(entity, Marker(99));
    });

    let entity = world.spawn();
    world.insert(entity, ReqA(1)).unwrap();

    assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(99)));
}

#[test]
fn required_component_auto_registers() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    // Don't manually register ReqB — register_required should do it
    world.register_required::<ReqA, ReqB>();

    let entity = world.spawn();
    world.insert(entity, ReqA(1)).unwrap();
    assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
}

#[test]
fn required_components_in_batch_insert() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_required::<ReqA, ReqB>();

    let entities = world.spawn_batch(3);
    let components = vec![ReqA(1), ReqA(2), ReqA(3)];
    world
        .insert_batch(entities.iter().copied().zip(components))
        .unwrap();

    for e in &entities {
        assert_eq!(world.get::<ReqB>(*e), Some(&ReqB(0)));
    }
}

#[test]
fn required_components_in_batch_tracked() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_required::<ReqA, ReqB>();

    let entities = world.spawn_batch(2);
    let components = vec![ReqA(1), ReqA(2)];
    world
        .insert_batch(entities.iter().copied().zip(components))
        .unwrap();

    for e in &entities {
        assert_eq!(world.get::<ReqB>(*e), Some(&ReqB(0)));
    }
}

#[test]
fn required_components_via_bundle() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_component::<Health>();
    world.register_required::<ReqA, ReqB>();

    let entity = world.spawn_with((ReqA(1), Health(100))).unwrap();
    assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
}

#[test]
fn multiple_required_components() {
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_required::<ReqA, ReqB>();
    world.register_required::<ReqA, ReqC>();

    let entity = world.spawn();
    world.insert(entity, ReqA(1)).unwrap();

    assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
    assert_eq!(world.get::<ReqC>(entity), Some(&ReqC(0)));
}

// --- Entity collect/remap tests ---

#[test]
fn collect_entities_from_parent() {
    use crate::std::components::{Children, Parent};

    let mut world = World::new();
    world.register_inspector::<Parent>();
    world.register_inspector_default::<Children>();

    let parent = world.spawn();
    let child = world.spawn();
    world.insert(child, Parent(parent)).unwrap();

    let mut collected = Vec::new();
    world.collect_entities_by_name(child, "Parent", &mut collected);
    assert_eq!(collected, vec![parent]);
}

#[test]
fn collect_entities_from_children() {
    use crate::std::components::{Children, Parent};

    let mut world = World::new();
    world.register_inspector::<Parent>();
    world.register_inspector_default::<Children>();

    let parent = world.spawn();
    let c1 = world.spawn();
    let c2 = world.spawn();
    world.insert(parent, Children(vec![c1, c2])).unwrap();

    let mut collected = Vec::new();
    world.collect_entities_by_name(parent, "Children", &mut collected);
    assert_eq!(collected, vec![c1, c2]);
}

#[test]
fn remap_entities_in_parent() {
    use crate::std::components::{Children, Parent};

    let mut world = World::new();
    world.register_inspector::<Parent>();
    world.register_inspector_default::<Children>();

    let old_parent = world.spawn();
    let new_parent = world.spawn();
    let child = world.spawn();
    world.insert(child, Parent(old_parent)).unwrap();

    world.remap_entities_by_name(child, "Parent", &mut |e| {
        if e == old_parent { new_parent } else { e }
    });

    assert_eq!(world.get::<Parent>(child), Some(&Parent(new_parent)));
}

#[test]
fn remap_entities_in_children() {
    use crate::std::components::{Children, Parent};

    let mut world = World::new();
    world.register_inspector::<Parent>();
    world.register_inspector_default::<Children>();

    let parent = world.spawn();
    let old_c1 = world.spawn();
    let old_c2 = world.spawn();
    let new_c1 = world.spawn();
    let new_c2 = world.spawn();
    world
        .insert(parent, Children(vec![old_c1, old_c2]))
        .unwrap();

    world.remap_entities_by_name(parent, "Children", &mut |e| {
        if e == old_c1 {
            new_c1
        } else if e == old_c2 {
            new_c2
        } else {
            e
        }
    });

    assert_eq!(
        world.get::<Children>(parent),
        Some(&Children(vec![new_c1, new_c2]))
    );
}

#[test]
fn collect_all_entities_gathers_from_all_components() {
    use crate::std::components::{Children, Parent};

    let mut world = World::new();
    world.register_inspector::<Parent>();
    world.register_inspector_default::<Children>();

    let parent = world.spawn();
    let c1 = world.spawn();
    let c2 = world.spawn();
    // Entity has both Parent (pointing at parent) and Children (containing c1, c2)
    let entity = world.spawn();
    world.insert(entity, Parent(parent)).unwrap();
    world.insert(entity, Children(vec![c1, c2])).unwrap();

    let mut collected = Vec::new();
    world.collect_all_entities(entity, &mut collected);
    assert_eq!(collected.len(), 3);
    assert!(collected.contains(&parent));
    assert!(collected.contains(&c1));
    assert!(collected.contains(&c2));
}

#[test]
fn collect_noop_for_non_entity_component() {
    let mut world = World::new();
    world.register_inspector_default::<crate::std::components::Transform>();

    let entity = world.spawn();
    world
        .insert(entity, crate::std::components::Transform::IDENTITY)
        .unwrap();

    let mut collected = Vec::new();
    world.collect_entities_by_name(entity, "Transform", &mut collected);
    assert!(collected.is_empty());
}

// --- Clone entity tests ---

#[test]
fn clone_entity_copies_components() {
    let mut world = World::new();
    crate::register_std_components(&mut world);

    let src = world.spawn();
    let t = crate::std::components::Transform::from_translation(redlilium_core::math::Vec3::new(
        1.0, 2.0, 3.0,
    ));
    world.insert(src, t).unwrap();
    world
        .insert(src, crate::std::components::Name::new("original"))
        .unwrap();

    let dst = world.clone_entity(src).unwrap();

    assert_ne!(src, dst);
    assert_eq!(
        world.get::<crate::std::components::Transform>(dst),
        Some(&t)
    );
    assert_eq!(
        world
            .get::<crate::std::components::Name>(dst)
            .map(|n| n.as_str()),
        Some("original"),
    );
}

#[test]
fn clone_entity_dead_source_returns_none() {
    let mut world = World::new();
    crate::register_std_components(&mut world);

    let src = world.spawn();
    world.despawn(src);

    assert!(world.clone_entity(src).is_none());
}

#[test]
fn clone_entity_tree_flat() {
    let mut world = World::new();
    crate::register_std_components(&mut world);

    let parent = world.spawn();
    world
        .insert(parent, crate::std::components::Name::new("parent"))
        .unwrap();
    world
        .insert(parent, crate::std::components::Transform::IDENTITY)
        .unwrap();

    let child_a = world.spawn();
    world
        .insert(child_a, crate::std::components::Name::new("child_a"))
        .unwrap();
    crate::std::hierarchy::set_parent(&mut world, child_a, parent);

    let child_b = world.spawn();
    world
        .insert(child_b, crate::std::components::Name::new("child_b"))
        .unwrap();
    crate::std::hierarchy::set_parent(&mut world, child_b, parent);

    // 3 original + 3 cloned = 6
    let entity_count_before = world.entity_count();
    let mapping = world.clone_entity_tree(parent);
    assert_eq!(mapping.len(), 3);
    assert_eq!(world.entity_count(), entity_count_before + 3);

    let new_parent = mapping[&parent];
    let new_child_a = mapping[&child_a];
    let new_child_b = mapping[&child_b];

    // Verify component data cloned
    assert_eq!(
        world
            .get::<crate::std::components::Name>(new_parent)
            .map(|n| n.as_str()),
        Some("parent"),
    );
    assert_eq!(
        world
            .get::<crate::std::components::Name>(new_child_a)
            .map(|n| n.as_str()),
        Some("child_a"),
    );

    // Verify hierarchy remapped
    let children = world.get::<crate::Children>(new_parent).unwrap();
    assert_eq!(children.0, vec![new_child_a, new_child_b]);

    let parent_of_a = world.get::<crate::Parent>(new_child_a).unwrap();
    assert_eq!(parent_of_a.0, new_parent);

    let parent_of_b = world.get::<crate::Parent>(new_child_b).unwrap();
    assert_eq!(parent_of_b.0, new_parent);

    // Cloned root should have no parent (original didn't)
    assert!(world.get::<crate::Parent>(new_parent).is_none());
}

#[test]
fn clone_entity_tree_deep() {
    let mut world = World::new();
    crate::register_std_components(&mut world);

    // root -> mid -> leaf
    let root = world.spawn();
    world
        .insert(root, crate::std::components::Name::new("root"))
        .unwrap();

    let mid = world.spawn();
    world
        .insert(mid, crate::std::components::Name::new("mid"))
        .unwrap();
    crate::std::hierarchy::set_parent(&mut world, mid, root);

    let leaf = world.spawn();
    world
        .insert(leaf, crate::std::components::Name::new("leaf"))
        .unwrap();
    crate::std::hierarchy::set_parent(&mut world, leaf, mid);

    let mapping = world.clone_entity_tree(root);
    assert_eq!(mapping.len(), 3);

    let new_root = mapping[&root];
    let new_mid = mapping[&mid];
    let new_leaf = mapping[&leaf];

    // root -> mid
    let root_children = world.get::<crate::Children>(new_root).unwrap();
    assert_eq!(root_children.0, vec![new_mid]);

    // mid -> leaf
    let mid_children = world.get::<crate::Children>(new_mid).unwrap();
    assert_eq!(mid_children.0, vec![new_leaf]);

    // leaf has parent = mid
    assert_eq!(world.get::<crate::Parent>(new_leaf).unwrap().0, new_mid);

    // mid has parent = root
    assert_eq!(world.get::<crate::Parent>(new_mid).unwrap().0, new_root);

    // root has no parent
    assert!(world.get::<crate::Parent>(new_root).is_none());
}

#[test]
fn clone_entity_tree_dead_root_returns_empty() {
    let mut world = World::new();
    crate::register_std_components(&mut world);

    let root = world.spawn();
    world.despawn(root);

    let mapping = world.clone_entity_tree(root);
    assert!(mapping.is_empty());
}

// --- Prefab extract + instantiate tests ---

#[test]
fn extract_and_instantiate_single_entity() {
    let mut world = World::new();
    crate::register_std_components(&mut world);

    let src = world.spawn();
    let t = crate::std::components::Transform::from_translation(redlilium_core::math::Vec3::new(
        5.0, 6.0, 7.0,
    ));
    world.insert(src, t).unwrap();
    world
        .insert(src, crate::std::components::Name::new("prefab_src"))
        .unwrap();

    let prefab = world.extract_prefab(src);
    assert_eq!(prefab.entity_count(), 1);

    let spawned = prefab.instantiate(&mut world);
    assert_eq!(spawned.len(), 1);

    let dst = spawned[0];
    assert_ne!(src, dst);
    assert_eq!(
        world.get::<crate::std::components::Transform>(dst),
        Some(&t)
    );
    assert_eq!(
        world
            .get::<crate::std::components::Name>(dst)
            .map(|n| n.as_str()),
        Some("prefab_src"),
    );
}

#[test]
fn extract_and_instantiate_tree_with_hierarchy() {
    let mut world = World::new();
    crate::register_std_components(&mut world);

    // Build: parent -> child_a, child_b
    let parent = world.spawn();
    world
        .insert(parent, crate::std::components::Name::new("parent"))
        .unwrap();
    let child_a = world.spawn();
    world
        .insert(child_a, crate::std::components::Name::new("child_a"))
        .unwrap();
    crate::std::hierarchy::set_parent(&mut world, child_a, parent);
    let child_b = world.spawn();
    world
        .insert(child_b, crate::std::components::Name::new("child_b"))
        .unwrap();
    crate::std::hierarchy::set_parent(&mut world, child_b, parent);

    let prefab = world.extract_prefab(parent);
    assert_eq!(prefab.entity_count(), 3);

    let spawned = prefab.instantiate(&mut world);
    assert_eq!(spawned.len(), 3);

    let new_parent = spawned[0];
    let new_child_a = spawned[1];
    let new_child_b = spawned[2];

    // Verify hierarchy is remapped
    let children = world.get::<crate::Children>(new_parent).unwrap();
    assert_eq!(children.0, vec![new_child_a, new_child_b]);

    assert_eq!(
        world.get::<crate::Parent>(new_child_a).unwrap().0,
        new_parent,
    );
    assert_eq!(
        world.get::<crate::Parent>(new_child_b).unwrap().0,
        new_parent,
    );

    // Root has no parent
    assert!(world.get::<crate::Parent>(new_parent).is_none());

    // Component data cloned
    assert_eq!(
        world
            .get::<crate::std::components::Name>(new_parent)
            .map(|n| n.as_str()),
        Some("parent"),
    );
}

#[test]
fn prefab_instantiate_multiple_times() {
    let mut world = World::new();
    crate::register_std_components(&mut world);

    let src = world.spawn();
    world
        .insert(src, crate::std::components::Name::new("template"))
        .unwrap();

    let prefab = world.extract_prefab(src);

    let a = prefab.instantiate(&mut world);
    let b = prefab.instantiate(&mut world);

    assert_ne!(a[0], b[0]);
    assert_eq!(
        world
            .get::<crate::std::components::Name>(a[0])
            .map(|n| n.as_str()),
        Some("template"),
    );
    assert_eq!(
        world
            .get::<crate::std::components::Name>(b[0])
            .map(|n| n.as_str()),
        Some("template"),
    );
}

#[test]
fn prefab_cross_world() {
    let mut world_a = World::new();
    crate::register_std_components(&mut world_a);

    let src = world_a.spawn();
    world_a
        .insert(src, crate::std::components::Name::new("cross"))
        .unwrap();
    world_a
        .insert(
            src,
            crate::std::components::Transform::from_translation(redlilium_core::math::Vec3::new(
                1.0, 2.0, 3.0,
            )),
        )
        .unwrap();

    let prefab = world_a.extract_prefab(src);

    // Instantiate into a completely different world
    let mut world_b = World::new();
    crate::register_std_components(&mut world_b);

    let spawned = prefab.instantiate(&mut world_b);
    assert_eq!(spawned.len(), 1);
    assert_eq!(
        world_b
            .get::<crate::std::components::Name>(spawned[0])
            .map(|n| n.as_str()),
        Some("cross"),
    );
    let t = world_b
        .get::<crate::std::components::Transform>(spawned[0])
        .unwrap();
    assert!((t.translation - redlilium_core::math::Vec3::new(1.0, 2.0, 3.0)).norm() < 1e-6);
}

#[test]
fn extract_prefab_dead_root_returns_empty() {
    let mut world = World::new();
    crate::register_std_components(&mut world);

    let root = world.spawn();
    world.despawn(root);

    let prefab = world.extract_prefab(root);
    assert!(prefab.is_empty());
}

// --- Bundle hook ordering tests ---

#[test]
fn bundle_hooks_see_all_components() {
    // When inserting (A, B) as a bundle, A's on_add hook should see B.
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();
    world.register_component::<Marker>();

    world.on_add::<Position>(|world, entity| {
        // Health should already be present because the bundle inserts both
        // before firing any hooks.
        let has_health = world.get::<Health>(entity).is_some();
        let _ = world.insert(entity, Marker(has_health as u32));
    });

    let entity = world.spawn();
    world
        .insert_bundle(entity, (Position { x: 1.0, y: 2.0 }, Health(100)))
        .unwrap();

    // Marker(1) means the hook saw Health present
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));
}

#[test]
fn bundle_required_not_overwritten_by_default() {
    // If bundle contains (A, B) and A requires B, the bundle's B should win
    // over the default B inserted by the required-component machinery.
    let mut world = World::new();
    world.register_component::<ReqA>();
    world.register_component::<ReqB>();
    world.register_required::<ReqA, ReqB>();

    let entity = world.spawn();
    world.insert_bundle(entity, (ReqA(1), ReqB(42))).unwrap();

    // ReqB should be 42 (from bundle), not 0 (from required default)
    assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(42)));
}

// --- Transaction cleanup tests ---

#[test]
fn spawn_with_unregistered_rolls_back_entity() {
    let mut world = World::new();
    world.register_component::<Position>();
    // Health is NOT registered

    let count_before = world.entity_count();
    let result = world.spawn_with((Position { x: 1.0, y: 2.0 }, Health(100)));

    assert!(result.is_err());
    // Entity should not exist — transaction cleaned up the spawn
    assert_eq!(world.entity_count(), count_before);
}

#[test]
fn spawn_with_success_commits() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();

    let entity = world
        .spawn_with((Position { x: 1.0, y: 2.0 }, Health(100)))
        .unwrap();

    assert!(world.is_alive(entity));
    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 1.0, y: 2.0 })
    );
    assert_eq!(world.get::<Health>(entity), Some(&Health(100)));
}

#[test]
fn insert_bundle_unregistered_rolls_back_components() {
    let mut world = World::new();
    world.register_component::<Position>();
    // Health is NOT registered

    let entity = world.spawn();
    let result = world.insert_bundle(entity, (Position { x: 1.0, y: 2.0 }, Health(100)));

    assert!(result.is_err());
    // Position should not have been inserted — transaction rolled back
    assert!(world.get::<Position>(entity).is_none());
}

#[test]
fn spawn_batch_with_partial_failure_rolls_back_all() {
    let mut world = World::new();
    world.register_component::<Position>();
    // Health NOT registered

    let count_before = world.entity_count();
    let result = world.spawn_batch_with(3, (Position { x: 1.0, y: 0.0 }, Health(100)));

    assert!(result.is_err());
    // All entities should be rolled back
    assert_eq!(world.entity_count(), count_before);
}

#[test]
fn transaction_hooks_fire_after_all_mutations() {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Health>();
    world.register_component::<Marker>();

    world.on_add::<Position>(|world, entity| {
        let has_health = world.get::<Health>(entity).is_some();
        let _ = world.insert(entity, Marker(has_health as u32));
    });

    let entity = world
        .spawn_with((Position { x: 1.0, y: 2.0 }, Health(100)))
        .unwrap();

    // Marker(1) means the hook saw Health present at commit time
    assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));
}

#[test]
fn insert_bundle_replaces_and_rolls_back() {
    let mut world = World::new();
    world.register_component::<Position>();
    // Health NOT registered

    let entity = world.spawn();
    world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();

    // Try to insert bundle that replaces Position and adds Health (unregistered)
    let result = world.insert_bundle(entity, (Position { x: 99.0, y: 99.0 }, Health(100)));
    assert!(result.is_err());

    // Position should be restored to original value
    assert_eq!(
        world.get::<Position>(entity),
        Some(&Position { x: 1.0, y: 2.0 })
    );
}

// ---- Entity visibility mask tests ----

#[test]
fn is_excluded_from_game_checks_disabled() {
    let mut world = World::new();
    let entity = world.spawn();

    assert!(!world.is_excluded_from_game(entity));

    world.set_entity_flags(entity, Entity::DISABLED);
    assert!(world.is_excluded_from_game(entity));
}

#[test]
fn is_excluded_from_game_checks_static() {
    let mut world = World::new();
    let entity = world.spawn();

    assert!(!world.is_excluded_from_game(entity));

    world.set_entity_flags(entity, Entity::STATIC);
    assert!(world.is_excluded_from_game(entity));
}

#[test]
fn is_excluded_from_game_checks_editor() {
    let mut world = World::new();
    let entity = world.spawn();

    assert!(!world.is_excluded_from_game(entity));

    world.set_entity_flags(entity, Entity::EDITOR);
    assert!(world.is_excluded_from_game(entity));
}

#[test]
fn is_excluded_from_game_checks_all_masks() {
    let mut world = World::new();
    let e_disabled = world.spawn();
    let e_static = world.spawn();
    let e_editor = world.spawn();
    let e_normal = world.spawn();

    world.set_entity_flags(e_disabled, Entity::DISABLED);
    world.set_entity_flags(e_static, Entity::STATIC);
    world.set_entity_flags(e_editor, Entity::EDITOR);

    assert!(world.is_excluded_from_game(e_disabled));
    assert!(world.is_excluded_from_game(e_static));
    assert!(world.is_excluded_from_game(e_editor));
    assert!(!world.is_excluded_from_game(e_normal));
}

// Phase 6: Schema hash validation tests

#[derive(Clone, Debug)]
struct TestComponentWithHash;

impl crate::Component for TestComponentWithHash {
    const NAME: &'static str = "TestComponentWithHash";

    fn schema_hash() -> String {
        "test_schema_hash_v1".to_string()
    }

    fn inspect_ui(
        &self,
        _ui: &mut egui::Ui,
        _world: &crate::World,
        _entity: crate::Entity,
    ) -> crate::InspectResult {
        None
    }

    fn serialize_component(
        &self,
        _ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<crate::serialize::Value, crate::serialize::SerializeError> {
        Ok(crate::serialize::Value::Null)
    }

    fn deserialize_component(
        _ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        Ok(TestComponentWithHash)
    }
}

#[test]
fn schema_hash_stored_in_snapshot_metadata() {
    let mut world = World::new();
    world.register_inspector::<TestComponentWithHash>();
    let entity = world.spawn();
    world.insert(entity, TestComponentWithHash).unwrap();

    let snapshot = world.serialize_world().expect("serialize world");

    // Verify: metadata contains schema hash for TestComponentWithHash
    assert!(
        snapshot
            .metadata
            .component_schemas
            .contains_key("TestComponentWithHash")
    );
    let hash = snapshot
        .metadata
        .component_schemas
        .get("TestComponentWithHash")
        .unwrap();
    assert_eq!(hash, "test_schema_hash_v1");
}

#[test]
fn schema_hash_consistent_across_serialize() {
    let mut world = World::new();
    world.register_inspector::<TestComponentWithHash>();
    let entity = world.spawn();
    world.insert(entity, TestComponentWithHash).unwrap();

    // Serialize twice
    let snap1 = world.serialize_world().expect("serialize 1");
    let snap2 = world.serialize_world().expect("serialize 2");

    // Verify: same hash both times
    let hash1 = snap1
        .metadata
        .component_schemas
        .get("TestComponentWithHash");
    let hash2 = snap2
        .metadata
        .component_schemas
        .get("TestComponentWithHash");
    assert_eq!(hash1, hash2);
}

// Phase 6, Step 3 (re-scoped by ADR-037): schema drift is detected but
// TOLERATED — name-keyed restore proceeds (added fields default, removed
// fields drop) with a warning, because cross-image scene seeding (editing
// world → play world under a rebuilt game) depends on that tolerance.

#[derive(Clone, Debug)]
struct ReorderedFieldComponent;

impl crate::Component for ReorderedFieldComponent {
    const NAME: &'static str = "ReorderedFieldComponent";

    // Version 1: hash representing fields [a, b, c]
    fn schema_hash() -> String {
        "field_order_v1_abc".to_string()
    }

    fn inspect_ui(
        &self,
        _ui: &mut egui::Ui,
        _world: &crate::World,
        _entity: crate::Entity,
    ) -> crate::InspectResult {
        None
    }

    fn serialize_component(
        &self,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<crate::serialize::Value, crate::serialize::SerializeError> {
        ctx.begin_struct(Self::NAME)?;
        ctx.end_struct()
    }

    fn deserialize_component(
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.begin_struct(Self::NAME)?;
        ctx.end_struct()?;
        Ok(ReorderedFieldComponent)
    }
}

#[derive(Clone, Debug)]
struct ReorderedFieldComponentV2;

impl crate::Component for ReorderedFieldComponentV2 {
    const NAME: &'static str = "ReorderedFieldComponent";

    // Version 2: DIFFERENT hash (fields reordered: [b, c, a])
    fn schema_hash() -> String {
        "field_order_v2_bca".to_string()
    }

    fn inspect_ui(
        &self,
        _ui: &mut egui::Ui,
        _world: &crate::World,
        _entity: crate::Entity,
    ) -> crate::InspectResult {
        None
    }

    fn serialize_component(
        &self,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<crate::serialize::Value, crate::serialize::SerializeError> {
        ctx.begin_struct(Self::NAME)?;
        ctx.end_struct()
    }

    fn deserialize_component(
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.begin_struct(Self::NAME)?;
        ctx.end_struct()?;
        Ok(ReorderedFieldComponentV2)
    }
}

#[test]
fn schema_drift_is_tolerated_on_restore() {
    // Setup: Snapshot with ReorderedFieldComponent (version 1 hash)
    let mut world1 = World::new();
    world1.register_inspector::<ReorderedFieldComponent>();
    let entity = world1.spawn();
    world1.insert(entity, ReorderedFieldComponent).unwrap();

    let snapshot = world1.serialize_world().expect("serialize");

    // Verify: snapshot carries the v1 hash.
    let stored_hash = snapshot
        .metadata
        .component_schemas
        .get("ReorderedFieldComponent")
        .unwrap();
    assert_eq!(stored_hash, "field_order_v1_abc");

    // Act: restore into a world whose component has a DIFFERENT hash.
    let mut world2 = World::new();
    world2.register_inspector::<ReorderedFieldComponentV2>();

    let restored = world2
        .deserialize_world_into(&snapshot)
        .expect("drifted schema restores tolerantly (with a warning), not an error");

    // The component deserialized by name despite the drift.
    assert_eq!(restored.len(), 1);
    assert!(
        world2
            .get::<ReorderedFieldComponentV2>(restored[0])
            .is_some(),
        "component restored by name under the drifted schema"
    );
}
