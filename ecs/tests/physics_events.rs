#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_core::math::Vec3;
use redlilium_ecs::physics::{
    CollisionEventSettings, CollisionPhase as Phase, CollisionStopReason as Reason, SensorSettings,
    TeleportVelocity,
};
use redlilium_ecs::*;

macro_rules! event_tests {
    () => {
        fn setup(tracking: bool, registered: bool, regular: bool) -> (World, Entity, Entity) {
            let mut w = World::new();
            register_std_components(&mut w);
            w.insert_resource(Physics::default());
            w.resource_mut::<Physics>().gravity = Default::default();
            w.resource_mut::<Physics>().integration_parameters.dt = 0.01;
            if registered {
                w.add_event::<Collision>();
            }
            let a = w
                .spawn_with((
                    Transform::IDENTITY,
                    Body::fixed(),
                    Collider::ball(1.0)
                        .with_sensor(Some(SensorSettings::default()))
                        .with_collision_events(tracking.then(CollisionEventSettings::default)),
                ))
                .unwrap();
            let b = w
                .spawn_with((
                    Transform::from_translation(Vec3::new(0.8, 0.0, 0.0)),
                    Body::dynamic(),
                    Collider::ball(0.5),
                ))
                .unwrap();
            sync(&mut w, regular);
            (w, a, b)
        }
        fn sync(w: &mut World, regular: bool) {
            let mut systems = SystemsContainer::new();
            if regular {
                systems.add(SyncRegular);
            } else {
                systems.add_exclusive(Sync);
            }
            assert!(EcsRunner::multi_thread(2).run(w, &systems).is_empty());
        }
        fn step(w: &mut World) -> Vec<SystemError> {
            let mut systems = SystemsContainer::new();
            systems.add(Step);
            EcsRunner::multi_thread(2).run(w, &systems)
        }
        fn advance(w: &mut World) {
            let errors = step(w);
            assert!(errors.is_empty(), "{errors:?}");
        }
        fn read(w: &World, cursor: &EventCursor<Collision>) -> Vec<Collision> {
            w.resource::<Events<Collision>>()
                .read(cursor)
                .copied()
                .collect()
        }
        fn phases(events: &[Collision]) -> Vec<Phase> {
            events.iter().map(|e| e.phase).collect()
        }
        fn move_to(w: &mut World, entity: Entity, x: f32) {
            let mut pose = Pose::default();
            pose.translation.x = x;
            w.resource_mut::<Physics>()
                .teleport(entity, pose, TeleportVelocity::Reset)
                .unwrap();
        }

        #[test]
        fn collision_type_matrix_covers_both_kinematic_modes_and_either_side() {
            use physics::CollisionTypes as Types;
            let options = [
                None,
                Some(Types::none()),
                Some(Types::all()),
                Some(Types {
                    dynamic_dynamic: true,
                    ..Types::none()
                }),
                Some(Types {
                    dynamic_kinematic: true,
                    ..Types::none()
                }),
                Some(Types {
                    dynamic_fixed: true,
                    ..Types::none()
                }),
                Some(Types {
                    kinematic_kinematic: true,
                    ..Types::none()
                }),
                Some(Types {
                    kinematic_fixed: true,
                    ..Types::none()
                }),
                Some(Types {
                    fixed_fixed: true,
                    ..Types::none()
                }),
            ];
            for regular in [false, true] {
                for (i, a_body) in [
                    Body::dynamic(),
                    Body::fixed(),
                    Body::kinematic_position(),
                    Body::kinematic_velocity(),
                ]
                .into_iter()
                .enumerate()
                {
                    for (j, b_body) in [
                        Body::dynamic(),
                        Body::fixed(),
                        Body::kinematic_position(),
                        Body::kinematic_velocity(),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        for option in options {
                            let t = option.unwrap_or_default();
                            let allowed = match (i, j) {
                                (0, 0) => t.dynamic_dynamic,
                                (0, 1) | (1, 0) => t.dynamic_fixed,
                                (0, _) | (_, 0) => t.dynamic_kinematic,
                                (1, 1) => t.fixed_fixed,
                                (1, _) | (_, 1) => t.kinematic_fixed,
                                _ => t.kinematic_kinematic,
                            };
                            let mut w = World::new();
                            register_std_components(&mut w);
                            w.insert_resource(Physics::default());
                            w.resource_mut::<Physics>().gravity = Default::default();
                            w.add_event::<Collision>();
                            // Alternate which collider requests detection. The other enables no pairs.
                            let (at, bt) = if regular {
                                (Some(Types::none()), option)
                            } else {
                                (option, Some(Types::none()))
                            };
                            let a = w
                                .spawn_with((
                                    Transform::IDENTITY,
                                    a_body.clone(),
                                    Collider::ball(1.0)
                                        .with_sensor(Some(SensorSettings::default()))
                                        .with_collision_events(Some(
                                            CollisionEventSettings::default(),
                                        ))
                                        .with_collision_types(at),
                                ))
                                .unwrap();
                            let b = w
                                .spawn_with((
                                    Transform::IDENTITY,
                                    b_body.clone(),
                                    Collider::ball(0.5).with_collision_types(bt),
                                ))
                                .unwrap();
                            sync(&mut w, regular);
                            advance(&mut w);
                            let cursor = EventCursor::new();
                            let events = read(&w, &cursor);
                            assert_eq!(
                                events.len(),
                                usize::from(allowed),
                                "types={option:?}, bodies={i}/{j}, regular={regular}"
                            );
                            let p = w.resource::<Physics>();
                            let ac = p.bodies()[p.body_for_entity(a).unwrap()].colliders()[0];
                            let bc = p.bodies()[p.body_for_entity(b).unwrap()].colliders()[0];
                            assert_eq!(
                                p.narrow_phase().intersection_pair(ac, bc).unwrap_or(false),
                                allowed
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn live_type_rules_refilter_stationary_overlaps_without_recreating_colliders() {
            use physics::CollisionTypes as Types;
            let kinematic_fixed = Some(Types {
                kinematic_fixed: true,
                ..Types::default()
            });
            for regular in [false, true] {
                let (mut w, a, b) = setup(true, true, regular);
                w.insert(b, Body::kinematic_position()).unwrap();
                sync(&mut w, regular);
                let cursor = EventCursor::new();
                advance(&mut w);
                assert!(read(&w, &cursor).is_empty());
                w.get_mut::<Collider>(a).unwrap().collision_types = kinematic_fixed;
                sync(&mut w, regular);
                assert!(read(&w, &cursor).is_empty());
                advance(&mut w);
                let started = read(&w, &cursor);
                assert_eq!(phases(&started), [Phase::Started]);
                w.get_mut::<Collider>(b).unwrap().collision_types = kinematic_fixed;
                sync(&mut w, regular);
                w.get_mut::<Collider>(a).unwrap().collision_types = None;
                sync(&mut w, regular);
                advance(&mut w);
                assert!(read(&w, &cursor).is_empty()); // the other collider still requests this pair
                w.get_mut::<Collider>(b).unwrap().collision_types = None;
                sync(&mut w, regular);
                advance(&mut w);
                let stopped = read(&w, &cursor);
                assert_eq!(phases(&stopped), [Phase::Stopped(Reason::FilteredOut)]);
                assert_eq!((started[0].a, started[0].b), (stopped[0].a, stopped[0].b));
                w.get_mut::<Collider>(a).unwrap().collision_types = kinematic_fixed;
                sync(&mut w, regular);
                advance(&mut w);
                let restarted = read(&w, &cursor);
                assert_eq!(phases(&restarted), [Phase::Started]);
                assert_eq!(
                    (started[0].a, started[0].b),
                    (restarted[0].a, restarted[0].b)
                );
                // Group rejection still wins even though type rules allow the pair.
                w.get_mut::<Collider>(b).unwrap().collision_groups =
                    Some(physics::CollisionGroups::new(1, 0));
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(
                    phases(&read(&w, &cursor)),
                    [Phase::Stopped(Reason::FilteredOut)]
                );
                w.get_mut::<Collider>(b).unwrap().collision_groups = None;
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(phases(&read(&w, &cursor)), [Phase::Started]);
            }
        }

        #[test]
        fn fixed_sensor_pairs_can_be_enabled_after_creation_and_keep_tracking_on_unrelated_edits() {
            for regular in [false, true] {
                let (mut w, a, b) = setup(true, true, regular);
                w.insert(b, Body::fixed()).unwrap();
                sync(&mut w, regular);
                let cursor = EventCursor::new();
                advance(&mut w);
                assert!(read(&w, &cursor).is_empty());
                w.get_mut::<Collider>(a).unwrap().collision_types =
                    Some(physics::CollisionTypes::all());
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(phases(&read(&w, &cursor)), [Phase::Started]);
                w.get_mut::<Collider>(a)
                    .unwrap()
                    .collision_types
                    .as_mut()
                    .unwrap()
                    .dynamic_dynamic = false;
                sync(&mut w, regular);
                advance(&mut w);
                assert!(read(&w, &cursor).is_empty());
                w.get_mut::<Collider>(a).unwrap().collision_types = None;
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(
                    phases(&read(&w, &cursor)),
                    [Phase::Stopped(Reason::FilteredOut)]
                );
            }
        }

        #[test]
        fn changing_body_types_closes_late_enabled_tracking_and_reopens_it() {
            for regular in [false, true] {
                let (mut w, a, b) = setup(false, true, regular);
                advance(&mut w); // Native contact starts with no event tracking.
                w.get_mut::<Collider>(a).unwrap().collision_events =
                    Some(CollisionEventSettings::default());
                sync(&mut w, regular);
                let cursor = EventCursor::new();
                advance(&mut w);
                let start = read(&w, &cursor);
                assert_eq!(phases(&start), [Phase::Started]);
                w.insert(b, Body::fixed()).unwrap();
                sync(&mut w, regular);
                advance(&mut w);
                let stopped = read(&w, &cursor);
                assert_eq!(phases(&stopped), [Phase::Stopped(Reason::FilteredOut)]);
                assert_eq!((start[0].a, start[0].b), (stopped[0].a, stopped[0].b));
                w.insert(b, Body::dynamic()).unwrap();
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(phases(&read(&w, &cursor)), [Phase::Started]);
            }
        }

        #[test]
        fn type_rules_disable_and_restore_solid_contacts() {
            for regular in [false, true] {
                let (mut w, a, b) = setup(true, true, regular);
                w.get_mut::<Collider>(a).unwrap().sensor = None;
                sync(&mut w, regular);
                let cursor = EventCursor::new();
                advance(&mut w);
                let started = read(&w, &cursor);
                assert_eq!(phases(&started), [Phase::Started]);
                // Both colliders must stop requesting the pair to reject it by type.
                for entity in [a, b] {
                    w.get_mut::<Collider>(entity).unwrap().collision_types =
                        Some(physics::CollisionTypes::none());
                }
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(
                    phases(&read(&w, &cursor)),
                    [Phase::Stopped(Reason::FilteredOut)]
                );
                {
                    let p = w.resource::<Physics>();
                    assert!(
                        !p.narrow_phase()
                            .contact_pair(started[0].a.collider, started[0].b.collider)
                            .is_some_and(|pair| pair.has_any_active_contact())
                    );
                }
                w.get_mut::<Collider>(a).unwrap().collision_types = None;
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(phases(&read(&w, &cursor)), [Phase::Started]);
            }
        }

        #[test]
        fn free_colliders_use_fixed_type_for_kinematic_triggers() {
            let (mut w, a, b) = setup(false, true, false);
            w.despawn(a);
            w.insert(b, Body::kinematic_velocity()).unwrap();
            sync(&mut w, false);
            let free = w.resource_mut::<Physics>().add_free_collider(
                RapierCollider::ball(1.0)
                    .sensor(true)
                    .active_collision_types(
                        physics::CollisionTypes {
                            kinematic_fixed: true,
                            ..Default::default()
                        }
                        .into(),
                    )
                    .active_events(ActiveEvents::COLLISION_EVENTS)
                    .build(),
            );
            let cursor = EventCursor::new();
            advance(&mut w);
            let started = read(&w, &cursor);
            assert_eq!(phases(&started), [Phase::Started]);
            assert!(started[0].a.collider == free || started[0].b.collider == free);
            w.resource_mut::<Physics>().remove_free_collider(free);
            advance(&mut w);
            assert_eq!(
                phases(&read(&w, &cursor)),
                [Phase::Stopped(Reason::Removed)]
            );
        }

        #[test]
        fn collision_readers_observe_each_transition_once_with_stable_participants() {
            for regular in [false, true] {
                let (mut w, a, b) = setup(true, true, regular);
                // Both flags enabled still generate a single event.
                w.get_mut::<Collider>(b).unwrap().collision_events =
                    Some(CollisionEventSettings::default());
                sync(&mut w, regular);
                let fast = EventCursor::new();
                let slow = EventCursor::new();
                advance(&mut w);
                let start = read(&w, &fast);
                assert_eq!(phases(&start), [Phase::Started]);
                assert_eq!(start[0].step, 1);
                assert!(start[0].a.entity == Some(a) || start[0].b.entity == Some(a));
                assert!(start[0].a.entity == Some(b) || start[0].b.entity == Some(b));
                assert_ne!(start[0].a.is_sensor, start[0].b.is_sensor);
                assert!(read(&w, &fast).is_empty());
                advance(&mut w);
                assert!(read(&w, &fast).is_empty());
                move_to(&mut w, b, 10.0);
                advance(&mut w);
                let stop = read(&w, &fast);
                assert_eq!(phases(&stop), [Phase::Stopped(Reason::Separated)]);
                assert_eq!(stop[0].step, 3);
                assert_eq!((start[0].a, start[0].b), (stop[0].a, stop[0].b));
                assert_eq!(read(&w, &slow), [start[0], stop[0]]);
            }
        }

        #[test]
        fn enabling_inside_overlap_and_disabling_last_observer_are_balanced() {
            for regular in [false, true] {
                let (mut w, a, b) = setup(false, true, regular);
                let cursor = EventCursor::new();
                advance(&mut w);
                assert!(read(&w, &cursor).is_empty());
                w.get_mut::<Collider>(a).unwrap().collision_events =
                    Some(CollisionEventSettings::default());
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(phases(&read(&w, &cursor)), [Phase::Started]);
                w.get_mut::<Collider>(b).unwrap().collision_events =
                    Some(CollisionEventSettings::default());
                sync(&mut w, regular);
                advance(&mut w);
                assert!(read(&w, &cursor).is_empty());
                w.get_mut::<Collider>(a).unwrap().collision_events = None;
                sync(&mut w, regular);
                advance(&mut w);
                assert!(read(&w, &cursor).is_empty());
                w.get_mut::<Collider>(b).unwrap().collision_events = None;
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(
                    phases(&read(&w, &cursor)),
                    [Phase::Stopped(Reason::TrackingDisabled)]
                );
                w.get_mut::<Collider>(a).unwrap().collision_events =
                    Some(CollisionEventSettings::default());
                sync(&mut w, regular);
                advance(&mut w);
                let start = read(&w, &cursor);
                assert_eq!(phases(&start), [Phase::Started]);
                w.despawn(b);
                sync(&mut w, regular);
                advance(&mut w);
                let end = read(&w, &cursor);
                assert_eq!(phases(&end), [Phase::Stopped(Reason::Removed)]);
                assert_eq!((start[0].a, start[0].b), (end[0].a, end[0].b));
            }
        }

        #[test]
        fn sensor_role_changes_close_old_pair_and_material_edits_do_not() {
            for regular in [false, true] {
                let (mut w, a, _) = setup(true, true, regular);
                let cursor = EventCursor::new();
                advance(&mut w);
                let first = read(&w, &cursor)[0];
                w.get_mut::<Collider>(a).unwrap().friction = 0.7;
                w.get_mut::<Collider>(a).unwrap().density = 2.0;
                w.get_mut::<Collider>(a).unwrap().shape = Collider::ball(1.1).shape;
                sync(&mut w, regular);
                advance(&mut w);
                assert!(read(&w, &cursor).is_empty());
                w.get_mut::<Collider>(a).unwrap().sensor = None;
                sync(&mut w, regular);
                advance(&mut w);
                let changed = read(&w, &cursor);
                assert_eq!(
                    phases(&changed),
                    [Phase::Stopped(Reason::Reconfigured), Phase::Started]
                );
                assert_eq!((first.a, first.b), (changed[0].a, changed[0].b));
                assert!(!changed[1].a.is_sensor && !changed[1].b.is_sensor);
                w.get_mut::<Collider>(a).unwrap().sensor = Some(SensorSettings::default());
                sync(&mut w, regular);
                advance(&mut w);
                let changed = read(&w, &cursor);
                assert_eq!(
                    phases(&changed),
                    [Phase::Stopped(Reason::Reconfigured), Phase::Started]
                );
                assert!(changed[1].a.is_sensor || changed[1].b.is_sensor);
            }
        }

        #[test]
        fn regular_sync_then_step_in_one_schedule_captures_pending_entity_owners() {
            for cancel in [false, true] {
                let mut w = World::new();
                register_std_components(&mut w);
                w.insert_resource(Physics::default());
                w.add_event::<Collision>();
                let a = w
                    .spawn_with((
                        Transform::IDENTITY,
                        Body::fixed(),
                        Collider::ball(1.0)
                            .with_sensor(Some(SensorSettings::default()))
                            .with_collision_events(Some(CollisionEventSettings::default())),
                    ))
                    .unwrap();
                let b = w
                    .spawn_with((Transform::IDENTITY, Body::dynamic(), Collider::ball(0.5)))
                    .unwrap();
                struct Cancel(Entity);
                impl System for Cancel {
                    type Result = ();
                    fn run(&self, ctx: &SystemContext<'_>) -> Result<(), SystemError> {
                        let entity = self.0;
                        ctx.commands(move |world| {
                            world.despawn(entity);
                        });
                        Ok(())
                    }
                }
                let mut systems = SystemsContainer::new();
                systems.add(SyncRegular);
                systems.add(Step);
                systems.add_edge::<SyncRegular, Step>().unwrap();
                if cancel {
                    systems.add(Cancel(b));
                    systems.add_edge::<Cancel, SyncRegular>().unwrap();
                }
                let errors = EcsRunner::multi_thread(2).run(&mut w, &systems);
                assert!(errors.is_empty(), "{errors:?}");
                let cursor = EventCursor::new();
                let started = read(&w, &cursor);
                assert_eq!(phases(&started), [Phase::Started]);
                assert!(started[0].a.entity == Some(a) || started[0].b.entity == Some(a));
                assert!(started[0].a.entity == Some(b) || started[0].b.entity == Some(b));
                if !cancel {
                    w.despawn(b);
                    sync(&mut w, true);
                }
                advance(&mut w);
                let stopped = read(&w, &cursor);
                assert_eq!(phases(&stopped), [Phase::Stopped(Reason::Removed)]);
                assert_eq!((started[0].a, started[0].b), (stopped[0].a, stopped[0].b));
            }
        }

        #[test]
        fn collision_masks_filter_contacts_and_sensors_bilaterally() {
            use physics::CollisionGroups as Groups;
            let a_groups = Some(Groups::new(1, 2));
            for regular in [false, true] {
                for sensor in [false, true] {
                    for (ag, bg, allowed) in [
                        (None, None, true),
                        (a_groups, None, true),
                        (a_groups, Some(Groups::new(2, 1)), true),
                        (a_groups, Some(Groups::new(2, 4)), false),
                        (Some(Groups::new(1, 4)), Some(Groups::new(2, 1)), false),
                        (Some(Groups::new(0, u32::MAX)), None, false),
                        (None, Some(Groups::new(u32::MAX, 0)), false),
                        (
                            Some(Groups::new(1 << 31, 3)),
                            Some(Groups::new(3, 1 << 31)),
                            true,
                        ),
                    ] {
                        let mut w = World::new();
                        register_std_components(&mut w);
                        w.insert_resource(Physics::default());
                        w.resource_mut::<Physics>().gravity = Default::default();
                        w.add_event::<Collision>();
                        let a = w
                            .spawn_with((
                                Transform::IDENTITY,
                                Body::fixed(),
                                Collider::ball(1.0)
                                    .with_sensor(sensor.then(SensorSettings::default))
                                    .with_collision_events(Some(CollisionEventSettings::default()))
                                    .with_collision_groups(ag),
                            ))
                            .unwrap();
                        let b = w
                            .spawn_with((
                                Transform::from_translation(Vec3::new(0.8, 0.0, 0.0)),
                                Body::dynamic(),
                                Collider::ball(0.5).with_collision_groups(bg),
                            ))
                            .unwrap();
                        sync(&mut w, regular);
                        let cursor = EventCursor::new();
                        advance(&mut w);
                        let events = read(&w, &cursor);
                        assert_eq!(
                            events.len(),
                            usize::from(allowed),
                            "{ag:?} {bg:?} sensor={sensor}"
                        );
                        if allowed {
                            assert_eq!(events[0].phase, Phase::Started);
                        }
                        let p = w.resource::<Physics>();
                        let ah = p.bodies()[p.body_for_entity(a).unwrap()].colliders()[0];
                        let bh = p.bodies()[p.body_for_entity(b).unwrap()].colliders()[0];
                        let active = if sensor {
                            p.narrow_phase().intersection_pair(ah, bh).unwrap_or(false)
                        } else {
                            p.narrow_phase()
                                .contact_pair(ah, bh)
                                .is_some_and(|pair| pair.has_any_active_contact())
                        };
                        assert_eq!(active, allowed);
                        if !allowed || sensor {
                            assert!((p.pose(b).unwrap().translation.x - 0.8).abs() < 1e-5);
                        }
                    }
                }
            }
        }

        #[test]
        fn group_edits_on_a_fixed_sensor_update_pairs_with_sleeping_bodies() {
            for regular in [false, true] {
                let (mut w, a, b) = setup(true, true, regular);
                let cursor = EventCursor::new();
                let body = w.resource::<Physics>().body_for_entity(b).unwrap();
                // Let Rapier deactivate the body and its island naturally.
                for _ in 0..300 {
                    advance(&mut w);
                }
                assert_eq!(phases(&read(&w, &cursor)), [Phase::Started]);
                assert!(w.resource::<Physics>().bodies()[body].is_sleeping());
                sync(&mut w, regular);
                assert!(w.resource::<Physics>().bodies()[body].is_sleeping());
                w.get_mut::<Collider>(a).unwrap().collision_groups =
                    Some(physics::CollisionGroups::new(1, 0));
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(
                    phases(&read(&w, &cursor)),
                    [Phase::Stopped(Reason::FilteredOut)]
                );
                w.get_mut::<Collider>(a).unwrap().collision_groups = None;
                sync(&mut w, regular);
                advance(&mut w);
                assert_eq!(phases(&read(&w, &cursor)), [Phase::Started]);
            }
        }

        #[test]
        fn editing_groups_preserves_handles_and_balances_observed_pairs() {
            use physics::CollisionGroups as Groups;
            for regular in [false, true] {
                for sensor in [false, true] {
                    for late_tracking in [false, true] {
                        let (mut w, a, b) = setup(!late_tracking, true, regular);
                        w.get_mut::<Collider>(a).unwrap().sensor =
                            sensor.then(SensorSettings::default);
                        sync(&mut w, regular);
                        let cursor = EventCursor::new();
                        advance(&mut w);
                        if late_tracking {
                            assert!(read(&w, &cursor).is_empty());
                            w.get_mut::<Collider>(a).unwrap().collision_events =
                                Some(CollisionEventSettings::default());
                            sync(&mut w, regular);
                            advance(&mut w);
                        }
                        let started = read(&w, &cursor);
                        assert_eq!(phases(&started), [Phase::Started]);
                        let body = w.resource::<Physics>().body_for_entity(b).unwrap();
                        let handles = w.resource::<Physics>().bodies()[body].colliders().to_vec();
                        w.get_mut::<Collider>(a).unwrap().collision_groups =
                            Some(Groups::new(1, 0));
                        sync(&mut w, regular);
                        assert!(read(&w, &cursor).is_empty());
                        advance(&mut w);
                        let stopped = read(&w, &cursor);
                        assert_eq!(phases(&stopped), [Phase::Stopped(Reason::FilteredOut)]);
                        assert_eq!((started[0].a, started[0].b), (stopped[0].a, stopped[0].b));
                        advance(&mut w);
                        assert!(read(&w, &cursor).is_empty());
                        // Restoring defaults permits the overlapping pair again.
                        w.get_mut::<Collider>(a).unwrap().collision_groups = None;
                        sync(&mut w, regular);
                        advance(&mut w);
                        assert_eq!(phases(&read(&w, &cursor)), [Phase::Started]);
                        assert_eq!(w.resource::<Physics>().body_for_entity(b), Some(body));
                        assert_eq!(w.resource::<Physics>().bodies()[body].colliders(), handles);
                        // A changed mask that still allows the pair must not restart it.
                        w.get_mut::<Collider>(a).unwrap().collision_groups =
                            Some(Groups::new(1, 2));
                        sync(&mut w, regular);
                        advance(&mut w);
                        assert!(read(&w, &cursor).is_empty());
                        // Filtering by the non-observing participant must close tracking too.
                        w.get_mut::<Collider>(b).unwrap().collision_groups =
                            Some(Groups::new(4, 1));
                        sync(&mut w, regular);
                        advance(&mut w);
                        assert_eq!(
                            phases(&read(&w, &cursor)),
                            [Phase::Stopped(Reason::FilteredOut)]
                        );
                        w.get_mut::<Collider>(b).unwrap().collision_groups =
                            Some(Groups::new(2, 1));
                        sync(&mut w, regular);
                        advance(&mut w);
                        assert_eq!(phases(&read(&w, &cursor)), [Phase::Started]);
                    }
                }
            }
        }

        #[test]
        fn ordinary_contacts_report_start_and_separation() {
            for regular in [false, true] {
                let (mut w, a, b) = setup(true, true, regular);
                w.get_mut::<Collider>(a).unwrap().sensor = None;
                sync(&mut w, regular);
                let cursor = EventCursor::new();
                advance(&mut w);
                let started = read(&w, &cursor);
                assert_eq!(phases(&started), [Phase::Started]);
                assert!(!started[0].a.is_sensor && !started[0].b.is_sensor);
                move_to(&mut w, b, 10.0);
                advance(&mut w);
                assert_eq!(
                    phases(&read(&w, &cursor)),
                    [Phase::Stopped(Reason::Separated)]
                );
            }
        }

        #[test]
        fn changing_one_of_two_sensors_preserves_snapshot_roles() {
            for regular in [false, true] {
                let (mut w, a, b) = setup(true, true, regular);
                w.get_mut::<Collider>(b).unwrap().sensor = Some(SensorSettings::default());
                sync(&mut w, regular);
                let cursor = EventCursor::new();
                advance(&mut w);
                let old = read(&w, &cursor)[0];
                assert!(old.a.is_sensor && old.b.is_sensor);
                w.get_mut::<Collider>(a).unwrap().sensor = None;
                sync(&mut w, regular);
                advance(&mut w);
                let changed = read(&w, &cursor);
                assert_eq!(
                    phases(&changed),
                    [Phase::Stopped(Reason::Reconfigured), Phase::Started]
                );
                assert_eq!((old.a, old.b), (changed[0].a, changed[0].b));
                assert_ne!(changed[1].a.is_sensor, changed[1].b.is_sensor);
                advance(&mut w);
                assert!(read(&w, &cursor).is_empty());
            }
        }

        #[test]
        fn complete_removal_releases_tracking_and_world_replacement_emits_no_mass_stops() {
            let (mut w, a, b) = setup(true, true, false);
            let cursor = EventCursor::new();
            advance(&mut w);
            read(&w, &cursor);
            w.despawn(a);
            w.despawn(b);
            sync(&mut w, false);
            advance(&mut w);
            assert_eq!(
                phases(&read(&w, &cursor)),
                [Phase::Stopped(Reason::Removed)]
            );
            w.remove_resource::<Events<Collision>>();
            advance(&mut w); // no enabled colliders or pairs left requiring the queue

            let (mut w, _, _) = setup(true, true, false);
            let cursor = EventCursor::new();
            advance(&mut w);
            read(&w, &cursor);
            w.insert_resource(Physics::default());
            assert!(read(&w, &cursor).is_empty());
        }

        #[test]
        fn removing_any_prerequisite_or_excluding_entity_preserves_end_identity() {
            for regular in [false, true] {
                for edit in [
                    |w: &mut World, e| {
                        w.remove::<Body>(e).unwrap();
                    },
                    |w: &mut World, e| {
                        w.remove::<Collider>(e).unwrap();
                    },
                    |w: &mut World, e| {
                        w.remove::<Transform>(e).unwrap();
                    },
                    |w: &mut World, e| {
                        w.set_entity_flags(e, Entity::DISABLED);
                    },
                    |w: &mut World, e| {
                        w.despawn(e);
                    },
                ] {
                    let (mut w, _, b) = setup(true, true, regular);
                    let cursor = EventCursor::new();
                    advance(&mut w);
                    let start = read(&w, &cursor)[0];
                    edit(&mut w, b);
                    sync(&mut w, regular);
                    assert!(read(&w, &cursor).is_empty()); // publication belongs to Step
                    advance(&mut w);
                    let stopped = read(&w, &cursor);
                    assert_eq!(phases(&stopped), [Phase::Stopped(Reason::Removed)]);
                    assert_eq!((start.a, start.b), (stopped[0].a, stopped[0].b));
                    advance(&mut w);
                    assert!(read(&w, &cursor).is_empty());
                }
            }
        }

        #[test]
        fn recreated_handles_and_entities_do_not_retarget_the_old_event() {
            for regular in [false, true] {
                let (mut w, _, b) = setup(true, true, regular);
                let cursor = EventCursor::new();
                advance(&mut w);
                let start = read(&w, &cursor)[0];
                let old_handle = w.resource::<Physics>().body_for_entity(b).unwrap();
                w.despawn(b);
                sync(&mut w, regular);
                let new = w
                    .spawn_with((
                        Transform::from_translation(Vec3::new(0.8, 0.0, 0.0)),
                        Body::dynamic(),
                        Collider::ball(0.5),
                    ))
                    .unwrap();
                assert_eq!(b.index(), new.index());
                assert_ne!(b, new);
                sync(&mut w, regular);
                assert_ne!(
                    w.resource::<Physics>().body_for_entity(new).unwrap(),
                    old_handle
                );
                advance(&mut w);
                let events = read(&w, &cursor);
                assert_eq!(
                    phases(&events),
                    [Phase::Stopped(Reason::Removed), Phase::Started]
                );
                assert_eq!((start.a, start.b), (events[0].a, events[0].b));
                assert!(events[1].a.entity == Some(new) || events[1].b.entity == Some(new));
                assert!(events.iter().all(|e| e.step == 2));
            }
        }

        #[test]
        fn free_colliders_report_none_entity_and_keep_identity_after_removal() {
            let (mut w, a, _) = setup(false, true, false);
            w.despawn(a);
            sync(&mut w, false);
            let free = w.resource_mut::<Physics>().add_free_collider(
                RapierCollider::ball(1.0)
                    .sensor(true)
                    .active_events(ActiveEvents::COLLISION_EVENTS)
                    .build(),
            );
            let cursor = EventCursor::new();
            advance(&mut w);
            let start = read(&w, &cursor);
            assert_eq!(phases(&start), [Phase::Started]);
            let participant = if start[0].a.collider == free {
                start[0].a
            } else {
                start[0].b
            };
            assert_eq!(participant.collider, free);
            assert_eq!(participant.entity, None);
            assert_eq!(participant.body, None);
            assert!(participant.is_sensor);
            assert!(w.resource_mut::<Physics>().remove_free_collider(free));
            advance(&mut w);
            let stopped = read(&w, &cursor);
            assert_eq!(phases(&stopped), [Phase::Stopped(Reason::Removed)]);
            assert_eq!((start[0].a, start[0].b), (stopped[0].a, stopped[0].b));
        }

        #[test]
        fn missing_queue_and_invalid_pose_do_not_advance_simulation_or_consume_pending_ends() {
            let (mut w, a, b) = setup(false, false, false);
            advance(&mut w);
            w.get_mut::<Collider>(a).unwrap().collision_events =
                Some(CollisionEventSettings::default());
            sync(&mut w, false);
            let before = w.resource::<Physics>().pose(b).unwrap();
            move_to(&mut w, b, 0.6);
            assert!(matches!(
                step(&mut w).as_slice(),
                [SystemError::InvalidConfiguration { .. }]
            ));
            assert_eq!(w.resource::<Physics>().pose(b), Some(before));
            w.add_event::<Collision>();
            let cursor = EventCursor::new();
            advance(&mut w);
            let started = read(&w, &cursor);
            assert_eq!(phases(&started), [Phase::Started]);
            assert_eq!(started[0].step, 2);
            w.despawn(a);
            sync(&mut w, false);
            w.get_mut::<Transform>(b).unwrap().scale.x = 2.0;
            assert!(matches!(
                step(&mut w).as_slice(),
                [SystemError::InvalidConfiguration { .. }]
            ));
            assert!(read(&w, &cursor).is_empty());
            w.get_mut::<Transform>(b).unwrap().scale.x = 1.0;
            advance(&mut w);
            let ended = read(&w, &cursor);
            assert_eq!(phases(&ended), [Phase::Stopped(Reason::Removed)]);
            assert_eq!(ended[0].step, 3);
        }

        #[test]
        fn fixed_substeps_accumulate_and_slow_readers_account_for_expired_events() {
            let (mut w, a, b) = setup(true, true, false);
            // Move out after the first substep using a game system ordered after physics.
            struct Leave {
                entity: Entity,
            }
            impl System for Leave {
                type Result = ();
                fn run(&self, ctx: &SystemContext<'_>) -> Result<(), SystemError> {
                    ctx.lock::<(ResMut<Physics>,)>().execute(|(mut p,)| {
                        let mut pose = Pose::default();
                        pose.translation.x = 10.0;
                        p.teleport(self.entity, pose, TeleportVelocity::Reset)
                            .unwrap();
                    });
                    Ok(())
                }
            }
            let mut schedules = Schedules::new();
            schedules.set_fixed_timestep(0.1);
            let fixed = schedules.get_mut::<FixedUpdate>();
            fixed.add(Step);
            fixed.add(Leave { entity: b });
            fixed.add_edge::<Step, Leave>().unwrap();
            let fast = EventCursor::new();
            let slow = EventCursor::new();
            schedules.run_frame(&mut w, &EcsRunner::single_thread(), 0.31);
            let all = read(&w, &fast);
            assert_eq!(
                phases(&all),
                [Phase::Started, Phase::Stopped(Reason::Separated)]
            );
            assert_eq!(all.iter().map(|e| e.step).collect::<Vec<_>>(), [1, 2]);
            assert!(all[0].a.entity == Some(a) || all[0].b.entity == Some(a));
            schedules.run_frame(&mut w, &EcsRunner::single_thread(), 0.0);
            assert!(read(&w, &fast).is_empty());
            schedules.run_frame(&mut w, &EcsRunner::single_thread(), 0.0);
            assert!(read(&w, &slow).is_empty());
            assert_eq!(slow.missed(), 2);
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod two_d {
    use super::*;
    use physics::{
        components2d::{Collider2D as Collider, RigidBody2D as Body},
        control2d::PhysicsPose2D as Pose,
        events2d::CollisionEvent2D as Collision,
        rapier2d::prelude::{ActiveEvents, ColliderBuilder as RapierCollider},
        systems2d::{
            StepPhysics2D as Step, SyncPhysicsBodies2D as Sync,
            SyncPhysicsBodiesSystem2D as SyncRegular,
        },
        world2d::PhysicsWorld2D as Physics,
    };
    event_tests!();
}
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod three_d {
    use super::*;
    use physics::{
        components3d::{Collider3D as Collider, RigidBody3D as Body},
        control3d::PhysicsPose3D as Pose,
        events3d::CollisionEvent3D as Collision,
        rapier3d::prelude::{ActiveEvents, ColliderBuilder as RapierCollider},
        systems3d::{
            StepPhysics3D as Step, SyncPhysicsBodies3D as Sync,
            SyncPhysicsBodiesSystem3D as SyncRegular,
        },
        world3d::PhysicsWorld3D as Physics,
    };
    event_tests!();
}
