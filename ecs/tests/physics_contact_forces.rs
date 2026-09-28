#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_core::math::Vec3;
use redlilium_ecs::physics::{ContactForceSettings, SensorSettings};
use redlilium_ecs::*;

macro_rules! force_tests {
    () => {
        fn sync(w: &mut World, regular: bool) -> Vec<SystemError> {
            let mut systems = SystemsContainer::new();
            if regular {
                systems.add(SyncRegular);
            } else {
                systems.add_exclusive(Sync);
            }
            EcsRunner::multi_thread(2).run(w, &systems)
        }
        fn step(w: &mut World) -> Vec<SystemError> {
            let mut systems = SystemsContainer::new();
            systems.add(Step);
            EcsRunner::multi_thread(2).run(w, &systems)
        }
        fn read(w: &World) -> Vec<Force> {
            w.resource::<Events<Force>>()
                .read(&EventCursor::new())
                .copied()
                .collect()
        }
        fn tracked(min_force: f32) -> Collider {
            Collider::ball(0.5).with_contact_force_events(Some(ContactForceSettings { min_force }))
        }
        fn world() -> World {
            let mut w = World::new();
            register_std_components(&mut w);
            w.insert_resource(Physics::default());
            w.resource_mut::<Physics>().gravity = Default::default();
            w.resource_mut::<Physics>().integration_parameters.dt = 0.01;
            w.add_event::<Force>();
            w
        }
        fn fixture(a: Collider, b: Collider, regular: bool, reverse: bool) -> (World, Entity, Entity) {
            let mut w = world();
            let fixed = (Transform::IDENTITY, Body::fixed(), a);
            let moving = (
                Transform::from_translation(Vec3::new(0.99, 0.0, 0.0)),
                Body::dynamic(),
                b,
            );
            let (a, b) = if reverse {
                let b = w.spawn_with(moving).unwrap();
                (w.spawn_with(fixed).unwrap(), b)
            } else {
                (w.spawn_with(fixed).unwrap(), w.spawn_with(moving).unwrap())
            };
            assert!(sync(&mut w, regular).is_empty());
            velocity(&mut w, b, -2.0);
            (w, a, b)
        }
        fn velocity(w: &mut World, e: Entity, x: f32) {
            let h = w.resource::<Physics>().body_for_entity(e).unwrap();
            let mut v = w.resource::<Physics>().bodies()[h].linvel();
            v.x = x as _;
            w.resource_mut::<Physics>()
                .body_motion(h)
                .unwrap()
                .set_linvel(v, true);
        }

        #[test]
        fn impulses_match_momentum_and_normals_follow_canonical_participants() {
            for regular in [false, true] {
                for reverse in [false, true] {
                    let (mut w, fixed, moving) = fixture(tracked(0.0), tracked(0.0), regular, reverse);
                    let h = w.resource::<Physics>().body_for_entity(moving).unwrap();
                    let mass = w.resource::<Physics>().bodies()[h].mass() as f64;
                    assert!(step(&mut w).is_empty());
                    let events = read(&w);
                    assert_eq!(
                        events.len(),
                        1,
                        "both participants must not duplicate an event"
                    );
                    let e = events[0];
                    let dv = w.resource::<Physics>().bodies()[h].linvel().x as f64 + 2.0;
                    assert!(
                        (e.normal_impulse - mass * dv).abs() < 1e-4,
                        "{e:?}, dv={dv}, mass={mass}"
                    );
                    assert!((e.dt - 0.01).abs() < 1e-8);
                    assert_eq!(e.average_force(), e.normal_impulse / e.dt);
                    assert_eq!(e.step, 1);
                    assert_eq!(e.strongest_contact.normal_impulse, e.normal_impulse);
                    assert!((e.strongest_contact.point.x - 0.5).abs() < 0.05, "{e:?}");
                    let sign = if e.a.body_entity == Some(fixed) { 1.0 } else { -1.0 };
                    assert!((e.strongest_contact.normal.x - sign).abs() < 1e-5, "{e:?}");
                    assert_eq!(e.strongest_contact.normal.y, 0.0);
                    assert!(e.a.collider.into_raw_parts() < e.b.collider.into_raw_parts());
                    // The queue owns snapshots, not references into removed Rapier objects.
                    w.despawn(fixed);
                    w.despawn(moving);
                    assert!(sync(&mut w, regular).is_empty());
                    assert_eq!(read(&w), events);
                    assert!(!w.is_alive(e.a.body_entity.unwrap()));
                }
            }
        }

        #[test]
        fn either_threshold_can_enable_reporting_and_disabled_side_has_no_vote() {
            for (a, b, expected) in [
                (tracked(f32::MAX), Collider::ball(0.5), 0),
                (Collider::ball(0.5), tracked(f32::MAX), 0),
                (tracked(f32::MAX), tracked(0.0), 1),
                (tracked(0.0), tracked(f32::MAX), 1),
                (Collider::ball(0.5), Collider::ball(0.5), 0),
            ] {
                let (mut w, _, _) = fixture(a, b, false, false);
                assert!(step(&mut w).is_empty());
                assert_eq!(read(&w).len(), expected);
            }
        }

        #[test]
        fn missing_force_queue_fails_before_teleport_and_does_not_consume_a_step() {
            let (mut w, _, b) = fixture(tracked(0.0), Collider::ball(0.5), false, false);
            w.remove_resource::<Events<Force>>();
            let old = w.resource::<Physics>().pose(b).unwrap();
            let mut next = old;
            next.translation.x = 0.98;
            w.resource_mut::<Physics>()
                .teleport(b, next, physics::TeleportVelocity::Preserve)
                .unwrap();
            let errors = step(&mut w);
            assert!(
                matches!(errors.as_slice(), [SystemError::InvalidConfiguration { message }] if message.contains("ContactForceEvent")),
                "{errors:?}"
            );
            assert_eq!(w.resource::<Physics>().pose(b).unwrap(), old);
            w.add_event::<Force>();
            assert!(step(&mut w).is_empty());
            assert_eq!(read(&w)[0].step, 1);
        }

        #[test]
        fn sensor_and_sleeping_pairs_do_not_replay_impulses() {
            let (mut w, _, _) = fixture(
                tracked(0.0).with_sensor(Some(SensorSettings::default())),
                tracked(0.0),
                false,
                false,
            );
            assert!(step(&mut w).is_empty());
            assert!(read(&w).is_empty());
            let (mut w, _, b) = fixture(tracked(0.0), Collider::ball(0.5), false, false);
            assert!(step(&mut w).is_empty());
            assert_eq!(read(&w).len(), 1);
            let h = w.resource::<Physics>().body_for_entity(b).unwrap();
            w.resource_mut::<Physics>().body_motion(h).unwrap().sleep();
            assert!(step(&mut w).is_empty());
            assert_eq!(read(&w).len(), 1);
        }

        #[test]
        fn live_enable_threshold_edits_and_disable_preserve_handles() {
            for regular in [false, true] {
                let (mut w, a, b) = fixture(Collider::ball(0.5), Collider::ball(0.5), regular, false);
                let ha = w.resource::<Physics>().body_for_entity(a).unwrap();
                let ca = w.resource::<Physics>().bodies()[ha].colliders()[0];
                assert!(step(&mut w).is_empty());
                for (settings, expected) in [
                    (Some(0.0), 1),
                    (Some(f32::MAX), 1),
                    (Some(0.0), 2),
                    (None, 2),
                ] {
                    let c = Collider::ball(0.5).with_contact_force_events(
                        settings.map(|min_force| ContactForceSettings { min_force }),
                    );
                    w.insert(a, c).unwrap();
                    assert!(sync(&mut w, regular).is_empty());
                    let mut pose = w.resource::<Physics>().pose(b).unwrap();
                    pose.translation.x = 0.99;
                    w.resource_mut::<Physics>()
                        .teleport(b, pose, physics::TeleportVelocity::Preserve)
                        .unwrap();
                    velocity(&mut w, b, -2.0);
                    assert!(step(&mut w).is_empty());
                    assert_eq!(read(&w).len(), expected);
                    assert_eq!(w.resource::<Physics>().bodies()[ha].colliders()[0], ca);
                }
                w.remove_resource::<Events<Force>>();
                assert!(step(&mut w).is_empty());
            }
        }

        #[test]
        fn regular_sync_before_deferred_publication_preserves_owners() {
            let mut w = world();
            w.resource_mut::<Physics>().gravity.x = -10.0;
            let a = w
                .spawn_with((Transform::IDENTITY, Body::fixed(), tracked(0.0)))
                .unwrap();
            let b = w
                .spawn_with((
                    Transform::from_translation(Vec3::new(0.99, 0.0, 0.0)),
                    Body::dynamic(),
                    Collider::ball(0.5),
                ))
                .unwrap();
            let mut systems = SystemsContainer::new();
            systems.add(SyncRegular);
            systems.add(Step);
            systems.add_edge::<SyncRegular, Step>().unwrap();
            assert!(EcsRunner::multi_thread(2).run(&mut w, &systems).is_empty());
            let events = read(&w);
            assert_eq!(events.len(), 1);
            assert!([events[0].a.body_entity, events[0].b.body_entity].contains(&Some(a)));
            assert!([events[0].a.body_entity, events[0].b.body_entity].contains(&Some(b)));
        }

        #[test]
        fn continuous_support_load_is_reported_every_awake_step() {
            let (mut w, _, b) = fixture(tracked(0.0), Collider::ball(0.5), false, false);
            velocity(&mut w, b, 0.0);
            w.resource_mut::<Physics>().gravity.x = -10.0;
            for _ in 0..3 {
                assert!(step(&mut w).is_empty());
            }
            let events = read(&w);
            assert_eq!(events.iter().map(|e| e.step).collect::<Vec<_>>(), [1, 2, 3]);
            assert!(events.iter().all(|e| e.average_force() > 0.0));
        }

        #[test]
        fn invalid_threshold_edit_rejects_whole_batch() {
            for regular in [false, true] {
                let (mut w, a, b) = fixture(tracked(0.0), Collider::ball(0.5), regular, false);
                let h = w.resource::<Physics>().body_for_entity(b).unwrap();
                w.insert(b, Body::dynamic().with_linear_damping(2.0))
                    .unwrap();
                w.insert(a, tracked(f32::NAN)).unwrap();
                assert!(!sync(&mut w, regular).is_empty());
                assert_eq!(w.resource::<Physics>().bodies()[h].linear_damping(), 0.0);
                assert!(step(&mut w).is_empty());
                assert_eq!(read(&w).len(), 1);
            }
        }

        #[test]
        fn several_contact_points_contribute_but_only_strongest_point_is_exposed() {
            let (mut w, _, b) = fixture(
                cube().with_contact_force_events(Some(ContactForceSettings::default())),
                cube(),
                false,
                false,
            );
            let h = w.resource::<Physics>().body_for_entity(b).unwrap();
            let mass = w.resource::<Physics>().bodies()[h].mass() as f64;
            assert!(step(&mut w).is_empty());
            let events = read(&w);
            assert_eq!(events.len(), 1);
            let e = events[0];
            let dv = w.resource::<Physics>().bodies()[h].linvel().x as f64 + 2.0;
            assert!((e.normal_impulse - mass * dv).abs() < 1e-4, "{e:?}");
            assert!(
                e.normal_impulse > e.strongest_contact.normal_impulse * 1.5,
                "{e:?}"
            );
        }

        #[test]
        fn standalone_colliders_keep_threshold_and_snapshot_without_ecs_owner() {
            for threshold in [0.0, f32::MAX] {
                let mut w = world();
                let free = w.resource_mut::<Physics>().add_free_collider(
                    RapierCollider::ball(0.5)
                        .active_events(ActiveEvents::CONTACT_FORCE_EVENTS)
                        .contact_force_event_threshold(threshold as _)
                        .build(),
                );
                let b = w
                    .spawn_with((
                        Transform::from_translation(Vec3::new(0.99, 0.0, 0.0)),
                        Body::dynamic(),
                        Collider::ball(0.5),
                    ))
                    .unwrap();
                assert!(sync(&mut w, true).is_empty());
                velocity(&mut w, b, -2.0);
                assert!(step(&mut w).is_empty());
                let events = read(&w);
                if threshold == 0.0 {
                    assert_eq!(events.len(), 1);
                    let e = events[0];
                    let participant = if e.a.collider == free { e.a } else { e.b };
                    assert_eq!(participant.body, None);
                    assert_eq!(participant.body_entity, None);
                } else {
                    assert!(events.is_empty());
                }
                assert!(w.resource_mut::<Physics>().remove_free_collider(free));
                assert_eq!(read(&w), events);
                w.remove_resource::<Events<Force>>();
                assert!(step(&mut w).is_empty());
            }
        }

        #[test]
        fn groups_reject_force_contacts_and_both_event_queues_are_independent() {
            let disabled = tracked(0.0).with_collision_groups(Some(physics::CollisionGroups::new(0, 0)));
            let (mut w, a, _) = fixture(disabled, tracked(0.0), false, false);
            assert!(step(&mut w).is_empty());
            assert!(read(&w).is_empty());
            w.insert(
                a,
                tracked(0.0).with_collision_events(Some(physics::CollisionEventSettings::default())),
            )
            .unwrap();
            assert!(sync(&mut w, false).is_empty());
            let errors = step(&mut w);
            assert!(
                matches!(errors.as_slice(), [SystemError::InvalidConfiguration { message }] if message.contains("CollisionEvent")),
                "{errors:?}"
            );
            w.add_event::<Collision>();
            assert!(step(&mut w).is_empty());
            assert_eq!(read(&w).len(), 1);
            assert_eq!(
                w.resource::<Events<Collision>>()
                    .read(&EventCursor::new())
                    .count(),
                1
            );
            w.remove_resource::<Events<Force>>();
            let errors = step(&mut w);
            assert!(
                matches!(errors.as_slice(), [SystemError::InvalidConfiguration { message }] if message.contains("ContactForceEvent")),
                "{errors:?}"
            );
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod two_d {
    use super::*;
    use physics::{
        components2d::{Collider2D as Collider, RigidBody2D as Body},
        events2d::{CollisionEvent2D as Collision, ContactForceEvent2D as Force},
        rapier2d::prelude::{ActiveEvents, ColliderBuilder as RapierCollider},
        systems2d::{
            StepPhysics2D as Step, SyncPhysicsBodies2D as Sync,
            SyncPhysicsBodiesSystem2D as SyncRegular,
        },
        world2d::PhysicsWorld2D as Physics,
    };
    fn cube() -> Collider {
        Collider::cuboid(0.5, 0.5)
    }
    force_tests!();
}
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod three_d {
    use super::*;
    use physics::{
        components3d::{Collider3D as Collider, RigidBody3D as Body},
        events3d::{CollisionEvent3D as Collision, ContactForceEvent3D as Force},
        rapier3d::prelude::{ActiveEvents, ColliderBuilder as RapierCollider},
        systems3d::{
            StepPhysics3D as Step, SyncPhysicsBodies3D as Sync,
            SyncPhysicsBodiesSystem3D as SyncRegular,
        },
        world3d::PhysicsWorld3D as Physics,
    };
    fn cube() -> Collider {
        Collider::cuboid(0.5, 0.5, 0.5)
    }
    force_tests!();
}
