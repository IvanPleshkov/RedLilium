#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_core::math::Vec3;
use redlilium_ecs::physics::{
    CcdSettings, CollisionEventSettings, CollisionGroups, CollisionPhase, CollisionStopReason,
    SensorSettings,
};
use redlilium_ecs::*;

macro_rules! ccd_tests {
    () => {
        fn sync(w: &mut World, regular: bool) {
            let mut systems = SystemsContainer::new();
            if regular {
                systems.add(SyncRegular);
            } else {
                systems.add_exclusive(Sync);
            }
            let errors = EcsRunner::multi_thread(2).run(w, &systems);
            assert!(errors.is_empty(), "{errors:?}");
        }
        fn step(w: &mut World) {
            let mut systems = SystemsContainer::new();
            systems.add(Step);
            let errors = EcsRunner::multi_thread(2).run(w, &systems);
            assert!(errors.is_empty(), "{errors:?}");
        }
        fn fixture(
            projectile: Body,
            target: Body,
            collider: Collider,
            regular: bool,
        ) -> (World, Entity, Entity) {
            let mut w = World::new();
            register_std_components(&mut w);
            w.insert_resource(Physics::default());
            w.resource_mut::<Physics>().gravity = Default::default();
            w.resource_mut::<Physics>().integration_parameters.dt = 1.0 / 60.0;
            w.add_event::<Collision>();
            let a = w
                .spawn_with((
                    Transform::from_translation(Vec3::new(-1.7, 0.0, 0.0)),
                    projectile,
                    Collider::ball(0.05),
                ))
                .unwrap();
            let b = w
                .spawn_with((Transform::IDENTITY, target, collider))
                .unwrap();
            sync(&mut w, regular);
            let h = w.resource::<Physics>().body_for_entity(a).unwrap();
            let mut velocity = w.resource::<Physics>().bodies()[h].linvel();
            velocity.x = 240.0; // Cross the target completely within one physics step.
            w.resource_mut::<Physics>()
                .body_motion(h)
                .unwrap()
                .set_linvel(velocity, true);
            (w, a, b)
        }
        fn moving_target(w: &mut World, b: Entity, kinematic: bool) {
            if kinematic {
                let mut input = Velocity::default();
                input.linear.y = 0.25;
                w.insert(b, input).unwrap();
            } else {
                let h = w.resource::<Physics>().body_for_entity(b).unwrap();
                let mut velocity = w.resource::<Physics>().bodies()[h].linvel();
                velocity.y = 0.25;
                w.resource_mut::<Physics>()
                    .body_motion(h)
                    .unwrap()
                    .set_linvel(velocity, true);
            }
        }

        #[test]
        fn extended_ccd_stops_fast_bodies_at_moving_targets() {
            for regular in [false, true] {
                for kinematic in [false, true] {
                    for enabled in [false, true] {
                        let (mut w, a, b) = fixture(
                            Body::dynamic().with_ccd(enabled.then(CcdSettings::default)),
                            if kinematic {
                                Body::kinematic_velocity()
                            } else {
                                Body::dynamic()
                            },
                            Collider::ball(0.2),
                            regular,
                        );
                        moving_target(&mut w, b, kinematic);
                        step(&mut w);
                        let x = w.resource::<Physics>().pose(a).unwrap().translation.x;
                        if enabled {
                            assert!(x < -0.1, "CCD missed moving target: x={x}");
                        } else {
                            assert!(
                                x > 1.0,
                                "fixture did not tunnel without extended CCD: x={x}"
                            );
                        }
                    }
                }
            }
        }

        #[test]
        fn default_ccd_covers_fixed_targets_and_global_zero_disables_it() {
            for extended in [false, true] {
                for global_enabled in [false, true] {
                    let (mut w, a, _) = fixture(
                        Body::dynamic().with_ccd(extended.then(CcdSettings::default)),
                        Body::fixed(),
                        Collider::ball(0.2),
                        false,
                    );
                    if !global_enabled {
                        w.resource_mut::<Physics>()
                            .integration_parameters
                            .max_ccd_substeps = 0;
                    }
                    step(&mut w);
                    let x = w.resource::<Physics>().pose(a).unwrap().translation.x;
                    assert_eq!(x < -0.1, global_enabled, "x={x}");
                    if !global_enabled {
                        assert!(x > 1.0, "x={x}");
                    }
                }
            }
        }

        #[test]
        fn collision_filters_and_bullet_pair_exclusion_are_respected() {
            for case in 0..4 {
                let bullets = case == 1;
                let (mut w, a, b) = fixture(
                    Body::dynamic().with_ccd(Some(CcdSettings::default())),
                    Body::dynamic().with_ccd(bullets.then(CcdSettings::default)),
                    Collider::ball(0.2)
                        .with_collision_groups((case == 0).then(|| CollisionGroups::new(1, 0))),
                    false,
                );
                if case >= 2 {
                    for e in [a, b] {
                        w.get_mut::<Collider>(e).unwrap().collision_types =
                            Some(physics::CollisionTypes::none());
                    }
                    sync(&mut w, false);
                }
                if case == 3 {
                    w.get_mut::<Collider>(a)
                        .unwrap()
                        .collision_types
                        .as_mut()
                        .unwrap()
                        .dynamic_dynamic = true;
                    sync(&mut w, false);
                }
                step(&mut w);
                let x = w.resource::<Physics>().pose(a).unwrap().translation.x;
                if case == 3 {
                    assert!(x < -0.1, "allowed pair blocked by type hook: x={x}");
                } else {
                    assert!(x > 1.0, "case={case}, x={x}");
                }
            }
        }

        #[test]
        fn fast_sensor_crossings_publish_balanced_events_without_stopping_motion() {
            for regular in [false, true] {
                let (mut w, a, b) = fixture(
                    Body::dynamic().with_ccd(Some(CcdSettings::default())),
                    Body::kinematic_velocity(),
                    Collider::ball(0.2)
                        .with_sensor(Some(SensorSettings::default()))
                        .with_collision_events(Some(CollisionEventSettings::default())),
                    regular,
                );
                moving_target(&mut w, b, true);
                let cursor = EventCursor::<Collision>::new();
                step(&mut w);
                assert!(w.resource::<Physics>().pose(a).unwrap().translation.x > 1.0);
                let events = w
                    .resource::<Events<Collision>>()
                    .read(&cursor)
                    .copied()
                    .collect::<Vec<_>>();
                assert_eq!(
                    events.iter().map(|e| e.phase).collect::<Vec<_>>(),
                    [
                        CollisionPhase::Started,
                        CollisionPhase::Stopped(CollisionStopReason::Separated),
                    ]
                );
                assert_eq!(events[0].step, events[1].step);
                assert_eq!((events[0].a, events[0].b), (events[1].a, events[1].b));
                assert!(events[0].a.body_entity == Some(a) || events[0].b.body_entity == Some(a));
                assert!(events[0].a.body_entity == Some(b) || events[0].b.body_entity == Some(b));
                step(&mut w);
                assert_eq!(w.resource::<Events<Collision>>().read(&cursor).count(), 0);
            }
        }

        #[test]
        fn ccd_edits_preserve_handles_motion_joints_and_noop_sleep() {
            for regular in [false, true] {
                let (mut w, a, b) =
                    fixture(Body::dynamic(), Body::fixed(), Collider::ball(0.2), regular);
                let joint = w
                    .spawn_with((Joint::fixed(a, b, Default::default(), Default::default()),))
                    .unwrap();
                let mut systems = SystemsContainer::new();
                systems.add_exclusive(JointSync);
                assert!(EcsRunner::single_thread().run(&mut w, &systems).is_empty());
                let p = w.resource::<Physics>();
                let h = p.body_for_entity(a).unwrap();
                let ch = p.bodies()[h].colliders()[0];
                let jh = p.joint_for_entity(joint).unwrap();
                let velocity = p.bodies()[h].linvel();
                let pose = p.pose(a).unwrap();
                assert!(!p.bodies()[h].is_ccd_enabled());
                drop(p);
                for enabled in [true, false] {
                    w.get_mut::<Body>(a).unwrap().ccd = enabled.then(CcdSettings::default);
                    sync(&mut w, regular);
                    let p = w.resource::<Physics>();
                    assert_eq!(p.body_for_entity(a), Some(h));
                    assert_eq!(p.bodies()[h].colliders(), &[ch]);
                    assert_eq!(p.joint_for_entity(joint), Some(jh));
                    assert_eq!(p.bodies()[h].is_ccd_enabled(), enabled);
                    assert_eq!(p.bodies()[h].linvel(), velocity);
                    assert_eq!(p.pose(a), Some(pose));
                }
                w.resource_mut::<Physics>().body_motion(h).unwrap().sleep();
                sync(&mut w, regular);
                assert!(w.resource::<Physics>().bodies()[h].is_sleeping());
                w.get_mut::<Body>(a).unwrap().ccd = Some(CcdSettings::default());
                sync(&mut w, regular);
                assert!(!w.resource::<Physics>().bodies()[h].is_sleeping());
                // Stored settings survive body-type edits and become effective again on dynamic bodies.
                w.insert(a, Body::fixed().with_ccd(Some(CcdSettings::default())))
                    .unwrap();
                sync(&mut w, regular);
                assert!(w.resource::<Physics>().bodies()[h].is_ccd_enabled());
                w.insert(a, Body::dynamic().with_ccd(Some(CcdSettings::default())))
                    .unwrap();
                sync(&mut w, regular);
                assert!(w.resource::<Physics>().bodies()[h].is_ccd_enabled());
            }
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod two_d {
    use super::*;
    use physics::{
        components2d::{Collider2D as Collider, ImpulseJoint2D as Joint, RigidBody2D as Body},
        control2d::KinematicVelocity2D as Velocity,
        events2d::CollisionEvent2D as Collision,
        systems2d::{
            StepPhysics2D as Step, SyncPhysicsBodies2D as Sync,
            SyncPhysicsBodiesSystem2D as SyncRegular, SyncPhysicsJoints2D as JointSync,
        },
        world2d::PhysicsWorld2D as Physics,
    };
    ccd_tests!();
}
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod three_d {
    use super::*;
    use physics::{
        components3d::{Collider3D as Collider, ImpulseJoint3D as Joint, RigidBody3D as Body},
        control3d::KinematicVelocity3D as Velocity,
        events3d::CollisionEvent3D as Collision,
        systems3d::{
            StepPhysics3D as Step, SyncPhysicsBodies3D as Sync,
            SyncPhysicsBodiesSystem3D as SyncRegular, SyncPhysicsJoints3D as JointSync,
        },
        world3d::PhysicsWorld3D as Physics,
    };
    ccd_tests!();
}
