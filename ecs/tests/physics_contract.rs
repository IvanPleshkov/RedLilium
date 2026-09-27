#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_core::math::Vec3;
use redlilium_ecs::physics::TeleportVelocity;
use redlilium_ecs::*;

macro_rules! contract_tests {
    () => {
        fn setup(kind: Body) -> (World, Entity) {
            let mut w = World::new();
            register_std_components(&mut w);
            let e = w
                .spawn_with((Transform::IDENTITY, kind, Collider::ball(0.5)))
                .unwrap();
            sync(&mut w, false).unwrap();
            w.resource_mut::<Physics>().gravity = Default::default();
            w.resource_mut::<Physics>().integration_parameters.dt = 0.1;
            (w, e)
        }
        fn sync(w: &mut World, regular: bool) -> Result<(), Vec<SystemError>> {
            let mut s = SystemsContainer::new();
            if regular {
                s.add(SyncRegular);
            } else {
                s.add_exclusive(Sync);
            }
            let errors = EcsRunner::single_thread().run(w, &s);
            if errors.is_empty() {
                Ok(())
            } else {
                Err(errors)
            }
        }
        fn run<S: System>(w: &mut World, system: S) {
            let mut s = SystemsContainer::new();
            s.add(system);
            let errors = EcsRunner::multi_thread(2).run(w, &s);
            assert!(errors.is_empty(), "{errors:?}");
        }
        fn joints(w: &mut World, regular: bool) {
            let mut s = SystemsContainer::new();
            if regular {
                s.add(JointSyncRegular);
            } else {
                s.add_exclusive(JointSync);
            }
            assert!(EcsRunner::single_thread().run(w, &s).is_empty());
        }
        #[test]
        fn transform_only_drives_fixed_bodies() {
            for fixed in [false, true] {
                let (mut w, e) = setup(if fixed {
                    Body::fixed()
                } else {
                    Body::dynamic()
                });
                w.get_mut::<Transform>(e).unwrap().translation.x = 8.0;
                sync(&mut w, false).unwrap();
                run(&mut w, Step);
                let expected = if fixed { 8.0 } else { 0.0 };
                assert_eq!(
                    w.resource::<Physics>().pose(e).unwrap().translation.x,
                    expected
                );
                assert_eq!(w.get::<Transform>(e).unwrap().translation.x, expected);
            }
        }
        #[test]
        fn position_kinematics_uses_target_and_holds_without_it() {
            let (mut w, e) = setup(Body::kinematic_position());
            let mut pose = Pose::default();
            pose.translation.x = 3.0;
            w.insert(e, Target::from(pose)).unwrap();
            w.get_mut::<Transform>(e).unwrap().translation.x = -99.0;
            run(&mut w, Step);
            assert!((w.resource::<Physics>().pose(e).unwrap().translation.x - 3.0).abs() < 1e-5);
            w.remove::<Target>(e).unwrap();
            w.get_mut::<Transform>(e).unwrap().translation.x = -99.0;
            run(&mut w, Step);
            assert!((w.resource::<Physics>().pose(e).unwrap().translation.x - 3.0).abs() < 1e-5);
        }
        #[test]
        fn velocity_kinematics_stops_without_input_and_teleport_clears_input() {
            let (mut w, e) = setup(Body::kinematic_velocity());
            let mut input = Velocity::default();
            input.linear.x = 2.0;
            w.insert(e, input).unwrap();
            run(&mut w, Step);
            assert!((w.resource::<Physics>().pose(e).unwrap().translation.x - 0.2).abs() < 1e-5);
            w.remove::<Velocity>(e).unwrap();
            run(&mut w, Step);
            assert!((w.resource::<Physics>().pose(e).unwrap().translation.x - 0.2).abs() < 1e-5);
            w.insert(e, input).unwrap();
            let mut pose = Pose::default();
            pose.translation.x = 10.0;
            w.resource_mut::<Physics>()
                .teleport(e, pose, TeleportVelocity::Reset)
                .unwrap();
            run(&mut w, Step);
            run(&mut w, Step);
            assert_eq!(w.resource::<Physics>().pose(e).unwrap().translation.x, 10.0);
            assert_eq!(w.get::<Velocity>(e).unwrap().linear.x, 0.0);
        }
        #[test]
        fn teleport_is_once_only_preserves_or_resets_velocity_and_updates_kinematic_target() {
            for policy in [TeleportVelocity::Preserve, TeleportVelocity::Reset] {
                let (mut w, e) = setup(Body::dynamic());
                let handle = w.resource::<Physics>().body_for_entity(e).unwrap();
                let mut velocity = w.resource::<Physics>().bodies()[handle].linvel();
                velocity.x = 2.0;
                w.resource_mut::<Physics>()
                    .body_motion(handle)
                    .unwrap()
                    .set_linvel(velocity, true);
                let mut pose = Pose::default();
                pose.translation.x = 10.0;
                w.resource_mut::<Physics>()
                    .teleport(e, pose, policy)
                    .unwrap();
                run(&mut w, Step);
                run(&mut w, Step);
                let expected = if policy == TeleportVelocity::Preserve {
                    10.4
                } else {
                    10.0
                };
                assert!(
                    (w.resource::<Physics>().pose(e).unwrap().translation.x - expected).abs()
                        < 1e-4
                );
            }
            let (mut w, e) = setup(Body::kinematic_position());
            w.insert(e, Target::from(Pose::default())).unwrap();
            let mut pose = Pose::default();
            pose.translation.x = 10.0;
            w.resource_mut::<Physics>()
                .teleport(e, pose, TeleportVelocity::Reset)
                .unwrap();
            run(&mut w, Step);
            run(&mut w, Step);
            assert_eq!(w.resource::<Physics>().pose(e).unwrap().translation.x, 10.0);
            assert_eq!(w.get::<Target>(e).unwrap().translation.x, 10.0);
        }
        #[test]
        fn invalid_teleport_keeps_the_previous_valid_request() {
            let (mut w, e) = setup(Body::dynamic());
            let mut pose = Pose::default();
            pose.translation.x = 10.0;
            w.resource_mut::<Physics>()
                .teleport(e, pose, TeleportVelocity::Reset)
                .unwrap();
            pose.translation.x = f32::NAN;
            assert!(matches!(
                w.resource_mut::<Physics>()
                    .teleport(e, pose, TeleportVelocity::Reset),
                Err(SystemError::InvalidConfiguration { .. })
            ));
            run(&mut w, Step);
            assert_eq!(w.resource::<Physics>().pose(e).unwrap().translation.x, 10.0);
        }
        #[test]
        fn body_type_edits_keep_identity_and_activate_the_new_control_mode() {
            for regular in [false, true] {
                let (mut w, e) = setup(Body::dynamic());
                let handle = w.resource::<Physics>().body_for_entity(e).unwrap();
                w.insert(e, Body::kinematic_position()).unwrap();
                let mut pose = Pose::default();
                pose.translation.x = 4.0;
                w.insert(e, Target::from(pose)).unwrap();
                sync(&mut w, regular).unwrap();
                run(&mut w, Step);
                assert_eq!(w.resource::<Physics>().body_for_entity(e).unwrap(), handle);
                assert!(
                    (w.resource::<Physics>().pose(e).unwrap().translation.x - 4.0).abs() < 1e-5
                );
                w.insert(e, Body::fixed()).unwrap();
                w.get_mut::<Transform>(e).unwrap().translation.x = 8.0;
                sync(&mut w, regular).unwrap();
                run(&mut w, Step);
                assert_eq!(w.resource::<Physics>().body_for_entity(e).unwrap(), handle);
                assert_eq!(w.resource::<Physics>().pose(e).unwrap().translation.x, 8.0);
            }
        }
        #[test]
        fn queued_teleport_does_not_follow_recreated_body() {
            let (mut w, e) = setup(Body::dynamic());
            let mut pose = Pose::default();
            pose.translation.x = 10.0;
            w.resource_mut::<Physics>()
                .teleport(e, pose, TeleportVelocity::Reset)
                .unwrap();
            w.remove::<Body>(e).unwrap();
            sync(&mut w, false).unwrap();
            w.insert(e, Body::dynamic()).unwrap();
            sync(&mut w, false).unwrap();
            run(&mut w, Step);
            assert_eq!(w.resource::<Physics>().pose(e).unwrap().translation.x, 0.0);
        }
        #[test]
        fn settings_update_in_place_and_unchanged_settings_leave_sleep_intact() {
            for regular in [false, true] {
                let (mut w, e) = setup(Body::dynamic());
                let h = w.resource::<Physics>().body_for_entity(e).unwrap();
                let ch = w.resource::<Physics>().bodies()[h].colliders()[0];
                w.resource_mut::<Physics>().body_motion(h).unwrap().sleep();
                sync(&mut w, regular).unwrap();
                assert!(w.resource::<Physics>().bodies()[h].is_sleeping());
                w.get_mut::<Body>(e).unwrap().gravity_scale = 0.0;
                w.get_mut::<Body>(e).unwrap().linear_damping = 0.5;
                let mut col = Collider::ball(1.0);
                col.density = 3.0;
                col.friction = 0.9;
                w.insert(e, col).unwrap();
                sync(&mut w, regular).unwrap();
                let p = w.resource::<Physics>();
                assert_eq!(p.body_for_entity(e).unwrap(), h);
                assert_eq!(p.bodies()[h].colliders()[0], ch);
                assert_eq!(p.bodies()[h].gravity_scale(), 0.0);
                assert_eq!(p.bodies()[h].linear_damping(), 0.5);
                assert!((p.colliders()[ch].friction() - 0.9).abs() < 1e-6);
                assert_eq!(p.colliders()[ch].density(), 3.0);
                assert_eq!(p.colliders()[ch].shape().as_ball().unwrap().radius, 1.0);
            }
        }
        #[test]
        fn joint_endpoint_changes_rebuild_only_the_joint() {
            for regular in [false, true] {
                let (mut w, a) = setup(Body::dynamic());
                let b = w
                    .spawn_with((
                        Transform::from_translation(Vec3::new(3.0, 0.0, 0.0)),
                        Body::fixed(),
                        Collider::ball(0.5),
                    ))
                    .unwrap();
                let c = w
                    .spawn_with((
                        Transform::from_translation(Vec3::new(6.0, 0.0, 0.0)),
                        Body::fixed(),
                        Collider::ball(0.5),
                    ))
                    .unwrap();
                sync(&mut w, regular).unwrap();
                let j = w
                    .spawn_with((Joint::fixed(a, b, Default::default(), Default::default()),))
                    .unwrap();
                joints(&mut w, regular);
                let old = w.resource::<Physics>().joint_for_entity(j).unwrap();
                joints(&mut w, regular);
                assert_eq!(w.resource::<Physics>().joint_for_entity(j).unwrap(), old);
                w.get_mut::<Joint>(j).unwrap().body2 = c;
                joints(&mut w, regular);
                let p = w.resource::<Physics>();
                let new = p.joint_for_entity(j).unwrap();
                assert_ne!(new, old);
                assert_eq!(p.bodies().len(), 3);
                assert_eq!(p.impulse_joints().len(), 1);
                assert_eq!(
                    p.impulse_joints().get(new).unwrap().body2(),
                    p.body_for_entity(c).unwrap()
                );
            }
        }
        #[test]
        fn hierarchy_and_scale_are_rejected_at_sync_and_again_at_step() {
            for regular in [false, true] {
                for parented in [false, true] {
                    let (mut w, e) = setup(Body::dynamic());
                    if parented {
                        let parent = w.spawn_with((Transform::IDENTITY,)).unwrap();
                        set_parent(&mut w, e, parent);
                    } else {
                        w.get_mut::<Transform>(e).unwrap().scale.x = 2.0;
                    }
                    assert!(matches!(
                        sync(&mut w, regular).unwrap_err().as_slice(),
                        [SystemError::InvalidConfiguration { .. }]
                    ));
                    let before = w.resource::<Physics>().pose(e).unwrap();
                    let mut s = SystemsContainer::new();
                    s.add(Step);
                    assert!(matches!(
                        EcsRunner::single_thread().run(&mut w, &s).as_slice(),
                        [SystemError::InvalidConfiguration { .. }]
                    ));
                    assert_eq!(w.resource::<Physics>().pose(e).unwrap(), before);
                }
            }
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod two_d {
    use super::*;
    use redlilium_ecs::physics::{
        components2d::{Collider2D as Collider, ImpulseJoint2D as Joint, RigidBody2D as Body},
        control2d::{
            KinematicTarget2D as Target, KinematicVelocity2D as Velocity, PhysicsPose2D as Pose,
        },
        systems2d::{
            StepPhysics2D as Step, SyncPhysicsBodies2D as Sync,
            SyncPhysicsBodiesSystem2D as SyncRegular, SyncPhysicsJoints2D as JointSync,
            SyncPhysicsJointsSystem2D as JointSyncRegular,
        },
        world2d::PhysicsWorld2D as Physics,
    };
    contract_tests!();
}
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod three_d {
    use super::*;
    use redlilium_ecs::physics::{
        components3d::{Collider3D as Collider, ImpulseJoint3D as Joint, RigidBody3D as Body},
        control3d::{
            KinematicTarget3D as Target, KinematicVelocity3D as Velocity, PhysicsPose3D as Pose,
        },
        systems3d::{
            InterpolatePhysics, RecordPhysicsPose, StepPhysics3D as Step,
            SyncPhysicsBodies3D as Sync, SyncPhysicsBodiesSystem3D as SyncRegular,
            SyncPhysicsJoints3D as JointSync, SyncPhysicsJointsSystem3D as JointSyncRegular,
        },
        world3d::{PhysicsInterpolation, PhysicsWorld3D as Physics},
    };
    contract_tests!();

    #[test]
    fn teleport_resets_history_and_record_ignores_presentation() {
        let (mut w, e) = setup(Body::dynamic());
        run(&mut w, RecordPhysicsPose);
        w.get_mut::<Transform>(e).unwrap().translation.x = -42.0;
        run(&mut w, RecordPhysicsPose);
        assert_eq!(
            w.get::<PhysicsInterpolation>(e).unwrap().cur_translation.x,
            0.0
        );
        let mut pose = Pose::default();
        pose.translation.x = 10.0;
        w.resource_mut::<Physics>()
            .teleport(e, pose, TeleportVelocity::Reset)
            .unwrap();
        run(&mut w, Step);
        run(&mut w, RecordPhysicsPose);
        let h = w.get::<PhysicsInterpolation>(e).unwrap();
        assert_eq!(h.prev_translation.x, 10.0);
        assert_eq!(h.cur_translation.x, 10.0);
        run(&mut w, InterpolatePhysics);
        assert_eq!(w.get::<Transform>(e).unwrap().translation.x, 10.0);
    }
}
