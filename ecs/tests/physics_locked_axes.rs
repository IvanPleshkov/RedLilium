#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_core::math::{Vec3, quat_from_rotation_z};
use redlilium_ecs::*;

macro_rules! axis_tests {
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
        fn setup(desc: Body, regular: bool) -> (World, Entity) {
            let mut w = World::new();
            register_std_components(&mut w);
            w.insert_resource(Physics::default());
            let mut transform = Transform::IDENTITY;
            transform.rotation = quat_from_rotation_z(0.6);
            let e = w.spawn_with((transform, desc, collider())).unwrap();
            sync(&mut w, regular);
            w.resource_mut::<Physics>().gravity = Default::default();
            w.resource_mut::<Physics>().integration_parameters.dt = 0.01;
            // Finish initial mass-property updates before testing impulses.
            step(&mut w);
            (w, e)
        }

        #[test]
        fn every_translation_axis_is_world_relative_and_blocks_velocity_force_and_impulse() {
            for regular in [false, true] {
                for axis in 0..DIM {
                    let (mut w, e) = setup(
                        Body::dynamic().with_locked_axes(Some(translation_lock(axis))),
                        regular,
                    );
                    let h = w.resource::<Physics>().body_for_entity(e).unwrap();
                    w.resource_mut::<Physics>().gravity = Vector::splat(9.0);
                    {
                        let mut p = w.resource_mut::<Physics>();
                        let mut motion = p.body_motion(h).unwrap();
                        motion.set_linvel(Vector::splat(2.0), true);
                        assert_eq!(motion.linvel()[axis], 0.0);
                        motion.add_force(Vector::splat(3.0), true);
                        motion.apply_impulse(Vector::splat(1.0), true);
                        motion.apply_impulse_at_point(Vector::splat(1.0), Vector::splat(0.2), true);
                        assert_eq!(motion.linvel()[axis], 0.0);
                    }
                    for _ in 0..15 {
                        step(&mut w);
                    }
                    let p = w.resource::<Physics>();
                    assert!(p.bodies()[h].translation()[axis].abs() < 1e-6);
                    assert_eq!(p.bodies()[h].linvel()[axis], 0.0);
                    for other in 0..DIM {
                        if other != axis {
                            assert!(p.bodies()[h].translation()[other] > 0.01);
                        }
                    }
                }
            }
        }

        #[test]
        fn every_rotation_axis_blocks_angular_velocity_and_torque_in_world_space() {
            for regular in [false, true] {
                for axis in 0..ANG_DIM {
                    let (mut w, e) = setup(
                        Body::dynamic().with_locked_axes(Some(rotation_lock(axis))),
                        regular,
                    );
                    let h = w.resource::<Physics>().body_for_entity(e).unwrap();
                    {
                        let mut p = w.resource_mut::<Physics>();
                        let mut motion = p.body_motion(h).unwrap();
                        motion.set_angvel(spin(1.0), true);
                        assert_eq!(angular_component(motion.angvel(), axis), 0.0);
                        motion.add_torque(spin(0.1), true);
                        motion.apply_torque_impulse(spin(0.01), true);
                        assert_eq!(angular_component(motion.angvel(), axis), 0.0);
                    }
                    for _ in 0..15 {
                        step(&mut w);
                        let p = w.resource::<Physics>();
                        assert!(
                            angular_component(p.bodies()[h].angvel(), axis).abs() < 1e-6,
                            "{:?}",
                            p.bodies()[h].angvel()
                        );
                    }
                    let p = w.resource::<Physics>();
                    for other in 0..ANG_DIM {
                        if other != axis {
                            assert!(angular_component(p.bodies()[h].angvel(), other).abs() > 0.01);
                        }
                    }
                }
            }
        }

        #[test]
        fn live_edits_project_only_locked_velocities_and_keep_body_collider_and_joint() {
            for regular in [false, true] {
                let (mut w, e) = setup(Body::dynamic(), regular);
                let other = w
                    .spawn_with((
                        Transform::from_translation(Vec3::new(10.0, 0.0, 0.0)),
                        Body::fixed(),
                        Collider::ball(0.5),
                    ))
                    .unwrap();
                sync(&mut w, regular);
                let j = w
                    .spawn_with((Joint::fixed(
                        e,
                        other,
                        Default::default(),
                        Default::default(),
                    ),))
                    .unwrap();
                let mut systems = SystemsContainer::new();
                systems.add_exclusive(JointSync);
                assert!(EcsRunner::single_thread().run(&mut w, &systems).is_empty());
                let h = w.resource::<Physics>().body_for_entity(e).unwrap();
                let c = w.resource::<Physics>().bodies()[h].colliders()[0];
                let joint = w.resource::<Physics>().joint_for_entity(j).unwrap();
                {
                    let mut p = w.resource_mut::<Physics>();
                    let mut motion = p.body_motion(h).unwrap();
                    motion.set_linvel(Vector::splat(2.0), true);
                    motion.set_angvel(spin(3.0), true);
                }
                let pose = w.resource::<Physics>().pose(e).unwrap();
                let locks = Locks {
                    translation_x: true,
                    ..Locks::rotations()
                };
                w.insert(e, Body::dynamic().with_locked_axes(Some(locks)))
                    .unwrap();
                sync(&mut w, regular);
                {
                    let p = w.resource::<Physics>();
                    assert_eq!(p.body_for_entity(e), Some(h));
                    assert_eq!(p.bodies()[h].colliders()[0], c);
                    assert_eq!(p.joint_for_entity(j), Some(joint));
                    assert_eq!(p.pose(e).unwrap(), pose);
                    assert_eq!(p.bodies()[h].linvel().x, 0.0);
                    assert_eq!(p.bodies()[h].linvel().y, 2.0);
                    assert_eq!(p.bodies()[h].angvel(), spin(0.0));
                }
                w.resource_mut::<Physics>().body_motion(h).unwrap().sleep();
                sync(&mut w, regular);
                assert!(w.resource::<Physics>().bodies()[h].is_sleeping());
                w.get_mut::<Body>(e).unwrap().locked_axes = None;
                sync(&mut w, regular);
                assert!(!w.resource::<Physics>().bodies()[h].is_sleeping());
                assert_eq!(w.resource::<Physics>().bodies()[h].linvel(), Vector::ZERO);
                assert_eq!(w.resource::<Physics>().bodies()[h].angvel(), spin(0.0));
                let mut p = w.resource_mut::<Physics>();
                let mut motion = p.body_motion(h).unwrap();
                motion.set_linvel(Vector::splat(2.0), true);
                motion.set_angvel(spin(3.0), true);
                assert_eq!(motion.linvel(), Vector::splat(2.0));
                assert_eq!(motion.angvel(), spin(3.0));
            }
        }

        #[test]
        fn kinematic_motion_bypasses_locks_and_switching_to_dynamic_clamps_velocity() {
            for regular in [false, true] {
                let locks = Some(Locks::all());
                let (mut w, e) = setup(Body::kinematic_velocity().with_locked_axes(locks), regular);
                let mut input = Velocity::default();
                input.linear.x = 2.0;
                w.insert(e, input).unwrap();
                step(&mut w);
                let h = w.resource::<Physics>().body_for_entity(e).unwrap();
                assert!(w.resource::<Physics>().pose(e).unwrap().translation.x > 0.0);
                assert_eq!(w.resource::<Physics>().bodies()[h].linvel().x, 2.0);
                w.insert(e, Body::dynamic().with_locked_axes(locks))
                    .unwrap();
                sync(&mut w, regular);
                assert_eq!(w.resource::<Physics>().bodies()[h].linvel(), Vector::ZERO);
                let p = w.resource::<Physics>().pose(e).unwrap();
                step(&mut w);
                assert_eq!(w.resource::<Physics>().pose(e).unwrap(), p);
                w.insert(e, Body::kinematic_position().with_locked_axes(locks))
                    .unwrap();
                sync(&mut w, regular);
                let mut target: Target = w.resource::<Physics>().pose(e).unwrap().into();
                target.translation.x = 5.0;
                w.insert(e, target).unwrap();
                step(&mut w);
                assert!(
                    (w.resource::<Physics>().pose(e).unwrap().translation.x - 5.0).abs() < 1e-5
                );
                w.insert(e, Body::fixed().with_locked_axes(locks)).unwrap();
                sync(&mut w, regular);
                w.get_mut::<Transform>(e).unwrap().translation.x = 7.0;
                step(&mut w);
                assert!(
                    (w.resource::<Physics>().pose(e).unwrap().translation.x - 7.0).abs() < 1e-5
                );
            }
        }

        #[test]
        fn locked_body_can_teleport_and_unlock_resumes_persistent_forces() {
            for regular in [false, true] {
                let (mut w, e) = setup(
                    Body::dynamic().with_locked_axes(Some(Locks::all())),
                    regular,
                );
                let h = w.resource::<Physics>().body_for_entity(e).unwrap();
                let mut pose = w.resource::<Physics>().pose(e).unwrap();
                pose.translation.x = 8.0;
                w.resource_mut::<Physics>()
                    .teleport(e, pose, physics::TeleportVelocity::Preserve)
                    .unwrap();
                w.resource_mut::<Physics>()
                    .body_motion(h)
                    .unwrap()
                    .add_force(Vector::X, true);
                step(&mut w);
                assert!(
                    (w.resource::<Physics>().pose(e).unwrap().translation.x - 8.0).abs() < 1e-5
                );
                assert_eq!(w.resource::<Physics>().bodies()[h].linvel(), Vector::ZERO);
                w.get_mut::<Body>(e).unwrap().locked_axes = None;
                sync(&mut w, regular);
                step(&mut w);
                assert!(w.resource::<Physics>().bodies()[h].linvel().x > 0.0);
                assert!(w.resource::<Physics>().pose(e).unwrap().translation.x > 8.0);
            }
        }

        #[test]
        fn contact_solver_cannot_move_a_fully_locked_dynamic_body() {
            for regular in [false, true] {
                let (mut w, e) = setup(
                    Body::dynamic().with_locked_axes(Some(Locks::all())),
                    regular,
                );
                w.insert(e, Collider::ball(1.0)).unwrap();
                let moving = w
                    .spawn_with((
                        Transform::from_translation(Vec3::new(-2.0, 0.0, 0.0)),
                        Body::dynamic(),
                        Collider::ball(0.2),
                    ))
                    .unwrap();
                sync(&mut w, regular);
                step(&mut w);
                let h = w.resource::<Physics>().body_for_entity(moving).unwrap();
                w.resource_mut::<Physics>()
                    .body_motion(h)
                    .unwrap()
                    .set_linvel(Vector::X * 15.0, true);
                let before = w.resource::<Physics>().pose(e).unwrap();
                for _ in 0..20 {
                    step(&mut w);
                }
                let p = w.resource::<Physics>();
                assert!((p.pose(e).unwrap().translation - before.translation).norm() < 1e-6);
                let locked = p.body_for_entity(e).unwrap();
                assert_eq!(p.bodies()[locked].linvel(), Vector::ZERO);
                assert_eq!(p.bodies()[locked].angvel(), spin(0.0));
                assert!(p.pose(moving).unwrap().translation.x < -1.0);
            }
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod two_d {
    use super::*;
    use physics::{
        LockedAxes2D as Locks,
        components2d::{Collider2D as Collider, ImpulseJoint2D as Joint, RigidBody2D as Body},
        control2d::{KinematicTarget2D as Target, KinematicVelocity2D as Velocity},
        rapier2d::prelude::{Real, Vector},
        systems2d::{
            StepPhysics2D as Step, SyncPhysicsBodies2D as Sync,
            SyncPhysicsBodiesSystem2D as SyncRegular, SyncPhysicsJoints2D as JointSync,
        },
        world2d::PhysicsWorld2D as Physics,
    };
    const DIM: usize = 2;
    const ANG_DIM: usize = 1;
    fn collider() -> Collider {
        Collider::cuboid(0.2, 0.4)
    }
    fn spin(v: Real) -> Real {
        v
    }
    fn angular_component(v: Real, _axis: usize) -> Real {
        v
    }
    fn translation_lock(axis: usize) -> Locks {
        let mut locks = Locks::default();
        match axis {
            0 => locks.translation_x = true,
            1 => locks.translation_y = true,
            _ => unreachable!(),
        };
        locks
    }
    fn rotation_lock(_: usize) -> Locks {
        Locks::rotations()
    }
    axis_tests!();
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod three_d {
    use super::*;
    use physics::{
        LockedAxes3D as Locks,
        components3d::{Collider3D as Collider, ImpulseJoint3D as Joint, RigidBody3D as Body},
        control3d::{KinematicTarget3D as Target, KinematicVelocity3D as Velocity},
        rapier3d::prelude::{Real, Vector},
        systems3d::{
            StepPhysics3D as Step, SyncPhysicsBodies3D as Sync,
            SyncPhysicsBodiesSystem3D as SyncRegular, SyncPhysicsJoints3D as JointSync,
        },
        world3d::PhysicsWorld3D as Physics,
    };
    const DIM: usize = 3;
    const ANG_DIM: usize = 3;
    fn collider() -> Collider {
        Collider::cuboid(0.2, 0.4, 0.8)
    }
    fn spin(v: Real) -> Vector {
        Vector::splat(v)
    }
    fn angular_component(v: Vector, axis: usize) -> Real {
        v[axis]
    }
    fn translation_lock(axis: usize) -> Locks {
        let mut locks = Locks::default();
        match axis {
            0 => locks.translation_x = true,
            1 => locks.translation_y = true,
            2 => locks.translation_z = true,
            _ => unreachable!(),
        };
        locks
    }
    fn rotation_lock(axis: usize) -> Locks {
        let mut locks = Locks::default();
        match axis {
            0 => locks.rotation_x = true,
            1 => locks.rotation_y = true,
            2 => locks.rotation_z = true,
            _ => unreachable!(),
        };
        locks
    }
    axis_tests!();

    #[test]
    fn gyroscopic_correction_is_disabled_only_while_rotation_is_constrained() {
        for regular in [false, true] {
            for locks in [Some(Locks::rotations()), None, Some(Locks::translations())] {
                let (mut w, e) = setup(Body::dynamic().with_locked_axes(locks), regular);
                let h = w.resource::<Physics>().body_for_entity(e).unwrap();
                assert_eq!(
                    w.resource::<Physics>().bodies()[h].gyroscopic_forces_enabled(),
                    locks != Some(Locks::rotations())
                );
                for next in [
                    Some(Locks {
                        rotation_y: true,
                        ..Default::default()
                    }),
                    None,
                    Some(Locks::translations()),
                ] {
                    w.get_mut::<Body>(e).unwrap().locked_axes = next;
                    sync(&mut w, regular);
                    assert_eq!(
                        w.resource::<Physics>().bodies()[h].gyroscopic_forces_enabled(),
                        !next.is_some_and(|l| l.rotation_y)
                    );
                }
            }
        }
    }
}
