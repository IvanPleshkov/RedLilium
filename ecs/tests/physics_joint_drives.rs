//! Joint authoring, in-place actuation and solver behavior in both precisions.
#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_ecs::physics::{JointLimits, JointMotor, JointMotorError, JointMotorModel};
use redlilium_ecs::*;

macro_rules! tests {
    () => {
        fn world() -> World {
            let mut w = World::new();
            register_std_components(&mut w);
            w.insert_resource(Physics::default());
            w.resource_mut::<Physics>().gravity = Default::default();
            w
        }
        fn sync(w: &mut World, regular: bool) -> Vec<SystemError> {
            let mut s = SystemsContainer::new();
            if regular {
                s.add(JointSyncRegular);
            } else {
                s.add_exclusive(JointSync);
            }
            EcsRunner::single_thread().run(w, &s)
        }
        fn valid(w: &mut World, regular: bool) {
            let errors = sync(w, regular);
            assert!(errors.is_empty(), "{errors:?}");
        }
        fn fixture(kind: Kind, regular: bool) -> (World, Entity, Entity, Entity) {
            let mut w = world();
            // Only the dynamic body has a collider, avoiding contact constraints.
            let a = w
                .spawn_with((Name::new("base"), Transform::IDENTITY, Body::fixed()))
                .unwrap();
            let b = w
                .spawn_with((
                    Name::new("moving"),
                    Transform::IDENTITY,
                    Body::dynamic(),
                    Collider::ball(0.5),
                ))
                .unwrap();
            let mut s = SystemsContainer::new();
            s.add_exclusive(BodySync);
            assert!(EcsRunner::single_thread().run(&mut w, &s).is_empty());
            let j = w
                .spawn_with((Name::new("joint"), Joint::new(a, b, kind)))
                .unwrap();
            valid(&mut w, regular);
            (w, a, b, j)
        }
        fn advance(w: &mut World, count: usize) {
            let mut s = SystemsContainer::new();
            s.add(Step);
            let runner = EcsRunner::single_thread();
            for _ in 0..count {
                let errors = runner.run(w, &s);
                assert!(errors.is_empty(), "{errors:?}");
            }
        }
        fn coordinate(w: &World, j: Entity) -> f64 {
            let p = w.resource::<Physics>();
            let d = w.get::<Joint>(j).unwrap();
            let native = p
                .impulse_joints()
                .get(p.joint_for_entity(j).unwrap())
                .unwrap();
            let f1 = p.bodies()[p.body_for_entity(d.body1).unwrap()].position()
                * native.data.local_frame1;
            let f2 = p.bodies()[p.body_for_entity(d.body2).unwrap()].position()
                * native.data.local_frame2;
            let relative = f1.inverse() * f2;
            if d.joint_type == Kind::Prismatic {
                relative.translation.x as f64
            } else {
                angle(relative.rotation)
            }
        }
        #[test]
        fn live_parameter_edits_keep_handle_and_unchanged_sync_keeps_sleep() {
            for regular in [false, true] {
                for kind in [Kind::Revolute, Kind::Prismatic] {
                    let (mut w, _, b, j) = fixture(kind, regular);
                    let h = w.resource::<Physics>().joint_for_entity(j).unwrap();
                    let bh = w.resource::<Physics>().body_for_entity(b).unwrap();
                    w.resource_mut::<Physics>().body_motion(bh).unwrap().sleep();
                    valid(&mut w, regular);
                    assert!(w.resource::<Physics>().bodies()[bh].is_sleeping());
                    w.get_mut::<Joint>(j).unwrap().motor =
                        Some(JointMotor::position(0.4, 40.0, 10.0));
                    valid(&mut w, regular);
                    assert!(!w.resource::<Physics>().bodies()[bh].is_sleeping());
                    w.get_mut::<Joint>(j)
                        .unwrap()
                        .set_motor_position_target(0.6)
                        .unwrap();
                    w.get_mut::<Joint>(j)
                        .unwrap()
                        .set_motor_velocity_target(0.1)
                        .unwrap();
                    w.get_mut::<Joint>(j).unwrap().limits = Some(JointLimits {
                        min: -0.8,
                        max: 0.8,
                    });
                    valid(&mut w, regular);
                    assert_eq!(w.resource::<Physics>().joint_for_entity(j), Some(h));
                    let p = w.resource::<Physics>();
                    let data = &p.impulse_joints().get(h).unwrap().data;
                    let axis = if kind == Kind::Prismatic {
                        Axis::LinX
                    } else {
                        Axis::AngX
                    };
                    assert!((data.motor(axis).unwrap().target_pos as f64 - 0.6).abs() < 1e-6);
                    assert!((data.motor(axis).unwrap().target_vel as f64 - 0.1).abs() < 1e-6);
                    drop(p);
                    w.get_mut::<Joint>(j).unwrap().motor = None;
                    w.get_mut::<Joint>(j).unwrap().limits = None;
                    valid(&mut w, regular);
                    let p = w.resource::<Physics>();
                    let data = &p.impulse_joints().get(h).unwrap().data;
                    assert!(data.motor(axis).is_none());
                    assert!(data.limits(axis).is_none());
                    drop(p);
                    w.get_mut::<Joint>(j).unwrap().local_frame1.translation.x = 0.1;
                    valid(&mut w, regular);
                    assert_ne!(w.resource::<Physics>().joint_for_entity(j), Some(h));
                }
            }
        }
        #[test]
        fn invalid_batch_and_target_edits_are_transactional() {
            for regular in [false, true] {
                let (mut w, a, b, j) = fixture(Kind::Revolute, regular);
                let good = Joint::new(a, b, Kind::Revolute)
                    .with_limits(Some(JointLimits {
                        min: -1.0,
                        max: 1.0,
                    }))
                    .with_motor(Some(JointMotor::position(0.0, 30.0, 5.0)));
                w.insert(j, good.clone()).unwrap();
                valid(&mut w, regular);
                let j2 = w.spawn_with((Joint::new(a, b, Kind::Prismatic),)).unwrap();
                valid(&mut w, regular);
                w.get_mut::<Joint>(j2).unwrap().motor = Some(JointMotor::velocity(2.0, 1.0));
                let mut cases = Vec::new();
                for limits in [
                    JointLimits {
                        min: 1.0,
                        max: -1.0,
                    },
                    JointLimits {
                        min: f32::NAN,
                        max: 0.0,
                    },
                    JointLimits {
                        min: -4.0,
                        max: 1.0,
                    },
                    JointLimits {
                        min: -::std::f32::consts::PI,
                        max: ::std::f32::consts::PI,
                    },
                ] {
                    cases.push(good.clone().with_limits(Some(limits)));
                }
                for motor in [
                    JointMotor::position(2.0, 1.0, 1.0),
                    JointMotor::position(0.0, 0.0, 1.0),
                    JointMotor::position(0.0, f32::INFINITY, 1.0),
                    JointMotor::velocity(0.0, 0.0),
                    JointMotor::velocity(f32::NAN, 1.0),
                    JointMotor::velocity(0.0, -1.0),
                    JointMotor::velocity(0.0, 1.0).with_max_effort(Some(-1.0)),
                    JointMotor::velocity(0.0, 1.0).with_max_effort(Some(f32::INFINITY)),
                ] {
                    cases.push(good.clone().with_motor(Some(motor)));
                }
                cases.push(Joint::new(a, b, Kind::Fixed).with_motor(good.motor));
                cases.push(Joint::new(a, b, Kind::Fixed).with_limits(good.limits));
                for bad in cases {
                    w.insert(j, bad.clone()).unwrap();
                    let errors = sync(&mut w, regular);
                    assert!(
                        matches!(
                            errors.as_slice(),
                            [SystemError::InvalidConfiguration { .. }]
                        ),
                        "{errors:?} {bad:?}"
                    );
                    let p = w.resource::<Physics>();
                    let data = &p
                        .impulse_joints()
                        .get(p.joint_for_entity(j2).unwrap())
                        .unwrap()
                        .data;
                    assert!(data.motor(Axis::LinX).is_none());
                }
                w.insert(j, good.clone()).unwrap();
                let mut desc = w.get_mut::<Joint>(j).unwrap();
                assert!(desc.set_motor_position_target(2.0).is_err());
                assert_eq!(*desc, good);
                assert!(desc.set_motor_velocity_target(f32::INFINITY).is_err());
                assert_eq!(*desc, good);
                desc.motor = None;
                assert_eq!(
                    desc.set_motor_velocity_target(1.0),
                    Err(JointMotorError::Disabled)
                );
                desc.motor = Some(JointMotor::velocity(1.0, 1.0));
                assert_eq!(
                    desc.set_motor_position_target(0.0),
                    Err(JointMotorError::RequiresPositionDrive)
                );
                valid(&mut w, regular);
            }
        }
        #[test]
        fn position_drives_converge_and_velocity_drives_obey_limits() {
            for regular in [false, true] {
                for kind in [Kind::Revolute, Kind::Prismatic] {
                    for model in [
                        JointMotorModel::AccelerationBased,
                        JointMotorModel::ForceBased,
                    ] {
                        let (mut w, _, _, j) = fixture(kind, regular);
                        w.get_mut::<Joint>(j).unwrap().motor =
                            Some(JointMotor::position(0.7, 50.0, 12.0).with_model(model));
                        valid(&mut w, regular);
                        advance(&mut w, 180);
                        assert!(
                            (coordinate(&w, j) - 0.7).abs() < 0.025,
                            "{:?} {:?}: {}",
                            kind,
                            model,
                            coordinate(&w, j)
                        );
                        w.get_mut::<Joint>(j).unwrap().motor =
                            Some(JointMotor::velocity(-2.0, 10.0).with_model(model));
                        w.get_mut::<Joint>(j).unwrap().limits = Some(JointLimits {
                            min: -0.3,
                            max: 0.9,
                        });
                        valid(&mut w, regular);
                        advance(&mut w, 120);
                        assert!(
                            (coordinate(&w, j) + 0.3).abs() < 0.04,
                            "{:?}: {}",
                            kind,
                            coordinate(&w, j)
                        );
                        // Equal limits lock the coordinate at an arbitrary value.
                        w.get_mut::<Joint>(j).unwrap().limits =
                            Some(JointLimits { min: 0.2, max: 0.2 });
                        valid(&mut w, regular);
                        advance(&mut w, 120);
                        assert!((coordinate(&w, j) - 0.2).abs() < 0.04);
                    }
                }
            }
        }
        #[test]
        fn unlimited_velocity_drive_and_disabling_motor_preserve_free_motion() {
            for kind in [Kind::Revolute, Kind::Prismatic] {
                let (mut w, _, b, j) = fixture(kind, true);
                w.get_mut::<Joint>(j).unwrap().motor = Some(JointMotor::velocity(2.0, 10.0));
                valid(&mut w, true);
                advance(&mut w, 240);
                let h = w.resource::<Physics>().body_for_entity(b).unwrap();
                let speed = |w: &World| {
                    let p = w.resource::<Physics>();
                    let body = &p.bodies()[h];
                    if kind == Kind::Prismatic {
                        body.linvel().x as f64
                    } else {
                        angular_velocity(body)
                    }
                };
                assert!((speed(&w) - 2.0).abs() < 0.02);
                w.get_mut::<Joint>(j).unwrap().motor = None;
                valid(&mut w, true);
                advance(&mut w, 60);
                assert!((speed(&w) - 2.0).abs() < 0.02);
                w.resource_mut::<Physics>().body_motion(h).unwrap().sleep();
                valid(&mut w, true);
                advance(&mut w, 1);
                assert!(w.resource::<Physics>().bodies()[h].is_sleeping());
            }
        }
        #[test]
        fn wide_angular_moves_use_explicit_intermediate_targets() {
            let (mut w, _, _, j) = fixture(Kind::Revolute, false);
            w.get_mut::<Joint>(j).unwrap().motor = Some(JointMotor::position(2.8, 30.0, 10.0));
            w.get_mut::<Joint>(j).unwrap().limits = Some(JointLimits {
                min: -3.0,
                max: 3.0,
            });
            valid(&mut w, false);
            advance(&mut w, 240);
            assert!((coordinate(&w, j) - 2.8).abs() < 0.03);
            w.get_mut::<Joint>(j)
                .unwrap()
                .set_motor_position_target(-2.8)
                .unwrap();
            valid(&mut w, false);
            advance(&mut w, 360);
            // The shortest rotation pushes against the +3 stop. An intermediate
            // target selects the longer path inside the configured limits.
            assert!((coordinate(&w, j) - 3.0).abs() < 0.03);
            w.get_mut::<Joint>(j)
                .unwrap()
                .set_motor_position_target(0.0)
                .unwrap();
            valid(&mut w, false);
            advance(&mut w, 240);
            assert!(coordinate(&w, j).abs() < 0.03);
            w.get_mut::<Joint>(j)
                .unwrap()
                .set_motor_position_target(-2.8)
                .unwrap();
            valid(&mut w, false);
            advance(&mut w, 240);
            assert!(
                (coordinate(&w, j) + 2.8).abs() < 0.03,
                "{}",
                coordinate(&w, j)
            );
        }
        #[test]
        fn effort_cap_and_zero_effort_work_for_force_and_torque() {
            for kind in [Kind::Revolute, Kind::Prismatic] {
                let (mut w, _, b, j) = fixture(kind, false);
                w.get_mut::<Joint>(j).unwrap().motor =
                    Some(JointMotor::velocity(100.0, 100.0).with_max_effort(Some(0.0)));
                valid(&mut w, false);
                advance(&mut w, 5);
                assert!(coordinate(&w, j).abs() < 1e-6);
                w.get_mut::<Joint>(j)
                    .unwrap()
                    .motor
                    .as_mut()
                    .unwrap()
                    .max_effort = Some(0.01);
                valid(&mut w, false);
                advance(&mut w, 1);
                let p = w.resource::<Physics>();
                let data = &p
                    .impulse_joints()
                    .get(p.joint_for_entity(j).unwrap())
                    .unwrap()
                    .data;
                let axis = if kind == Kind::Prismatic {
                    Axis::LinX
                } else {
                    Axis::AngX
                };
                let impulse = data.motor(axis).unwrap().impulse.abs() as f64;
                assert!(
                    impulse > 0.0 && impulse <= 0.01 * p.integration_parameters.dt as f64 + 1e-7,
                    "{impulse}"
                );
                let bh = p.body_for_entity(b).unwrap();
                assert!(!p.bodies()[bh].is_sleeping());
            }
        }
        #[test]
        fn independent_frames_define_zero_and_slider_axis() {
            for kind in [Kind::Revolute, Kind::Prismatic] {
                let (mut w, _, b, j) = fixture(kind, false);
                // Frame 1 rotated +90 degrees, frame 2 identity: zero hinge angle
                // means body 2 rotated +90 degrees; slider moves along world Y.
                w.get_mut::<Joint>(j).unwrap().local_frame1 = rotated_frame();
                w.get_mut::<Joint>(j).unwrap().motor = Some(JointMotor::position(
                    if kind == Kind::Prismatic { 0.6 } else { 0.0 },
                    50.0,
                    12.0,
                ));
                valid(&mut w, false);
                advance(&mut w, 240);
                let expected = if kind == Kind::Prismatic { 0.6 } else { 0.0 };
                assert!(
                    (coordinate(&w, j) - expected).abs() < 0.03,
                    "{}",
                    coordinate(&w, j)
                );
                if kind == Kind::Prismatic {
                    let p = w.resource::<Physics>();
                    let body = &p.bodies()[p.body_for_entity(b).unwrap()];
                    assert!((body.translation().y as f64 - 0.6).abs() < 0.03);
                    assert!(body.translation().x.abs() < 0.03);
                }
            }
        }
        #[test]
        fn scene_roundtrip_preserves_frames_settings_targets_and_remaps_bodies() {
            let (mut w, _, _, j) = fixture(Kind::Prismatic, false);
            let mut desc = w.get_mut::<Joint>(j).unwrap();
            desc.local_frame1 = rotated_frame();
            desc.local_frame2.translation.x = 0.4;
            desc.limits = Some(JointLimits {
                min: -0.2,
                max: 0.8,
            });
            desc.motor = Some(
                JointMotor::position(0.5, 30.0, 8.0)
                    .with_max_effort(Some(4.0))
                    .with_model(JointMotorModel::ForceBased),
            );
            desc.set_motor_velocity_target(0.2).unwrap();
            let expected = desc.clone();
            drop(desc);
            let snapshot = w.serialize_world().unwrap();
            let mut snapshots = vec![snapshot.clone()];
            #[cfg(feature = "serialize-ron")]
            snapshots.push(
                serialize::decode(
                    &serialize::encode(&snapshot, serialize::Format::Ron).unwrap(),
                    serialize::Format::Ron,
                )
                .unwrap(),
            );
            #[cfg(feature = "serialize-bincode")]
            snapshots.push(
                serialize::decode(
                    &serialize::encode(&snapshot, serialize::Format::Bincode).unwrap(),
                    serialize::Format::Bincode,
                )
                .unwrap(),
            );
            for snapshot in snapshots.drain(..) {
                let mut restored = world();
                for _ in 0..10 {
                    restored.spawn();
                }
                let entities = restored.deserialize_world_into(&snapshot).unwrap();
                let actual = entities
                    .iter()
                    .find_map(|&e| restored.get::<Joint>(e))
                    .unwrap();
                assert_ne!(actual.body1, expected.body1);
                assert_ne!(actual.body2, expected.body2);
                assert_eq!(actual.local_frame1, expected.local_frame1);
                assert_eq!(actual.local_frame2, expected.local_frame2);
                assert_eq!(actual.motor, expected.motor);
                assert_eq!(actual.limits, expected.limits);
                assert!(entities.contains(&actual.body1));
                assert!(entities.contains(&actual.body2));
            }
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod dim2 {
    use super::*;
    use redlilium_ecs::physics::{
        components2d::{
            Collider2D as Collider, ImpulseJoint2D as Joint, JointFrame2D as Frame,
            JointType2D as Kind, RigidBody2D as Body,
        },
        rapier2d::prelude::{JointAxis as Axis, Rotation},
        systems2d::{
            StepPhysics2D as Step, SyncPhysicsBodies2D as BodySync,
            SyncPhysicsJoints2D as JointSync, SyncPhysicsJointsSystem2D as JointSyncRegular,
        },
        world2d::PhysicsWorld2D as Physics,
    };
    fn rotated_frame() -> Frame {
        Frame {
            rotation: ::std::f32::consts::FRAC_PI_2,
            ..Default::default()
        }
    }
    fn angle(rotation: Rotation) -> f64 {
        rotation.angle() as f64
    }
    fn angular_velocity(body: &redlilium_ecs::physics::rapier2d::prelude::RigidBody) -> f64 {
        body.angvel() as f64
    }
    tests!();
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod dim3 {
    use super::*;
    use redlilium_ecs::physics::{
        components3d::{
            Collider3D as Collider, ImpulseJoint3D as Joint, JointFrame3D as Frame,
            JointType3D as Kind, RigidBody3D as Body,
        },
        rapier3d::prelude::{JointAxis as Axis, Rotation},
        systems3d::{
            StepPhysics3D as Step, SyncPhysicsBodies3D as BodySync,
            SyncPhysicsJoints3D as JointSync, SyncPhysicsJointsSystem3D as JointSyncRegular,
        },
        world3d::PhysicsWorld3D as Physics,
    };
    fn rotated_frame() -> Frame {
        Frame {
            rotation: redlilium_core::math::quat_from_rotation_z(::std::f32::consts::FRAC_PI_2),
            ..Default::default()
        }
    }
    fn angle(rotation: Rotation) -> f64 {
        rotation.to_scaled_axis().x as f64
    }
    fn angular_velocity(body: &redlilium_ecs::physics::rapier3d::prelude::RigidBody) -> f64 {
        body.angvel().x as f64
    }
    tests!();
}
