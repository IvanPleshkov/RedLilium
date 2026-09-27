#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_ecs::*;

macro_rules! descriptor_tests {
    () => {
        fn world() -> World {
            let mut w = World::new();
            register_std_components(&mut w);
            w.insert_resource(Physics::default());
            w
        }
        fn spawn(w: &mut World) -> Entity {
            w.spawn_with((Transform::IDENTITY, Body::dynamic(), Collider::ball(0.5)))
                .unwrap()
        }
        fn sync(w: &mut World, regular: bool, joints: bool) -> Vec<SystemError> {
            let mut systems = SystemsContainer::new();
            match (regular, joints) {
                (false, false) => {
                    systems.add_exclusive(Sync);
                }
                (true, false) => {
                    systems.add(SyncRegular);
                }
                (false, true) => {
                    systems.add_exclusive(JointSync);
                }
                (true, true) => {
                    systems.add(JointSyncRegular);
                }
            }
            EcsRunner::multi_thread(2).run(w, &systems)
        }
        fn invalid(errors: Vec<SystemError>, field: &str) {
            assert!(
                matches!(errors.as_slice(), [SystemError::InvalidConfiguration { message }]
                if message.contains(field) && message.contains("physics entity")),
                "{errors:?}"
            );
        }
        fn named(w: &World, entities: &[Entity], name: &str) -> Entity {
            entities
                .iter()
                .copied()
                .find(|&e| w.get::<Name>(e).is_some_and(|n| n.0 == name))
                .unwrap()
        }

        #[test]
        fn scene_roundtrip_preserves_descriptors_and_remaps_joint_endpoints() {
            let mut w = world();
            let mut bodies = Vec::new();
            for (i, body) in [
                Body::dynamic(),
                Body::fixed(),
                Body::kinematic_position(),
                Body::kinematic_velocity(),
            ]
            .into_iter()
            .enumerate()
            {
                let body = body
                    .with_linear_damping(0.3)
                    .with_angular_damping(0.7)
                    .with_gravity_scale(-0.5);
                let shapes = shapes();
                let collider = shapes[i % shapes.len()]
                    .clone()
                    .with_friction(0.8)
                    .with_restitution(0.4)
                    .with_density(2.5)
                    .with_sensor(Some(physics::SensorSettings::default()))
                    .with_collision_events((i % 2 == 0).then(physics::CollisionEventSettings::default))
                    .with_collision_groups(match i {
                        0 => None,
                        1 => Some(physics::CollisionGroups::default()),
                        2 => Some(physics::CollisionGroups::new(0, 0)),
                        _ => Some(physics::CollisionGroups::new(1 << 31, 0x8000_0003)),
                    });
                let e = w
                    .spawn_with((
                        Name::new(format!("body{i}")),
                        Transform::IDENTITY,
                        body.clone(),
                        collider.clone(),
                    ))
                    .unwrap();
                bodies.push((e, body, collider));
            }
            let joints = joint_types()
                .into_iter()
                .enumerate()
                .map(|(i, joint_type)| {
                    let joint = Joint {
                        body1: bodies[0].0,
                        body2: bodies[1].0,
                        joint_type,
                    };
                    w.spawn_with((Name::new(format!("joint{i}")), joint.clone()))
                        .unwrap();
                    joint
                })
                .collect::<Vec<_>>();
            assert!(sync(&mut w, false, false).is_empty());
            assert!(sync(&mut w, false, true).is_empty());
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
                let mut target = world();
                // Force restored entity IDs to differ from the serialized IDs.
                for _ in 0..16 {
                    target.spawn();
                }
                let entities = target.deserialize_world_into(&snapshot).unwrap();
                for (i, (old, body, collider)) in bodies.iter().enumerate() {
                    let e = named(&target, &entities, &format!("body{i}"));
                    assert_ne!(e, *old);
                    assert_eq!(target.get::<Body>(e), Some(body));
                    assert_eq!(target.get::<Collider>(e), Some(collider));
                    assert!(target.get::<BodyHandle>(e).is_none());
                }
                for (i, joint) in joints.iter().enumerate() {
                    let e = named(&target, &entities, &format!("joint{i}"));
                    let restored = target.get::<Joint>(e).unwrap();
                    assert_eq!(restored.joint_type, joint.joint_type);
                    assert_eq!(restored.body1, named(&target, &entities, "body0"));
                    assert_eq!(restored.body2, named(&target, &entities, "body1"));
                    assert!(target.get::<JointHandle>(e).is_none());
                }
                assert!(sync(&mut target, true, false).is_empty());
                assert!(sync(&mut target, true, true).is_empty());
                assert_eq!(target.resource::<Physics>().bodies().len(), bodies.len());
                assert_eq!(
                    target.resource::<Physics>().impulse_joints().len(),
                    joints.len()
                );
            }
        }

        #[test]
        fn prefab_joint_references_follow_each_instance() {
            let mut w = world();
            let a = spawn(&mut w);
            let b = spawn(&mut w);
            let j = w
                .spawn_with((Joint::fixed(a, b, Default::default(), Default::default()),))
                .unwrap();
            set_parent(&mut w, b, a);
            set_parent(&mut w, j, a);
            let prefab = w.serialize_prefab(a).unwrap();
            for _ in 0..2 {
                let entities = w.deserialize_prefab(&prefab).unwrap();
                let joint = entities
                    .iter()
                    .find_map(|&e| w.get::<Joint>(e))
                    .unwrap()
                    .clone();
                assert!(entities.contains(&joint.body1));
                assert!(entities.contains(&joint.body2));
                assert_ne!(joint.body1, a);
                assert_ne!(joint.body2, b);
                assert_eq!(w.get::<Body>(joint.body1), w.get::<Body>(a));
                assert_eq!(w.get::<Collider>(joint.body2), w.get::<Collider>(b));
            }
        }

        #[test]
        fn invalid_new_descriptors_never_reach_rapier() {
            for regular in [false, true] {
                for (body, collider, field) in invalid_settings() {
                    let mut w = world();
                    let e = w.spawn_with((Transform::IDENTITY, body, collider)).unwrap();
                    invalid(sync(&mut w, regular, false), field);
                    assert!(w.resource::<Physics>().bodies().is_empty());
                    assert!(w.resource::<Physics>().colliders().is_empty());
                    assert!(w.get::<BodyHandle>(e).is_none());
                }
            }
        }

        fn invalid_settings() -> Vec<(Body, Collider, &'static str)> {
            let mut result = Vec::new();
            for v in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                result.push((
                    Body::dynamic().with_linear_damping(v),
                    Collider::default(),
                    "linear_damping",
                ));
                result.push((
                    Body::dynamic().with_angular_damping(v),
                    Collider::default(),
                    "angular_damping",
                ));
                result.push((
                    Body::default(),
                    Collider::default().with_friction(v),
                    "friction",
                ));
                result.push((
                    Body::default(),
                    Collider::default().with_density(v),
                    "density",
                ));
                result.push((
                    Body::default(),
                    Collider::default().with_restitution(v),
                    "restitution",
                ));
                result.push((Body::default(), Collider::ball(v), "radius"));
                result.push((Body::default(), Collider::capsule_y(v, 0.5), "half_height"));
                result.push((Body::default(), Collider::capsule_y(0.5, v), "radius"));
                result.push((
                    Body::default(),
                    Collider {
                        shape: Shape::Cuboid {
                            half_extents: Vector::repeat(v),
                        },
                        ..Collider::default()
                    },
                    "half_extents",
                ));
                if !v.is_finite() {
                    result.push((
                        Body::dynamic().with_gravity_scale(v),
                        Collider::default(),
                        "gravity_scale",
                    ));
                }
            }
            result.push((Body::default(), Collider::ball(0.0), "radius"));
            result.push((
                Body::default(),
                Collider {
                    shape: Shape::Cuboid {
                        half_extents: Vector::zeros(),
                    },
                    ..Collider::default()
                },
                "half_extents",
            ));
            result.push((
                Body::default(),
                Collider::default().with_restitution(1.01),
                "restitution",
            ));
            result.extend(
                invalid_shapes()
                    .into_iter()
                    .map(|(c, field)| (Body::default(), c, field)),
            );
            result
        }

        #[test]
        fn invalid_edit_preserves_entire_sync_batch_and_can_be_corrected() {
            for regular in [false, true] {
                let mut w = world();
                let a = spawn(&mut w);
                let b = spawn(&mut w);
                assert!(sync(&mut w, regular, false).is_empty());
                let ha = w.resource::<Physics>().body_for_entity(a).unwrap();
                let hb = w.resource::<Physics>().body_for_entity(b).unwrap();
                let cb = w.resource::<Physics>().bodies()[hb].colliders()[0];
                w.resource_mut::<Physics>().body_motion(ha).unwrap().sleep();
                w.get_mut::<Body>(a).unwrap().linear_damping = 0.5;
                for (body, collider, field) in invalid_settings() {
                    w.insert(b, body).unwrap();
                    w.insert(b, collider).unwrap();
                    invalid(sync(&mut w, regular, false), field);
                    let p = w.resource::<Physics>();
                    assert_eq!(p.body_for_entity(a).unwrap(), ha);
                    assert_eq!(p.body_for_entity(b).unwrap(), hb);
                    assert_eq!(p.bodies()[ha].linear_damping(), 0.0);
                    assert!(p.bodies()[ha].is_sleeping());
                    assert_eq!(p.bodies()[hb].colliders()[0], cb);
                    assert_eq!(p.colliders()[cb].shape().as_ball().unwrap().radius, 0.5);
                }
                w.insert(b, Body::dynamic().with_gravity_scale(-1.0))
                    .unwrap();
                w.insert(
                    b,
                    Collider::capsule_y(0.0, 0.5)
                        .with_density(0.0)
                        .with_friction(0.0)
                        .with_restitution(1.0),
                )
                .unwrap();
                assert!(sync(&mut w, regular, false).is_empty());
                assert_eq!(w.resource::<Physics>().bodies()[ha].linear_damping(), 0.5);
                assert_eq!(w.resource::<Physics>().bodies()[hb].gravity_scale(), -1.0);
            }
        }

        #[test]
        fn invalid_joint_edits_preserve_existing_joint_and_recover() {
            for regular in [false, true] {
                let mut w = world();
                let a = spawn(&mut w);
                let b = spawn(&mut w);
                assert!(sync(&mut w, regular, false).is_empty());
                let valid = Joint::fixed(a, b, Vector::zeros(), Vector::zeros());
                let j = w.spawn_with((valid.clone(),)).unwrap();
                assert!(sync(&mut w, regular, true).is_empty());
                let h = w.resource::<Physics>().joint_for_entity(j).unwrap();
                let mut cases = vec![(
                    Joint::fixed(a, a, Vector::zeros(), Vector::zeros()),
                    "endpoints",
                )];
                for joint_type in joint_types() {
                    let mut jt = joint_type.clone();
                    anchors(&mut jt).0.x = f32::NAN;
                    cases.push((
                        Joint {
                            joint_type: jt,
                            ..valid.clone()
                        },
                        "anchors",
                    ));
                    let mut jt = joint_type;
                    anchors(&mut jt).1.x = f32::INFINITY;
                    cases.push((
                        Joint {
                            joint_type: jt,
                            ..valid.clone()
                        },
                        "anchors",
                    ));
                }
                for v in [0.0, f32::NAN, f32::INFINITY] {
                    for mut jt in joint_types() {
                        if let Some(axis) = axis(&mut jt) {
                            *axis = Vector::repeat(v);
                            cases.push((
                                Joint {
                                    joint_type: jt,
                                    ..valid.clone()
                                },
                                "axis",
                            ));
                        }
                    }
                }
                for (bad, field) in cases {
                    w.insert(j, bad.clone()).unwrap();
                    invalid(sync(&mut w, regular, true), field);
                    let p = w.resource::<Physics>();
                    assert_eq!(p.joint_for_entity(j).unwrap(), h);
                    assert_eq!(p.impulse_joints().len(), 1);
                    assert_eq!(p.impulse_joints().get(h).unwrap().body2, p.body_for_entity(b).unwrap());
                    // Creation must reject the same descriptor too.
                    let mut fresh = world();
                    fresh.spawn_with((bad,)).unwrap();
                    invalid(sync(&mut fresh, regular, true), field);
                    assert!(fresh.resource::<Physics>().impulse_joints().is_empty());
                }
                w.insert(j, valid).unwrap();
                assert!(sync(&mut w, regular, true).is_empty());
                assert_eq!(w.resource::<Physics>().joint_for_entity(j).unwrap(), h);
                // An unavailable endpoint is a pending joint, not a configuration error.
                let pending = w.spawn();
                w.get_mut::<Joint>(j).unwrap().body2 = pending;
                assert!(sync(&mut w, regular, true).is_empty());
                assert!(w.resource::<Physics>().impulse_joints().is_empty());
            }
        }

        #[test]
        fn joint_axes_are_normalized_even_at_extreme_finite_magnitudes() {
            for regular in [false, true] {
                for magnitude in [3.0, f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
                    let mut w = world();
                    let a = spawn(&mut w);
                    let b = spawn(&mut w);
                    assert!(sync(&mut w, regular, false).is_empty());
                    for mut jt in joint_types() {
                        if let Some(axis) = axis(&mut jt) {
                            *axis = Vector::zeros();
                            axis.y = -magnitude;
                            let j = w
                                .spawn_with((Joint {
                                    body1: a,
                                    body2: b,
                                    joint_type: jt,
                                },))
                                .unwrap();
                            assert!(sync(&mut w, regular, true).is_empty());
                            let p = w.resource::<Physics>();
                            let data = &p.impulse_joints().get(p.joint_for_entity(j).unwrap()).unwrap().data;
                            let axis = data.local_frame1.rotation * RapierVector::X;
                            assert!(axis.x.abs() < 1e-5, "{axis:?}");
                            assert!((axis.y + 1.0).abs() < 1e-5, "{axis:?}");
                        }
                    }
                }
            }
        }

        #[test]
        #[allow(deprecated)]
        fn legacy_builder_publishes_mappings_and_next_sync_reuses_bodies() {
            let mut w = world();
            let e = spawn(&mut w);
            build(&mut w).unwrap();
            let handle = w.get::<BodyHandle>(e).unwrap().0;
            assert_eq!(w.resource::<Physics>().body_for_entity(e), Some(handle));
            assert_eq!(w.resource::<Physics>().entity_for_body(handle), Some(e));
            assert!(sync(&mut w, false, false).is_empty());
            assert_eq!(w.get::<BodyHandle>(e).unwrap().0, handle);
            assert_eq!(w.resource::<Physics>().bodies().len(), 1);
        }

        #[test]
        #[allow(deprecated)]
        fn legacy_builder_validates_before_replacing_resource() {
            let mut w = world();
            let a = spawn(&mut w);
            assert!(sync(&mut w, false, false).is_empty());
            let h = w.resource::<Physics>().body_for_entity(a).unwrap();
            w.get_mut::<Collider>(a).unwrap().density = -1.0;
            invalid(vec![build(&mut w).unwrap_err()], "density");
            assert_eq!(w.resource::<Physics>().body_for_entity(a).unwrap(), h);
            assert_eq!(w.get::<BodyHandle>(a).unwrap().0, h);
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod two_d {
    use super::*;
    use redlilium_core::math::Vec2 as Vector;
    use redlilium_ecs::physics::rapier2d::prelude::Vector as RapierVector;
    #[allow(deprecated)]
    use redlilium_ecs::physics::{
        components2d::{
            Collider2D as Collider, ColliderShape2D as Shape, ImpulseJoint2D as Joint,
            JointType2D as JointType, RigidBody2D as Body, build_physics_world_2d as build,
        },
        systems2d::{
            SyncPhysicsBodies2D as Sync, SyncPhysicsBodiesSystem2D as SyncRegular,
            SyncPhysicsJoints2D as JointSync, SyncPhysicsJointsSystem2D as JointSyncRegular,
        },
        world2d::{
            ImpulseJoint2DHandle as JointHandle, PhysicsWorld2D as Physics,
            RigidBody2DHandle as BodyHandle,
        },
    };
    fn shapes() -> Vec<Collider> {
        vec![
            Collider::ball(0.7),
            Collider::cuboid(0.3, 0.6),
            Collider::capsule_y(0.8, 0.4),
        ]
    }
    fn invalid_shapes() -> Vec<(Collider, &'static str)> {
        vec![]
    }
    fn joint_types() -> Vec<JointType> {
        let anchor1 = Vector::new(0.2, 0.3);
        let anchor2 = Vector::new(-0.4, 0.6);
        vec![
            JointType::Revolute { anchor1, anchor2 },
            JointType::Fixed { anchor1, anchor2 },
            JointType::Prismatic {
                anchor1,
                anchor2,
                axis: Vector::new(2.0, 3.0),
            },
        ]
    }
    fn anchors(j: &mut JointType) -> (&mut Vector, &mut Vector) {
        match j {
            JointType::Revolute { anchor1, anchor2 }
            | JointType::Fixed { anchor1, anchor2 }
            | JointType::Prismatic {
                anchor1, anchor2, ..
            } => (anchor1, anchor2),
        }
    }
    fn axis(j: &mut JointType) -> Option<&mut Vector> {
        match j {
            JointType::Prismatic { axis, .. } => Some(axis),
            _ => None,
        }
    }
    descriptor_tests!();
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod three_d {
    use super::*;
    use redlilium_core::math::Vec3 as Vector;
    use redlilium_ecs::physics::rapier3d::prelude::Vector as RapierVector;
    #[allow(deprecated)]
    use redlilium_ecs::physics::{
        components3d::{
            Collider3D as Collider, ColliderShape3D as Shape, ImpulseJoint3D as Joint,
            JointType3D as JointType, RigidBody3D as Body, build_physics_world_3d as build,
        },
        systems3d::{
            SyncPhysicsBodies3D as Sync, SyncPhysicsBodiesSystem3D as SyncRegular,
            SyncPhysicsJoints3D as JointSync, SyncPhysicsJointsSystem3D as JointSyncRegular,
        },
        world3d::{
            ImpulseJoint3DHandle as JointHandle, PhysicsWorld3D as Physics,
            RigidBody3DHandle as BodyHandle,
        },
    };
    fn shapes() -> Vec<Collider> {
        vec![
            Collider::ball(0.7),
            Collider::cuboid(0.3, 0.6, 0.9),
            Collider::capsule_y(0.8, 0.4),
            Collider::cylinder(0.8, 0.4),
        ]
    }
    fn invalid_shapes() -> Vec<(Collider, &'static str)> {
        [-1.0, 0.0, f32::NAN, f32::INFINITY]
            .into_iter()
            .flat_map(|v| {
                [
                    (Collider::cylinder(v, 0.5), "half_height"),
                    (Collider::cylinder(0.5, v), "radius"),
                ]
            })
            .collect()
    }
    fn joint_types() -> Vec<JointType> {
        let anchor1 = Vector::new(0.2, 0.3, 0.4);
        let anchor2 = Vector::new(-0.4, 0.6, -0.2);
        let axis = Vector::new(2.0, 3.0, -4.0);
        vec![
            JointType::Spherical { anchor1, anchor2 },
            JointType::Revolute {
                anchor1,
                anchor2,
                axis,
            },
            JointType::Fixed { anchor1, anchor2 },
            JointType::Prismatic {
                anchor1,
                anchor2,
                axis,
            },
        ]
    }
    fn anchors(j: &mut JointType) -> (&mut Vector, &mut Vector) {
        match j {
            JointType::Spherical { anchor1, anchor2 }
            | JointType::Revolute {
                anchor1, anchor2, ..
            }
            | JointType::Fixed { anchor1, anchor2 }
            | JointType::Prismatic {
                anchor1, anchor2, ..
            } => (anchor1, anchor2),
        }
    }
    fn axis(j: &mut JointType) -> Option<&mut Vector> {
        match j {
            JointType::Revolute { axis, .. } | JointType::Prismatic { axis, .. } => Some(axis),
            _ => None,
        }
    }
    descriptor_tests!();
}
