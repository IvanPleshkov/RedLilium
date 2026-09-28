//! Run the same lifecycle contract against both dimensions and sync variants.
#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]

use redlilium_core::math::Vec3;
use redlilium_ecs::*;

fn world() -> World {
    let mut world = World::new();
    register_std_components(&mut world);
    world
}
fn run(world: &mut World, systems: &SystemsContainer, multi: bool) {
    let runner = if multi {
        EcsRunner::multi_thread(2)
    } else {
        EcsRunner::single_thread()
    };
    assert!(runner.run(world, systems).is_empty());
}

// Explicit edges order command enqueueing; no exclusive barrier flushes between
// this mutation and the regular sync system under test.
struct Edit(Entity, fn(&mut World, Entity));
impl System for Edit {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        let (entity, edit) = (self.0, self.1);
        ctx.commands(move |world| edit(world, entity));
        Ok(())
    }
}

macro_rules! lifecycle_tests {
    () => {
        fn body(world: &mut World) -> Entity {
            world
                .spawn_with((
                    Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
                    Body::dynamic(),
                    Collider::ball(0.5),
                ))
                .unwrap()
        }
        fn joint(world: &mut World, a: Entity, b: Entity) -> Entity {
            world
                .spawn_with((Joint::fixed(a, b, Default::default(), Default::default()),))
                .unwrap()
        }
        fn sync_bodies(world: &mut World, regular: bool, multi: bool) {
            let mut systems = SystemsContainer::new();
            if regular {
                systems.add(SyncBodiesRegular);
            } else {
                systems.add_exclusive(SyncBodies);
            }
            run(world, &systems, multi);
        }
        fn sync_joints(world: &mut World, regular: bool, multi: bool) {
            let mut systems = SystemsContainer::new();
            if regular {
                systems.add(SyncJointsRegular);
            } else {
                systems.add_exclusive(SyncJoints);
            }
            run(world, &systems, multi);
        }
        #[test]
        fn regular_sync_step_and_record_use_new_bodies_without_a_command_flush() {
            for multi in [false, true] {
                for position_based in [false, true] {
                    let mut w = world();
                    let mut physics = Physics::default();
                    physics.gravity = Default::default();
                    physics.integration_parameters.dt = 0.1;
                    w.insert_resource(physics);
                    let body = if position_based {
                        Body::kinematic_position()
                    } else {
                        Body::kinematic_velocity()
                    };
                    let e = w
                        .spawn_with((Transform::IDENTITY, body, Collider::ball(0.5)))
                        .unwrap();
                    if position_based {
                        let mut pose = Pose::default();
                        pose.translation.x = 0.2;
                        w.insert(e, Target::from(pose)).unwrap();
                    } else {
                        let mut velocity = Velocity::default();
                        velocity.linear.x = 2.0;
                        w.insert(e, velocity).unwrap();
                    }
                    let mut s = SystemsContainer::new();
                    s.add(SyncBodiesRegular);
                    s.add(Step);
                    s.add(Record);
                    s.add_edge::<SyncBodiesRegular, Step>().unwrap();
                    s.add_edge::<Step, Record>().unwrap();
                    run(&mut w, &s, multi);
                    assert!(
                        (w.resource::<Physics>().pose(e).unwrap().translation.x - 0.2).abs() < 1e-5
                    );
                    assert!((w.get::<Transform>(e).unwrap().translation.x - 0.2).abs() < 1e-5);
                    let history = w.get::<History>(e).unwrap();
                    assert!((history.prev_translation.x - 0.2).abs() < 1e-5);
                    assert!((history.cur_translation.x - 0.2).abs() < 1e-5);
                    run(&mut w, &s, multi);
                    let history = w.get::<History>(e).unwrap();
                    let expected = if position_based { 0.2 } else { 0.4 };
                    assert!((history.prev_translation.x - 0.2).abs() < 1e-5);
                    assert!((history.cur_translation.x - expected).abs() < 1e-5);
                }
            }
        }

        #[test]
        fn first_step_validates_new_kinematic_input_before_advancing_any_body() {
            for multi in [false, true] {
                let mut w = world();
                w.insert_resource(Physics::default());
                let dynamic = body(&mut w);
                let mut velocity = Velocity::default();
                velocity.linear.x = f32::NAN;
                w.spawn_with((Transform::IDENTITY, Body::kinematic_velocity(), velocity))
                    .unwrap();
                let mut s = SystemsContainer::new();
                s.add(SyncBodiesRegular);
                s.add(Step);
                s.add_edge::<SyncBodiesRegular, Step>().unwrap();
                let runner = if multi {
                    EcsRunner::multi_thread(2)
                } else {
                    EcsRunner::single_thread()
                };
                let errors = runner.run(&mut w, &s);
                assert!(
                    matches!(
                        errors.as_slice(),
                        [SystemError::InvalidConfiguration { .. }]
                    ),
                    "{errors:?}"
                );
                assert_eq!(
                    w.resource::<Physics>().pose(dynamic).unwrap().translation.y,
                    10.0
                );
            }
        }

        struct RepeatJointSync {
            joint: Entity,
            replacement_body: Option<Entity>,
        }
        impl System for RepeatJointSync {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                SyncJointsRegular.run(ctx)?;
                ctx.lock::<(Res<Physics>,)>().execute(|(physics,)| {
                    assert!(physics.joint_for_entity(self.joint).is_some());
                });
                if let Some(body) = self.replacement_body {
                    ctx.lock::<(Write<Joint>,)>().execute(|(mut joints,)| {
                        joints.get_mut(self.joint.index()).unwrap().body2 = body;
                    });
                }
                SyncJointsRegular.run(ctx)
            }
        }

        #[test]
        fn repeated_joint_sync_before_publication_preserves_single_native_owner() {
            for multi in [false, true] {
                for rebuild in [false, true] {
                    let mut w = world();
                    w.insert_resource(Physics::default());
                    let a = body(&mut w);
                    let b = body(&mut w);
                    let c = body(&mut w);
                    let j = joint(&mut w, a, b);
                    let mut s = SystemsContainer::new();
                    s.add(SyncBodiesRegular);
                    s.add(RepeatJointSync {
                        joint: j,
                        replacement_body: rebuild.then_some(c),
                    });
                    s.add_edge::<SyncBodiesRegular, RepeatJointSync>().unwrap();
                    run(&mut w, &s, multi);
                    {
                        let physics = w.resource::<Physics>();
                        assert_eq!(physics.impulse_joints().len(), 1);
                        let handle = physics.joint_for_entity(j).unwrap();
                        assert_eq!(w.get::<JointHandle>(j).unwrap().0, handle);
                        assert_eq!(
                            physics.impulse_joints().get(handle).unwrap().body2(),
                            physics
                                .body_for_entity(if rebuild { c } else { b })
                                .unwrap()
                        );
                    }
                    w.despawn(j);
                    sync_joints(&mut w, true, multi);
                    assert!(w.resource::<Physics>().impulse_joints().is_empty());
                    assert!(w.resource::<Physics>().joint_for_entity(j).is_none());
                }
            }
        }

        #[test]
        fn removing_each_required_component_cleans_body_colliders_and_joint_and_allows_recreation()
        {
            let removals: [fn(&mut World, Entity); 2] = [
                |w, e| {
                    w.remove::<Body>(e).unwrap();
                },
                |w, e| {
                    w.remove::<Transform>(e).unwrap();
                },
            ];
            for multi in [false, true] {
                for regular in [false, true] {
                    for remove in removals {
                        let mut w = world();
                        w.insert_resource(Physics::default());
                        let a = body(&mut w);
                        let b = body(&mut w);
                        sync_bodies(&mut w, regular, multi);
                        let j = joint(&mut w, a, b);
                        sync_joints(&mut w, regular, multi);
                        assert_eq!(w.resource::<Physics>().impulse_joints().len(), 1);
                        remove(&mut w, a);
                        sync_bodies(&mut w, regular, multi);
                        assert!(w.get::<BodyHandle>(a).is_none());
                        assert!(w.get::<JointHandle>(j).is_none());
                        {
                            let p = w.resource::<Physics>();
                            assert_eq!(p.bodies().len(), 1);
                            assert_eq!(p.colliders().len(), 1);
                            assert_eq!(p.impulse_joints().len(), 0);
                            assert!(p.body_for_entity(a).is_none());
                            assert_eq!(
                                p.entity_for_body(w.get::<BodyHandle>(b).unwrap().0),
                                Some(b)
                            );
                            assert!(p.joint_for_entity(j).is_none());
                        }
                        w.insert(a, Body::dynamic()).unwrap();
                        w.insert(a, Collider::ball(0.5)).unwrap();
                        w.insert(a, Transform::IDENTITY).unwrap();
                        sync_bodies(&mut w, regular, multi);
                        sync_joints(&mut w, regular, multi);
                        let p = w.resource::<Physics>();
                        assert_eq!(p.bodies().len(), 2);
                        assert_eq!(p.colliders().len(), 2);
                        assert!(p.impulse_joints().contains(p.joint_for_entity(j).unwrap()));
                    }
                }
            }
        }
        #[test]
        fn deferred_body_creation_rejects_recycled_dead_or_incomplete_entity() {
            let edits: [fn(&mut World, Entity); 4] = [
                |w, e| {
                    assert!(w.despawn(e));
                },
                |w, e| {
                    assert!(w.despawn(e));
                    let new = w.spawn_with((Transform::IDENTITY,)).unwrap();
                    assert_eq!(new.index(), e.index());
                    assert_ne!(new, e);
                },
                |w, e| {
                    w.remove::<Transform>(e).unwrap();
                },
                |w, e| {
                    w.set_entity_flags(e, Entity::DISABLED);
                },
            ];
            for multi in [false, true] {
                for edit in edits {
                    let mut w = world();
                    w.insert_resource(Physics::default());
                    let e = body(&mut w);
                    let mut s = SystemsContainer::new();
                    s.add(Edit(e, edit));
                    s.add(SyncBodiesRegular);
                    s.add_edge::<Edit, SyncBodiesRegular>().unwrap();
                    run(&mut w, &s, multi);
                    if let Some(entity) = w.entity_at_index(e.index()) {
                        assert!(w.get::<BodyHandle>(entity).is_none());
                    }
                    let p = w.resource::<Physics>();
                    assert!(p.bodies().is_empty());
                    assert!(p.colliders().is_empty());
                    assert!(p.body_for_entity(e).is_none());
                }
            }
        }
        #[test]
        fn deferred_joint_creation_rejects_recycled_dead_or_incomplete_entity() {
            let edits: [fn(&mut World, Entity); 4] = [
                |w, e| {
                    assert!(w.despawn(e));
                },
                |w, e| {
                    assert!(w.despawn(e));
                    let new = w.spawn();
                    assert_eq!(new.index(), e.index());
                    assert_ne!(new, e);
                },
                |w, e| {
                    w.remove::<Joint>(e).unwrap();
                },
                |w, e| {
                    w.set_entity_flags(e, Entity::DISABLED);
                },
            ];
            for multi in [false, true] {
                for edit in edits {
                    let mut w = world();
                    let a = body(&mut w);
                    let b = body(&mut w);
                    sync_bodies(&mut w, false, multi);
                    let j = joint(&mut w, a, b);
                    let mut s = SystemsContainer::new();
                    s.add(Edit(j, edit));
                    s.add(SyncJointsRegular);
                    s.add_edge::<Edit, SyncJointsRegular>().unwrap();
                    run(&mut w, &s, multi);
                    if let Some(entity) = w.entity_at_index(j.index()) {
                        assert!(w.get::<JointHandle>(entity).is_none());
                    }
                    let p = w.resource::<Physics>();
                    assert_eq!(p.bodies().len(), 2);
                    assert!(p.impulse_joints().is_empty());
                    assert!(p.joint_for_entity(j).is_none());
                }
            }
        }
        #[test]
        fn joint_cleanup_uses_actual_endpoints_after_descriptor_edit() {
            for multi in [false, true] {
                for regular in [false, true] {
                    let mut w = world();
                    w.insert_resource(Physics::default());
                    let a = body(&mut w);
                    let b = body(&mut w);
                    let c = body(&mut w);
                    sync_bodies(&mut w, regular, multi);
                    let j = joint(&mut w, a, b);
                    sync_joints(&mut w, regular, multi);
                    w.get_mut::<Joint>(j).unwrap().body1 = c;
                    w.despawn(a);
                    sync_bodies(&mut w, regular, multi);
                    assert!(w.get::<JointHandle>(j).is_none());
                    assert!(w.resource::<Physics>().joint_for_entity(j).is_none());
                    sync_joints(&mut w, regular, multi);
                    let p = w.resource::<Physics>();
                    assert!(p.impulse_joints().contains(p.joint_for_entity(j).unwrap()));
                }
            }
        }
        #[test]
        fn ecs_cleanup_preserves_caller_owned_free_colliders() {
            for regular in [false, true] {
                let mut w = world();
                w.insert_resource(Physics::default());
                let free = w
                    .resource_mut::<Physics>()
                    .add_free_collider(ColliderBuilder::ball(1.0).build());
                let e = body(&mut w);
                sync_bodies(&mut w, regular, false);
                let handle = w.get::<BodyHandle>(e).unwrap().0;
                w.despawn(e);
                sync_bodies(&mut w, regular, false);
                let p = w.resource::<Physics>();
                assert!(p.bodies().is_empty());
                assert!(p.body_for_entity(e).is_none());
                assert!(p.entity_for_body(handle).is_none());
                assert_eq!(p.colliders().len(), 1);
                assert!(p.colliders().contains(free));
            }
        }
        #[test]
        fn joint_can_be_removed_and_recreated_through_its_descriptor() {
            for regular in [false, true] {
                let mut w = world();
                let a = body(&mut w);
                let b = body(&mut w);
                sync_bodies(&mut w, false, false);
                let j = joint(&mut w, a, b);
                sync_joints(&mut w, regular, false);
                let old = w.get::<JointHandle>(j).unwrap().0;
                let descriptor = w.get::<Joint>(j).unwrap().clone();
                w.remove::<Joint>(j).unwrap();
                sync_joints(&mut w, regular, false);
                assert!(w.get::<JointHandle>(j).is_none());
                assert!(w.resource::<Physics>().joint_for_entity(j).is_none());
                w.insert(j, descriptor).unwrap();
                sync_joints(&mut w, regular, false);
                let p = w.resource::<Physics>();
                let new = p.joint_for_entity(j).unwrap();
                assert_ne!(old, new);
                assert!(p.impulse_joints().contains(new));
                assert_eq!(w.get::<JointHandle>(j).unwrap().0, new);
            }
        }
        #[test]
        fn fixed_step_tracks_schedule_rate_and_preserves_manual_dt_without_time() {
            let mut w = world();
            let mut physics = Physics::default();
            physics.integration_parameters.dt = 0.125;
            w.insert_resource(physics);
            let mut s = SystemsContainer::new();
            s.add(Step);
            run(&mut w, &s, false);
            assert_eq!(w.resource::<Physics>().integration_parameters.dt, 0.125);
            let mut schedules = Schedules::new();
            schedules.get_mut::<FixedUpdate>().add(Step);
            let runner = EcsRunner::single_thread();
            for dt in [0.1, 0.02] {
                schedules.set_fixed_timestep(dt);
                schedules.run_frame(&mut w, &runner, dt);
                assert!(
                    (w.resource::<Physics>().integration_parameters.dt as f64 - dt).abs() < 1e-6
                );
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
        rapier2d::prelude::ColliderBuilder,
        systems2d::{
            RecordPhysicsPose2D as Record, StepPhysics2D as Step,
            SyncPhysicsBodies2D as SyncBodies, SyncPhysicsBodiesSystem2D as SyncBodiesRegular,
            SyncPhysicsJoints2D as SyncJoints, SyncPhysicsJointsSystem2D as SyncJointsRegular,
        },
        world2d::{
            ImpulseJoint2DHandle as JointHandle, PhysicsInterpolation2D as History,
            PhysicsWorld2D as Physics, RigidBody2DHandle as BodyHandle,
        },
    };
    lifecycle_tests!();

    #[test]
    fn initial_rotation_survives_creation_and_step_with_z_layer_preserved() {
        for regular in [false, true] {
            let mut w = world();
            w.insert_resource(Physics::default());
            let mut transform =
                Transform::from_rotation(redlilium_core::math::quat_from_rotation_z(0.5));
            transform.translation.z = 7.0;
            let e = w
                .spawn_with((transform, Body::dynamic(), Collider::cuboid(2.0, 0.5)))
                .unwrap();
            sync_bodies(&mut w, regular, false);
            let angle = {
                let p = w.resource::<Physics>();
                p.bodies()[p.body_for_entity(e).unwrap()].rotation().angle()
            };
            assert!((angle - 0.5).abs() < 1e-6);
            let mut s = SystemsContainer::new();
            s.add(Step);
            run(&mut w, &s, false);
            let t = w.get::<Transform>(e).unwrap();
            assert_eq!(t.translation.z, 7.0);
            assert!((t.rotation.coords - transform.rotation.coords).norm() < 1e-6);
        }
    }
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod three_d {
    use super::*;
    use redlilium_ecs::physics::{
        components3d::{Collider3D as Collider, ImpulseJoint3D as Joint, RigidBody3D as Body},
        control3d::{
            KinematicTarget3D as Target, KinematicVelocity3D as Velocity, PhysicsPose3D as Pose,
        },
        rapier3d::prelude::ColliderBuilder,
        systems3d::{
            InterpolatePhysics, RecordPhysicsPose, RecordPhysicsPose as Record,
            StepPhysics3D as Step, SyncPhysicsBodies3D as SyncBodies,
            SyncPhysicsBodiesSystem3D as SyncBodiesRegular, SyncPhysicsJoints3D as SyncJoints,
            SyncPhysicsJointsSystem3D as SyncJointsRegular,
        },
        world3d::{
            ImpulseJoint3DHandle as JointHandle, PhysicsInterpolation,
            PhysicsInterpolation as History, PhysicsWorld3D as Physics,
            RigidBody3DHandle as BodyHandle,
        },
    };
    lifecycle_tests!();

    #[test]
    fn removed_body_releases_transform_and_recreated_body_gets_fresh_history() {
        for multi in [false, true] {
            for regular in [false, true] {
                let mut w = world();
                w.insert_resource(Physics::default());
                let e = body(&mut w);
                sync_bodies(&mut w, regular, multi);
                let mut record = SystemsContainer::new();
                record.add(RecordPhysicsPose);
                run(&mut w, &record, multi);
                assert!(w.get::<PhysicsInterpolation>(e).is_some());
                w.remove::<Body>(e).unwrap();
                sync_bodies(&mut w, regular, multi);
                assert!(w.get::<PhysicsInterpolation>(e).is_none());
                w.get_mut::<Transform>(e).unwrap().translation.x = 123.0;
                let mut interpolate = SystemsContainer::new();
                interpolate.add(InterpolatePhysics);
                run(&mut w, &interpolate, multi);
                assert_eq!(w.get::<Transform>(e).unwrap().translation.x, 123.0);
                w.insert(e, Body::dynamic()).unwrap();
                sync_bodies(&mut w, regular, multi);
                run(&mut w, &record, multi);
                let history = w.get::<PhysicsInterpolation>(e).unwrap();
                assert_eq!(history.prev_translation.x, 123.0);
                assert_eq!(history.cur_translation.x, 123.0);
            }
        }
    }

    #[test]
    fn deferred_pose_seed_does_not_follow_a_recycled_slot() {
        for multi in [false, true] {
            let mut w = world();
            let e = body(&mut w);
            sync_bodies(&mut w, false, multi);
            let mut s = SystemsContainer::new();
            s.add(Edit(e, |w, old| {
                w.despawn(old);
                let new = w.spawn();
                assert_eq!(new.index(), old.index());
            }));
            s.add(RecordPhysicsPose);
            s.add_edge::<Edit, RecordPhysicsPose>().unwrap();
            run(&mut w, &s, multi);
            let new = w.entity_at_index(e.index()).unwrap();
            assert!(w.get::<PhysicsInterpolation>(new).is_none());
        }
    }
}
