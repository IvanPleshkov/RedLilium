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
        fn removing_each_required_component_cleans_body_colliders_and_joint_and_allows_recreation()
        {
            let removals: [fn(&mut World, Entity); 3] = [
                |w, e| {
                    w.remove::<Body>(e).unwrap();
                },
                |w, e| {
                    w.remove::<Collider>(e).unwrap();
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
                        assert_eq!(w.resource::<Physics>().impulse_joints.len(), 1);
                        remove(&mut w, a);
                        sync_bodies(&mut w, regular, multi);
                        assert!(w.get::<BodyHandle>(a).is_none());
                        assert!(w.get::<JointHandle>(j).is_none());
                        {
                            let p = w.resource::<Physics>();
                            assert_eq!(p.bodies.len(), 1);
                            assert_eq!(p.colliders.len(), 1);
                            assert_eq!(p.impulse_joints.len(), 0);
                            assert!(!p.entity_to_body.contains_key(&a));
                            assert_eq!(p.body_to_entity.len(), 1);
                            assert!(!p.entity_to_joint.contains_key(&j));
                        }
                        w.insert(a, Body::dynamic()).unwrap();
                        w.insert(a, Collider::ball(0.5)).unwrap();
                        w.insert(a, Transform::IDENTITY).unwrap();
                        sync_bodies(&mut w, regular, multi);
                        sync_joints(&mut w, regular, multi);
                        let p = w.resource::<Physics>();
                        assert_eq!(p.bodies.len(), 2);
                        assert_eq!(p.colliders.len(), 2);
                        assert!(p.impulse_joints.contains(p.entity_to_joint[&j]));
                    }
                }
            }
        }
        #[test]
        fn deferred_body_creation_rejects_recycled_dead_or_incomplete_entity() {
            let edits: [fn(&mut World, Entity); 5] = [
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
                    w.remove::<Collider>(e).unwrap();
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
                    assert!(p.bodies.is_empty());
                    assert!(p.colliders.is_empty());
                    assert!(p.entity_to_body.is_empty());
                    assert!(p.body_to_entity.is_empty());
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
                    assert_eq!(p.bodies.len(), 2);
                    assert!(p.impulse_joints.is_empty());
                    assert!(p.entity_to_joint.is_empty());
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
                    assert!(w.resource::<Physics>().entity_to_joint.is_empty());
                    sync_joints(&mut w, regular, multi);
                    let p = w.resource::<Physics>();
                    assert!(p.impulse_joints.contains(p.entity_to_joint[&j]));
                }
            }
        }
        #[test]
        fn sync_joints_recovers_from_a_joint_removed_directly_in_rapier() {
            for regular in [false, true] {
                let mut w = world();
                let a = body(&mut w);
                let b = body(&mut w);
                sync_bodies(&mut w, false, false);
                let j = joint(&mut w, a, b);
                sync_joints(&mut w, regular, false);
                let old = w.get::<JointHandle>(j).unwrap().0;
                w.resource_mut::<Physics>().remove_impulse_joint(old, true);
                sync_joints(&mut w, regular, false);
                let p = w.resource::<Physics>();
                let new = p.entity_to_joint[&j];
                assert_ne!(old, new);
                assert!(p.impulse_joints.contains(new));
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
        systems2d::{
            StepPhysics2D as Step, SyncPhysicsBodies2D as SyncBodies,
            SyncPhysicsBodiesSystem2D as SyncBodiesRegular, SyncPhysicsJoints2D as SyncJoints,
            SyncPhysicsJointsSystem2D as SyncJointsRegular,
        },
        world2d::{
            ImpulseJoint2DHandle as JointHandle, PhysicsWorld2D as Physics,
            RigidBody2DHandle as BodyHandle,
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
                p.bodies[p.entity_to_body[&e]].rotation().angle()
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
        systems3d::{
            InterpolatePhysics, RecordPhysicsPose, StepPhysics3D as Step,
            SyncPhysicsBodies3D as SyncBodies, SyncPhysicsBodiesSystem3D as SyncBodiesRegular,
            SyncPhysicsJoints3D as SyncJoints, SyncPhysicsJointsSystem3D as SyncJointsRegular,
        },
        world3d::{
            ImpulseJoint3DHandle as JointHandle, PhysicsInterpolation, PhysicsWorld3D as Physics,
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
