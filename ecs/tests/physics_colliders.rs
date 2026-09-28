//! Independent collider ownership, local geometry and identity in both backends.
#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use physics::{
    CollisionEventSettings, CollisionPhase, CollisionStopReason, CollisionTypes, RayCastOptions,
    SensorSettings, ShapeCastOptions,
};
use redlilium_core::math::quat_from_rotation_z;
use redlilium_ecs::*;

struct Edit(Entity, fn(&mut World, Entity));
impl System for Edit {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        let (entity, f) = (self.0, self.1);
        ctx.commands(move |w| f(w, entity));
        Ok(())
    }
}
macro_rules! tests {
    () => {
        fn world() -> World {
            let mut w = World::new();
            register_std_components(&mut w);
            w.insert_resource(Physics::default());
            w.resource_mut::<Physics>().gravity = Default::default();
            w.add_event::<Event>();
            w
        }
        fn sync(w: &mut World, regular: bool) -> Vec<SystemError> {
            let mut s = SystemsContainer::new();
            if regular {
                s.add(SyncRegular);
            } else {
                s.add_exclusive(Sync);
            }
            EcsRunner::multi_thread(2).run(w, &s)
        }
        fn step(w: &mut World) {
            let mut s = SystemsContainer::new();
            s.add(Step);
            let errors = EcsRunner::multi_thread(2).run(w, &s);
            assert!(errors.is_empty(), "{errors:?}");
        }
        fn body(w: &mut World) -> Entity {
            w.spawn_with((Body::dynamic(), Transform::IDENTITY))
                .unwrap()
        }
        fn tracked() -> Collider {
            Collider::ball(0.5)
                .with_sensor(Some(SensorSettings {}))
                .with_collision_events(Some(CollisionEventSettings {}))
                .with_collision_types(Some(CollisionTypes::all()))
        }
        fn compound() -> Collider {
            Collider::compound(vec![
                Part::ball(0.4).with_local_pose(local(0.0, 0.0, 0.0)),
                Part::cuboid(vec(0.25, 0.6).map(|x| if x == 0.0 { 0.25 } else { x }))
                    .with_local_pose(local(2.0, 0.0, 0.4)),
            ])
            .with_local_pose(local(1.0, 0.0, 0.0))
        }

        #[test]
        fn local_pose_composes_with_body_and_compound_parts_and_hits_identify_both_entities() {
            for regular in [false, true] {
                let mut w = world();
                let b = body(&mut w);
                w.get_mut::<Transform>(b).unwrap().rotation =
                    quat_from_rotation_z(::std::f32::consts::FRAC_PI_2);
                let c = w.spawn_with((Owner { body: b }, compound())).unwrap();
                let unrelated = body(&mut w);
                set_parent(&mut w, c, unrelated);
                let mut visual =
                    Transform::from_translation(redlilium_core::math::Vec3::repeat(100.0));
                visual.scale = redlilium_core::math::Vec3::repeat(5.0);
                w.insert(c, visual).unwrap();
                let sensor = w
                    .spawn_with((
                        Owner { body: b },
                        tracked().with_local_pose(local(-2.0, 0.0, 0.0)),
                    ))
                    .unwrap();
                assert!(sync(&mut w, regular).is_empty());
                step(&mut w);
                let p = w.resource::<Physics>();
                let bh = p.body_for_entity(b).unwrap();
                let ch = p.collider_for_entity(c).unwrap();
                assert_eq!(w.get::<CH>(c).unwrap().0, ch);
                assert_eq!(p.bodies()[bh].colliders().len(), 2);
                assert_eq!(p.entity_for_collider(ch), Some(c));
                let hit = p
                    .cast_ray(
                        vec(-5.0, 1.0),
                        vec(10.0, 0.0),
                        RayCastOptions::default(),
                        Filter::default().exclude_sensors(),
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(hit.target.collider_entity, Some(c));
                assert_eq!(hit.target.body_entity, Some(b));
                let hit = p
                    .cast_ray(
                        vec(-5.0, -2.0),
                        vec(10.0, 0.0),
                        RayCastOptions::default(),
                        Filter::default(),
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(hit.target.collider_entity, Some(sensor));
                // Local collider pose shifts centre of mass relative to the body origin.
                assert!(p.bodies()[bh].mass() > 0.0);
                assert!(p.bodies()[bh].local_center_of_mass().x.abs() > 0.01);
                let hit = p
                    .cast_shape(
                        &compound().shape,
                        query(-5.0, 1.0),
                        vec(10.0, 0.0),
                        ShapeCastOptions::default(),
                        Filter::default().exclude_sensors(),
                    )
                    .unwrap();
                assert!(hit.is_some());
                let mut hits = Vec::new();
                let _ = p
                    .visit_overlaps(
                        &compound().shape,
                        query(0.0, 1.0),
                        Filter::default().exclude_sensors(),
                        |t| {
                            hits.push(t);
                            ::std::ops::ControlFlow::Continue(())
                        },
                    )
                    .unwrap();
                assert_eq!(hits.len(), 1);
            }
        }

        #[test]
        fn collider_removal_and_live_edits_preserve_body_joints_and_other_colliders() {
            for regular in [false, true] {
                let mut w = world();
                let b = body(&mut w);
                let other = body(&mut w);
                let c = w.spawn_with((Owner { body: b }, compound())).unwrap();
                w.insert(b, Collider::ball(0.2)).unwrap();
                let j = w
                    .spawn_with((Joint::fixed(
                        b,
                        other,
                        Default::default(),
                        Default::default(),
                    ),))
                    .unwrap();
                assert!(sync(&mut w, regular).is_empty());
                let mut s = SystemsContainer::new();
                s.add_exclusive(JointSync);
                assert!(EcsRunner::single_thread().run(&mut w, &s).is_empty());
                let bh = w.resource::<Physics>().body_for_entity(b).unwrap();
                let ch = w.resource::<Physics>().collider_for_entity(c).unwrap();
                let jh = w.resource::<Physics>().joint_for_entity(j).unwrap();
                step(&mut w);
                w.resource_mut::<Physics>().body_motion(bh).unwrap().sleep();
                assert!(sync(&mut w, regular).is_empty());
                assert!(w.resource::<Physics>().bodies()[bh].is_sleeping());
                let shared = w.resource::<Physics>().colliders()[ch]
                    .shared_shape()
                    .clone();
                w.get_mut::<Collider>(c).unwrap().local_pose = local(0.0, 3.0, 0.7);
                assert!(sync(&mut w, regular).is_empty());
                assert!(::std::ptr::eq(
                    shared.as_ref(),
                    w.resource::<Physics>().colliders()[ch].shape()
                ));
                w.get_mut::<Collider>(c).unwrap().shape = Collider::ball(0.8).shape;
                assert!(sync(&mut w, regular).is_empty());
                assert_eq!(w.resource::<Physics>().collider_for_entity(c), Some(ch));
                w.resource_mut::<Physics>()
                    .body_motion(bh)
                    .unwrap()
                    .set_linvel(native_vec(2.0, 0.0), true);
                w.remove::<Collider>(b).unwrap();
                w.remove::<Collider>(c).unwrap();
                assert!(sync(&mut w, regular).is_empty());
                let p = w.resource::<Physics>();
                assert_eq!(p.body_for_entity(b), Some(bh));
                assert_eq!(p.bodies()[bh].linvel(), native_vec(2.0, 0.0));
                assert_eq!(p.joint_for_entity(j), Some(jh));
                assert!(p.bodies()[bh].colliders().is_empty());
                assert!(w.get::<CH>(c).is_none());
                assert!(w.get::<CH>(b).is_none());
            }
        }

        #[test]
        fn missing_excluded_and_removed_owners_suspend_colliders_without_despawning_entities() {
            for regular in [false, true] {
                let mut w = world();
                let b = body(&mut w);
                let c = w.spawn_with((Owner { body: b }, compound())).unwrap();
                assert!(sync(&mut w, regular).is_empty());
                let ch = w.resource::<Physics>().collider_for_entity(c).unwrap();
                w.remove::<Body>(b).unwrap();
                assert!(sync(&mut w, regular).is_empty());
                assert!(w.is_alive(c));
                assert!(w.get::<Collider>(c).is_some());
                assert!(w.get::<CH>(c).is_none());
                assert!(w.resource::<Physics>().colliders().is_empty());
                w.insert(b, Body::dynamic()).unwrap();
                assert!(sync(&mut w, regular).is_empty());
                assert_ne!(w.resource::<Physics>().collider_for_entity(c), Some(ch));
                w.set_entity_flags(b, Entity::DISABLED);
                assert!(sync(&mut w, regular).is_empty());
                assert!(w.resource::<Physics>().colliders().is_empty());
                w.set_entity_flags(b, 0);
                assert!(sync(&mut w, regular).is_empty());
                w.despawn(b);
                let replacement = body(&mut w);
                assert_eq!(replacement.index(), b.index());
                assert!(sync(&mut w, regular).is_empty());
                assert!(w.resource::<Physics>().collider_for_entity(c).is_none());
                assert!(w.get::<CH>(c).is_none());
            }
        }

        #[test]
        fn reattachment_ends_old_contact_and_captures_new_body_identity() {
            for regular in [false, true] {
                let mut w = world();
                let a = body(&mut w);
                let b = body(&mut w);
                let obstacle = body(&mut w);
                w.insert(obstacle, Collider::ball(1.0)).unwrap();
                let c = w.spawn_with((Owner { body: a }, tracked())).unwrap();
                let mut cursor = EventCursor::<Event>::new();
                assert!(sync(&mut w, regular).is_empty());
                step(&mut w);
                let old = w
                    .resource::<Events<Event>>()
                    .read(&mut cursor)
                    .copied()
                    .collect::<Vec<_>>();
                assert_eq!(old.len(), 1);
                let ch = w.resource::<Physics>().collider_for_entity(c).unwrap();
                w.get_mut::<Owner>(c).unwrap().body = b;
                assert!(sync(&mut w, regular).is_empty());
                step(&mut w);
                let events = w
                    .resource::<Events<Event>>()
                    .read(&mut cursor)
                    .copied()
                    .collect::<Vec<_>>();
                assert_eq!(events.len(), 2);
                assert_eq!(
                    events[0].phase,
                    CollisionPhase::Stopped(CollisionStopReason::Removed)
                );
                let old_part = [events[0].a, events[0].b]
                    .into_iter()
                    .find(|p| p.collider_entity == Some(c))
                    .unwrap();
                assert_eq!(old_part.body_entity, Some(a));
                assert_eq!(old_part.collider, ch);
                assert_eq!(events[1].phase, CollisionPhase::Started);
                let new_part = [events[1].a, events[1].b]
                    .into_iter()
                    .find(|p| p.collider_entity == Some(c))
                    .unwrap();
                assert_eq!(new_part.body_entity, Some(b));
                assert_ne!(new_part.collider, ch);
            }
        }

        #[test]
        fn force_events_capture_independent_collider_and_body_entities() {
            let mut w = world();
            w.add_event::<Force>();
            let fixed = w.spawn_with((Body::fixed(), Transform::IDENTITY)).unwrap();
            let moving = body(&mut w);
            w.get_mut::<Transform>(moving).unwrap().translation.x = 1.5;
            let a = w
                .spawn_with((Owner { body: fixed }, Collider::ball(1.0)))
                .unwrap();
            let b = w
                .spawn_with((
                    Owner { body: moving },
                    Collider::ball(1.0).with_contact_force_events(Some(
                        physics::ContactForceSettings { min_force: 0.0 },
                    )),
                ))
                .unwrap();
            assert!(sync(&mut w, true).is_empty());
            let bh = w.resource::<Physics>().body_for_entity(moving).unwrap();
            w.resource_mut::<Physics>()
                .body_motion(bh)
                .unwrap()
                .set_linvel(native_vec(-1.0, 0.0), true);
            step(&mut w);
            let cursor = EventCursor::<Force>::new();
            let events = w
                .resource::<Events<Force>>()
                .read(&cursor)
                .copied()
                .collect::<Vec<_>>();
            assert!(!events.is_empty());
            for e in events {
                let ca = [e.a, e.b]
                    .into_iter()
                    .find(|p| p.collider_entity == Some(a))
                    .unwrap();
                let cb = [e.a, e.b]
                    .into_iter()
                    .find(|p| p.collider_entity == Some(b))
                    .unwrap();
                assert_eq!(ca.body_entity, Some(fixed));
                assert_eq!(cb.body_entity, Some(moving));
            }
        }

        #[test]
        fn validation_is_atomic_for_compounds_and_local_poses() {
            for regular in [false, true] {
                let mut w = world();
                let b = body(&mut w);
                let c = w.spawn_with((Owner { body: b }, compound())).unwrap();
                assert!(sync(&mut w, regular).is_empty());
                let ch = w.resource::<Physics>().collider_for_entity(c).unwrap();
                let before = w.get::<Collider>(c).unwrap().clone();
                for bad in [
                    Collider::compound(vec![]),
                    Collider::compound(vec![Part::ball(-1.0)]),
                    compound().with_local_pose(local(f32::NAN, 0.0, 0.0)),
                    Collider::compound(vec![Part::ball(1.0).with_local_pose(local(
                        0.0,
                        f32::INFINITY,
                        0.0,
                    ))]),
                ] {
                    w.insert(c, bad).unwrap();
                    let new = w
                        .spawn_with((Body::dynamic(), Transform::IDENTITY))
                        .unwrap();
                    let errors = sync(&mut w, regular);
                    assert!(
                        matches!(
                            errors.as_slice(),
                            [SystemError::InvalidConfiguration { .. }]
                        ),
                        "{errors:?}"
                    );
                    assert_eq!(w.resource::<Physics>().collider_for_entity(c), Some(ch));
                    assert!(w.resource::<Physics>().body_for_entity(new).is_none());
                    w.despawn(new);
                }
                w.insert(c, before).unwrap();
                assert!(sync(&mut w, regular).is_empty());
            }
        }

        #[test]
        fn deferred_publication_cancels_changed_ownership_but_keeps_a_body_without_colliders() {
            for edit in [
                (|w: &mut World, e| {
                    w.remove::<Collider>(e).unwrap();
                }) as fn(&mut World, Entity),
                |w, e| {
                    w.get_mut::<Owner>(e).unwrap().body = Entity::DANGLING;
                },
                |w, e| {
                    let b = w.get::<Owner>(e).unwrap().body;
                    w.despawn(b);
                },
                |w, e| {
                    w.despawn(e);
                },
            ] {
                let mut w = world();
                let b = body(&mut w);
                let c = w.spawn_with((Owner { body: b }, compound())).unwrap();
                let mut s = SystemsContainer::new();
                s.add(Edit(c, edit));
                s.add(SyncRegular);
                s.add_edge::<Edit, SyncRegular>().unwrap();
                assert!(EcsRunner::multi_thread(2).run(&mut w, &s).is_empty());
                assert!(w.resource::<Physics>().colliders().is_empty());
                if w.is_alive(b) {
                    assert!(w.resource::<Physics>().body_for_entity(b).is_some());
                }
            }
            // Removing the colocated collider in earlier commands no longer cancels its body.
            let mut w = world();
            let b = body(&mut w);
            w.insert(b, Collider::ball(1.0)).unwrap();
            let mut s = SystemsContainer::new();
            s.add(Edit(b, |w, e| {
                w.remove::<Collider>(e).unwrap();
            }));
            s.add(SyncRegular);
            s.add_edge::<Edit, SyncRegular>().unwrap();
            assert!(EcsRunner::single_thread().run(&mut w, &s).is_empty());
            assert!(w.get::<BH>(b).is_some());
            assert!(w.resource::<Physics>().colliders().is_empty());
        }

        #[test]
        fn scene_and_prefab_remap_explicit_owners_and_preserve_compound_geometry() {
            let mut w = world();
            let b = body(&mut w);
            w.insert(b, Name::new("body")).unwrap();
            let c = w
                .spawn_with((Name::new("collider"), Owner { body: b }, compound()))
                .unwrap();
            // Hierarchy selects the prefab contents, but doesn't supply physical ownership.
            set_parent(&mut w, c, b);
            let prefab = w.serialize_prefab(b).unwrap();
            for _ in 0..2 {
                let entities = w.deserialize_prefab(&prefab).unwrap();
                let owner_entity = *entities
                    .iter()
                    .find(|e| w.get::<Body>(**e).is_some())
                    .unwrap();
                let collider_entity = *entities
                    .iter()
                    .find(|e| w.get::<Owner>(**e).is_some())
                    .unwrap();
                assert_eq!(w.get::<Owner>(collider_entity).unwrap().body, owner_entity);
                assert_ne!(owner_entity, b);
                assert_eq!(w.get::<Collider>(collider_entity), Some(&compound()));
            }
            assert!(sync(&mut w, false).is_empty());
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
            for snapshot in snapshots {
                let mut target = world();
                for _ in 0..16 {
                    target.spawn();
                }
                let entities = target.deserialize_world_into(&snapshot).unwrap();
                for &e in &entities {
                    if let Some(owner) = target.get::<Owner>(e) {
                        assert!(entities.contains(&owner.body));
                        assert!(target.get::<Body>(owner.body).is_some());
                        assert_eq!(target.get::<Collider>(e), Some(&compound()));
                        assert!(target.get::<CH>(e).is_none());
                    }
                }
                assert!(sync(&mut target, true).is_empty());
                assert_eq!(target.resource::<Physics>().colliders().len(), 3);
            }
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod dim2 {
    use super::*;
    use physics::components2d::{
        Collider2D as Collider, ColliderBody2D as Owner, ColliderPart2D as Part,
        ColliderPose2D as Local, ImpulseJoint2D as Joint, RigidBody2D as Body,
    };
    use physics::events2d::{CollisionEvent2D as Event, ContactForceEvent2D as Force};
    use physics::systems2d::{
        StepPhysics2D as Step, SyncPhysicsBodies2D as Sync,
        SyncPhysicsBodiesSystem2D as SyncRegular, SyncPhysicsJoints2D as JointSync,
    };
    use physics::world2d::{
        Collider2DHandle as CH, PhysicsWorld2D as Physics, RigidBody2DHandle as BH,
    };
    fn native_vec(x: f32, y: f32) -> physics::rapier2d::prelude::Vector {
        physics::rapier2d::prelude::Vector::new(x as _, y as _)
    }
    use physics::control2d::PhysicsPose2D as QueryPose;
    use physics::queries2d::QueryFilter as Filter;
    fn vec(x: f32, y: f32) -> redlilium_core::math::Vec2 {
        redlilium_core::math::Vec2::new(x, y)
    }
    fn local(x: f32, y: f32, angle: f32) -> Local {
        Local {
            translation: vec(x, y),
            rotation: angle,
        }
    }
    fn query(x: f32, y: f32) -> QueryPose {
        QueryPose {
            translation: vec(x, y),
            ..Default::default()
        }
    }
    tests!();
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod dim3 {
    use super::*;
    use physics::components3d::{
        Collider3D as Collider, ColliderBody3D as Owner, ColliderPart3D as Part,
        ColliderPose3D as Local, ImpulseJoint3D as Joint, RigidBody3D as Body,
    };
    use physics::events3d::{CollisionEvent3D as Event, ContactForceEvent3D as Force};
    use physics::systems3d::{
        StepPhysics3D as Step, SyncPhysicsBodies3D as Sync,
        SyncPhysicsBodiesSystem3D as SyncRegular, SyncPhysicsJoints3D as JointSync,
    };
    use physics::world3d::{
        Collider3DHandle as CH, PhysicsWorld3D as Physics, RigidBody3DHandle as BH,
    };
    fn native_vec(x: f32, y: f32) -> physics::rapier3d::prelude::Vector {
        physics::rapier3d::prelude::Vector::new(x as _, y as _, 0.0)
    }
    use physics::control3d::PhysicsPose3D as QueryPose;
    use physics::queries3d::QueryFilter as Filter;
    fn vec(x: f32, y: f32) -> redlilium_core::math::Vec3 {
        redlilium_core::math::Vec3::new(x, y, 0.0)
    }
    fn local(x: f32, y: f32, angle: f32) -> Local {
        Local {
            translation: vec(x, y),
            rotation: quat_from_rotation_z(angle),
        }
    }
    fn query(x: f32, y: f32) -> QueryPose {
        QueryPose {
            translation: vec(x, y),
            ..Default::default()
        }
    }
    tests!();
}
