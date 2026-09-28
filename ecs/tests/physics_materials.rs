//! Contact combination, live updates and authored material persistence.
#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_ecs::physics::CoefficientCombineRule as Rule;
use redlilium_ecs::*;
const RULES: [Rule; 6] = [
    Rule::Average,
    Rule::Min,
    Rule::Multiply,
    Rule::Max,
    Rule::ClampedSum,
    Rule::GeometricMean,
];
const COMBINED: [f64; 6] = [0.5, 0.2, 0.16, 0.8, 1.0, 0.4];
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
}
macro_rules! tests {
    () => {
        fn world() -> World {
            let mut w = World::new();
            register_std_components(&mut w);
            w.insert_resource(Physics::default());
            w
        }
        fn sync(w: &mut World, regular: bool) -> Vec<SystemError> {
            let mut s = SystemsContainer::new();
            if regular {
                s.add(SyncRegular);
            } else {
                s.add_exclusive(Sync);
            }
            EcsRunner::single_thread().run(w, &s)
        }
        fn valid(w: &mut World, regular: bool) {
            let errors = sync(w, regular);
            assert!(errors.is_empty(), "{errors:?}");
        }
        fn fixture(regular: bool, rule: Rule) -> (World, Entity, Entity) {
            let mut w = world();
            let a = w
                .spawn_with((
                    Transform::from_translation(redlilium_core::math::Vec3::new(0.0, -1.0, 0.0)),
                    Body::fixed(),
                    floor()
                        .with_friction(0.2)
                        .with_restitution(0.2)
                        .with_friction_combine_rule(Some(rule))
                        .with_restitution_combine_rule(Some(rule)),
                ))
                .unwrap();
            let b = w
                .spawn_with((
                    Transform::IDENTITY,
                    Body::dynamic(),
                    Collider::ball(0.5).with_friction(0.8).with_restitution(0.8),
                ))
                .unwrap();
            valid(&mut w, regular);
            (w, a, b)
        }
        fn check_contact(w: &mut World, a: Entity, b: Entity, friction: f64, restitution: f64) {
            let mut s = SystemsContainer::new();
            s.add(Step);
            assert!(EcsRunner::single_thread().run(w, &s).is_empty());
            let p = w.resource::<Physics>();
            let pair = p
                .narrow_phase()
                .contact_pair(
                    p.collider_for_entity(a).unwrap(),
                    p.collider_for_entity(b).unwrap(),
                )
                .expect("touching colliders");
            let mut count = 0;
            for manifold in pair.manifolds() {
                if !manifold.data.solver_contacts.is_empty() {
                    close(manifold.data.friction as f64, friction);
                    close(manifold.data.restitution as f64, restitution);
                    count += 1;
                }
            }
            assert!(count > 0, "expected an active contact");
        }
        #[test]
        fn creation_and_live_contact_rules_use_priority_independently() {
            for regular in [false, true] {
                for (i, rule1) in RULES.into_iter().enumerate() {
                    let (mut w, a, b) = fixture(regular, rule1);
                    check_contact(&mut w, a, b, COMBINED[i], COMBINED[i]);
                    let ch = w.resource::<Physics>().collider_for_entity(b).unwrap();
                    let bh = w.resource::<Physics>().body_for_entity(b).unwrap();
                    for (j, rule2) in RULES.into_iter().enumerate() {
                        w.get_mut::<Collider>(b).unwrap().friction_combine_rule = Some(rule2);
                        valid(&mut w, regular);
                        // Restitution retains its own rule selection, unrelated to friction.
                        check_contact(&mut w, a, b, COMBINED[i.max(j)], COMBINED[i]);
                        assert_eq!(w.resource::<Physics>().collider_for_entity(b), Some(ch));
                        assert_eq!(w.resource::<Physics>().body_for_entity(b), Some(bh));
                    }
                    w.get_mut::<Collider>(a).unwrap().friction_combine_rule = None;
                    w.get_mut::<Collider>(a).unwrap().restitution_combine_rule = None;
                    w.get_mut::<Collider>(b).unwrap().friction_combine_rule = None;
                    valid(&mut w, regular);
                    check_contact(&mut w, a, b, 0.5, 0.5);
                }
            }
        }
        #[test]
        fn rule_edit_wakes_body_but_noop_keeps_sleep_and_invalid_batch_is_rejected() {
            for regular in [false, true] {
                let (mut w, a, b) = fixture(regular, Rule::Average);
                check_contact(&mut w, a, b, 0.5, 0.5);
                let bh = w.resource::<Physics>().body_for_entity(b).unwrap();
                w.resource_mut::<Physics>().body_motion(bh).unwrap().sleep();
                valid(&mut w, regular);
                assert!(w.resource::<Physics>().bodies()[bh].is_sleeping());
                w.get_mut::<Collider>(b).unwrap().friction_combine_rule = Some(Rule::Min);
                w.get_mut::<Collider>(a).unwrap().friction = f32::NAN;
                assert!(matches!(
                    sync(&mut w, regular).as_slice(),
                    [SystemError::InvalidConfiguration { .. }]
                ));
                assert!(w.resource::<Physics>().bodies()[bh].is_sleeping());
                let p = w.resource::<Physics>();
                let ch = p.collider_for_entity(b).unwrap();
                assert_eq!(
                    p.colliders()[ch].friction_combine_rule(),
                    NativeRule::Average
                );
                drop(p);
                w.get_mut::<Collider>(a).unwrap().friction = 0.2;
                valid(&mut w, regular);
                assert!(!w.resource::<Physics>().bodies()[bh].is_sleeping());
                check_contact(&mut w, a, b, 0.2, 0.5);
            }
        }
        #[test]
        fn editing_static_surface_refreshes_existing_sleeping_contact() {
            for regular in [false, true] {
                let (mut w, a, b) = fixture(regular, Rule::Average);
                check_contact(&mut w, a, b, 0.5, 0.5);
                let bh = w.resource::<Physics>().body_for_entity(b).unwrap();
                w.resource_mut::<Physics>().body_motion(bh).unwrap().sleep();
                w.get_mut::<Collider>(a).unwrap().friction_combine_rule = Some(Rule::Min);
                w.get_mut::<Collider>(a).unwrap().restitution_combine_rule = Some(Rule::Max);
                valid(&mut w, regular);
                check_contact(&mut w, a, b, 0.2, 0.8);
                assert!(!w.resource::<Physics>().bodies()[bh].is_sleeping());
                // Existing scalar coefficient edits need the same invalidation.
                w.get_mut::<Collider>(a).unwrap().friction = 0.1;
                w.get_mut::<Collider>(a).unwrap().restitution = 0.9;
                valid(&mut w, regular);
                check_contact(&mut w, a, b, 0.1, 0.9);
            }
        }
        #[test]
        fn all_rule_overrides_roundtrip_through_scene_formats() {
            for rule in RULES.into_iter().map(Some).chain([None]) {
                let mut w = world();
                let collider = Collider::ball(0.5)
                    .with_friction_combine_rule(rule)
                    .with_restitution_combine_rule(rule);
                w.spawn_with((collider.clone(),)).unwrap();
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
                    let entities = target.deserialize_world_into(&snapshot).unwrap();
                    assert_eq!(target.get::<Collider>(entities[0]), Some(&collider));
                }
            }
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod dim2 {
    use super::*;
    use redlilium_ecs::physics::{
        components2d::{Collider2D as Collider, RigidBody2D as Body},
        rapier2d::prelude::CoefficientCombineRule as NativeRule,
        systems2d::{
            StepPhysics2D as Step, SyncPhysicsBodies2D as Sync,
            SyncPhysicsBodiesSystem2D as SyncRegular,
        },
        world2d::PhysicsWorld2D as Physics,
    };
    fn floor() -> Collider {
        Collider::cuboid(10.0, 0.5)
    }
    tests!();
}

#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod dim3 {
    use super::*;
    use redlilium_ecs::physics::{
        components3d::{Collider3D as Collider, RigidBody3D as Body},
        rapier3d::prelude::CoefficientCombineRule as NativeRule,
        systems3d::{
            StepPhysics3D as Step, SyncPhysicsBodies3D as Sync,
            SyncPhysicsBodiesSystem3D as SyncRegular,
        },
        world3d::PhysicsWorld3D as Physics,
    };
    fn floor() -> Collider {
        Collider::cuboid(10.0, 0.5, 10.0)
    }
    tests!();
}
