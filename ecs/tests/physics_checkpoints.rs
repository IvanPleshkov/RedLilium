#![cfg(any(
    feature = "physics-2d",
    feature = "physics-2d-f32",
    feature = "physics-3d",
    feature = "physics-3d-f32"
))]
use redlilium_core::math::Vec3;
use redlilium_ecs::physics::{
    CcdSettings, CollisionEventSettings, ContactForceSettings, JointMotor, PhysicsCheckpointError,
    PhysicsEventDelivery, TeleportVelocity,
};
use redlilium_ecs::*;

macro_rules! tests {
    () => {
        #[cfg(target_arch = "wasm32")]
        pub(super) fn run_contracts() {
            sphere_exact_continuation_reusable_checkpoint_and_clock();
            persistent_contacts_sleep_and_wake_replay_exactly();
            removed_joint_restored_after_game_restores_descriptor();
            ccd_repeated_restore_and_invalid_identity_are_atomic();
            pending_teleport_forces_and_kinematic_inputs_survive_repeated_restore();
            resource_busy_and_unsynchronized_capture_return_errors();
            restore_does_not_invoke_game_lifecycle_hooks();
            legacy_queues_clear_and_tick_events_remain_readable_without_cursors();
        }
        fn fixture() -> (World, Schedules, EcsRunner, Entity) {
            let mut world = World::new();
            register_std_components(&mut world);
            let mut physics = Physics::default();
            physics.gravity = Default::default();
            physics.event_delivery = PhysicsEventDelivery::TickOnly;
            world.insert_resource(physics);
            let body = world
                .spawn_with((
                    Transform::IDENTITY,
                    Body::dynamic(),
                    Collider::ball(0.5)
                        .with_collision_events(Some(CollisionEventSettings::default())),
                ))
                .unwrap();
            let mut schedules = Schedules::new();
            schedules.set_fixed_timestep(1.0 / 128.0);
            let fixed = schedules.get_mut::<FixedUpdate>();
            fixed.add_exclusive(BodySync);
            fixed.add_exclusive(JointSync);
            fixed.add(Step);
            fixed.add(Record);
            fixed.add_edge::<BodySync, JointSync>().unwrap();
            fixed.add_edge::<JointSync, Step>().unwrap();
            fixed.add_edge::<Step, Record>().unwrap();
            (world, schedules, EcsRunner::single_thread(), body)
        }
        fn advance(world: &mut World, schedules: &mut Schedules, runner: &EcsRunner, n: usize) {
            for _ in 0..n {
                schedules.run_fixed_tick(world, runner).unwrap();
            }
        }
        fn capture(world: &mut World) -> Checkpoint {
            Bindings::with(world, |physics, bindings| {
                physics.capture_checkpoint(bindings)
            })
            .unwrap()
        }
        fn restore(
            world: &mut World,
            checkpoint: &Checkpoint,
        ) -> Result<(), PhysicsCheckpointError> {
            Bindings::with(world, |physics, bindings| {
                physics.restore_checkpoint(checkpoint, bindings)
            })
        }
        fn state(world: &World) -> String {
            let p = world.resource::<Physics>();
            let bodies: Vec<_> = p
                .bodies()
                .iter()
                .map(|(h, b)| {
                    format!(
                        "{h:?}|{:?}|{:?}|{:?}|{}",
                        b.position(),
                        b.linvel(),
                        b.angvel(),
                        b.is_sleeping()
                    )
                })
                .collect();
            let poses: Vec<_> = world
                .read::<Interpolation>()
                .unwrap()
                .iter()
                .map(|(i, h)| format!("{i}:{h:?}"))
                .collect();
            format!(
                "{bodies:?}|{:?}|{:?}|{poses:?}",
                p.impulse_joints(),
                p.last_step_events()
            )
        }
        fn velocity(world: &World, entity: Entity, x: f64) {
            let mut physics = world.resource_mut::<Physics>();
            let h = physics.body_for_entity(entity).unwrap();
            let mut v = physics.bodies()[h].linvel();
            v.x = x as _;
            physics.body_motion(h).unwrap().set_linvel(v, true);
        }
        fn continuation(
            world: &mut World,
            schedules: &mut Schedules,
            runner: &EcsRunner,
            n: usize,
        ) -> Vec<String> {
            (0..n)
                .map(|_| {
                    advance(world, schedules, runner, 1);
                    state(world)
                })
                .collect()
        }
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        fn sphere_exact_continuation_reusable_checkpoint_and_clock() {
            let (mut w, mut s, r, e) = fixture();
            // Materialize without consuming a physics tick.
            run_exclusive_system_once(&mut BodySync, &mut w).unwrap();
            velocity(&w, e, 8.0);
            advance(&mut w, &mut s, &r, 64);
            let checkpoint = capture(&mut w);
            let clock = s.capture_fixed_clock(&w).unwrap();
            let saved = state(&w);
            let expected = continuation(&mut w, &mut s, &r, 32);
            for checkpoint in [&checkpoint, &checkpoint.clone(), &checkpoint] {
                restore(&mut w, checkpoint).unwrap();
                s.restore_fixed_clock(&mut w, &clock).unwrap();
                assert_eq!(state(&w), saved);
                let p = w.resource::<Physics>();
                let h = p.body_for_entity(e).unwrap();
                assert_eq!(p.bodies()[h].linvel().x, 8.0);
                assert_eq!(p.bodies()[h].position().translation.x, 4.0);
                drop(p);
                assert_eq!(continuation(&mut w, &mut s, &r, 32), expected);
                assert_eq!(w.resource::<FixedTime>().tick(), 96);
                let p = w.resource::<Physics>();
                assert_eq!(
                    p.bodies()[p.body_for_entity(e).unwrap()]
                        .position()
                        .translation
                        .x,
                    6.0
                );
            }
        }
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        fn persistent_contacts_sleep_and_wake_replay_exactly() {
            let (mut w, mut s, r, e) = fixture();
            w.resource_mut::<Physics>().gravity.y = -9.81;
            w.spawn_with((
                Transform::from_translation(Vec3::new(0.0, -10.5, 0.0)),
                Body::fixed(),
                Collider::ball(10.0)
                    .with_contact_force_events(Some(ContactForceSettings::default())),
            ))
            .unwrap();
            advance(&mut w, &mut s, &r, 256);
            assert!(
                w.resource::<Physics>()
                    .narrow_phase()
                    .contact_pairs()
                    .any(|p| p.has_any_active_contact())
            );
            let checkpoint = capture(&mut w);
            let clock = s.capture_fixed_clock(&w).unwrap();
            let expected = continuation(&mut w, &mut s, &r, 64);
            restore(&mut w, &checkpoint).unwrap();
            s.restore_fixed_clock(&mut w, &clock).unwrap();
            assert_eq!(continuation(&mut w, &mut s, &r, 64), expected);
            let h = w.resource::<Physics>().body_for_entity(e).unwrap();
            w.resource_mut::<Physics>().body_motion(h).unwrap().sleep();
            let sleeping = capture(&mut w);
            let clock = s.capture_fixed_clock(&w).unwrap();
            velocity(&w, e, 1.0);
            let expected = continuation(&mut w, &mut s, &r, 40);
            restore(&mut w, &sleeping).unwrap();
            s.restore_fixed_clock(&mut w, &clock).unwrap();
            assert!(w.resource::<Physics>().bodies()[h].is_sleeping());
            velocity(&w, e, 1.0);
            assert_eq!(continuation(&mut w, &mut s, &r, 40), expected);
        }
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        fn removed_joint_restored_after_game_restores_descriptor() {
            let (mut w, mut s, r, e) = fixture();
            w.resource_mut::<Physics>().gravity.y = -9.81;
            let anchor = w.spawn_with((Transform::IDENTITY, Body::fixed())).unwrap();
            let mut joint = Joint::new(anchor, e, Kind::Prismatic);
            joint.motor = Some(JointMotor::position(0.75, 40.0, 10.0));
            let joint_entity = w.spawn_with((joint.clone(),)).unwrap();
            advance(&mut w, &mut s, &r, 64);
            let checkpoint = capture(&mut w);
            let clock = s.capture_fixed_clock(&w).unwrap();
            let original_handle = w
                .resource::<Physics>()
                .joint_for_entity(joint_entity)
                .unwrap();
            assert!(w.resource::<Physics>().pose(e).unwrap().translation.x > 0.1);
            let expected = continuation(&mut w, &mut s, &r, 64);
            w.remove::<Joint>(joint_entity).unwrap();
            advance(&mut w, &mut s, &r, 4);
            assert!(
                w.resource::<Physics>()
                    .joint_for_entity(joint_entity)
                    .is_none()
            );
            let before = state(&w);
            assert!(matches!(
                restore(&mut w, &checkpoint),
                Err(PhysicsCheckpointError::DescriptorsChanged(_))
            ));
            assert_eq!(state(&w), before);
            w.insert(joint_entity, joint).unwrap();
            restore(&mut w, &checkpoint).unwrap();
            s.restore_fixed_clock(&mut w, &clock).unwrap();
            assert_eq!(
                w.resource::<Physics>().joint_for_entity(joint_entity),
                Some(original_handle)
            );
            assert_eq!(continuation(&mut w, &mut s, &r, 64), expected);
        }
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        fn ccd_repeated_restore_and_invalid_identity_are_atomic() {
            let (mut w, mut s, r, e) = fixture();
            w.get_mut::<Body>(e).unwrap().ccd = Some(CcdSettings::default());
            w.spawn_with((
                Transform::from_translation(Vec3::new(3.0, 0.0, 0.0)),
                Body::fixed(),
                Collider::ball(0.1),
            ))
            .unwrap();
            run_exclusive_system_once(&mut BodySync, &mut w).unwrap();
            velocity(&w, e, 240.0);
            let checkpoint = capture(&mut w);
            let clock = s.capture_fixed_clock(&w).unwrap();
            let expected = continuation(&mut w, &mut s, &r, 12);
            assert!(
                w.resource::<Physics>().pose(e).unwrap().translation.x < 3.0,
                "CCD must stop the fast body"
            );
            for _ in 0..3 {
                restore(&mut w, &checkpoint).unwrap();
                s.restore_fixed_clock(&mut w, &clock).unwrap();
                assert_eq!(continuation(&mut w, &mut s, &r, 12), expected);
            }
            let (mut other, _, _, _) = fixture();
            assert!(matches!(
                restore(&mut other, &checkpoint),
                Err(PhysicsCheckpointError::WrongPhysicsResource)
            ));
            assert!(other.adopt_resource_from::<Physics>(&mut w));
            assert!(matches!(
                restore(&mut other, &checkpoint),
                Err(PhysicsCheckpointError::WrongWorld)
            ));
            assert!(w.adopt_resource_from::<Physics>(&mut other));
            w.despawn(e);
            let replacement = w.spawn();
            assert_ne!(e, replacement);
            let before = state(&w);
            assert!(matches!(
                restore(&mut w, &checkpoint),
                Err(PhysicsCheckpointError::EntityChanged(_))
            ));
            assert_eq!(state(&w), before);
        }
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        fn pending_teleport_forces_and_kinematic_inputs_survive_repeated_restore() {
            let (mut w, mut s, r, e) = fixture();
            let mut motion = Motion::default();
            motion.linear.x = 2.0;
            let k = w
                .spawn_with((Transform::IDENTITY, Body::kinematic_velocity(), motion))
                .unwrap();
            let target: Target = Pose::default().into();
            let kt = w
                .spawn_with((Transform::IDENTITY, Body::kinematic_position(), target))
                .unwrap();
            advance(&mut w, &mut s, &r, 2);
            let mut pose = Pose::default();
            pose.translation.x = 10.0;
            w.resource_mut::<Physics>()
                .teleport(e, pose, TeleportVelocity::Preserve)
                .unwrap();
            let h = w.resource::<Physics>().body_for_entity(e).unwrap();
            let mut force = w.resource::<Physics>().bodies()[h].linvel();
            force.x = 10.0;
            w.resource_mut::<Physics>()
                .body_motion(h)
                .unwrap()
                .add_force(force, true);
            let checkpoint = capture(&mut w);
            let clock = s.capture_fixed_clock(&w).unwrap();
            let saved = state(&w);
            let expected = continuation(&mut w, &mut s, &r, 16);
            assert!(w.resource::<Physics>().pose(e).unwrap().translation.x > 10.0);
            for _ in 0..2 {
                w.remove::<Motion>(k).unwrap();
                w.remove::<Target>(kt).unwrap();
                restore(&mut w, &checkpoint).unwrap();
                s.restore_fixed_clock(&mut w, &clock).unwrap();
                assert_eq!(state(&w), saved);
                assert_eq!(w.get::<Motion>(k).unwrap().linear.x, 2.0);
                assert!(w.get::<Target>(kt).is_some());
                assert_eq!(continuation(&mut w, &mut s, &r, 16), expected);
            }
        }
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        fn resource_busy_and_unsynchronized_capture_return_errors() {
            let (mut w, mut s, r, e) = fixture();
            advance(&mut w, &mut s, &r, 1);
            let resource = w.resource_shared::<Physics>();
            let guard = resource.read();
            assert!(matches!(
                Bindings::with(&mut w, |p, b| p.capture_checkpoint(b)),
                Err(PhysicsCheckpointError::Busy(_))
            ));
            drop(guard);
            w.remove::<Body>(e).unwrap();
            assert!(matches!(
                Bindings::with(&mut w, |p, b| p.capture_checkpoint(b)),
                Err(PhysicsCheckpointError::Unsynchronized(_))
            ));
        }
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        fn restore_does_not_invoke_game_lifecycle_hooks() {
            let (mut w, mut s, r, e) = fixture();
            advance(&mut w, &mut s, &r, 2);
            let checkpoint = capture(&mut w);
            w.insert_resource(0usize);
            w.on_insert::<BodyHandle>(|w, _| *w.resource_mut::<usize>() += 1);
            w.on_remove::<BodyHandle>(|w, _| *w.resource_mut::<usize>() += 1);
            w.remove::<BodyHandle>(e).unwrap();
            let calls = *w.resource::<usize>();
            restore(&mut w, &checkpoint).unwrap();
            assert!(w.get::<BodyHandle>(e).is_some());
            assert_eq!(*w.resource::<usize>(), calls);
        }
        #[cfg_attr(not(target_arch = "wasm32"), test)]
        fn legacy_queues_clear_and_tick_events_remain_readable_without_cursors() {
            let (mut w, mut s, r, _) = fixture();
            w.add_event::<Collision>();
            w.resource_mut::<Physics>().event_delivery = PhysicsEventDelivery::FrameQueues;
            let wall = w
                .spawn_with((
                    Transform::from_translation(Vec3::new(0.8, 0.0, 0.0)),
                    Body::fixed(),
                    Collider::ball(0.5),
                ))
                .unwrap();
            advance(&mut w, &mut s, &r, 1);
            assert!(!w.resource::<Events<Collision>>().is_empty());
            let batch = w.resource::<Physics>().last_step_events().clone();
            let checkpoint = capture(&mut w);
            let before = state(&w);
            w.get_mut::<Transform>(wall).unwrap().translation.x = 2.0;
            assert!(matches!(
                restore(&mut w, &checkpoint),
                Err(PhysicsCheckpointError::DescriptorsChanged(
                    "fixed body Transform"
                ))
            ));
            assert_eq!(state(&w), before);
            assert_eq!(w.get::<Transform>(wall).unwrap().translation.x, 2.0);
            assert!(!w.resource::<Events<Collision>>().is_empty());
            w.get_mut::<Transform>(wall).unwrap().translation.x = 0.8;
            advance(&mut w, &mut s, &r, 3);
            restore(&mut w, &checkpoint).unwrap();
            assert!(w.resource::<Events<Collision>>().is_empty());
            assert_eq!(*w.resource::<Physics>().last_step_events(), batch);
        }
    };
}

#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod dim2 {
    use super::*;
    use redlilium_ecs::physics::{
        components2d::{
            Collider2D as Collider, ImpulseJoint2D as Joint, JointType2D as Kind,
            RigidBody2D as Body,
        },
        control2d::{
            KinematicTarget2D as Target, KinematicVelocity2D as Motion, PhysicsPose2D as Pose,
        },
        events2d::CollisionEvent2D as Collision,
        systems2d::{
            RecordPhysicsPose2D as Record, StepPhysics2D as Step, SyncPhysicsBodies2D as BodySync,
            SyncPhysicsJoints2D as JointSync,
        },
        world2d::{
            PhysicsBindings2D as Bindings, PhysicsCheckpoint2D as Checkpoint,
            PhysicsInterpolation2D as Interpolation, PhysicsWorld2D as Physics,
            RigidBody2DHandle as BodyHandle,
        },
    };
    tests!();
}
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod dim3 {
    use super::*;
    use redlilium_ecs::physics::{
        components3d::{
            Collider3D as Collider, ImpulseJoint3D as Joint, JointType3D as Kind,
            RigidBody3D as Body,
        },
        control3d::{
            KinematicTarget3D as Target, KinematicVelocity3D as Motion, PhysicsPose3D as Pose,
        },
        events3d::CollisionEvent3D as Collision,
        systems3d::{
            RecordPhysicsPose as Record, StepPhysics3D as Step, SyncPhysicsBodies3D as BodySync,
            SyncPhysicsJoints3D as JointSync,
        },
        world3d::{
            PhysicsBindings3D as Bindings, PhysicsCheckpoint3D as Checkpoint,
            PhysicsInterpolation as Interpolation, PhysicsWorld3D as Physics,
            RigidBody3DHandle as BodyHandle,
        },
    };
    tests!();
}

/// Same contract cases, invoked by the headless WASM example.
#[cfg(target_arch = "wasm32")]
pub fn run_wasm_contracts() {
    #[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
    dim2::run_contracts();
    #[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
    dim3::run_contracts();
}
