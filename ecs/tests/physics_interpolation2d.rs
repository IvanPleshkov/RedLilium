#![cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
use redlilium_core::math::{Vec2, Vec3, quat_from_rotation_z};
use redlilium_ecs::physics::{
    TeleportVelocity,
    components2d::{Collider2D, RigidBody2D},
    control2d::PhysicsPose2D,
    rapier2d::prelude::Vector,
    systems2d::{
        InterpolatePhysics2D, RecordPhysicsPose2D, StepPhysics2D, SyncPhysicsBodies2D,
        SyncPhysicsBodiesSystem2D,
    },
    world2d::{PhysicsInterpolation2D, PhysicsWorld2D, RigidBody2DHandle},
};
use redlilium_ecs::*;
fn run<S: System<Result = ()> + 'static>(w: &mut World, system: S) {
    let mut systems = SystemsContainer::new();
    systems.add(system);
    let errors = EcsRunner::single_thread().run(w, &systems);
    assert!(errors.is_empty(), "{errors:?}");
}
fn sync(w: &mut World, regular: bool) {
    let mut systems = SystemsContainer::new();
    if regular {
        systems.add(SyncPhysicsBodiesSystem2D);
    } else {
        systems.add_exclusive(SyncPhysicsBodies2D);
    }
    let errors = EcsRunner::single_thread().run(w, &systems);
    assert!(errors.is_empty(), "{errors:?}");
}
fn setup(body: RigidBody2D) -> (World, Entity) {
    let mut w = World::new();
    register_std_components(&mut w);
    w.insert_resource(PhysicsWorld2D::default());
    w.resource_mut::<PhysicsWorld2D>().gravity = Vector::ZERO;
    let e = w
        .spawn_with((
            body,
            Collider2D::ball(0.5),
            Transform::from_translation(Vec3::new(0.0, 0.0, 7.0)),
        ))
        .unwrap();
    sync(&mut w, false);
    (w, e)
}
fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
}
fn velocity(w: &mut World, e: Entity, x: f32) {
    let h = w.resource::<PhysicsWorld2D>().body_for_entity(e).unwrap();
    w.resource_mut::<PhysicsWorld2D>()
        .body_motion(h)
        .unwrap()
        .set_linvel(Vector::new(x as _, 0.0), true);
}
fn render_half(w: &mut World) {
    let mut schedules = Schedules::new();
    schedules.set_fixed_timestep(0.02);
    schedules.get_mut::<PostUpdate>().add(InterpolatePhysics2D);
    schedules.run_frame(w, &EcsRunner::single_thread(), 0.01);
}
#[test]
fn seed_and_record_use_authoritative_pose_and_missing_time_shows_latest() {
    let (mut w, e) = setup(RigidBody2D::dynamic());
    w.get_mut::<Transform>(e).unwrap().translation.x = 123.0;
    run(&mut w, RecordPhysicsPose2D);
    let h = w.get::<PhysicsInterpolation2D>(e).unwrap();
    assert_eq!(h.prev_translation, h.cur_translation);
    close(h.cur_translation.x, 0.0);
    velocity(&mut w, e, 6.0);
    run(&mut w, StepPhysics2D);
    run(&mut w, RecordPhysicsPose2D);
    let h = w.get::<PhysicsInterpolation2D>(e).unwrap();
    close(h.prev_translation.x, 0.0);
    close(h.cur_translation.x, 0.1);
    run(&mut w, InterpolatePhysics2D);
    close(w.get::<Transform>(e).unwrap().translation.x, 0.1);
    close(w.get::<Transform>(e).unwrap().translation.z, 7.0);
}
#[test]
fn shortest_arc_blending_preserves_depth_and_scale() {
    for (previous, current) in [(170.0_f32, -170.0_f32), (-170.0, 170.0)] {
        let (mut w, e) = setup(RigidBody2D::dynamic());
        w.insert(
            e,
            PhysicsInterpolation2D {
                prev_translation: Vec2::new(0.0, 2.0),
                cur_translation: Vec2::new(4.0, 6.0),
                prev_rotation: previous.to_radians(),
                cur_rotation: current.to_radians(),
            },
        )
        .unwrap();
        // Interpolation is presentation only and must leave unrelated fields alone.
        w.get_mut::<Transform>(e).unwrap().scale = Vec3::new(2.0, 3.0, 4.0);
        render_half(&mut w);
        let t = w.get::<Transform>(e).unwrap();
        close(t.translation.x, 2.0);
        close(t.translation.y, 4.0);
        close(t.translation.z, 7.0);
        assert_eq!(t.scale, Vec3::new(2.0, 3.0, 4.0));
        close(
            t.rotation
                .coords
                .dot(&quat_from_rotation_z(::std::f32::consts::PI).coords)
                .abs(),
            1.0,
        );
        close(
            w.resource::<PhysicsWorld2D>()
                .pose(e)
                .unwrap()
                .translation
                .x,
            0.0,
        );
    }
}
#[test]
fn catch_up_steps_keep_the_last_two_poses_without_render_feedback() {
    let (mut w, e) = setup(RigidBody2D::dynamic());
    velocity(&mut w, e, 1.0);
    let mut schedules = Schedules::new();
    schedules.set_fixed_timestep(0.02);
    let fixed = schedules.get_mut::<FixedUpdate>();
    fixed.add(StepPhysics2D);
    fixed.add(RecordPhysicsPose2D);
    fixed
        .add_edge::<StepPhysics2D, RecordPhysicsPose2D>()
        .unwrap();
    schedules.get_mut::<PostUpdate>().add(InterpolatePhysics2D);
    let runner = EcsRunner::single_thread();
    schedules.run_frame(&mut w, &runner, 0.055);
    let h = w.get::<PhysicsInterpolation2D>(e).unwrap();
    close(h.prev_translation.x, 0.02);
    close(h.cur_translation.x, 0.04);
    close(w.get::<Transform>(e).unwrap().translation.x, 0.035);
    schedules.run_frame(&mut w, &runner, 0.01);
    close(
        w.resource::<PhysicsWorld2D>()
            .pose(e)
            .unwrap()
            .translation
            .x,
        0.06,
    );
    close(w.get::<Transform>(e).unwrap().translation.x, 0.045);
}
#[test]
fn teleports_reset_history_for_dynamic_and_kinematic_bodies() {
    for body in [
        RigidBody2D::dynamic(),
        RigidBody2D::kinematic_position(),
        RigidBody2D::kinematic_velocity(),
    ] {
        let (mut w, e) = setup(body);
        run(&mut w, RecordPhysicsPose2D);
        let pose = PhysicsPose2D {
            translation: Vec2::new(10.0, 3.0),
            rotation: 1.2,
        };
        w.resource_mut::<PhysicsWorld2D>()
            .teleport(e, pose, TeleportVelocity::Reset)
            .unwrap();
        run(&mut w, StepPhysics2D);
        run(&mut w, RecordPhysicsPose2D);
        render_half(&mut w);
        let h = w.get::<PhysicsInterpolation2D>(e).unwrap();
        close(h.prev_translation.x, 10.0);
        close(h.cur_translation.x, 10.0);
        close(h.prev_rotation, 1.2);
        close(h.cur_rotation, 1.2);
        let t = w.get::<Transform>(e).unwrap();
        close(t.translation.x, 10.0);
        close(t.translation.z, 7.0);
    }
}
#[test]
fn fixed_bodies_are_not_interpolated_and_type_changes_reset_history() {
    for regular in [false, true] {
        let (mut w, e) = setup(RigidBody2D::dynamic());
        run(&mut w, RecordPhysicsPose2D);
        w.insert(e, RigidBody2D::fixed()).unwrap();
        sync(&mut w, regular);
        w.get_mut::<Transform>(e).unwrap().translation.x = 15.0;
        run(&mut w, StepPhysics2D);
        run(&mut w, RecordPhysicsPose2D);
        render_half(&mut w);
        close(w.get::<Transform>(e).unwrap().translation.x, 15.0);
        w.insert(e, RigidBody2D::kinematic_position()).unwrap();
        sync(&mut w, regular);
        run(&mut w, StepPhysics2D);
        run(&mut w, RecordPhysicsPose2D);
        let h = w.get::<PhysicsInterpolation2D>(e).unwrap();
        close(h.prev_translation.x, 15.0);
        close(h.cur_translation.x, 15.0);
    }
    let (mut w, e) = setup(RigidBody2D::fixed());
    run(&mut w, RecordPhysicsPose2D);
    assert!(w.get::<PhysicsInterpolation2D>(e).is_none());
}
#[test]
fn removal_and_recreation_release_and_reseed_history() {
    for regular in [false, true] {
        let (mut w, e) = setup(RigidBody2D::dynamic());
        run(&mut w, RecordPhysicsPose2D);
        let old = w.get::<RigidBody2DHandle>(e).unwrap().0;
        w.remove::<RigidBody2D>(e).unwrap();
        sync(&mut w, regular);
        assert!(w.get::<PhysicsInterpolation2D>(e).is_none());
        w.get_mut::<Transform>(e).unwrap().translation.x = 25.0;
        run(&mut w, InterpolatePhysics2D);
        close(w.get::<Transform>(e).unwrap().translation.x, 25.0);
        w.insert(e, RigidBody2D::dynamic()).unwrap();
        sync(&mut w, regular);
        run(&mut w, RecordPhysicsPose2D);
        assert_ne!(w.get::<RigidBody2DHandle>(e).unwrap().0, old);
        let h = w.get::<PhysicsInterpolation2D>(e).unwrap();
        close(h.prev_translation.x, 25.0);
        close(h.cur_translation.x, 25.0);
    }
}
struct Recycle(Entity);
impl System for Recycle {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        let e = self.0;
        ctx.commands(move |w| {
            w.despawn(e);
            let new = w.spawn();
            assert_eq!(new.index(), e.index());
        });
        Ok(())
    }
}
#[test]
fn deferred_seeding_does_not_follow_recycled_entity() {
    for multi in [false, true] {
        let (mut w, e) = setup(RigidBody2D::dynamic());
        let mut systems = SystemsContainer::new();
        systems.add(Recycle(e));
        systems.add(RecordPhysicsPose2D);
        systems.add_edge::<Recycle, RecordPhysicsPose2D>().unwrap();
        let runner = if multi {
            EcsRunner::multi_thread(2)
        } else {
            EcsRunner::single_thread()
        };
        assert!(runner.run(&mut w, &systems).is_empty());
        let new = w.entity_at_index(e.index()).unwrap();
        assert!(w.get::<PhysicsInterpolation2D>(new).is_none());
    }
}
