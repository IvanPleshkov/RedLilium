use redlilium_core::math::{Mat4, Vec3, quat_from_rotation_x, quat_from_rotation_y};
use redlilium_core::scene::{CameraProjection, NodeTransform, Scene, SceneCamera, SceneNode};
use redlilium_ecs::{
    ComputePool, EcsRunner, IoRuntime, SystemsContainer, World, run_system_blocking,
};

use redlilium_ecs::std::components::*;
use redlilium_ecs::std::systems::*;
use redlilium_ecs::{register_std_components, spawn_scene};

// ---------------------------------------------------------------------------
// Full pipeline: spawn → systems → query
// ---------------------------------------------------------------------------

#[test]
fn full_frame_pipeline() {
    let mut world = World::new();
    register_std_components(&mut world);

    // Advance tick so that insert stamps ticks_changed > 0,
    // allowing Changed<Transform> to detect them (since_tick=0, tick>0).
    world.advance_tick();

    // Spawn a camera at (0, 5, 10) looking toward origin
    let cam_entity = world.spawn();
    world
        .insert(
            cam_entity,
            Transform::from_translation(Vec3::new(0.0, 5.0, 10.0)),
        )
        .unwrap();
    world.insert(cam_entity, GlobalTransform::IDENTITY).unwrap();
    world
        .insert(
            cam_entity,
            Camera::perspective(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 1000.0),
        )
        .unwrap();

    // Spawn a few objects at different positions
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(5.0, 0.0, -5.0),
        Vec3::new(-3.0, 2.0, -10.0),
    ];
    let mut objects = Vec::new();
    for pos in &positions {
        let e = world.spawn();
        world.insert(e, Transform::from_translation(*pos)).unwrap();
        world.insert(e, GlobalTransform::IDENTITY).unwrap();
        world.insert(e, Visibility::VISIBLE).unwrap();
        objects.push(e);
    }

    // Build systems container with dependencies
    let mut container = SystemsContainer::new();
    container.add(UpdateGlobalTransforms);
    container.add(UpdateCameraMatrices);
    container
        .add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>()
        .unwrap();

    // Run with single-threaded runner
    let runner = EcsRunner::single_thread();
    runner.run(&mut world, &container);

    // Verify camera matrices were computed
    let cameras = world.read::<Camera>().unwrap();
    let cam = cameras.get(cam_entity.index()).unwrap();
    assert_ne!(cam.view_matrix, Mat4::identity());
    assert_ne!(cam.projection_matrix, Mat4::identity());

    // Verify the view matrix places the camera at (0, 5, 10)
    let inv = cam.view_matrix.try_inverse().unwrap();
    let cam_pos = Vec3::new(inv[(0, 3)], inv[(1, 3)], inv[(2, 3)]);
    assert!((cam_pos - Vec3::new(0.0, 5.0, 10.0)).norm() < 1e-4);

    // Verify object global transforms match their local transforms
    drop(cameras);
    let globals = world.read::<GlobalTransform>().unwrap();
    for (i, &obj) in objects.iter().enumerate() {
        let gt = globals.get(obj.index()).unwrap();
        assert!(
            (gt.translation() - positions[i]).norm() < 1e-6,
            "Object {i} global transform mismatch"
        );
    }
}

// ---------------------------------------------------------------------------
// Multi-threaded execution
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn multi_thread_execution() {
    let mut world = World::new();
    register_std_components(&mut world);

    // Advance tick so that insert stamps ticks_changed > 0.
    world.advance_tick();

    // Spawn 100 entities with transforms
    for i in 0..100 {
        let e = world.spawn();
        let angle = (i as f32) * 0.1;
        world
            .insert(
                e,
                Transform::new(
                    Vec3::new(i as f32, 0.0, 0.0),
                    quat_from_rotation_y(angle),
                    Vec3::new(1.0, 1.0, 1.0),
                ),
            )
            .unwrap();
        world.insert(e, GlobalTransform::IDENTITY).unwrap();
    }

    let mut container = SystemsContainer::new();
    container.add(UpdateGlobalTransforms);

    let runner = EcsRunner::multi_thread(4);
    runner.run(&mut world, &container);

    // Verify all global transforms were updated
    let transforms = world.read::<Transform>().unwrap();
    let globals = world.read::<GlobalTransform>().unwrap();
    for (idx, transform) in transforms.iter() {
        let global = globals.get(idx).unwrap();
        let expected = transform.to_matrix();
        assert!(
            (global.0 - expected).norm() < 1e-6,
            "Entity at index {idx} has incorrect global transform"
        );
    }
}

// ---------------------------------------------------------------------------
// Scene spawning with full system pipeline
// ---------------------------------------------------------------------------

#[test]
fn spawn_scene_and_run_systems() {
    let mut world = World::new();
    redlilium_ecs::register_std_components(&mut world);

    let scene = Scene::new()
        .with_name("TestScene")
        .with_cameras(vec![SceneCamera {
            name: Some("MainCam".to_string()),
            projection: CameraProjection::Perspective {
                yfov: 1.0,
                aspect: Some(16.0 / 9.0),
                znear: 0.1,
                zfar: Some(500.0),
            },
        }])
        .with_nodes(vec![
            SceneNode::new()
                .with_name("root")
                .with_transform(NodeTransform::IDENTITY.with_translation([5.0, 0.0, 0.0]))
                .with_children(vec![
                    SceneNode::new()
                        .with_name("camera_node")
                        .with_transform(NodeTransform::IDENTITY.with_translation([0.0, 10.0, 0.0]))
                        .with_camera(0),
                    SceneNode::new().with_name("mesh_node").with_transform(
                        NodeTransform::IDENTITY.with_rotation([0.0, 0.383, 0.0, 0.924]),
                    ),
                ]),
        ]);

    // Advance tick so that insert stamps ticks_changed > 0.
    world.advance_tick();

    let roots = spawn_scene(&mut world, &scene);

    assert_eq!(roots.len(), 1);
    // root + camera_node + mesh_node = 3 entities
    assert_eq!(world.entity_count(), 3);

    // Run systems
    let mut container = SystemsContainer::new();
    container.add(UpdateGlobalTransforms);
    container.add(UpdateCameraMatrices);
    container
        .add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>()
        .unwrap();

    let runner = EcsRunner::single_thread();
    runner.run(&mut world, &container);

    // Verify root entity
    let root = roots[0];
    let gt = world.get::<GlobalTransform>(root).unwrap();
    assert!((gt.translation() - Vec3::new(5.0, 0.0, 0.0)).norm() < 1e-5);
    let name = world.get::<Name>(root).unwrap();
    assert_eq!(name.as_str(), "root");

    // Find the camera entity by querying Camera component
    let cameras_storage = world.read::<Camera>().unwrap();
    let mut cam_count = 0;
    for (_, cam) in cameras_storage.iter() {
        cam_count += 1;
        // Camera matrices should be computed
        assert_ne!(cam.projection_matrix, Mat4::identity());
        assert_ne!(cam.view_matrix, Mat4::identity());
    }
    assert_eq!(cam_count, 1);
}

// ---------------------------------------------------------------------------
// Visibility filtering pattern
// ---------------------------------------------------------------------------

#[test]
fn visibility_filtering_with_systems() {
    let mut world = World::new();
    register_std_components(&mut world);

    // Advance tick so that insert stamps ticks_changed > 0.
    world.advance_tick();

    // Spawn 5 entities, hide every other one
    let mut entities = Vec::new();
    for i in 0..5 {
        let e = world.spawn();
        world
            .insert(
                e,
                Transform::from_translation(Vec3::new(i as f32, 0.0, 0.0)),
            )
            .unwrap();
        world.insert(e, GlobalTransform::IDENTITY).unwrap();
        world
            .insert(
                e,
                if i % 2 == 0 {
                    Visibility::VISIBLE
                } else {
                    Visibility::HIDDEN
                },
            )
            .unwrap();
        entities.push(e);
    }

    // Run transform system via run_blocking
    let compute = ComputePool::new(IoRuntime::new());
    let io = IoRuntime::new();
    run_system_blocking(&UpdateGlobalTransforms, &world, &compute, &io).unwrap();

    // Query visible entities (the rendering pattern)
    let globals = world.read::<GlobalTransform>().unwrap();
    let visibility = world.read::<Visibility>().unwrap();

    let visible_positions: Vec<Vec3> = globals
        .iter()
        .filter(|(idx, _)| visibility.get(*idx).is_some_and(|v| v.is_visible()))
        .map(|(_, gt)| gt.translation())
        .collect();

    assert_eq!(visible_positions.len(), 3); // indices 0, 2, 4
    assert!(visible_positions.contains(&Vec3::new(0.0, 0.0, 0.0)));
    assert!(visible_positions.contains(&Vec3::new(2.0, 0.0, 0.0)));
    assert!(visible_positions.contains(&Vec3::new(4.0, 0.0, 0.0)));
}

// ---------------------------------------------------------------------------
// Multiple system ticks (simulating a game loop)
// ---------------------------------------------------------------------------

#[test]
fn multiple_frame_simulation() {
    let mut world = World::new();
    register_std_components(&mut world);

    // Advance tick so that insert stamps ticks_changed > 0.
    world.advance_tick();

    let entity = world.spawn();
    world
        .insert(entity, Transform::from_translation(Vec3::zeros()))
        .unwrap();
    world.insert(entity, GlobalTransform::IDENTITY).unwrap();

    let mut container = SystemsContainer::new();
    container.add(UpdateGlobalTransforms);

    let runner = EcsRunner::single_thread();

    // Simulate 10 frames of movement
    for frame in 0..10 {
        // "Move" the entity each frame
        {
            let mut transforms = world.write::<Transform>().unwrap();
            let mut t = transforms.get_mut(entity.index()).unwrap();
            t.translation = Vec3::new(frame as f32, 0.0, 0.0);
        }

        runner.run(&mut world, &container);

        // Verify global transform tracks the local transform
        let globals = world.read::<GlobalTransform>().unwrap();
        let gt = globals.get(entity.index()).unwrap();
        assert!(
            (gt.translation().x - frame as f32).abs() < 1e-6,
            "Frame {frame}: expected x={}, got x={}",
            frame,
            gt.translation().x
        );
    }
}

// ---------------------------------------------------------------------------
// Light + transform interaction pattern
// ---------------------------------------------------------------------------

#[test]
fn light_direction_from_transform() {
    let mut world = World::new();
    register_std_components(&mut world);

    // Advance tick so that insert stamps ticks_changed > 0.
    world.advance_tick();

    // Create a directional light pointing down (-Y rotation)
    let sun = world.spawn();
    let rotation = quat_from_rotation_x(-std::f32::consts::FRAC_PI_4); // 45° downward
    world
        .insert(sun, Transform::from_rotation(rotation))
        .unwrap();
    world.insert(sun, GlobalTransform::IDENTITY).unwrap();
    world
        .insert(
            sun,
            DirectionalLight::new(Vec3::new(1.0, 0.98, 0.9), 100000.0),
        )
        .unwrap();
    world.insert(sun, Name::new("Sun")).unwrap();

    // Create point lights at various positions
    let light_positions = [
        Vec3::new(5.0, 3.0, 0.0),
        Vec3::new(-5.0, 3.0, 0.0),
        Vec3::new(0.0, 3.0, 5.0),
        Vec3::new(0.0, 3.0, -5.0),
    ];
    for (i, pos) in light_positions.iter().enumerate() {
        let e = world.spawn();
        world.insert(e, Transform::from_translation(*pos)).unwrap();
        world.insert(e, GlobalTransform::IDENTITY).unwrap();
        world
            .insert(
                e,
                PointLight::new(Vec3::new(1.0, 1.0, 1.0), 100.0).with_range(20.0),
            )
            .unwrap();
        world
            .insert(e, Name::new(format!("PointLight_{i}")))
            .unwrap();
    }

    // Run transform system via run_blocking
    let compute = ComputePool::new(IoRuntime::new());
    let io = IoRuntime::new();
    run_system_blocking(&UpdateGlobalTransforms, &world, &compute, &io).unwrap();

    // Query directional light direction from its global transform
    let globals = world.read::<GlobalTransform>().unwrap();
    let dir_lights = world.read::<DirectionalLight>().unwrap();

    for (idx, _light) in dir_lights.iter() {
        let gt = globals.get(idx).unwrap();
        let direction = gt.forward();
        // 45° downward from -Z: direction should have negative Y and negative Z
        assert!(direction.y < 0.0, "Sun should point downward");
        assert!(direction.z < 0.0, "Sun should point forward-ish");
    }
    drop(globals);
    drop(dir_lights);

    // Query point light positions from their global transforms
    let globals = world.read::<GlobalTransform>().unwrap();
    let point_lights = world.read::<PointLight>().unwrap();

    let mut light_count = 0;
    for (idx, light) in point_lights.iter() {
        let gt = globals.get(idx).unwrap();
        let pos = gt.translation();
        assert_eq!(light.range, 20.0);
        assert!(pos.y > 0.0, "All point lights should be above ground");
        light_count += 1;
    }
    assert_eq!(light_count, 4);
}

// ---------------------------------------------------------------------------
// register_std_components prevents panics on empty queries
// ---------------------------------------------------------------------------

#[test]
fn register_prevents_empty_world_panic() {
    let mut world = World::new();
    register_std_components(&mut world);

    let mut container = SystemsContainer::new();
    container.add(UpdateGlobalTransforms);
    container.add(UpdateCameraMatrices);
    container
        .add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>()
        .unwrap();

    // Should not panic even with zero entities
    let runner = EcsRunner::single_thread();
    runner.run(&mut world, &container);
}

// ---------------------------------------------------------------------------
// Prefab serialization of a child entity
// ---------------------------------------------------------------------------

/// A prefab cut from the middle of a hierarchy must not carry the root's
/// `Parent` (it references an entity outside the subtree — attachment
/// context, not content). Regression: delete-undo of a child entity failed
/// with "entity reference points outside the deserialized set".
#[test]
fn child_prefab_roundtrip_drops_external_parent() {
    let mut world = World::new();
    register_std_components(&mut world);

    let parent = world.spawn();
    let child = world.spawn();
    let grandchild = world.spawn();
    for e in [parent, child, grandchild] {
        world.insert(e, Transform::default()).unwrap();
    }
    redlilium_ecs::set_parent(&mut world, child, parent);
    redlilium_ecs::set_parent(&mut world, grandchild, child);

    let prefab = world.serialize_prefab(child).unwrap();
    // The root of the cut must not serialize its Parent; internal hierarchy
    // (child -> grandchild) is content and stays.
    assert!(
        !prefab.entities[0]
            .components
            .iter()
            .any(|c| c.type_name == "Parent"),
        "root's external Parent must be dropped"
    );

    let spawned = world
        .deserialize_prefab(&prefab)
        .expect("child prefab must deserialize");
    assert_eq!(spawned.len(), 2);
    // The clone is a root (no parent); its own subtree is intact.
    assert!(world.get::<Parent>(spawned[0]).is_none());
    assert_eq!(
        world.get::<Parent>(spawned[1]).map(|p| p.0),
        Some(spawned[0])
    );
}

// ---------------------------------------------------------------------------
// Per-system change detection (issue #21)
// ---------------------------------------------------------------------------

mod change_detection {
    use redlilium_ecs::{
        EcsRunner, MaybeAdded, MaybeChanged, Read, System, SystemContext, SystemError,
        SystemsContainer, World, Write,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct Marker(u32);

    /// Bumps every `Marker` once, on its first run only.
    struct MutateOnce {
        runs: Arc<AtomicU32>,
    }
    impl System for MutateOnce {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            if self.runs.fetch_add(1, Ordering::SeqCst) == 0 {
                ctx.lock::<(Write<Marker>,)>().execute(|(mut markers,)| {
                    for (_, mut m) in markers.iter_mut() {
                        m.0 += 1;
                    }
                });
            }
            Ok(())
        }
    }

    /// Counts entities whose `Marker` changed since this system's last run.
    struct CountChanged {
        seen: Arc<AtomicU32>,
    }
    impl System for CountChanged {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            ctx.lock::<(Read<Marker>, MaybeChanged<Marker>)>()
                .execute(|(markers, changed)| {
                    for (idx, _) in markers.iter() {
                        if changed.matches(idx) {
                            self.seen.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                });
            Ok(())
        }
    }

    /// Counts entities whose `Marker` was added since this system's last run.
    struct CountAdded {
        seen: Arc<AtomicU32>,
    }
    impl System for CountAdded {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            ctx.lock::<(Read<Marker>, MaybeAdded<Marker>)>()
                .execute(|(markers, added)| {
                    for (idx, _) in markers.iter() {
                        if added.matches(idx) {
                            self.seen.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                });
            Ok(())
        }
    }

    /// Spawns one `Marker` entity via deferred commands, on its first run only.
    struct SpawnViaCommandsOnce {
        runs: Arc<AtomicU32>,
    }
    impl System for SpawnViaCommandsOnce {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            if self.runs.fetch_add(1, Ordering::SeqCst) == 0 {
                ctx.spawn_with((Marker(7),));
            }
            Ok(())
        }
    }

    fn setup() -> (World, Arc<AtomicU32>, Arc<AtomicU32>) {
        let mut world = World::new();
        world.register_component::<Marker>();
        let e = world.spawn();
        world.insert(e, Marker(0)).unwrap();
        (
            world,
            Arc::new(AtomicU32::new(0)),
            Arc::new(AtomicU32::new(0)),
        )
    }

    /// Audit C2, scenario 1: a system ordered BEFORE the mutator must see the
    /// mutation on its next run. With the old frame-global `since = tick - 1`
    /// window it never did.
    #[test]
    fn changed_visible_to_system_ordered_before_mutator() {
        let (mut world, seen, runs) = setup();

        let mut systems = SystemsContainer::new();
        systems.add(CountChanged { seen: seen.clone() });
        systems.add(MutateOnce { runs });
        // Observer runs strictly before the mutator every frame.
        systems.add_edge::<CountChanged, MutateOnce>().unwrap();

        let runner = EcsRunner::single_thread();

        // Frame 1: the setup-time insert (stamped at world tick 1) is visible
        // to the never-run observer (last_run 0); the mutator then writes.
        runner.run(&mut world, &systems);
        assert_eq!(
            seen.load(Ordering::SeqCst),
            1,
            "setup insert visible on first run"
        );

        // Frame 2: the observer must see the mutation made AFTER it ran in
        // frame 1.
        runner.run(&mut world, &systems);
        assert_eq!(
            seen.load(Ordering::SeqCst),
            2,
            "frame-1 mutation visible in frame 2"
        );

        // Frame 3: nothing mutated since — no re-trigger.
        runner.run(&mut world, &systems);
        assert_eq!(seen.load(Ordering::SeqCst), 2, "no spurious re-detection");
    }

    /// Audit C2, scenario 2: components inserted by deferred commands (applied
    /// at end of frame) must be visible to Added filters next frame.
    #[test]
    fn command_applied_insert_visible_via_added() {
        let mut world = World::new();
        world.register_component::<Marker>();
        let seen = Arc::new(AtomicU32::new(0));
        let runs = Arc::new(AtomicU32::new(0));

        let mut systems = SystemsContainer::new();
        systems.add(CountAdded { seen: seen.clone() });
        systems.add(SpawnViaCommandsOnce { runs });

        let runner = EcsRunner::single_thread();

        // Frame 1: spawn queued, applied after both systems ran.
        runner.run(&mut world, &systems);
        assert_eq!(seen.load(Ordering::SeqCst), 0);

        // Frame 2: the command-applied insert must be visible.
        runner.run(&mut world, &systems);
        assert_eq!(
            seen.load(Ordering::SeqCst),
            1,
            "command insert visible next frame"
        );

        // Frame 3: no re-trigger.
        runner.run(&mut world, &systems);
        assert_eq!(seen.load(Ordering::SeqCst), 1);
    }

    /// Audit C2, scenario 3 (editor Render pattern): a separate schedule run
    /// after the mutating schedule must see the current frame's changes.
    #[test]
    fn later_schedule_sees_current_frame_changes() {
        let (mut world, seen, runs) = setup();

        let mut update = SystemsContainer::new();
        update.add(MutateOnce { runs });
        let mut render = SystemsContainer::new();
        render.add(CountChanged { seen: seen.clone() });

        let runner = EcsRunner::single_thread();

        // One host frame: Update mutates, Render (a separate container run
        // right after) must see both the setup insert and the mutation.
        runner.run(&mut world, &update);
        runner.run(&mut world, &render);
        assert_eq!(
            seen.load(Ordering::SeqCst),
            1,
            "render sees same-frame mutation"
        );

        // Next host frame: nothing new.
        runner.run(&mut world, &update);
        runner.run(&mut world, &render);
        assert_eq!(
            seen.load(Ordering::SeqCst),
            1,
            "no re-detection in later frames"
        );
    }

    /// Same as scenario 1, driven by the multi-threaded runner (atomic tick
    /// assignment and last_run updates through &SystemsContainer).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn changed_visible_before_mutator_multi_thread() {
        let (mut world, seen, runs) = setup();

        let mut systems = SystemsContainer::new();
        systems.add(CountChanged { seen: seen.clone() });
        systems.add(MutateOnce { runs });
        systems.add_edge::<CountChanged, MutateOnce>().unwrap();

        let runner = EcsRunner::multi_thread(2);
        runner.run(&mut world, &systems);
        assert_eq!(seen.load(Ordering::SeqCst), 1);
        runner.run(&mut world, &systems);
        assert_eq!(seen.load(Ordering::SeqCst), 2);
        runner.run(&mut world, &systems);
        assert_eq!(seen.load(Ordering::SeqCst), 2);
    }
}

// ---------------------------------------------------------------------------
// Runner robustness (issue #16)
// ---------------------------------------------------------------------------

mod runner_robustness {
    use redlilium_ecs::{
        EcsRunner, MainThreadResMut, OnAdd, System, SystemContext, SystemError, SystemsContainer,
        Triggers, World,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::mpsc;

    struct Marker(#[allow(dead_code)] u32);

    /// Issue #16 (B1): a panic inside main-thread-dispatched work used to
    /// unwind the multi runner's coordination loop while the requesting
    /// worker blocked forever on its result channel — a permanent hang. Now
    /// the panic is caught and reported as a system error.
    #[test]
    fn main_thread_work_panic_is_reported_not_hung() {
        struct MainThreadState(#[allow(dead_code)] u32);

        struct PanicsOnMainThread;
        impl System for PanicsOnMainThread {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                ctx.lock::<(MainThreadResMut<MainThreadState>,)>()
                    .execute(|_state| {
                        panic!("boom on the main thread");
                    });
                Ok(())
            }
        }

        let mut world = World::new();
        world.insert_main_thread_resource(MainThreadState(0));

        let mut systems = SystemsContainer::new();
        systems.add(PanicsOnMainThread);

        // Run in a thread with a deadline so a regression fails the test
        // instead of hanging the suite forever.
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let runner = EcsRunner::multi_thread(2);
            let errors = runner.run(&mut world, &systems);
            let _ = tx.send(errors.len());
        });
        let error_count = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("multi runner hung on a main-thread panic (issue #16 B1)");
        assert!(error_count > 0, "the panic must surface as a system error");
    }

    /// Issue #16 (C3): one runner drives several schedules; previous-tick
    /// results used to be stored in a single slot validated only by node
    /// count, so a system could receive a foreign schedule's result through
    /// `reuse_result`.
    #[test]
    fn prev_results_do_not_leak_across_schedules() {
        struct Producer {
            output: u32,
            reused: Arc<AtomicU32>,
        }
        impl System for Producer {
            type Result = u32;
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<u32, SystemError> {
                Ok(self.output)
            }
            fn reuse_result(&self, prev: u32) {
                self.reused.store(prev, Ordering::SeqCst);
            }
        }

        let mut world = World::new();
        let reused_a = Arc::new(AtomicU32::new(0));
        let reused_b = Arc::new(AtomicU32::new(0));

        // Two schedules with identical shape (1 system, same Result type).
        let mut schedule_a = SystemsContainer::new();
        schedule_a.add(Producer {
            output: 1,
            reused: reused_a.clone(),
        });
        let mut schedule_b = SystemsContainer::new();
        schedule_b.add(Producer {
            output: 2,
            reused: reused_b.clone(),
        });

        let runner = EcsRunner::single_thread();

        // Frame 1: A then B. B must NOT receive A's result for reuse.
        runner.run(&mut world, &schedule_a);
        runner.run(&mut world, &schedule_b);
        assert_eq!(
            reused_b.load(Ordering::SeqCst),
            0,
            "B reused a foreign schedule's result"
        );

        // Frame 2: each schedule gets its own previous result back.
        runner.run(&mut world, &schedule_a);
        runner.run(&mut world, &schedule_b);
        assert_eq!(reused_a.load(Ordering::SeqCst), 1);
        assert_eq!(reused_b.load(Ordering::SeqCst), 2);
    }

    /// Issue #16 (C4): enabling the same trigger buffer twice used to
    /// register a duplicate swap fn (a double swap wipes `readable` right
    /// after it was filled) and a duplicate observer (double counting).
    #[test]
    fn enable_triggers_is_idempotent() {
        let mut world = World::new();
        world.register_component::<Marker>();
        world.enable_add_triggers::<Marker>();
        world.enable_add_triggers::<Marker>(); // second call must be a no-op

        struct Noop;
        impl System for Noop {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                Ok(())
            }
        }

        let e = world.spawn();
        world.insert(e, Marker(1)).unwrap();
        // A runner pass (empty containers early-return) flushes the queued
        // observer triggers at end of run.
        let mut noop = SystemsContainer::new();
        noop.add(Noop);
        let runner = EcsRunner::single_thread();
        runner.run(&mut world, &noop);
        world.update_triggers();

        let triggers = world.resource::<Triggers<OnAdd<Marker>>>();
        assert_eq!(
            triggers.len(),
            1,
            "duplicate observer or double swap detected"
        );
    }
}

// ---------------------------------------------------------------------------
// Events: per-reader cursors, exactly-once delivery through run_frame
// ---------------------------------------------------------------------------

mod events {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use redlilium_ecs::{
        EcsRunner, EventCursor, Events, PostUpdate, PreUpdate, Res, ResMut, Schedules, System,
        SystemContext, SystemError, Update, World,
    };

    struct Damage(u32);

    /// Sends Damage(1) on frame 1 and Damage(2) on frame 2, nothing after.
    struct Sender {
        frame: AtomicU32,
    }
    impl System for Sender {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            let frame = self.frame.fetch_add(1, Ordering::SeqCst) + 1;
            ctx.lock::<(ResMut<Events<Damage>>,)>()
                .execute(|(mut events,)| {
                    if frame <= 2 {
                        events.send(Damage(frame));
                    }
                });
            Ok(())
        }
    }

    /// Accumulates every Damage value it reads (sum and count).
    struct Reader {
        cursor: EventCursor<Damage>,
        sum: Arc<AtomicU32>,
        count: Arc<AtomicU32>,
    }
    impl Reader {
        fn new(sum: Arc<AtomicU32>, count: Arc<AtomicU32>) -> Self {
            Self {
                cursor: EventCursor::new(),
                sum,
                count,
            }
        }
    }
    impl System for Reader {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            ctx.lock::<(Res<Events<Damage>>,)>().execute(|(events,)| {
                for damage in events.read(&self.cursor) {
                    self.sum.fetch_add(damage.0, Ordering::SeqCst);
                    self.count.fetch_add(1, Ordering::SeqCst);
                }
            });
            Ok(())
        }
    }

    /// A reader ordered BEFORE the sender (PreUpdate) and one ordered AFTER
    /// it (PostUpdate) must both see every event exactly once.
    #[test]
    fn exactly_once_before_and_after_sender() {
        let mut world = World::new();
        world.add_event::<Damage>();

        let pre_sum = Arc::new(AtomicU32::new(0));
        let pre_count = Arc::new(AtomicU32::new(0));
        let post_sum = Arc::new(AtomicU32::new(0));
        let post_count = Arc::new(AtomicU32::new(0));

        let mut schedules = Schedules::new();
        schedules
            .get_mut::<PreUpdate>()
            .add(Reader::new(pre_sum.clone(), pre_count.clone()));
        schedules.get_mut::<Update>().add(Sender {
            frame: AtomicU32::new(0),
        });
        schedules
            .get_mut::<PostUpdate>()
            .add(Reader::new(post_sum.clone(), post_count.clone()));

        let runner = EcsRunner::single_thread();
        for _ in 0..4 {
            schedules.run_frame(&mut world, &runner, 1.0 / 60.0);
        }

        // Sender emitted Damage(1) and Damage(2): sum 3, two events.
        // The PostUpdate reader sees each in its send frame; the PreUpdate
        // reader sees each one frame later; neither sees anything twice.
        assert_eq!(post_sum.load(Ordering::SeqCst), 3);
        assert_eq!(post_count.load(Ordering::SeqCst), 2);
        assert_eq!(pre_sum.load(Ordering::SeqCst), 3);
        assert_eq!(pre_count.load(Ordering::SeqCst), 2);
    }

    /// Same setup on the multi-threaded runner.
    #[test]
    fn exactly_once_multi_thread() {
        let mut world = World::new();
        world.add_event::<Damage>();

        let sum = Arc::new(AtomicU32::new(0));
        let count = Arc::new(AtomicU32::new(0));

        let mut schedules = Schedules::new();
        schedules
            .get_mut::<PreUpdate>()
            .add(Reader::new(sum.clone(), count.clone()));
        schedules.get_mut::<Update>().add(Sender {
            frame: AtomicU32::new(0),
        });

        let runner = EcsRunner::multi_thread(2);
        for _ in 0..4 {
            schedules.run_frame(&mut world, &runner, 1.0 / 60.0);
        }

        assert_eq!(sum.load(Ordering::SeqCst), 3);
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }
}
