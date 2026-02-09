use glam::{Mat4, Quat, Vec3};
use redlilium_core::scene::{CameraProjection, NodeTransform, Scene, SceneCamera, SceneNode};
use redlilium_ecs::{ComputePool, Schedule, StringTable, ThreadPool, World, run_system_blocking};

use ecs_std::components::*;
use ecs_std::systems::*;
use ecs_std::{register_std_components, spawn_scene};

// ---------------------------------------------------------------------------
// Full pipeline: spawn → systems → query
// ---------------------------------------------------------------------------

#[test]
fn full_frame_pipeline() {
    let mut world = World::new();
    register_std_components(&mut world);

    // Spawn a camera at (0, 5, 10) looking toward origin
    let cam_entity = world.spawn();
    world.insert(
        cam_entity,
        Transform::from_translation(Vec3::new(0.0, 5.0, 10.0)),
    );
    world.insert(cam_entity, GlobalTransform::IDENTITY);
    world.insert(
        cam_entity,
        Camera::perspective(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 1000.0),
    );

    // Spawn a few objects at different positions
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(5.0, 0.0, -5.0),
        Vec3::new(-3.0, 2.0, -10.0),
    ];
    let mut objects = Vec::new();
    for pos in &positions {
        let e = world.spawn();
        world.insert(e, Transform::from_translation(*pos));
        world.insert(e, GlobalTransform::IDENTITY);
        world.insert(e, Visibility::VISIBLE);
        objects.push(e);
    }

    // Build and run schedule
    let mut schedule = Schedule::new();
    schedule.add(UpdateGlobalTransforms);
    schedule
        .add(UpdateCameraMatrices)
        .after::<UpdateGlobalTransforms>();
    schedule.build();

    // Run the schedule
    let compute = ComputePool::new();
    schedule.run(&world, &compute);

    // Verify camera matrices were computed
    let cameras = world.read::<Camera>();
    let cam = cameras.get(cam_entity.index()).unwrap();
    assert_ne!(cam.view_matrix, Mat4::IDENTITY);
    assert_ne!(cam.projection_matrix, Mat4::IDENTITY);

    // Verify the view matrix places the camera at (0, 5, 10)
    let cam_pos = cam.view_matrix.inverse().w_axis.truncate();
    assert!((cam_pos - Vec3::new(0.0, 5.0, 10.0)).length() < 1e-4);

    // Verify object global transforms match their local transforms
    drop(cameras);
    let globals = world.read::<GlobalTransform>();
    for (i, &obj) in objects.iter().enumerate() {
        let gt = globals.get(obj.index()).unwrap();
        assert!(
            (gt.translation() - positions[i]).length() < 1e-6,
            "Object {i} global transform mismatch"
        );
    }
}

// ---------------------------------------------------------------------------
// Parallel schedule execution
// ---------------------------------------------------------------------------

#[test]
fn parallel_schedule_execution() {
    let mut world = World::new();
    register_std_components(&mut world);

    // Spawn 100 entities with transforms
    for i in 0..100 {
        let e = world.spawn();
        let angle = (i as f32) * 0.1;
        world.insert(
            e,
            Transform::new(
                Vec3::new(i as f32, 0.0, 0.0),
                Quat::from_rotation_y(angle),
                Vec3::ONE,
            ),
        );
        world.insert(e, GlobalTransform::IDENTITY);
    }

    let mut schedule = Schedule::new();
    schedule.add(UpdateGlobalTransforms);
    schedule.build();

    let pool = ThreadPool::new(4);
    let compute = ComputePool::new();
    schedule.run_parallel(&world, &pool, &compute);

    // Verify all global transforms were updated
    let transforms = world.read::<Transform>();
    let globals = world.read::<GlobalTransform>();
    for (idx, transform) in transforms.iter() {
        let global = globals.get(idx).unwrap();
        let expected = transform.to_matrix();
        assert!(
            (global.0 - expected).abs_diff_eq(Mat4::ZERO, 1e-6),
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
    let mut strings = StringTable::new();

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

    let roots = spawn_scene(&mut world, &scene, &mut strings);

    assert_eq!(roots.len(), 1);
    // root + camera_node + mesh_node = 3 entities
    assert_eq!(world.entity_count(), 3);

    // Run systems
    let mut schedule = Schedule::new();
    schedule.add(UpdateGlobalTransforms);
    schedule
        .add(UpdateCameraMatrices)
        .after::<UpdateGlobalTransforms>();
    schedule.build();
    let compute = ComputePool::new();
    schedule.run(&world, &compute);

    // Verify root entity
    let root = roots[0];
    let gt = world.get::<GlobalTransform>(root).unwrap();
    assert!((gt.translation() - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
    let name = world.get::<Name>(root).unwrap();
    assert_eq!(strings.get(name.id()), "root");

    // Find the camera entity by querying Camera component
    let cameras_storage = world.read::<Camera>();
    let mut cam_count = 0;
    for (_, cam) in cameras_storage.iter() {
        cam_count += 1;
        // Camera matrices should be computed
        assert_ne!(cam.projection_matrix, Mat4::IDENTITY);
        assert_ne!(cam.view_matrix, Mat4::IDENTITY);
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

    // Spawn 5 entities, hide every other one
    let mut entities = Vec::new();
    for i in 0..5 {
        let e = world.spawn();
        world.insert(
            e,
            Transform::from_translation(Vec3::new(i as f32, 0.0, 0.0)),
        );
        world.insert(e, GlobalTransform::IDENTITY);
        world.insert(
            e,
            if i % 2 == 0 {
                Visibility::VISIBLE
            } else {
                Visibility::HIDDEN
            },
        );
        entities.push(e);
    }

    // Run transform system via run_blocking
    let compute = ComputePool::new();
    run_system_blocking(&UpdateGlobalTransforms, &world, &compute);

    // Query visible entities (the rendering pattern)
    let globals = world.read::<GlobalTransform>();
    let visibility = world.read::<Visibility>();

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

    let entity = world.spawn();
    world.insert(entity, Transform::from_translation(Vec3::ZERO));
    world.insert(entity, GlobalTransform::IDENTITY);

    let mut schedule = Schedule::new();
    schedule.add(UpdateGlobalTransforms);
    schedule.build();

    let compute = ComputePool::new();

    // Simulate 10 frames of movement
    for frame in 0..10 {
        // "Move" the entity each frame
        {
            let mut transforms = world.write::<Transform>();
            let t = transforms.get_mut(entity.index()).unwrap();
            t.translation = Vec3::new(frame as f32, 0.0, 0.0);
        }

        schedule.run(&world, &compute);

        // Verify global transform tracks the local transform
        let globals = world.read::<GlobalTransform>();
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
    let mut strings = StringTable::new();
    register_std_components(&mut world);

    // Create a directional light pointing down (-Y rotation)
    let sun = world.spawn();
    let rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4); // 45° downward
    world.insert(sun, Transform::from_rotation(rotation));
    world.insert(sun, GlobalTransform::IDENTITY);
    world.insert(
        sun,
        DirectionalLight::new(Vec3::new(1.0, 0.98, 0.9), 100000.0),
    );
    world.insert(sun, Name::new(strings.intern("Sun")));

    // Create point lights at various positions
    let light_positions = [
        Vec3::new(5.0, 3.0, 0.0),
        Vec3::new(-5.0, 3.0, 0.0),
        Vec3::new(0.0, 3.0, 5.0),
        Vec3::new(0.0, 3.0, -5.0),
    ];
    for (i, pos) in light_positions.iter().enumerate() {
        let e = world.spawn();
        world.insert(e, Transform::from_translation(*pos));
        world.insert(e, GlobalTransform::IDENTITY);
        world.insert(e, PointLight::new(Vec3::ONE, 100.0).with_range(20.0));
        world.insert(e, Name::new(strings.intern(&format!("PointLight_{i}"))));
    }

    // Run transform system via run_blocking
    let compute = ComputePool::new();
    run_system_blocking(&UpdateGlobalTransforms, &world, &compute);

    // Query directional light direction from its global transform
    let globals = world.read::<GlobalTransform>();
    let dir_lights = world.read::<DirectionalLight>();

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
    let globals = world.read::<GlobalTransform>();
    let point_lights = world.read::<PointLight>();

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

    // This would panic without registration since no entities have these components
    let mut schedule = Schedule::new();
    schedule.add(UpdateGlobalTransforms);
    schedule
        .add(UpdateCameraMatrices)
        .after::<UpdateGlobalTransforms>();
    schedule.build();

    // Should not panic even with zero entities
    let compute = ComputePool::new();
    schedule.run(&world, &compute);
}
