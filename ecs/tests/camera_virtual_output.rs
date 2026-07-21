//! Pixel-level e2e for the camera graphics stack (ADR-029, #74).
//!
//! Two cameras with `CameraOutput` specs run through the real Render
//! schedule (`EnsureCameraTargets` → `CameraRender`); the offscreen
//! camera's color output — published as a **virtual texture asset** — is
//! read back through the frame graph and must contain that camera's clear
//! color. This proves the whole chain on the GPU: spec → derived target →
//! per-camera pass → published virtual identity. (That a *material* resolves
//! `TextureSource::Virtual` to this same texture is unit-covered in
//! `texture_manager`/`EnsureCameraTargets` tests — the binding path is
//! shared with file textures.)
#![cfg(feature = "rendering")]

use std::sync::{Arc, Mutex};

use redlilium_assets::Guid;
use redlilium_ecs::rendering::loaders::TextureSource;
use redlilium_ecs::{
    Camera, CameraOutput, CameraRender, CameraTarget, EcsRunner, EnsureCameraTargets, FrameRing,
    MainViewport, PipelineCache, PipelineRegistry, Render, RenderSchedule, ScenePass, Schedules,
    SizePolicy, TextureManager, World,
};
use redlilium_graphics::{
    BufferDescriptor, BufferUsage, GraphicsInstance, RenderGraph, TransferConfig,
    TransferOperation, TransferPass,
};

/// 64×64 → 256-byte rows: readback needs no row padding.
const SIZE: u32 = 64;

#[test]
fn offscreen_camera_clear_lands_in_virtual_texture() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping pixel verification");
        return;
    }

    // World with the resources the Render systems consume.
    let mut world = World::new();
    redlilium_ecs::register_std_components(&mut world);
    redlilium_ecs::register_rendering_components(&mut world);
    world.insert_resource(TextureManager::new(device.clone()));
    world.insert_resource(RenderSchedule::empty());
    world.insert_resource(ScenePass::default());
    world.insert_resource(PipelineRegistry::default());
    world.insert_resource(PipelineCache::new(device.clone()));
    world.insert_resource(FrameRing::new(&device, 1 << 18, "test_frame_ring").expect("frame ring"));
    world.insert_resource(MainViewport::new(SIZE, SIZE));

    // Primary camera (Screen, black) + offscreen camera (red) publishing its
    // color under a virtual asset identity.
    let output_guid = Guid::stable("test/offscreen_camera_output");
    let primary = world.spawn();
    world
        .insert(primary, Camera::perspective(1.0, 1.0, 0.1, 100.0))
        .unwrap();
    world.insert(primary, CameraOutput::screen()).unwrap();

    let offscreen = world.spawn();
    world
        .insert(offscreen, Camera::perspective(1.0, 1.0, 0.1, 100.0))
        .unwrap();
    world
        .insert(
            offscreen,
            CameraOutput::offscreen(SizePolicy::Fixed(SIZE, SIZE), Some(output_guid))
                .with_clear_color([1.0, 0.0, 0.0, 1.0]),
        )
        .unwrap();

    // The real Render schedule wiring (subset of the runtime's).
    let mut schedules = Schedules::new();
    {
        let render = schedules.get_mut::<Render>();
        render.add_exclusive(EnsureCameraTargets);
        render.add(CameraRender);
        render
            .add_edge::<EnsureCameraTargets, CameraRender>()
            .expect("no cycle");
    }
    let runner = EcsRunner::single_thread();

    // Frame bracket, exactly as a host drives it.
    let mut pipeline = device.create_pipeline(1);
    let mut schedule = pipeline.begin_frame().expect("begin_frame");
    world
        .resource_mut::<RenderSchedule>()
        .set(schedule.acquire_graph());
    schedules.run_schedule::<Render>(&mut world, &runner);
    let mut graph = world
        .resource_mut::<RenderSchedule>()
        .take()
        .expect("graph back from the Render schedule");

    // Read the *virtual texture* — the asset identity consumers use — not
    // the CameraTarget directly.
    let virtual_texture = {
        let textures = world.resource::<TextureManager>();
        textures
            .get(&TextureSource::Virtual(output_guid))
            .expect("offscreen output published")
            .texture
            .clone()
    };
    let byte_size = (SIZE * SIZE * 4) as u64;
    let readback = device
        .create_buffer(&BufferDescriptor::new(
            byte_size,
            BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        ))
        .expect("readback buffer");
    let pixels = Arc::new(Mutex::new(Vec::new()));
    let mut transfer = TransferPass::new("virtual_readback".into());
    transfer.set_transfer_config(
        TransferConfig::new()
            .with_operation(TransferOperation::readback_texture_whole(
                virtual_texture,
                readback.clone(),
            ))
            .with_operation(TransferOperation::readback_buffer(
                readback.clone(),
                0..byte_size as usize,
                pixels.clone(),
            )),
    );
    graph.add_transfer_pass(transfer);

    schedule.render(graph);
    pipeline.end_frame(schedule);
    pipeline.wait_idle().expect("wait_idle");
    // Recycle the slot so the post-fence readback processing fills `pixels`.
    let mut schedule = pipeline.begin_frame().expect("begin_frame 2");
    schedule.render(RenderGraph::new());
    pipeline.end_frame(schedule);
    pipeline.wait_idle().expect("wait_idle 2");

    let pixels = pixels.lock().unwrap();
    assert_eq!(pixels.len(), byte_size as usize, "readback completed");
    // Sample a few points: all must be the offscreen camera's red clear.
    for (x, y) in [(0, 0), (SIZE / 2, SIZE / 2), (SIZE - 1, SIZE - 1)] {
        let at = ((y * SIZE + x) * 4) as usize;
        let px = &pixels[at..at + 4];
        assert_eq!(
            px,
            [255, 0, 0, 255],
            "virtual texture must hold the offscreen clear at ({x},{y})"
        );
    }

    // The primary camera's own derived target exists and is viewport-sized.
    let target = world
        .get::<CameraTarget>(primary)
        .expect("primary target derived");
    assert_eq!((target.color.width(), target.color.height()), (SIZE, SIZE));
}
