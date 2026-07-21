//! Pixel-level e2e for the standard deferred PBR/IBL pipeline (#144), the
//! **no-environment fallback** (#145).
//!
//! An offscreen camera with `RenderPath("deferred")` and no `CameraEnvironment`
//! (nor any environment manager in the world) runs through the real Render
//! schedule (`EnsureCameraTargets` → `CameraRender`). With no environment the
//! skybox pass is skipped and image-based lighting is zero (the black-cube
//! fallback); with no meshes either, every pixel resolves to the camera's
//! clear color, transformed by the display pass. The contract this guards:
//! the deferred chain (RenderPath resolution → G-buffer + scene_color
//! derivation → resolve → display) runs cleanly with no environment — a
//! uniform, finite, non-black image, no panic/NaN. The environment-lit path
//! (sky + IBL) is covered by the editor golden tests, which load a real
//! `environment` asset.
#![cfg(feature = "rendering")]

use std::sync::{Arc, Mutex};

use redlilium_assets::Guid;
use redlilium_ecs::rendering::loaders::TextureSource;
use redlilium_ecs::{
    Camera, CameraOutput, CameraRender, DEFERRED_PIPELINE, EcsRunner, EnsureCameraTargets,
    FrameRing, MainViewport, PipelineCache, PipelineRegistry, Render, RenderPath, RenderSchedule,
    ScenePass, Schedules, SizePolicy, TextureManager, World,
};
use redlilium_graphics::{
    BufferDescriptor, BufferUsage, GraphicsInstance, TransferConfig, TransferOperation,
    TransferPass,
};

/// 64×64 → 256-byte rows: readback needs no row padding.
const SIZE: u32 = 64;

#[test]
fn deferred_camera_no_environment_fallback() {
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
    world.insert_resource(FrameRing::new(&device, 1 << 16, "test_frame_ring").expect("frame ring"));
    world.insert_resource(MainViewport::new(SIZE, SIZE));

    // Offscreen camera on the deferred path, publishing its color output
    // under a virtual asset identity. Clear color = magenta; with no
    // environment and no meshes the fallback fills every pixel with it.
    let output_guid = Guid::stable("test/deferred_camera_output");
    let camera = world.spawn();
    world
        .insert(camera, Camera::perspective(1.0, 1.0, 0.1, 100.0))
        .unwrap();
    world
        .insert(camera, RenderPath::named(DEFERRED_PIPELINE))
        .unwrap();
    world
        .insert(
            camera,
            CameraOutput::offscreen(SizePolicy::Fixed(SIZE, SIZE), Some(output_guid))
                .with_clear_color([1.0, 0.0, 1.0, 1.0]),
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
    let graph = world
        .resource_mut::<RenderSchedule>()
        .take()
        .expect("graph back from the Render schedule");
    schedule.submit(graph);

    // The deferred path must have derived the G-buffer for the camera.
    {
        use redlilium_ecs::rendering::deferred::GBUFFER_ALBEDO;
        let targets = world
            .get::<redlilium_ecs::PipelineTargets>(camera)
            .expect("deferred pipeline derived PipelineTargets");
        assert!(
            targets.get(GBUFFER_ALBEDO).is_some(),
            "G-buffer albedo derived"
        );
    }

    // Read back the published virtual texture.
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
    let mut transfer = TransferPass::new("deferred_readback".into());
    transfer.set_transfer_config(
        TransferConfig::new()
            .with_operation(TransferOperation::readback_texture_whole(
                virtual_texture,
                readback.clone(),
            ))
            .with_operation(TransferOperation::readback_buffer(
                readback,
                0..byte_size as usize,
                pixels.clone(),
            )),
    );
    let mut readback_graph = schedule.acquire_graph();
    readback_graph.add_transfer_pass(transfer);
    schedule.submit(readback_graph);
    pipeline.end_frame(schedule);
    pipeline.wait_idle().expect("wait_idle");
    // Recycle the slot so the post-fence readback processing fills `pixels`.
    let mut schedule = pipeline.begin_frame().expect("drain frame");
    schedule.render(redlilium_graphics::RenderGraph::new());
    pipeline.end_frame(schedule);
    pipeline.wait_idle().expect("wait_idle 2");

    let data = pixels.lock().unwrap().clone();
    assert_eq!(data.len(), byte_size as usize, "readback size");

    // Fallback contract (#145): no environment + no meshes ⇒ every pixel is
    // the camera clear color through the display transform — a single,
    // finite, non-black color. This guards that the chain runs cleanly with
    // no environment (no panic, no NaN, no garbage), not any specific look.
    let first = [data[0], data[1], data[2], data[3]];
    let mut distinct = std::collections::HashSet::new();
    for px in data.chunks_exact(4) {
        distinct.insert([px[0], px[1], px[2], px[3]]);
    }
    assert_eq!(
        distinct.len(),
        1,
        "no-environment fallback must be uniform (no sky, no geometry), got {} colors",
        distinct.len()
    );
    assert!(
        first[0] > 0 || first[1] > 0 || first[2] > 0,
        "fallback clear color must be visible (non-black), got {first:?}"
    );
}
