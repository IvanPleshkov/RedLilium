//! Pixel-level e2e for the standard deferred PBR/IBL pipeline (#144).
//!
//! An offscreen camera with `RenderPath("deferred")` runs through the real
//! Render schedule (`EnsureCameraTargets` → `CameraRender`). Even with no
//! meshes in the world the deferred path must produce a picture: the G-buffer
//! pass clears, the skybox pass renders the baked sky cubemap (its IBL upload
//! rides the same first frame's graph), and the resolve pass composites — so
//! the readback must contain sky pixels, not the camera's clear color. This
//! proves the whole chain: RenderPath resolution → G-buffer derivation into
//! PipelineTargets → IBL upload through the frame graph → skybox/resolve
//! materials specialized for the target format.
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
fn deferred_camera_renders_skybox_background() {
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
    // under a virtual asset identity. Clear color = magenta, which the sky
    // must overwrite everywhere.
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

    // The sky must overwrite the magenta clear everywhere: no pixel may stay
    // at the clear color, and the image must not be uniform garbage — the
    // sky has gradients, so expect some pixel diversity.
    let mut clear_pixels = 0usize;
    let mut distinct = std::collections::HashSet::new();
    for px in data.chunks_exact(4) {
        distinct.insert([px[0], px[1], px[2]]);
        // Magenta clear in any plausible encoding: strong red+blue, no green.
        if px[0] > 200 && px[1] < 30 && px[2] > 200 {
            clear_pixels += 1;
        }
    }
    assert_eq!(
        clear_pixels, 0,
        "no pixel may remain at the camera clear color (skybox must cover)"
    );
    assert!(
        distinct.len() >= 8,
        "sky background should have gradients, got {} distinct colors",
        distinct.len()
    );
}
