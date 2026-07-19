//! Golden-image tests for the standard deferred PBR/IBL path (#126).
//!
//! One deterministic scene (ground plane, dielectric cube, metal sphere,
//! textured checker sphere — the std `pbr` material instances over generated
//! meshes) renders through the full production stack: VFS mount → asset DB →
//! managers → deferred pipeline → readback. The resulting image is checked
//! two ways:
//!
//! 1. **Golden files** (`tests/golden/*.png`) — pixel drift over time fails
//!    the test. Regenerate deliberately with
//!    `REDLILIUM_GOLDEN_UPDATE=1 cargo test -p redlilium-editor golden`.
//! 2. **Cross-format agreement** — the camera renders the same scene into
//!    all three [`OutputFormat`]s (sRGB-typed, plain unorm, linear HDR), and
//!    their display-encoded images must agree. Each format takes a different
//!    shader output-transform path (`SRGB_FRAMEBUFFER` / manual encode /
//!    `HDR_OUTPUT`), so a double gamma, a missing tonemap, or an encode
//!    mismatch — the color-bug classes that shipped before — shows up as a
//!    disagreement *without* consulting any golden file.
//!
//! The std mount is read-only here: the committed `assets.db` is loaded, but
//! never scanned or persisted — a test must not dirty the asset databases.

#![cfg(test)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use redlilium_core::color::{f16_to_f32, srgb_encode, tonemap_pbr_neutral};
use redlilium_core::math::{Vec3, quat_looking_along};
use redlilium_ecs::rendering::loaders::TextureSource;
use redlilium_ecs::{
    CameraOutput, DirectionalLight, EcsRunner, GlobalTransform, MaterialInstanceSource,
    MeshGenerator, MeshRenderer, MeshSource, OutputFormat, PointLight, Primitive, Render,
    RenderSchedule, SizePolicy, TextureManager, Transform, Visibility,
};
use redlilium_graphics::{
    BufferDescriptor, BufferUsage, FramePipeline, GraphicsInstance, TextureFormat, TransferConfig,
    TransferOperation, TransferPass,
};
use redlilium_vfs::{FileSystemProvider, Vfs};

use crate::core::{EditorWorld, EditorWorldParams, create_editor_world_empty};
use crate::scene_view::SceneViewState;

/// Square target; 256×4 = 1024-byte rows (and ×8 for f16) are already
/// 256-aligned, so the readback needs no row un-padding.
const SIZE: u32 = 256;
const FIXED_DT: f64 = 1.0 / 60.0;
const GOLDEN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden");
const STD_ASSETS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../std-assets");

/// Max ignored per-channel difference (8-bit) against a golden.
const GOLDEN_TOLERANCE: u8 = 3;
/// Fraction of pixels allowed above [`GOLDEN_TOLERANCE`] (driver noise on
/// hard geometry edges), and the hard per-channel cap for those.
const GOLDEN_OUTLIERS: f64 = 0.01;
const GOLDEN_MAX_DIFF: u8 = 24;

/// Cross-format thresholds are looser: the HDR image quantizes radiance to
/// f16 before the CPU applies the tonemap the GPU applies in f32.
const CROSS_TOLERANCE: u8 = 6;
const CROSS_OUTLIERS: f64 = 0.02;

#[test]
fn deferred_golden_images_across_output_formats() {
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping golden test");
        return;
    }

    // Read-only std mount (absolute path — the unit-test cwd is `editor/`).
    let mut vfs = Vfs::new();
    vfs.mount("std", FileSystemProvider::new(STD_ASSETS_DIR));
    let engine = redlilium_runtime::EngineContext::with_vfs(device.clone(), vfs);
    engine.load_mount_db("std", STD_ASSETS_DIR);

    let mut scene_view = SceneViewState::new(device.clone(), TextureFormat::Bgra8UnormSrgb);
    let mut ew = create_editor_world_empty(
        &EditorWorldParams {
            remote: false,
            egui: false,
        },
        &engine,
        &mut scene_view,
        1.0,
    );
    spawn_golden_scene(&mut ew.world);

    let output_guid = redlilium_assets::Guid::stable("test/golden_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Srgb, output_guid);

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    // Pump frames until the asset pipeline is quiet (meshes, materials,
    // shaders, textures all resident), then a few calm frames on top.
    let mut calm = 0u32;
    for tick_no in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            log::info!("golden: assets idle after {tick_no} ticks");
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle in 600 ticks");

    // Render + read back the same scene in every output format. Each
    // format change re-derives the camera target and respecializes the
    // fullscreen materials; give it a few frames to settle.
    let mut images = Vec::new();
    for format in [
        OutputFormat::Srgb,
        OutputFormat::Standard,
        OutputFormat::Hdr,
    ] {
        set_output(&mut ew.world, camera, format, output_guid);
        for _ in 0..3 {
            tick(&mut ew, &mut pipeline, &runner);
        }
        let raw = read_back(&ew, &device, &mut pipeline, output_guid, format);
        images.push((format, to_display_rgba8(&raw, format)));
    }
    drop(pipeline);

    for (format, image) in &images {
        compare_or_update(&format!("deferred_{}.png", format_slug(*format)), image);
    }

    // Cross-format agreement — see the module docs. Runs after the golden
    // comparison so a genuine drift reports against the goldens first.
    let srgb = &images[0].1;
    for (format, image) in &images[1..] {
        assert_images_close(
            srgb,
            image,
            CROSS_TOLERANCE,
            CROSS_OUTLIERS,
            GOLDEN_MAX_DIFF,
            &format!("Srgb vs {format:?} output disagree"),
        );
    }
}

/// The fixed scene: everything the color path exercises, nothing that moves.
fn spawn_golden_scene(world: &mut redlilium_ecs::World) {
    let pbr = MaterialInstanceSource {
        guid: redlilium_assets::Guid::stable("materials/pbr.matinst"),
    };
    let metal = MaterialInstanceSource {
        guid: redlilium_assets::Guid::stable("materials/pbr_metal.matinst"),
    };
    let checker = MaterialInstanceSource {
        guid: redlilium_assets::Guid::stable("materials/pbr_checker.matinst"),
    };

    let mut spawn = |mesh: MeshSource, material: MaterialInstanceSource, transform: Transform| {
        let entity = world.spawn();
        world.insert(entity, transform).unwrap();
        world
            .insert(entity, GlobalTransform(transform.to_matrix()))
            .unwrap();
        world.insert(entity, Visibility::VISIBLE).unwrap();
        world
            .insert(entity, MeshRenderer::single(Primitive::new(mesh, material)))
            .unwrap();
    };

    // Ground plane (scaled cube), dielectric.
    spawn(
        MeshSource::Generated(MeshGenerator::cube(0.5)),
        pbr.clone(),
        Transform::new(
            Vec3::new(0.0, -0.05, 0.0),
            redlilium_core::math::Quat::identity(),
            Vec3::new(10.0, 0.1, 10.0),
        ),
    );
    // Dielectric cube front and center.
    spawn(
        MeshSource::Generated(MeshGenerator::cube(0.5)),
        pbr,
        Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
    );
    // Metal sphere — specular IBL (the prefilter mip convention) at a glance.
    spawn(
        MeshSource::Generated(MeshGenerator::sphere(0.7, 32, 16)),
        metal,
        Transform::from_translation(Vec3::new(2.0, 0.7, -0.5)),
    );
    // Textured checker sphere — sRGB base color decode + ORM packing + AO.
    spawn(
        MeshSource::Generated(MeshGenerator::sphere(0.6, 32, 16)),
        checker,
        Transform::from_translation(Vec3::new(-1.6, 0.6, 0.8)),
    );

    // Key sun (#146: lights are ECS components) — same direction/energy as
    // the retired built-in constant, so the lit look carries over.
    let sun = world.spawn();
    let sun_transform =
        Transform::from_rotation(quat_looking_along(Vec3::new(-0.75, -0.40, -0.75)));
    world.insert(sun, sun_transform).unwrap();
    world
        .insert(sun, GlobalTransform(sun_transform.to_matrix()))
        .unwrap();
    world.insert(sun, Visibility::VISIBLE).unwrap();
    world
        .insert(sun, DirectionalLight::new(Vec3::new(1.0, 0.98, 0.95), 1.6))
        .unwrap();

    // Warm point light between cube and checker sphere — exercises the
    // inverse-square + range-window path.
    let point = world.spawn();
    let point_transform = Transform::from_translation(Vec3::new(-0.8, 1.6, 1.4));
    world.insert(point, point_transform).unwrap();
    world
        .insert(point, GlobalTransform(point_transform.to_matrix()))
        .unwrap();
    world.insert(point, Visibility::VISIBLE).unwrap();
    world
        .insert(
            point,
            PointLight::new(Vec3::new(1.0, 0.55, 0.25), 3.0).with_range(8.0),
        )
        .unwrap();
}

fn set_output(
    world: &mut redlilium_ecs::World,
    camera: redlilium_ecs::Entity,
    format: OutputFormat,
    guid: redlilium_assets::Guid,
) {
    world
        .insert(
            camera,
            CameraOutput::offscreen(SizePolicy::Fixed(SIZE, SIZE), Some(guid))
                .with_clear_color([1.0, 0.0, 1.0, 1.0])
                .with_format(format),
        )
        .unwrap();
}

/// One frame, the way the headless shell ticks: CPU schedules, then the
/// Render schedule bracketed by the frame pipeline, asset-upload transfer
/// graphs submitted first.
fn tick(ew: &mut EditorWorld, pipeline: &mut FramePipeline, runner: &EcsRunner) {
    ew.debug_drawer.read().advance_tick();
    ew.schedules.run_frame(&mut ew.world, runner, FIXED_DT);

    let mut schedule = pipeline.begin_frame().expect("begin_frame");
    let graph = schedule.acquire_graph();
    ew.world.resource_mut::<RenderSchedule>().set(graph);
    ew.schedules.run_schedule::<Render>(&mut ew.world, runner);
    let mut schedule_res = ew.world.resource_mut::<RenderSchedule>();
    let transfer_graphs = schedule_res.take_transfer_graphs();
    let graph = schedule_res
        .take()
        .expect("graph back from the Render schedule");
    drop(schedule_res);
    for transfer in transfer_graphs {
        schedule.submit(transfer);
    }
    schedule.render(graph);
    pipeline.end_frame(schedule);
    ew.window_input.write().begin_frame();
}

/// Read the published camera output back to the CPU (raw texture bytes).
fn read_back(
    ew: &EditorWorld,
    device: &Arc<redlilium_graphics::GraphicsDevice>,
    pipeline: &mut FramePipeline,
    guid: redlilium_assets::Guid,
    format: OutputFormat,
) -> Vec<u8> {
    let texture = ew
        .world
        .resource::<TextureManager>()
        .get(&TextureSource::Virtual(guid))
        .expect("camera output published")
        .texture
        .clone();
    assert_eq!(texture.format(), format.color_format(), "target format");

    let bytes_px: u64 = match format {
        OutputFormat::Hdr => 8,
        _ => 4,
    };
    let byte_size = u64::from(SIZE) * u64::from(SIZE) * bytes_px;
    let buffer = device
        .create_buffer(&BufferDescriptor::new(
            byte_size,
            BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        ))
        .expect("readback buffer");
    let result = Arc::new(Mutex::new(Vec::new()));
    let mut transfer = TransferPass::new("golden_readback".into());
    transfer.set_transfer_config(
        TransferConfig::new()
            .with_operation(TransferOperation::readback_texture_whole(
                texture,
                buffer.clone(),
            ))
            .with_operation(TransferOperation::readback_buffer(
                buffer,
                0..byte_size as usize,
                result.clone(),
            )),
    );
    let mut schedule = pipeline.begin_frame().expect("readback frame");
    let mut graph = schedule.acquire_graph();
    graph.add_transfer_pass(transfer);
    schedule.render(graph);
    pipeline.end_frame(schedule);
    pipeline.wait_idle().expect("wait_idle");
    // Recycle every slot so the post-fence readback processing fills
    // `result` (the pipeline has two frames in flight).
    for _ in 0..2 {
        let mut schedule = pipeline.begin_frame().expect("drain frame");
        schedule.render(redlilium_graphics::RenderGraph::new());
        pipeline.end_frame(schedule);
        pipeline.wait_idle().expect("drain wait_idle");
    }

    let data = result.lock().unwrap().clone();
    assert_eq!(data.len(), byte_size as usize, "readback size");
    data
}

/// Convert raw target bytes to a display-encoded RGBA8 image — the space
/// PNGs live in and all three formats can be compared in.
fn to_display_rgba8(raw: &[u8], format: OutputFormat) -> image::RgbaImage {
    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    match format {
        // sRGB-typed target: bytes are already display-encoded, BGRA order.
        OutputFormat::Srgb => {
            for px in raw.chunks_exact(4) {
                pixels.extend_from_slice(&[px[2], px[1], px[0], 255]);
            }
        }
        // Plain unorm target: the shader encoded manually; bytes are
        // display-encoded RGBA.
        OutputFormat::Standard => {
            for px in raw.chunks_exact(4) {
                pixels.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        // Linear-HDR target: apply the same SDR transform the shaders apply
        // on SDR targets (tonemap + sRGB encode).
        OutputFormat::Hdr => {
            for texel in raw.chunks_exact(8) {
                let rgb = [
                    f16_to_f32(u16::from_le_bytes([texel[0], texel[1]])),
                    f16_to_f32(u16::from_le_bytes([texel[2], texel[3]])),
                    f16_to_f32(u16::from_le_bytes([texel[4], texel[5]])),
                ];
                for c in tonemap_pbr_neutral(rgb) {
                    pixels.push((srgb_encode(c).clamp(0.0, 1.0) * 255.0).round() as u8);
                }
                pixels.push(255);
            }
        }
    }
    image::RgbaImage::from_raw(SIZE, SIZE, pixels).expect("image size")
}

fn format_slug(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Srgb => "srgb",
        OutputFormat::Standard => "standard",
        OutputFormat::Hdr => "hdr",
    }
}

/// Compare against the committed golden, or rewrite it when
/// `REDLILIUM_GOLDEN_UPDATE=1`.
fn compare_or_update(name: &str, image: &image::RgbaImage) {
    let path = Path::new(GOLDEN_DIR).join(name);
    if std::env::var("REDLILIUM_GOLDEN_UPDATE").is_ok() {
        std::fs::create_dir_all(GOLDEN_DIR).expect("golden dir");
        image.save(&path).expect("write golden");
        eprintln!("golden updated: {}", path.display());
        return;
    }
    let golden = image::open(&path)
        .unwrap_or_else(|e| {
            panic!(
                "golden '{name}' unreadable ({e}) — generate with \
                 REDLILIUM_GOLDEN_UPDATE=1 cargo test -p redlilium-editor golden"
            )
        })
        .to_rgba8();
    assert_eq!(golden.dimensions(), image.dimensions(), "{name} dimensions");
    assert_images_close(
        &golden,
        image,
        GOLDEN_TOLERANCE,
        GOLDEN_OUTLIERS,
        GOLDEN_MAX_DIFF,
        &format!("image drifted from golden '{name}'"),
    );
}

/// Tolerance-based comparison: per-channel differences up to `tolerance`
/// are noise; pixels above it must stay under the `outliers` fraction (hard
/// geometry edges) and never exceed `max_diff`.
fn assert_images_close(
    a: &image::RgbaImage,
    b: &image::RgbaImage,
    tolerance: u8,
    outliers: f64,
    max_diff: u8,
    what: &str,
) {
    let mut over = 0usize;
    let mut worst = 0u8;
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let diff = (0..3).map(|c| pa.0[c].abs_diff(pb.0[c])).max().unwrap_or(0);
        worst = worst.max(diff);
        if diff > tolerance {
            over += 1;
        }
    }
    let total = (a.width() * a.height()) as f64;
    let over_frac = over as f64 / total;
    assert!(
        over_frac <= outliers && worst <= max_diff,
        "{what}: {over} px ({:.2}%) differ by more than {tolerance} \
         (allowed {:.2}%), worst channel diff {worst} (cap {max_diff})",
        over_frac * 100.0,
        outliers * 100.0,
    );
}
