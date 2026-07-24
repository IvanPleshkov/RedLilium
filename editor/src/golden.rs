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

use redlilium_core::color::{f16_to_f32, srgb_encode};
use redlilium_core::math::{Vec3, quat_looking_along};
use redlilium_ecs::rendering::loaders::TextureSource;
use redlilium_ecs::{
    CameraAmbientOcclusion, CameraAutoExposure, CameraBloom, CameraEnvironment, CameraExposure,
    CameraOutput, DirectionalLight, DisplayHeadroom, EcsRunner, GlobalTransform,
    MaterialInstanceSource, MeshGenerator, MeshRenderer, MeshSource, OutputFormat, PointLight,
    Primitive, Render, RenderSchedule, SizePolicy, TextureManager, Transform, Visibility,
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
    let _ = env_logger::builder().is_test(true).try_init();
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
    // The IBL environment is now a per-camera asset (#145); the std default
    // is the same baked sunrise set the pipeline used to embed.
    ew.world
        .insert(camera, CameraEnvironment::default())
        .unwrap();

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    // Pump frames until the asset pipeline is quiet (meshes, materials,
    // shaders, textures, the environment cubemaps all resident), then a few
    // calm frames on top.
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

/// End-to-end proof of the display-headroom plumbing (#154): the same
/// over-exposed scene, rendered to a linear-HDR target, must pin its peak at
/// paper-white (1.0) when `DisplayHeadroom` is 1 and roll that same highlight
/// *up* into the extended range when it is 4 — while sub-knee mids stay put.
/// Guards that `exposure.z` reaches the shader and actually lifts the ceiling,
/// independent of any golden image.
#[test]
fn deferred_hdr_headroom_extends_highlights() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping headroom test");
        return;
    }

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

    let output_guid = redlilium_assets::Guid::stable("test/headroom_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Hdr, output_guid);
    ew.world
        .insert(camera, CameraEnvironment::default())
        .unwrap();
    // Over-expose so highlights sit well above the compression knee regardless
    // of scene content — makes the roll-off differential unambiguous.
    ew.world.insert(camera, CameraExposure::new(8.0)).unwrap();

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    let mut calm = 0u32;
    for _ in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle");

    // H=1: the shoulder pins the peak at/below paper-white (only f16 slop over).
    ew.world.insert_resource(DisplayHeadroom(1.0));
    for _ in 0..3 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let peak_sdr = hdr_peak_channel(&read_back(
        &ew,
        &device,
        &mut pipeline,
        output_guid,
        OutputFormat::Hdr,
    ));

    // H=4: the same highlight rolls up into the extended range, bounded by H.
    ew.world.insert_resource(DisplayHeadroom(4.0));
    for _ in 0..3 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let peak_hdr = hdr_peak_channel(&read_back(
        &ew,
        &device,
        &mut pipeline,
        output_guid,
        OutputFormat::Hdr,
    ));
    drop(pipeline);

    assert!(
        peak_sdr <= 1.05,
        "H=1 pins the peak to paper-white, got {peak_sdr}"
    );
    assert!(
        peak_hdr > peak_sdr + 0.5,
        "H=4 extends highlights above paper-white ({peak_sdr} -> {peak_hdr})"
    );
    assert!(
        peak_hdr <= 4.0 + 1e-2,
        "roll-off stays within the headroom, got {peak_hdr}"
    );
}

/// Largest finite linear channel (max over RGB) in a raw `Rgba16Float`
/// readback — the peak the display-headroom roll-off produced.
fn hdr_peak_channel(raw: &[u8]) -> f32 {
    let mut peak = 0.0f32;
    for texel in raw.chunks_exact(8) {
        for i in 0..3 {
            let c = f16_to_f32(u16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]));
            if c.is_finite() && c > peak {
                peak = c;
            }
        }
    }
    peak
}

/// The temporal contract's observable (#147): the G-buffer velocity target
/// is exactly zero for a static scene, matches the projected NDC delta on
/// the frame an entity moves, and returns to zero the frame after motion
/// stops (prev-model history caught up).
#[test]
fn deferred_velocity_buffer_tracks_motion() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping velocity test");
        return;
    }

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

    // One cube front and center — the golden scene's hero position.
    let start = Vec3::new(0.0, 0.5, 0.0);
    let moved = Vec3::new(0.25, 0.5, 0.0);
    let cube = ew.world.spawn();
    let transform = Transform::from_translation(start);
    ew.world.insert(cube, transform).unwrap();
    ew.world
        .insert(cube, GlobalTransform(transform.to_matrix()))
        .unwrap();
    ew.world.insert(cube, Visibility::VISIBLE).unwrap();
    ew.world
        .insert(
            cube,
            MeshRenderer::single(Primitive::new(
                MeshSource::Generated(MeshGenerator::cube(0.5)),
                MaterialInstanceSource {
                    guid: redlilium_assets::Guid::stable("materials/pbr.matinst"),
                },
            )),
        )
        .unwrap();

    let output_guid = redlilium_assets::Guid::stable("test/velocity_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Srgb, output_guid);

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    let mut calm = 0u32;
    for _ in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle");

    let velocity_texture = ew
        .world
        .get::<redlilium_ecs::PipelineTargets>(camera)
        .expect("deferred targets derived")
        .get(redlilium_ecs::rendering::deferred::GBUFFER_VELOCITY)
        .expect("velocity target derived")
        .clone();

    // Static scene: history equals current everywhere — exact zeros.
    let static_velocity = decode_velocity(&read_back_texture(
        &device,
        &mut pipeline,
        velocity_texture.clone(),
        4,
    ));
    let static_max = max_magnitude(&static_velocity);
    assert!(
        static_max < 1e-4,
        "static scene must be still: {static_max}"
    );

    // Move the cube one frame's worth: the velocity texels on it must match
    // the NDC delta of the projected cube center.
    let expected = {
        let vp = ew
            .world
            .get::<redlilium_ecs::Camera>(camera)
            .expect("camera")
            .view_projection();
        let project = |p: Vec3| {
            let clip = vp * redlilium_core::math::Vec4::new(p.x, p.y, p.z, 1.0);
            [clip.x / clip.w, clip.y / clip.w]
        };
        let curr = project(moved);
        let prev = project(start);
        [curr[0] - prev[0], curr[1] - prev[1]]
    };
    ew.world
        .insert(cube, Transform::from_translation(moved))
        .unwrap();
    tick(&mut ew, &mut pipeline, &runner);
    let moving_velocity = decode_velocity(&read_back_texture(
        &device,
        &mut pipeline,
        velocity_texture.clone(),
        4,
    ));
    let hot: Vec<[f32; 2]> = moving_velocity
        .iter()
        .copied()
        .filter(|v| (v[0] * v[0] + v[1] * v[1]).sqrt() > 1e-4)
        .collect();
    assert!(hot.len() > 50, "moving cube covers texels: {}", hot.len());
    let mean = [
        hot.iter().map(|v| v[0]).sum::<f32>() / hot.len() as f32,
        hot.iter().map(|v| v[1]).sum::<f32>() / hot.len() as f32,
    ];
    let dot = mean[0] * expected[0] + mean[1] * expected[1];
    let norm = |v: [f32; 2]| (v[0] * v[0] + v[1] * v[1]).sqrt();
    let cosine = dot / (norm(mean) * norm(expected)).max(1e-12);
    assert!(
        cosine > 0.95,
        "velocity direction disagrees: mean {mean:?}, expected {expected:?}"
    );
    let ratio = norm(mean) / norm(expected).max(1e-12);
    assert!(
        (0.5..2.0).contains(&ratio),
        "velocity magnitude off: mean {mean:?}, expected {expected:?} (ratio {ratio})"
    );

    // Motion stopped: history caught up next frame — zeros again.
    tick(&mut ew, &mut pipeline, &runner);
    let settled = decode_velocity(&read_back_texture(
        &device,
        &mut pipeline,
        velocity_texture.clone(),
        4,
    ));
    let settled_max = max_magnitude(&settled);
    assert!(settled_max < 1e-4, "history must catch up: {settled_max}");

    // The contract's central trap: with jitter enabled, a static scene must
    // STILL read zero velocity — velocity math uses the unjittered pair, so
    // sub-pixel sampling never masquerades as motion.
    ew.world
        .insert(camera, redlilium_ecs::TemporalJitter::default())
        .unwrap();
    for _ in 0..2 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let jittered = decode_velocity(&read_back_texture(
        &device,
        &mut pipeline,
        velocity_texture,
        4,
    ));
    let jittered_max = max_magnitude(&jittered);
    assert!(
        jittered_max < 1e-4,
        "jitter leaked into velocity: {jittered_max}"
    );
}

/// TAA (#148) on a static jittered scene must converge to an image close to
/// the non-TAA golden (same scene, edges anti-aliased) and then hold it:
/// two reads a couple frames apart must be nearly identical — a broken
/// history (bad reprojection, leaking jitter, no clipping) shows up as
/// flicker or drift long before it is visible in a single image.
#[test]
fn deferred_taa_accumulates_stably() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping TAA test");
        return;
    }

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

    let output_guid = redlilium_assets::Guid::stable("test/taa_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Srgb, output_guid);
    ew.world
        .insert(camera, redlilium_ecs::TemporalJitter::default())
        .unwrap();
    ew.world
        .insert(camera, CameraEnvironment::default())
        .unwrap();

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    let mut calm = 0u32;
    for _ in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle");

    // Converge: blend 0.1 remembers ~10 frames; 24 is comfortably settled.
    for _ in 0..24 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let converged = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );

    // Convergence sanity: close to the non-TAA golden. Edges legitimately
    // differ (anti-aliasing is the point), so the thresholds are loose —
    // this guards against explosions, black frames, and runaway feedback,
    // not pixel identity.
    if std::env::var("REDLILIUM_TAA_DUMP").is_ok() {
        let dump = std::env::temp_dir().join("redlilium_taa_dump.png");
        converged.save(&dump).expect("dump");
        eprintln!("TAA debug dump: {}", dump.display());
    }
    let golden = image::open(Path::new(GOLDEN_DIR).join("deferred_srgb.png"))
        .expect("deferred_srgb golden exists")
        .to_rgba8();
    assert_images_close(
        &golden,
        &converged,
        8,
        0.10,
        200,
        "TAA-converged image far from the reference",
    );

    // Stability: with the scene static, two frames at different jitter
    // phases must agree — the accumulated history dominates. High-contrast
    // edge texels legitimately re-clip between jitter phases (variance
    // clipping at gamma 1), so the peak cap is looser than the goldens' —
    // it guards against gross flicker and feedback, not edge shimmer.
    for _ in 0..2 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let later = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );
    assert_images_close(
        &converged,
        &later,
        3,
        0.02,
        48,
        "converged TAA output flickers between frames",
    );
}

/// Disocclusion no-ghost (#148 depth-reject + variance clip): a camera truck
/// uncovers sky from behind the foreground objects, and their history must not
/// smear onto the freshly revealed sky. The frame is captured mid-truck at the
/// final pose and again once fully settled at that *same* pose. The comparison
/// is restricted to the sky — zero screen velocity under a pure translation, so
/// read straight off the velocity buffer — where static sky is identical
/// between the two captures; any sky pixel that disagrees is a temporal ghost
/// trailing a foreground silhouette. Masking to the sky isolates the ghost from
/// the legitimate motion-softening on geometry edges, so no calibrated
/// whole-image threshold is needed. A swapped/degenerate reprojection or a
/// broken history rejection lights up a whole trailing band and fails here.
#[test]
fn deferred_taa_disocclusion_leaves_no_ghost() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping disocclusion test");
        return;
    }

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

    let output_guid = redlilium_assets::Guid::stable("test/taa_ghost_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Srgb, output_guid);
    ew.world
        .insert(camera, redlilium_ecs::TemporalJitter::default())
        .unwrap();
    ew.world
        .insert(camera, CameraEnvironment::default())
        .unwrap();
    // Drive the camera transform directly for a deterministic truck; the
    // free-fly integrator would otherwise overwrite it every tick.
    let _ = ew.world.remove::<redlilium_ecs::FreeFlyCamera>(camera);

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    // Base framing = the default orbit pose; truck sideways along its own right
    // axis so the foreground objects parallax against the sky.
    let base = *ew.world.get::<Transform>(camera).expect("camera transform");
    // Right axis = column 0 of the pose matrix (unit, camera scale is 1).
    let bm = base.to_matrix();
    let right = Vec3::new(bm[(0, 0)], bm[(1, 0)], bm[(2, 0)]);
    // ~3 px/frame at this framing: enough to uncover a sky strip, slow enough
    // that the adaptive blend stays on the long memory, where a ghost would
    // persist for ~10 frames instead of being washed out by fast convergence.
    const DX: f32 = 0.03;
    const PAN_FRAMES: i32 = 8;
    let pose = |i: i32| {
        Transform::new(
            base.translation + right * (DX * i as f32),
            base.rotation,
            base.scale,
        )
    };

    // Settle at the base pose (also lets the async assets finish loading).
    ew.world.insert(camera, pose(0)).unwrap();
    let mut calm = 0u32;
    for _ in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle");
    for _ in 0..24 {
        tick(&mut ew, &mut pipeline, &runner);
    }

    // Truck to the final pose; the last frame is captured mid-motion.
    for i in 1..=PAN_FRAMES {
        ew.world.insert(camera, pose(i)).unwrap();
        tick(&mut ew, &mut pipeline, &runner);
    }
    let panned = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );
    // Sky mask from the same frame: zero screen velocity under a pure truck.
    let velocity_texture = ew
        .world
        .get::<redlilium_ecs::PipelineTargets>(camera)
        .expect("deferred targets")
        .get(redlilium_ecs::rendering::deferred::GBUFFER_VELOCITY)
        .expect("velocity target")
        .clone();
    let velocity = decode_velocity(&read_back_texture(
        &device,
        &mut pipeline,
        velocity_texture,
        4,
    ));

    // Hold the final pose until the history fully settles: any transient ghost
    // has decayed, so this is the ground-truth sky at that pose.
    for _ in 0..30 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let settled = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );

    // Compare over *deep* sky only. A pixel is sky when its screen velocity is
    // ~zero (far plane under a pure truck); deep sky additionally requires every
    // neighbour in a small radius to be sky, which erodes the object↔sky
    // silhouette boundary out of the mask. That boundary is where the legitimate
    // TAA edge-convergence (mid-motion vs settled) lives — excluding it leaves a
    // clean baseline, while a disocclusion ghost smears several pixels *into*
    // open sky and so still lands inside the eroded mask.
    const SKY_EPS: f32 = 1e-3; // velocity below this ⇒ sky (far plane at zfar)
    const ERODE: i32 = 2; // radius of geometry-free neighbourhood required
    const GHOST_DIFF: u8 = 20; // per-channel display delta that counts as a ghost
    let is_sky = |x: i32, y: i32| -> bool {
        if x < 0 || y < 0 || x >= SIZE as i32 || y >= SIZE as i32 {
            return false;
        }
        let v = velocity[(y as u32 * SIZE + x as u32) as usize];
        (v[0] * v[0] + v[1] * v[1]).sqrt() < SKY_EPS
    };
    let deep_sky = |x: i32, y: i32| -> bool {
        for dy in -ERODE..=ERODE {
            for dx in -ERODE..=ERODE {
                if !is_sky(x + dx, y + dy) {
                    return false;
                }
            }
        }
        true
    };
    let mut sky = 0usize;
    let mut ghosted = 0usize;
    let mut worst = 0u8;
    for y in 0..SIZE as i32 {
        for x in 0..SIZE as i32 {
            if !deep_sky(x, y) {
                continue;
            }
            sky += 1;
            let p = panned.get_pixel(x as u32, y as u32).0;
            let s = settled.get_pixel(x as u32, y as u32).0;
            let d = (0..3).map(|c| p[c].abs_diff(s[c])).max().unwrap();
            worst = worst.max(d);
            if d > GHOST_DIFF {
                ghosted += 1;
            }
        }
    }
    assert!(
        sky > 5000,
        "truck must expose enough deep sky to judge: {sky}"
    );
    // Validated on-device (M3/MoltenVK): with the current shader the eroded
    // deep sky is spotless — 0 pixels differ by >GHOST_DIFF, worst delta 3.
    // Bypassing the history rejection (variance clip + depth reject) lights up
    // ~86 deep-sky pixels (worst ~89) — a whole trailing band. The cap sits well
    // clear of the clean baseline and well under the broken signal.
    assert!(
        ghosted < 30,
        "TAA ghost trail on revealed sky: {ghosted} deep-sky px differ >{GHOST_DIFF} (worst {worst}, of {sky})"
    );
}

/// SSAO (#150) must only *darken* the image-based ambient, and must visibly
/// darken contacts. Renders the fixed scene without AO, then with a strong
/// `CameraAmbientOcclusion`, and asserts three properties that pin the effect
/// without depending on its exact tuning: the balance darkens, essentially no
/// pixel brightens (AO can only occlude), and a meaningful fraction darkens
/// (creases/contacts). A golden image guards the look on top.
#[test]
fn deferred_ssao_darkens_ambient() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping SSAO test");
        return;
    }

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

    let output_guid = redlilium_assets::Guid::stable("test/ssao_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Srgb, output_guid);
    ew.world
        .insert(camera, CameraEnvironment::default())
        .unwrap();

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    let mut calm = 0u32;
    for _ in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle");

    // Baseline: no SSAO.
    for _ in 0..3 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let no_ao = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );

    // Enable a strong SSAO (bigger radius / more intensity than the default,
    // so the differential is unmistakable and decoupled from tuning). A few
    // frames to derive the targets, rebuild the materials, and render.
    ew.world
        .insert(camera, CameraAmbientOcclusion::new(0.6, 1.5, 2.0))
        .unwrap();
    for _ in 0..5 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let with_ao = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );
    drop(pipeline);

    // Per-pixel luma delta (with_ao − no_ao). SSAO scales the ambient by a
    // factor ≤ 1, and tonemap+encode are monotonic, so a correct pass never
    // brightens a pixel.
    let total = (SIZE * SIZE) as f64;
    let mut brighter = 0usize;
    let mut darker = 0usize;
    let mut sum_delta = 0i64;
    for (a, b) in no_ao.pixels().zip(with_ao.pixels()) {
        let la = a.0[0] as i32 + a.0[1] as i32 + a.0[2] as i32;
        let lb = b.0[0] as i32 + b.0[1] as i32 + b.0[2] as i32;
        let d = lb - la;
        sum_delta += d as i64;
        if d > 6 {
            brighter += 1;
        }
        if d < -6 {
            darker += 1;
        }
    }
    assert!(
        sum_delta < 0,
        "SSAO did not darken the image on balance (sum {sum_delta})"
    );
    assert!(
        (brighter as f64 / total) < 0.01,
        "SSAO brightened {brighter} px — it must only occlude"
    );
    assert!(
        (darker as f64 / total) > 0.01,
        "SSAO darkened too few pixels ({darker}) — contacts not occluded"
    );

    // Golden regression on the SSAO look.
    compare_or_update("deferred_ssao.png", &with_ao);
}

/// Bloom (#151) must spread bright highlights into a visible halo without
/// blowing the image out or punching black holes in it. Renders the fixed
/// scene without bloom, then with a strong `CameraBloom`, and asserts three
/// tuning-independent properties that pin the two failure modes that bit
/// early cuts — an Inf-driven NaN block (a bright HDR sky exceeding f16 range
/// turning into a black square) and an unbounded additive composite washing
/// the frame white: a meaningful fraction brightens (the halo), no
/// previously-bright pixel collapses to black (no NaN), and the frame does
/// not turn mostly white (bounded `lerp` composite, not `scene + bloom`). A
/// golden image guards the look on top.
#[test]
fn deferred_bloom_brightens_highlights() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping bloom test");
        return;
    }

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

    let output_guid = redlilium_assets::Guid::stable("test/bloom_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Srgb, output_guid);
    ew.world
        .insert(camera, CameraEnvironment::default())
        .unwrap();

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    let mut calm = 0u32;
    for _ in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle");

    // Baseline: no bloom.
    for _ in 0..3 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let no_bloom = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );

    // A strong bloom so the differential is unmistakable. A few frames to
    // derive the mip chain, build the materials, and render.
    ew.world.insert(camera, CameraBloom::new(0.15)).unwrap();
    for _ in 0..5 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let with_bloom = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );
    drop(pipeline);

    // Per-pixel luma (channel sum, 0..765).
    let luma = |p: &image::Rgba<u8>| p.0[0] as i32 + p.0[1] as i32 + p.0[2] as i32;
    let total = (SIZE * SIZE) as f64;
    let mut brighter = 0usize; // halo spread
    let mut black_holes = 0usize; // bright → near-black (the NaN block)
    let mut white_with = 0usize; // near-white after bloom (blowout)
    for (a, b) in no_bloom.pixels().zip(with_bloom.pixels()) {
        let (la, lb) = (luma(a), luma(b));
        if lb - la > 6 {
            brighter += 1;
        }
        if la > 90 && lb < 15 {
            black_holes += 1;
        }
        if lb >= 750 {
            white_with += 1;
        }
    }
    assert!(
        (brighter as f64 / total) > 0.02,
        "bloom brightened too few pixels ({brighter}) — no visible glow"
    );
    // Inf → NaN block: a bright pixel must never collapse to black.
    assert!(
        (black_holes as f64 / total) < 0.001,
        "bloom punched {black_holes} bright pixels to black — NaN (Inf in the \
         Karis average)"
    );
    // Unbounded composite: the frame must not turn mostly white.
    assert!(
        (white_with as f64 / total) < 0.5,
        "bloom blew the frame out ({white_with} near-white px) — the composite \
         must be a bounded lerp, not an add"
    );

    // Golden regression on the bloom look.
    compare_or_update("deferred_bloom.png", &with_bloom);
}

/// Auto-exposure (#153) must adapt toward a target and route the metered
/// multiplier to the display. Renders the fixed scene with
/// `CameraAutoExposure`, lets the eye adaptation converge, and asserts two
/// tuning-independent properties: the adaptation reaches a **stable** fixed
/// point (one more frame barely moves — no flicker), and the
/// exposure-compensation knob scales the result **monotonically** (−2 vs +2
/// stops, which leave the metered value untouched, brighten the image). A
/// golden image guards the converged look.
#[test]
fn deferred_auto_exposure_adapts_and_tracks_compensation() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping auto-exposure test");
        return;
    }

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

    let output_guid = redlilium_assets::Guid::stable("test/auto_exposure_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Srgb, output_guid);
    ew.world
        .insert(camera, CameraEnvironment::default())
        .unwrap();

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    let mut calm = 0u32;
    for _ in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle");

    let capture = |ew: &EditorWorld, pipeline: &mut FramePipeline| {
        to_display_rgba8(
            &read_back(ew, &device, pipeline, output_guid, OutputFormat::Srgb),
            OutputFormat::Srgb,
        )
    };
    let mean = |img: &image::RgbaImage| -> f64 {
        let sum: u64 = img
            .pixels()
            .map(|p| p.0[0] as u64 + p.0[1] as u64 + p.0[2] as u64)
            .sum();
        sum as f64 / (SIZE * SIZE) as f64
    };

    // Enable auto-exposure and let the eye adaptation converge (dt = 1/60, the
    // default speeds settle well within this budget).
    ew.world
        .insert(camera, CameraAutoExposure::default())
        .unwrap();
    for _ in 0..150 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let converged = capture(&ew, &mut pipeline);

    // Stability: one more frame must barely move — the metered value reached
    // its fixed point (no per-frame flicker).
    tick(&mut ew, &mut pipeline, &runner);
    let settled = capture(&ew, &mut pipeline);
    let drift = (mean(&converged) - mean(&settled)).abs();
    assert!(
        drift < 3.0,
        "auto-exposure still drifting after convergence (Δmean {drift:.2}) — not a stable fixed point"
    );

    // Compensation tracks: it does not touch the histogram, so −2 vs +2 stops
    // must scale the image monotonically brighter. Compensation is read fresh
    // each frame; a couple of frames flush the ring.
    ew.world
        .insert(
            camera,
            CameraAutoExposure {
                compensation: -2.0,
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..2 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let dark = capture(&ew, &mut pipeline);
    ew.world
        .insert(
            camera,
            CameraAutoExposure {
                compensation: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..2 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let bright = capture(&ew, &mut pipeline);
    drop(pipeline);

    let (md, mb) = (mean(&dark), mean(&bright));
    assert!(
        mb > md + 10.0,
        "exposure compensation did not brighten the image (+2 EV {mb:.1} vs -2 EV {md:.1}) — \
         the metered exposure buffer is not reaching the display"
    );

    // Golden regression on the converged (compensation-0) look.
    compare_or_update("deferred_auto_exposure.png", &converged);
}

/// Motion blur (#149) must run end-to-end on the GPU — the TileMax /
/// NeighborMax / reconstruction passes create, bind, and execute without a
/// hazard — and be a no-op on a static scene: with no motion the neighbourhood
/// blur vector is ~0, so the reconstruction returns the source pixel. Render the
/// golden scene with TAA, capture it, then opt into MotionBlur and confirm the
/// image is unchanged (a few LSBs of extra-resample tolerance).
#[test]
fn deferred_motion_blur_runs_static_noop() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping motion blur test");
        return;
    }

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

    let output_guid = redlilium_assets::Guid::stable("test/mb_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Srgb, output_guid);
    ew.world
        .insert(camera, redlilium_ecs::TemporalJitter::default())
        .unwrap();

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    let mut calm = 0u32;
    for _ in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle");
    for _ in 0..24 {
        tick(&mut ew, &mut pipeline, &runner);
    }

    // TAA only.
    let without = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );

    // Opt into motion blur: the pipeline rebuilds its MB targets/materials
    // (which resets the TAA history), runs TileMax -> NeighborMax ->
    // reconstruction, and re-converges. Tick a whole multiple of the jitter
    // cycle (8) so the capture lands on the *same* jitter phase as `without` —
    // otherwise the two converged frames differ by TAA's phase edge-shimmer,
    // not by anything motion blur did.
    ew.world
        .insert(camera, redlilium_ecs::MotionBlur::default())
        .unwrap();
    for _ in 0..32 {
        tick(&mut ew, &mut pipeline, &runner);
    }
    let with = to_display_rgba8(
        &read_back(&ew, &device, &mut pipeline, output_guid, OutputFormat::Srgb),
        OutputFormat::Srgb,
    );

    assert_images_close(
        &without,
        &with,
        4,
        0.03,
        24,
        "motion blur changed a static image (should be a no-op)",
    );
}

/// The camera-only velocity path (`velocity_complete`, #149) must be
/// *temporally stable* under a smooth camera pan — the property motion blur
/// depends on and TAA does not. TAA averages the per-frame velocity away;
/// motion blur consumes it directly, so any frame-to-frame jitter in the
/// background (sky) velocity surfaces as shimmer once TAA stops hiding it.
///
/// This isolates the velocity path from scene content: an **empty** scene means
/// the G-buffer albedo alpha clears to 0 everywhere, so every texel is
/// "background" (ADR-039) and `GBUFFER_VELOCITY` is *entirely* the
/// `velocity_complete` reprojection. The camera is panned by a **constant** yaw
/// about a **fixed** eye. Under a constant rotation rate about a fixed point the
/// induced optical flow is time-invariant — the far-plane point a pixel sees
/// always sits at the same relative angle one frame back, at the same distance
/// from the shared eye — so two consecutive frames must produce a near-identical
/// velocity field (equal in exact arithmetic; float noise otherwise).
///
/// TemporalJitter is on: velocity uses the *unjittered* pair, so sub-pixel
/// jitter must not leak in and destabilise the field. A pass proves the
/// reprojection is clean, and therefore that any sky shimmer seen under motion
/// blur comes from *upstream* camera motion (variable frame time, an unsmoothed
/// look target), not this shader.
#[test]
fn deferred_camera_pan_velocity_is_temporally_stable() {
    let _ = env_logger::builder().is_test(true).try_init();
    let instance = GraphicsInstance::new().expect("graphics instance");
    let device = instance.create_device().expect("graphics device");
    if device.name() == "Dummy Adapter" {
        eprintln!("dummy backend does not render; skipping pan velocity test");
        return;
    }

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
    // Empty scene on purpose: no meshes -> albedo alpha clears to 0 everywhere
    // -> every texel is background -> GBUFFER_VELOCITY is purely the camera-only
    // reprojection under scrutiny.

    let output_guid = redlilium_assets::Guid::stable("test/pan_velocity_camera_output");
    let camera = ew.editor_camera;
    set_output(&mut ew.world, camera, OutputFormat::Srgb, output_guid);
    ew.world
        .insert(camera, redlilium_ecs::TemporalJitter::default())
        .unwrap();

    // Fixed eye; the look direction yaws a constant step per frame. A fixed eye
    // keeps the flow a pure rotation, whose field is exactly frame-invariant.
    let eye = Vec3::new(0.0, 1.0, 3.0);
    const DTHETA: f32 = 0.02; // rad/frame (~1.15 deg) — far above sub-pixel jitter.
    let set_pan = |world: &mut redlilium_ecs::World, frame: i32| {
        let theta = frame as f32 * DTHETA;
        let forward = Vec3::new(theta.sin(), 0.0, -theta.cos());
        let t = Transform::new(eye, quat_looking_along(forward), Vec3::new(1.0, 1.0, 1.0));
        world.insert(camera, t).unwrap();
        world
            .insert(camera, GlobalTransform(t.to_matrix()))
            .unwrap();
    };

    let runner = EcsRunner::single_thread();
    ew.schedules.run_startup(&mut ew.world, &runner);
    let mut pipeline = device.create_pipeline(2);

    // Hold theta = 0 while the pipeline's shaders/targets go resident.
    set_pan(&mut ew.world, 0);
    let mut calm = 0u32;
    for _ in 0..600 {
        tick(&mut ew, &mut pipeline, &runner);
        calm = if crate::remote_commands::assets_idle(&ew.world) {
            calm + 1
        } else {
            0
        };
        if calm >= 3 {
            break;
        }
    }
    assert!(calm >= 3, "asset pipeline never went idle");
    for _ in 0..5 {
        tick(&mut ew, &mut pipeline, &runner);
    }

    let velocity_texture = ew
        .world
        .get::<redlilium_ecs::PipelineTargets>(camera)
        .expect("deferred targets derived")
        .get(redlilium_ecs::rendering::deferred::GBUFFER_VELOCITY)
        .expect("velocity target derived")
        .clone();

    // Warm the pan up so `prev` is itself a mid-pan step (constant omega on both
    // sides of the capture), then capture two consecutive frames.
    let mut frame = 1;
    for _ in 0..8 {
        set_pan(&mut ew.world, frame);
        tick(&mut ew, &mut pipeline, &runner);
        frame += 1;
    }
    let vel_a = decode_velocity(&read_back_texture(
        &device,
        &mut pipeline,
        velocity_texture.clone(),
        4,
    ));
    set_pan(&mut ew.world, frame);
    tick(&mut ew, &mut pipeline, &runner);
    let vel_b = decode_velocity(&read_back_texture(
        &device,
        &mut pipeline,
        velocity_texture,
        4,
    ));

    // The pan must actually produce motion, or the test is vacuous (e.g. the
    // background was never filled).
    let mag_a = max_magnitude(&vel_a);
    assert!(
        mag_a > 0.01,
        "camera pan produced no background velocity: {mag_a}"
    );

    // Compare on a centre crop: away from the frustum edge the rotational flow
    // is cleanly defined, with no near-plane / behind-camera projection spikes.
    let lo = SIZE / 4;
    let hi = SIZE - SIZE / 4;
    let mut delta_max = 0.0f32;
    let mut mag_center = 0.0f32;
    for y in lo..hi {
        for x in lo..hi {
            let i = (y * SIZE + x) as usize;
            let a = vel_a[i];
            let b = vel_b[i];
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
            delta_max = delta_max.max(d);
            mag_center = mag_center.max((a[0] * a[0] + a[1] * a[1]).sqrt());
        }
    }
    assert!(
        mag_center > 0.01,
        "centre pan velocity too small: {mag_center}"
    );
    let ratio = delta_max / mag_center;
    eprintln!(
        "pan velocity: |v|~{mag_center:.4} NDC, frame-to-frame delta_max {delta_max:.5} \
         ({:.2}% of |v|)",
        ratio * 100.0
    );
    assert!(
        ratio < 0.05,
        "camera-only velocity flickers frame-to-frame under a constant pan: delta_max \
         {delta_max:.5} is {:.1}% of |v| {mag_center:.4} (threshold 5%)",
        ratio * 100.0
    );
}

/// Decode an Rg16Float readback into per-texel `[vx, vy]`.
fn decode_velocity(raw: &[u8]) -> Vec<[f32; 2]> {
    raw.chunks_exact(4)
        .map(|texel| {
            [
                f16_to_f32(u16::from_le_bytes([texel[0], texel[1]])),
                f16_to_f32(u16::from_le_bytes([texel[2], texel[3]])),
            ]
        })
        .collect()
}

fn max_magnitude(velocity: &[[f32; 2]]) -> f32 {
    velocity
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
        .fold(0.0, f32::max)
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
    read_back_texture(device, pipeline, texture, bytes_px)
}

/// Read any COPY_SRC texture of the test's SIZE back to the CPU.
fn read_back_texture(
    device: &Arc<redlilium_graphics::GraphicsDevice>,
    pipeline: &mut FramePipeline,
    texture: Arc<redlilium_graphics::Texture>,
    bytes_px: u64,
) -> Vec<u8> {
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
        // Linear-HDR target: the display pass already applied the headroom
        // roll-off on the GPU (#154), and the test path has no DisplayHeadroom
        // resource, so H=1 — the surface holds PBR-Neutral-tonemapped *linear*
        // (1.0 = paper white). To get a comparable display image we only sRGB-
        // encode it (do NOT tonemap again — the GPU already did, unlike the old
        // raw-linear-clamp path).
        OutputFormat::Hdr => {
            for texel in raw.chunks_exact(8) {
                let rgb = [
                    f16_to_f32(u16::from_le_bytes([texel[0], texel[1]])),
                    f16_to_f32(u16::from_le_bytes([texel[2], texel[3]])),
                    f16_to_f32(u16::from_le_bytes([texel[4], texel[5]])),
                ];
                for c in rgb {
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
