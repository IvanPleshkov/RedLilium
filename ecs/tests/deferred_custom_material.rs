//! Application-authored cutout/deformation, through the stock dispatcher and
//! deferred resolve/TAA/blur. Uses public APIs, including the offline shader path.
#![cfg(feature = "rendering")]
use redlilium_assets::{AssetDb, AssetPath, AssetProcessor, Guid};
use redlilium_ecs::rendering::GBufferContract;
use redlilium_ecs::rendering::{deferred::*, loaders::*, shading::*};
use redlilium_ecs::{
    Camera, CameraOutput, CameraRender, DEFERRED_PIPELINE, EcsRunner, EnsureCameraTargets, Entity,
    FrameRing, GameTime, MainViewport, MaterialAssetManager, MaterialInstanceManager, MeshRenderer,
    MotionBlur, PipelineCache, PipelineRegistry, PipelineTargets, Primitive, RealTime, Render,
    RenderPath, RenderSchedule, ScenePass, Schedules, ShaderManager, ShadingRegistry, SizePolicy,
    TemporalJitter, TemporalState, TextureManager, World, register_rendering_components,
    register_std_components,
};
use redlilium_graphics::*;
use redlilium_vfs::{MemoryProvider, Vfs};
use std::sync::Arc;
#[path = "fixtures/deforming_cutout_baked.rs"]
mod baked;

const SIZE: u32 = 64;
fn layout() -> Arc<VertexLayout> {
    Arc::new(
        VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(32))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(20))
            .with_attribute(VertexAttribute::texcoord0(12)),
    )
}
fn vertices(shear: f32, previous: f32) -> Vec<f32> {
    let mut result = vec![];
    for (x, y, u, v) in [
        (-0.75, -0.75, 0., 1.),
        (0.75, -0.75, 1., 1.),
        (0.75, 0.75, 1., 0.),
        (-0.75, -0.75, 0., 1.),
        (0.75, 0.75, 1., 0.),
        (-0.75, 0.75, 0., 0.),
    ] {
        let t = (y + 0.75) / 1.5;
        result.extend([x + shear * t, y, -1., u, v, x + previous * t, y, -1.]);
    }
    result
}

struct Probe {
    world: World,
    camera: Entity,
    device: Arc<GraphicsDevice>,
    pipeline: FramePipeline,
    schedules: Schedules,
    mesh: Arc<Mesh>,
    image: Arc<Texture>,
}
impl Probe {
    fn new(device: Arc<GraphicsDevice>) -> Self {
        baked::register().unwrap();
        assert_eq!(
            baked::SOURCE,
            include_str!("fixtures/deforming_cutout.slang"),
            "rebake fixture"
        );
        let mut world = World::new();
        register_std_components(&mut world);
        register_rendering_components(&mut world);
        let shader_guid = Guid::stable("game/test/deforming_cutout.slang");
        let mut cache = PipelineCache::new(device.clone());
        cache.register_deferred_shader(shader_guid, GBufferContract::V1);
        let shader = Arc::new(Shader {
            source: baked::SOURCE.as_bytes().to_vec(),
        });
        cache
            .get_or_build_deferred(
                shader_guid,
                &shader,
                &VariantKey::default(),
                &layout(),
                TextureFormat::Depth32Float,
            )
            .unwrap();
        world.insert_resource(cache);
        world.insert_resource(RenderSchedule::empty());
        world.insert_resource(ScenePass::default());
        world.insert_resource(PipelineRegistry::default());
        world.insert_resource(FrameRing::new(&device, 1 << 18, "probe").unwrap());
        world.insert_resource(MainViewport::new(SIZE, SIZE));
        world.insert_resource(GameTime::default());
        world.insert_resource(RealTime::default());
        let image = device
            .create_texture(&TextureDescriptor::new_2d(
                2,
                1,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            ))
            .unwrap();
        let image_guid = Guid::new();
        let mut textures = TextureManager::new(device.clone());
        textures.publish_virtual(image_guid, image.clone()).unwrap();
        let mut registry = ShadingRegistry::with_builtins();
        registry
            .register(ShadingModel {
                id: "test_cutout".into(),
                shader: shader_guid,
                schema: vec![
                    PropDef {
                        name: "tint".into(),
                        default: PropValue::Vec4([1.; 4]),
                    },
                    PropDef {
                        name: "image".into(),
                        default: PropValue::Texture(TextureSource::Virtual(image_guid)),
                    },
                ],
            })
            .unwrap();
        let mut vfs = Vfs::new();
        vfs.mount("test", MemoryProvider::new());
        let mut processor = AssetProcessor::new(vfs, device.clone());
        let mut db = AssetDb::new();
        let template = db.register_path(AssetPath::new("test", "cutout.material"), "material", 0);
        db.set_settings(
            &template,
            Some(
                ron::to_string(&MaterialData {
                    shading_model: "test_cutout".into(),
                    features: vec![],
                    properties: vec![],
                })
                .unwrap(),
            ),
        );
        let mut shaders = ShaderManager::new();
        shaders.publish(shader_guid, shader);
        let mut materials = MaterialAssetManager::new();
        let mut instances = MaterialInstanceManager::new(device.clone());
        let instance_guid = Guid::new();
        instances.publish_virtual(
            instance_guid,
            MaterialInstanceData {
                parent: template,
                overrides: vec![],
            },
        );
        for _ in 0..8 {
            instances.drive(
                &mut processor,
                &db,
                &mut materials,
                &mut shaders,
                &mut textures,
                &registry,
            );
            for (_, task) in processor.drain_tasks() {
                use std::{
                    future::Future,
                    task::{Context, Poll, Waker},
                };
                assert!(matches!(
                    std::pin::pin!(task)
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop())),
                    Poll::Ready(_)
                ));
            }
            processor.collect();
            processor.flush_gpu();
        }
        let mesh = device
            .create_mesh(&MeshDescriptor::new(layout()).with_vertex_count(6))
            .unwrap();
        let mut primitive = Primitive::new(
            MeshSource::External(1),
            MaterialInstanceSource {
                guid: instance_guid,
            },
        );
        primitive.mesh.resolve(mesh.clone());
        primitive.material.resolve(
            instances
                .get(instance_guid)
                .expect("resolved custom material")
                .clone(),
        );
        let entity = world.spawn();
        world
            .insert(
                entity,
                MeshRenderer::single(primitive).with_casts_shadows(false),
            )
            .unwrap();
        world.insert_resource(instances);
        world.insert_resource(textures);
        let camera = world.spawn();
        world
            .insert(camera, Camera::orthographic(1., 1., 0.1, 10.))
            .unwrap();
        world
            .insert(
                camera,
                CameraOutput::offscreen(SizePolicy::Fixed(SIZE, SIZE), None)
                    .with_clear_color([0.05, 0., 0.2, 1.]),
            )
            .unwrap();
        world
            .insert(camera, RenderPath::named(DEFERRED_PIPELINE))
            .unwrap();
        world.insert(camera, TemporalJitter::default()).unwrap();
        world.insert(camera, MotionBlur::default()).unwrap();
        let mut schedules = Schedules::new();
        let render = schedules.get_mut::<Render>();
        render.add_exclusive(EnsureCameraTargets);
        render.add(CameraRender);
        render
            .add_edge::<EnsureCameraTargets, CameraRender>()
            .unwrap();
        let pipeline = device.create_pipeline(1);
        Self {
            world,
            camera,
            device,
            pipeline,
            schedules,
            mesh,
            image,
        }
    }

    fn frame(&mut self, shear: f32, previous: f32, color: [u8; 3]) -> Vec<Readback> {
        let frame = if self.world.has_resource::<TemporalState>() {
            self.world.resource::<TemporalState>().frame()
        } else {
            0
        };
        self.world
            .resource_mut::<GameTime>()
            .set_scale(match frame {
                1 => 0.25,
                2 => 0.0,
                _ => 1.0,
            });
        let dt = match frame {
            1 => 1. / 120.,
            2 => 1. / 30.,
            _ => 1. / 60.,
        };
        self.schedules
            .run_frame(&mut self.world, &EcsRunner::single_thread(), dt);
        let mut schedule = self.pipeline.begin_frame().unwrap();
        let mut graph = schedule.acquire_graph();
        let mut upload = TransferPass::new("game_vertex_upload".into());
        upload.set_transfer_config(TransferConfig::new().with_operations(vec![
            TransferOperation::write_buffer(self.mesh.vertex_buffer(0).unwrap().clone(),0,
                Arc::from(bytemuck::cast_slice::<f32,u8>(&vertices(shear,previous)))),
            TransferOperation::upload_texture_data(&self.device,self.image.clone(),
                &[color[0],color[1],color[2],0,color[0],color[1],color[2],255]).unwrap(),
        ]));
        graph.add_transfer_pass(upload);
        self.world
            .resource_mut::<MaterialInstanceManager>()
            .flush_uploads(&mut graph);
        self.world.resource_mut::<RenderSchedule>().set(graph);
        self.schedules
            .run_schedule::<Render>(&mut self.world, &EcsRunner::single_thread());
        let mut graph = self.world.resource_mut::<RenderSchedule>().take().unwrap();
        let targets = self.world.get::<PipelineTargets>(self.camera).unwrap();
        let write = 1 - (self.world.resource::<TemporalState>().frame() % 2) as usize;
        let readbacks = [
            GBUFFER_ALBEDO,
            GBUFFER_VELOCITY,
            SCENE_COLOR,
            TAA_HISTORY[write],
            MB_OUTPUT,
        ]
        .map(|name| {
            inspect_texture(
                &self.device,
                &mut graph,
                targets.get(name).expect(name).clone(),
            )
        });
        schedule.submit(graph).unwrap();
        self.pipeline.end_frame(schedule);
        readbacks.into()
    }
}

// Inspection is itself a graphics pass: no COPY_SRC requirement or extra usage
// flags on production targets. textureLoad preserves the actual linear values.
fn inspect_texture(
    device: &Arc<GraphicsDevice>,
    graph: &mut RenderGraph,
    source: Arc<Texture>,
) -> Readback {
    let shader = r#"
@group(0) @binding(0) var image: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    var p = array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
    return vec4f(p[i],0,1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return textureLoad(image,vec2i(p.xy),0);
}"#;
    let mesh = device
        .create_mesh(&MeshDescriptor::new(VertexLayout::position_only()).with_vertex_count(3))
        .unwrap();
    let material = device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(redlilium_graphics::ShaderSource::vertex(shader, "vs"))
                .with_shader(redlilium_graphics::ShaderSource::fragment(shader, "fs"))
                .with_binding_layout(Arc::new(BindingLayout {
                    entries: vec![BindingLayoutEntry::new(0, BindingType::Texture)],
                    label: None,
                }))
                .with_vertex_layout(mesh.layout().clone())
                .with_color_format(TextureFormat::Rgba16Float)
                .with_depth(None)
                .with_cull_mode(CullMode::None),
        )
        .unwrap();
    let group = device
        .create_binding_group(
            material.binding_layouts()[0].clone(),
            BindingGroupDescriptor::new().with_texture(0, source.clone()),
        )
        .unwrap();
    let size = source.size().width;
    let output = device
        .create_texture(&TextureDescriptor::new_2d(
            size,
            size,
            TextureFormat::Rgba16Float,
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        ))
        .unwrap();
    let mut pass = GraphicsPass::new("inspect".into());
    pass.set_render_targets(RenderTargetConfig::new().with_color(
        ColorAttachment::from_texture(output.clone()).with_clear_color(0., 0., 0., 0.),
    ));
    pass.add_draw_command(DrawCommand::new(
        mesh,
        Arc::new(MaterialInstance::new(material).with_binding_group(group)),
    ));
    graph.add_graphics_pass(pass);
    let byte_size = size as u64 * size as u64 * 8;
    let buffer = device
        .create_buffer(&BufferDescriptor::new(
            byte_size,
            BufferUsage::COPY_DST | BufferUsage::MAP_READ,
        ))
        .unwrap();
    let result = Readback::new();
    let mut pass = TransferPass::new("inspect_readback".into());
    pass.set_transfer_config(TransferConfig::new().with_operations(vec![
        TransferOperation::readback_texture_whole(output, buffer.clone()),
        TransferOperation::readback_buffer(buffer, 0..byte_size as usize, result.clone()),
    ]));
    graph.add_transfer_pass(pass);
    result
}
fn pixel(bytes: &[u8], size: usize, x: usize, y: usize) -> [f32; 4] {
    std::array::from_fn(|c| {
        let i = (y * size + x) * 8 + c * 2;
        let bits = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        let sign = if bits & 0x8000 != 0 { -1. } else { 1. };
        let exponent = ((bits >> 10) & 31) as i32;
        assert_ne!(exponent, 31, "non-finite GPU output");
        sign * if exponent == 0 {
            (bits & 1023) as f32 * 2f32.powi(-24)
        } else {
            (1. + (bits & 1023) as f32 / 1024.) * 2f32.powi(exponent - 15)
        }
    })
}
fn verify(data: &[Vec<u8>], size: usize, moving: bool, reset: bool, color: [u8; 3]) {
    let x = size * 3 / 4;
    let y = size / 2;
    assert_eq!(
        pixel(&data[0], size, size / 4, y)[3],
        0.,
        "transparent texels must remain background"
    );
    let velocity = pixel(&data[1], size, x, y);
    if moving {
        assert!(
            (velocity[0] - 0.125).abs() < 0.015,
            "deformation velocity {velocity:?}"
        );
    } else {
        assert!(
            velocity[0].abs() < 0.0001,
            "static/jitter velocity {velocity:?}"
        );
    }
    assert!(velocity[1].abs() < 0.0001);
    let scene = pixel(&data[2], size, x, y);
    for c in 0..3 {
        assert!(
            (scene[c] - color[c] as f32 / 255.).abs() < 0.006,
            "unlit resolve: {scene:?}"
        );
    }
    for target in data {
        for y in 0..size {
            for x in 0..size {
                pixel(target, size, x, y);
            }
        }
    }
    if moving {
        assert!(
            data[3]
                .chunks_exact(8)
                .zip(data[4].chunks_exact(8))
                .filter(|(a, b)| a[..6] != b[..6])
                .count()
                > 8,
            "stock motion blur must filter the moving silhouette"
        );
    }
    if reset {
        // TAA history alpha stores depth, not scene opacity. Compare RGB.
        for y in 0..size {
            for x in 0..size {
                let scene = pixel(&data[2], size, x, y);
                let taa = pixel(&data[3], size, x, y);
                let blur = pixel(&data[4], size, x, y);
                for c in 0..3 {
                    assert!(
                        (scene[c] - taa[c]).abs() < 0.001,
                        "reset TAA at {x},{y}: {scene:?} vs {taa:?}"
                    );
                    assert!(
                        (taa[c] - blur[c]).abs() < 0.001,
                        "reset blur at {x},{y}: {taa:?} vs {blur:?}"
                    );
                }
            }
        }
    }
}
#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() {
    let _ = env_logger::builder().is_test(true).try_init();
    let device = GraphicsInstance::with_parameters(
        InstanceParameters::new().with_backend(BackendType::Wgpu),
    )
    .unwrap()
    .create_device()
    .unwrap();
    assert_ne!(device.name(), "Dummy Adapter");
    eprintln!("RL-008 GPU: {}", device.name());
    let mut probe = Probe::new(device);
    for (index, (shear, previous, reset, color, size)) in [
        (0., 0., true, [128, 64, 32], 64),
        (0.25, 0., false, [128, 64, 32], 64),
        (0.25, 0.25, false, [128, 64, 32], 64),
        (0., 0., true, [32, 128, 64], 64),
        (0., 0., true, [64, 32, 128], 96),
    ]
    .into_iter()
    .enumerate()
    {
        if index == 3 {
            probe.world.resource_mut::<TemporalState>().reset_history();
        }
        if index == 4 {
            probe
                .world
                .insert(
                    probe.camera,
                    CameraOutput::offscreen(SizePolicy::Fixed(size, size), None),
                )
                .unwrap();
        }
        let reads = probe.frame(shear, previous, color);
        probe.pipeline.wait_idle().unwrap();
        probe.pipeline.recycle_all_graphs();
        // The next submission/poll drives native asynchronous buffer-map callbacks.
        let mut drain = probe.pipeline.begin_frame().unwrap();
        drain.render(RenderGraph::new()).unwrap();
        probe.pipeline.end_frame(drain);
        probe.pipeline.wait_idle().unwrap();
        let data: Vec<_> = reads
            .iter()
            .map(|r| r.take_result().expect("readback complete").unwrap())
            .collect();
        verify(&data, size as usize, index == 1, reset, color);
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn custom_cutout_deformation_and_temporal_history() {
    run_native();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(
    inline_js = "export function next_tick() { return new Promise(r => setTimeout(r, 0)); }"
)]
extern "C" {
    fn next_tick() -> js_sys::Promise;
}

/// Same pixel assertions on WebGPU. Yield to browser callbacks at every fence
/// and buffer mapping; never block the event loop.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_browser() -> String {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    let device = GraphicsInstance::with_parameters_async(
        InstanceParameters::new().with_backend(BackendType::Wgpu),
    )
    .await
    .unwrap()
    .create_device()
    .unwrap();
    let name = device.name().to_owned();
    let mut probe = Probe::new(device);
    for (index, (shear, previous, reset, color, size)) in [
        (0., 0., true, [128, 64, 32], 64),
        (0.25, 0., false, [128, 64, 32], 64),
        (0.25, 0.25, false, [128, 64, 32], 64),
        (0., 0., true, [32, 128, 64], 64),
        (0., 0., true, [64, 32, 128], 96),
    ]
    .into_iter()
    .enumerate()
    {
        if index == 3 {
            probe.world.resource_mut::<TemporalState>().reset_history();
        }
        if index == 4 {
            probe
                .world
                .insert(
                    probe.camera,
                    CameraOutput::offscreen(SizePolicy::Fixed(size, size), None),
                )
                .unwrap();
        }
        let reads = probe.frame(shear, previous, color);
        let start = web_time::Instant::now();
        while !probe.pipeline.try_recycle_all_graphs() {
            assert!(start.elapsed().as_secs() < 30, "GPU fence timed out");
            wasm_bindgen_futures::JsFuture::from(next_tick())
                .await
                .unwrap();
        }
        while reads.iter().any(|r| r.status() == ReadbackStatus::Pending) {
            assert!(start.elapsed().as_secs() < 30, "GPU readback timed out");
            wasm_bindgen_futures::JsFuture::from(next_tick())
                .await
                .unwrap();
        }
        // Surface any asynchronous WebGPU validation error before pixel checks.
        probe
            .device
            .create_buffer(&BufferDescriptor::new(4, BufferUsage::COPY_DST))
            .unwrap();
        let data: Vec<_> = reads
            .iter()
            .map(|r| r.take_result().unwrap().unwrap())
            .collect();
        verify(&data, size as usize, index == 1, reset, color);
    }
    format!(
        "RL-008 WebGPU PASS ({name}): cutout, deformation velocity, unlit, TAA, blur, reset, resize"
    )
}

#[test]
fn deferred_rejection_cannot_reuse_an_unvalidated_specialization() {
    let device = GraphicsInstance::with_parameters(
        InstanceParameters::new().with_backend(BackendType::Wgpu),
    )
    .unwrap()
    .create_device()
    .unwrap();
    let mut cache = PipelineCache::new(device);
    let guid = Guid::new();
    let forward = Arc::new(Shader {
        source: include_bytes!("../../std-assets/shaders/opaque_color.slang").to_vec(),
    });
    let layout = VertexLayout::position_normal();
    cache
        .get_or_build(
            guid,
            &forward,
            &VariantKey::default(),
            &layout,
            &GBUFFER_FORMATS,
            TextureFormat::Depth32Float,
        )
        .unwrap();
    assert!(
        cache
            .get_or_build_deferred(
                guid,
                &forward,
                &VariantKey::default(),
                &layout,
                TextureFormat::Depth32Float
            )
            .is_err()
    );
    cache.register_deferred_shader(guid, GBufferContract::V1);
    assert!(
        cache
            .get_or_build_deferred(
                guid,
                &forward,
                &VariantKey::default(),
                &layout,
                TextureFormat::Depth32Float
            )
            .is_err()
    );
    let valid = Arc::new(Shader {
        source: include_bytes!("../../std-assets/shaders/deferred_gbuffer.slang").to_vec(),
    });
    let good = cache
        .get_or_build_deferred(
            guid,
            &valid,
            &VariantKey::default(),
            &layout,
            TextureFormat::Depth32Float,
        )
        .unwrap();
    let last_good = cache
        .get_or_build_deferred(
            guid,
            &forward,
            &VariantKey::default(),
            &layout,
            TextureFormat::Depth32Float,
        )
        .unwrap();
    assert!(
        Arc::ptr_eq(&good, &last_good),
        "bad hot reload retains only a validated pipeline"
    );
}
