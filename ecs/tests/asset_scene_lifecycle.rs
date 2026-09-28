//! Scene lifetime through ECS loading systems, async executors and submitted
//! transfer graphs. Dummy tests Arc/resource ownership, not hardware VRAM or
//! drawing; no manager is recreated between unload/reload cycles.
#![cfg(all(feature = "rendering", not(target_arch = "wasm32")))]

use redlilium_assets::{AssetDb, AssetPath, AssetProcessor, AssetRecord, Guid};
use redlilium_ecs::rendering::{
    ResolvedEnvironment, ResolvedInstance,
    loaders::{EnvironmentData, MaterialData, MaterialInstanceData, TextureSource},
    shading::{PropValue, ShadingRegistry},
};
use redlilium_ecs::{
    AssetGpuFlush, AssetPump, CameraEnvironment, EcsRunner, Entity, EnvironmentManager,
    MaterialAssetManager, MaterialInstanceLoad, MaterialInstanceManager, MaterialInstanceSource,
    MeshGenerator, MeshLoad, MeshManager, MeshRenderer, MeshSource, Primitive, RenderSchedule,
    ShaderManager, SystemsContainer, TextureManager, VertexLayoutManager, World,
    register_rendering_components, register_std_components,
};
use redlilium_graphics::{
    BackendType, BoundResource, FramePipeline, GraphicsInstance, InstanceParameters, Mesh,
    RenderGraph,
};
use redlilium_vfs::{MemoryProvider, Vfs};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

struct Fixture {
    world: World,
    runner: EcsRunner,
    systems: SystemsContainer,
    pipeline: FramePipeline,
    mesh: Guid,
    layout: Guid,
    material: Guid,
    instance: Guid,
    texture: Guid,
    cube: Guid,
    environment: Guid,
    shader: Guid,
    upload_graphs: usize,
}

impl Fixture {
    fn new(multi: bool) -> Self {
        let device = GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap();
        let runner = if multi {
            EcsRunner::multi_thread(2)
        } else {
            EcsRunner::single_thread()
        };
        let io = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let mut vfs = Vfs::new();
        vfs.mount("scene", MemoryProvider::new());
        let registry = ShadingRegistry::with_builtins();
        let shader = registry.get("opaque_textured").unwrap().shader;
        let mut db = AssetDb::new();
        db.insert(
            shader,
            AssetRecord {
                path: AssetPath::new("scene", "surface.slang"),
                kind: "shader".into(),
                source_hash: 0,
                settings: None,
                references: BTreeMap::new(),
            },
        )
        .unwrap();
        io.block_on(vfs.write(
            "scene/surface.slang",
            include_bytes!("../../std-assets/shaders/opaque_textured.slang").to_vec(),
        ))
        .unwrap();
        let mesh = db.register_path(AssetPath::new("scene", "sphere.rmesh"), "mesh", 0);
        let layout = db.register_path(
            AssetPath::new("scene", "sphere.vlayout"),
            "vertex_layout",
            0,
        );
        let generator = MeshGenerator::sphere(1.0, 8, 4);
        io.block_on(vfs.write(
            "scene/sphere.rmesh",
            bincode::serialize(&generator.build()).unwrap(),
        ))
        .unwrap();
        db.set_settings(
            &layout,
            Some(ron::to_string(generator.layout().as_ref()).unwrap()),
        );
        db.set_reference(&mesh, "layout", Some(layout));
        let texture = db.register_path(AssetPath::new("scene", "albedo.png"), "texture", 0);
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbaImage::from_pixel(2, 2, image::Rgba([100, 150, 200, 255]))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        io.block_on(vfs.write("scene/albedo.png", png.into_inner()))
            .unwrap();
        db.set_settings(&texture, Some("(generate_mips:false)".into()));
        let cube = db.register_path(AssetPath::new("scene", "sky.ktx2"), "texture", 0);
        io.block_on(vfs.write(
            "scene/sky.ktx2",
            include_bytes!("../../std-assets/textures/ibl/irradiance_cube.ktx2").to_vec(),
        ))
        .unwrap();
        let environment = db.register_path(AssetPath::new("scene", "sky.env"), "environment", 0);
        db.set_settings(
            &environment,
            Some(
                ron::to_string(&EnvironmentData {
                    irradiance: cube,
                    prefilter: cube,
                    sky: cube,
                })
                .unwrap(),
            ),
        );
        let material = db.register_path(AssetPath::new("scene", "surface.material"), "material", 0);
        db.set_settings(
            &material,
            Some(
                ron::to_string(&MaterialData {
                    shading_model: "opaque_textured".into(),
                    features: vec![],
                    properties: vec![(
                        "base_texture".into(),
                        PropValue::Texture(TextureSource::File(texture)),
                    )],
                })
                .unwrap(),
            ),
        );
        let instance = db.register_path(
            AssetPath::new("scene", "surface.matinst"),
            "material_instance",
            0,
        );
        db.set_settings(
            &instance,
            Some(
                ron::to_string(&MaterialInstanceData {
                    parent: material,
                    overrides: vec![],
                })
                .unwrap(),
            ),
        );

        let mut world = World::new();
        register_std_components(&mut world);
        register_rendering_components(&mut world);
        world.insert_resource(AssetProcessor::new(vfs, device.clone()));
        world.insert_resource(db);
        world.insert_resource(registry);
        world.insert_resource(MeshManager::new());
        world.insert_resource(VertexLayoutManager::new());
        world.insert_resource(ShaderManager::new());
        world.insert_resource(MaterialAssetManager::new());
        world.insert_resource(MaterialInstanceManager::new(device.clone()));
        world.insert_resource(TextureManager::new(device.clone()));
        world.insert_resource(EnvironmentManager::new());
        world.insert_resource(RenderSchedule::empty());
        let mut systems = SystemsContainer::new();
        systems.add_exclusive(MeshLoad::default());
        systems.add(MaterialInstanceLoad);
        systems.add(AssetPump);
        systems.add(AssetGpuFlush);
        systems
            .add_edge::<MeshLoad, MaterialInstanceLoad>()
            .unwrap();
        systems
            .add_edge::<MaterialInstanceLoad, AssetPump>()
            .unwrap();
        systems.add_edge::<AssetPump, AssetGpuFlush>().unwrap();
        Self {
            world,
            runner,
            systems,
            pipeline: device.create_pipeline(2),
            mesh,
            layout,
            material,
            instance,
            texture,
            cube,
            environment,
            shader,
            upload_graphs: 0,
        }
    }

    fn spawn_scene(&mut self) -> [Entity; 3] {
        let mut spawn_mesh = || {
            self.world
                .spawn_with((MeshRenderer::single(Primitive::new(
                    MeshSource::File(self.mesh),
                    MaterialInstanceSource {
                        guid: self.instance,
                    },
                )),))
                .unwrap()
        };
        let first = spawn_mesh();
        let second = spawn_mesh();
        let camera = self
            .world
            .spawn_with((CameraEnvironment::new(self.environment),))
            .unwrap();
        [first, second, camera]
    }

    fn tick(&mut self) {
        self.world
            .resource_mut::<RenderSchedule>()
            .set(RenderGraph::new());
        assert!(self.runner.run(&mut self.world, &self.systems).is_empty());
        // Drain background work deterministically; keep the executors alive
        // across frames/scenes. Next frame collects their completions.
        self.runner
            .graceful_shutdown(Duration::from_secs(5))
            .unwrap();
        let graphs = {
            let mut render = self.world.resource_mut::<RenderSchedule>();
            self.world
                .resource_mut::<MaterialInstanceManager>()
                .flush_uploads(render.graph_mut().unwrap());
            let mut graphs = render.take_transfer_graphs();
            self.upload_graphs += graphs.len();
            graphs.push(render.take().unwrap());
            graphs
        };
        let mut frame = self.pipeline.begin_frame().unwrap();
        for graph in graphs {
            frame.submit(graph).unwrap();
        }
        self.pipeline.end_frame(frame);
    }

    fn load(&mut self, scene: &[Entity; 3]) -> Owners {
        for _ in 0..32 {
            self.tick();
            let first = self.world.get::<MeshRenderer>(scene[0]).unwrap();
            let second = self.world.get::<MeshRenderer>(scene[1]).unwrap();
            let camera = self.world.get::<CameraEnvironment>(scene[2]).unwrap();
            if let (Some(mesh), Some(instance), Some(environment)) = (
                first.primitives[0].mesh(),
                first.primitives[0].material(),
                camera.environment.get(),
            ) {
                if let (Some(other_mesh), Some(other_instance)) =
                    (second.primitives[0].mesh(), second.primitives[0].material())
                {
                    assert!(Arc::ptr_eq(&mesh, &other_mesh));
                    assert!(Arc::ptr_eq(&instance, &other_instance));
                    assert!(self.world.resource::<AssetProcessor>().is_idle());
                    return Owners {
                        mesh,
                        instance,
                        environment: environment.clone(),
                    };
                }
            }
        }
        panic!("scene asset references did not resolve in 32 frames");
    }

    fn unload(&mut self, scene: [Entity; 3]) {
        for entity in scene {
            assert!(self.world.despawn(entity));
        }
        // Submitted graphs intentionally own resources until the GPU fence.
        self.pipeline.wait_idle().unwrap();
        self.pipeline.recycle_all_graphs();
        assert!(self.world.resource::<AssetProcessor>().is_idle());
    }

    fn collect(&mut self) -> usize {
        let w = &self.world;
        w.resource_mut::<MaterialInstanceManager>().collect_unused()
            + w.resource_mut::<EnvironmentManager>().collect_unused()
            + w.resource_mut::<MaterialAssetManager>().collect_unused()
            + w.resource_mut::<MeshManager>().collect_unused()
            + w.resource_mut::<TextureManager>().collect_unused()
            + w.resource_mut::<VertexLayoutManager>().collect_unused()
            + w.resource_mut::<ShaderManager>().collect_unused()
    }

    fn release(&mut self) {
        let w = &self.world;
        w.resource_mut::<MaterialInstanceManager>()
            .release(self.instance);
        w.resource_mut::<EnvironmentManager>()
            .release(self.environment);
        w.resource_mut::<MaterialAssetManager>()
            .release(self.material);
        w.resource_mut::<MeshManager>()
            .release(&MeshSource::File(self.mesh));
        w.resource_mut::<TextureManager>()
            .release(&TextureSource::File(self.texture));
        w.resource_mut::<TextureManager>()
            .release(&TextureSource::File(self.cube));
        w.resource_mut::<VertexLayoutManager>().release(self.layout);
        w.resource_mut::<ShaderManager>().release(self.shader);
    }
}

// Represents an independent consumer such as a preview or another live scene.
struct Owners {
    mesh: Arc<Mesh>,
    instance: Arc<ResolvedInstance>,
    environment: Arc<ResolvedEnvironment>,
}

struct Watch {
    name: &'static str,
    alive: Box<dyn Fn() -> bool>,
}
fn watch<T: 'static>(name: &'static str, value: &Arc<T>) -> Watch {
    let weak = Arc::downgrade(value);
    Watch {
        name,
        alive: Box::new(move || weak.upgrade().is_some()),
    }
}
impl Owners {
    fn watch(&self) -> Vec<Watch> {
        let mut watches = vec![
            watch("mesh", &self.mesh),
            watch("layout", self.mesh.layout()),
            watch("material instance", &self.instance),
            watch("material", &self.instance.parent),
            watch("shader", &self.instance.shader),
            watch("environment", &self.environment),
            watch("cubemap", &self.environment.sky.texture),
            watch("environment texture resolution", &self.environment.sky),
        ];
        for buffer in self.mesh.vertex_buffers() {
            watches.push(watch("vertex buffer", buffer));
        }
        if let Some(buffer) = self.mesh.index_buffer() {
            watches.push(watch("index buffer", buffer));
        }
        for (_, texture) in &self.instance.textures {
            watches.push(watch("texture resolution", texture));
            watches.push(watch("texture", &texture.texture));
            watches.push(watch("sampler", &texture.sampler));
        }
        for entry in &self.instance.props.entries {
            if let BoundResource::Buffer(buffer) = &entry.resource {
                watches.push(watch("material uniform buffer", buffer));
            }
        }
        watches
    }
}

#[test]
fn scene_unload_collect_reload_releases_resources_in_both_executors() {
    for multi in [false, true] {
        let mut fixture = Fixture::new(multi);
        for cycle in 0..3 {
            let scene = fixture.spawn_scene();
            let uploads_before = fixture.upload_graphs;
            let owners = fixture.load(&scene);
            assert!(
                fixture.upload_graphs > uploads_before,
                "cycle {cycle} loads new GPU resources"
            );
            let watches = owners.watch();
            drop(owners);
            fixture.unload(scene);
            assert_eq!(
                fixture.collect(),
                8,
                "one mesh/layout/instance/material/shader/environment and two textures"
            );
            for watch in watches {
                assert!(
                    !(watch.alive)(),
                    "cycle {cycle}: {} retained after collection",
                    watch.name
                );
            }
            assert_eq!(fixture.collect(), 0, "collection reaches a fixed point");
            assert!(
                fixture
                    .world
                    .resource::<MeshManager>()
                    .get(&MeshSource::File(fixture.mesh))
                    .is_none()
            );
            assert!(
                fixture
                    .world
                    .resource::<MaterialInstanceManager>()
                    .get(fixture.instance)
                    .is_none()
            );
        }
    }
}

#[test]
fn another_consumer_preserves_released_scene_assets_without_duplicate_uploads() {
    for multi in [false, true] {
        let mut fixture = Fixture::new(multi);
        let scene = fixture.spawn_scene();
        let preview = fixture.load(&scene);
        let watches = preview.watch();
        fixture.unload(scene);
        fixture.release();
        assert_eq!(fixture.collect(), 0);
        let uploads_before = fixture.upload_graphs;
        let scene = fixture.spawn_scene();
        let reloaded = fixture.load(&scene);
        assert!(Arc::ptr_eq(&preview.mesh, &reloaded.mesh));
        assert!(Arc::ptr_eq(&preview.instance, &reloaded.instance));
        assert!(Arc::ptr_eq(&preview.environment, &reloaded.environment));
        assert_eq!(
            fixture.upload_graphs, uploads_before,
            "live weak entries require no upload"
        );
        drop(reloaded);
        fixture.unload(scene);
        drop(preview);
        assert_eq!(fixture.collect(), 8);
        for watch in watches {
            assert!(
                !(watch.alive)(),
                "{} outlived its last consumer",
                watch.name
            );
        }
        assert_eq!(fixture.collect(), 0);
    }
}
