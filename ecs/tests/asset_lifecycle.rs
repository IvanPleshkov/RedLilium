#![cfg(feature = "rendering")]
use redlilium_assets::*;
use redlilium_graphics::{BackendType, GraphicsDevice, GraphicsInstance, InstanceParameters};
use redlilium_vfs::{MemoryProvider, Vfs};
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Waker},
};
fn block<T>(f: impl Future<Output = T>) -> T {
    let mut f = std::pin::pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    match f.as_mut().poll(&mut cx) {
        Poll::Ready(t) => t,
        Poll::Pending => panic!("expected immediately ready"),
    }
}
fn device() -> Arc<GraphicsDevice> {
    GraphicsInstance::with_parameters(InstanceParameters::new().with_backend(BackendType::Dummy))
        .unwrap()
        .create_device()
        .unwrap()
}
fn processor() -> AssetProcessor {
    AssetProcessor::new(Vfs::new(), device())
}
fn pump(p: &mut AssetProcessor) {
    for (_, f) in p.drain_tasks() {
        block(f);
    }
    p.collect();
}

#[test]
fn environment_recovers_after_texture_is_fixed() {
    use redlilium_ecs::rendering::{
        EnvironmentManager, TextureManager,
        loaders::{EnvironmentData, EnvironmentSource, TextureSource},
    };
    let mut vfs = Vfs::new();
    vfs.mount("a", MemoryProvider::new());
    let dev = device();
    let mut p = AssetProcessor::new(vfs.clone(), dev.clone());
    let mut db = AssetDb::new();
    let t = db.register_path(AssetPath::new("a", "x.ktx2"), "texture", 0);
    let e = db.register_path(AssetPath::new("a", "x.env"), "environment", 0);
    db.set_settings(
        &e,
        Some(
            ron::to_string(&EnvironmentData {
                irradiance: t,
                prefilter: t,
                sky: t,
            })
            .unwrap(),
        ),
    );
    let source = EnvironmentSource { guid: e };
    let mut env = EnvironmentManager::new();
    let mut tex = TextureManager::new(dev);
    for _ in 0..8 {
        env.request(&source);
        env.drive(&mut p, &db, &mut tex);
        tex.drive(&mut p, &db);
        pump(&mut p);
        p.flush_gpu();
    }
    assert!(tex.is_failed(&TextureSource::File(t)));
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../std-assets/textures/ibl/irradiance_cube.ktx2"
    ))
    .unwrap();
    block(vfs.write("a/x.ktx2", bytes)).unwrap();
    tex.invalidate_file(t);
    for _ in 0..15 {
        env.request(&source);
        env.drive(&mut p, &db, &mut tex);
        tex.drive(&mut p, &db);
        pump(&mut p);
        p.flush_gpu();
    }
    assert!(
        tex.get(&TextureSource::File(t)).is_some(),
        "texture recovered"
    );
    let mut fresh = EnvironmentManager::new();
    for _ in 0..8 {
        fresh.request(&source);
        fresh.drive(&mut p, &db, &mut tex);
        pump(&mut p);
    }
    assert!(fresh.get(e).is_some(), "fresh resolver succeeds");
    assert!(
        env.get(e).is_some(),
        "dependency repair must resume the original environment"
    );
}

#[test]
fn layout_reload_never_confuses_reused_allocations() {
    use redlilium_core::mesh::VertexLayout;
    use redlilium_ecs::rendering::VertexLayoutManager;
    let mut p = processor();
    let mut db = AssetDb::new();
    let g = db.register_path(AssetPath::new("a", "x.vlayout"), "vertex_layout", 0);
    let layouts = [
        VertexLayout::position_only(),
        VertexLayout::position_normal(),
    ];
    let mut m = VertexLayoutManager::new();
    for l in &layouts {
        m.intern((**l).clone());
    }
    for i in 0..64 {
        let wanted = &layouts[i % 2];
        db.set_settings(&g, Some(ron::to_string(wanted.as_ref()).unwrap()));
        m.invalidate(g);
        m.get_or_request(&mut p, &db, g);
        pump(&mut p);
        let actual = m.get_or_request(&mut p, &db, g).unwrap();
        assert_eq!(actual.as_ref(), wanted.as_ref(), "reload {i}");
    }
}

#[derive(Clone, redlilium_ecs::Component)]
struct MeshHolder {
    mesh: redlilium_ecs::AssetRef<redlilium_ecs::MeshSource>,
}
#[test]
fn new_and_edited_asset_refs_load_without_manual_rescan() {
    use redlilium_ecs::*;
    for multi in [false, true] {
        let mut w = World::new();
        register_std_components(&mut w);
        register_rendering_components(&mut w);
        w.register_inspector::<MeshHolder>();
        w.insert_resource(MeshManager::new());
        w.insert_resource(VertexLayoutManager::new());
        w.insert_resource(processor());
        w.insert_resource(AssetDb::new());
        let mut systems = SystemsContainer::new();
        systems.add_exclusive(MeshLoad::default());
        let runner = if multi {
            EcsRunner::multi_thread(2)
        } else {
            EcsRunner::single_thread()
        };
        let tick = |w: &mut World| {
            assert!(runner.run(w, &systems).is_empty());
            let mut processor = w.resource_mut::<AssetProcessor>();
            pump(&mut processor);
            processor.flush_gpu(); // dummy backend: test resource resolution, not pixels
        };
        tick(&mut w);
        let entity = w
            .spawn_with((MeshHolder {
                mesh: AssetRef::new(MeshSource::Generated(MeshGenerator::cube(0.5))),
            },))
            .unwrap();
        for _ in 0..6 {
            tick(&mut w);
        }
        let first = w
            .get::<MeshHolder>(entity)
            .unwrap()
            .mesh
            .get()
            .cloned()
            .unwrap();
        assert!(w.resource::<AssetProcessor>().is_idle());
        w.get_mut::<MeshHolder>(entity).unwrap().mesh =
            AssetRef::new(MeshSource::Generated(MeshGenerator::quad(1.0, 1.0)));
        for _ in 0..6 {
            tick(&mut w);
        }
        let second = w
            .get::<MeshHolder>(entity)
            .unwrap()
            .mesh
            .get()
            .cloned()
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        // New references to an already resident resource must also resolve.
        let another = w
            .spawn_with((MeshHolder {
                mesh: AssetRef::new(MeshSource::Generated(MeshGenerator::quad(1.0, 1.0))),
            },))
            .unwrap();
        tick(&mut w);
        assert!(Arc::ptr_eq(
            w.get::<MeshHolder>(another).unwrap().mesh.get().unwrap(),
            &second
        ));
    }
}

#[test]
fn shader_repair_retries_failed_feature_validation() {
    use redlilium_ecs::rendering::{
        MaterialAssetManager, ShaderManager,
        loaders::{MaterialData, Shader},
        shading::{FeatureValue, ShadingRegistry},
    };
    let mut p = processor();
    let mut db = AssetDb::new();
    let registry = ShadingRegistry::with_builtins();
    let shader = registry.get("opaque").unwrap().shader;
    let material = db.register_path(AssetPath::new("a", "surface.material"), "material", 0);
    db.set_settings(
        &material,
        Some(
            ron::to_string(&MaterialData {
                shading_model: "opaque".into(),
                properties: vec![],
                features: vec![("DETAIL".into(), FeatureValue::Bool(true))],
            })
            .unwrap(),
        ),
    );
    let mut shaders = ShaderManager::new();
    shaders.publish(
        shader,
        Arc::new(Shader {
            source: b"// missing feature".to_vec(),
        }),
    );
    let mut materials = MaterialAssetManager::new();
    for _ in 0..4 {
        assert!(
            materials
                .get_or_request(&mut p, &db, &mut shaders, &registry, material)
                .is_none()
        );
        pump(&mut p);
    }
    assert!(p.is_idle());
    let repaired = Arc::new(Shader {
        source: b"//#pragma variant DETAIL\n".to_vec(),
    });
    shaders.publish(shader, repaired.clone());
    let material = materials
        .get_or_request(&mut p, &db, &mut shaders, &registry, material)
        .unwrap();
    assert!(Arc::ptr_eq(&material.shader, &repaired));
}

#[test]
fn material_instance_recovers_after_its_texture_dependency_is_repaired() {
    use redlilium_ecs::rendering::{
        MaterialAssetManager, MaterialInstanceManager, ShaderManager, TextureManager,
        loaders::{
            MaterialData, MaterialInstanceData, MaterialInstanceSource, Shader, TextureSource,
        },
        shading::{PropValue, ShadingRegistry},
    };
    let mut vfs = Vfs::new();
    vfs.mount("a", MemoryProvider::new());
    let dev = device();
    let mut p = AssetProcessor::new(vfs.clone(), dev.clone());
    let mut db = AssetDb::new();
    let registry = ShadingRegistry::with_builtins();
    let texture = db.register_path(AssetPath::new("a", "image.png"), "texture", 0);
    let material = db.register_path(AssetPath::new("a", "surface.material"), "material", 0);
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
    let mut shaders = ShaderManager::new();
    shaders.publish(
        registry.get("opaque_textured").unwrap().shader,
        Arc::new(Shader { source: vec![] }),
    );
    let mut materials = MaterialAssetManager::new();
    let mut textures = TextureManager::new(dev.clone());
    let mut instances = MaterialInstanceManager::new(dev);
    let instance = Guid::new();
    instances.publish_virtual(
        instance,
        MaterialInstanceData {
            parent: material,
            overrides: vec![],
        },
    );
    let tick = |p: &mut AssetProcessor,
                textures: &mut TextureManager,
                instances: &mut MaterialInstanceManager,
                materials: &mut MaterialAssetManager,
                shaders: &mut ShaderManager,
                db: &AssetDb| {
        textures.drive(p, db);
        instances.drive(p, db, materials, shaders, textures, &registry);
        pump(p);
        p.flush_gpu();
    };
    for _ in 0..12 {
        tick(
            &mut p,
            &mut textures,
            &mut instances,
            &mut materials,
            &mut shaders,
            &db,
        );
    }
    assert!(textures.is_failed(&TextureSource::File(texture)));
    assert!(instances.get(instance).is_none());
    let mut image_bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgba8(1, 1)
        .write_to(&mut image_bytes, image::ImageFormat::Png)
        .unwrap();
    block(vfs.write("a/image.png", image_bytes.into_inner())).unwrap();
    textures.invalidate_file(texture);
    for _ in 0..12 {
        tick(
            &mut p,
            &mut textures,
            &mut instances,
            &mut materials,
            &mut shaders,
            &db,
        );
    }
    assert!(instances.get(instance).is_some());
    // Request remains idempotent after recovery.
    let resolved = instances.get(instance).unwrap().clone();
    instances.request(&MaterialInstanceSource { guid: instance });
    tick(
        &mut p,
        &mut textures,
        &mut instances,
        &mut materials,
        &mut shaders,
        &db,
    );
    assert!(Arc::ptr_eq(instances.get(instance).unwrap(), &resolved));
    // A repaired parent can remove a failed texture dependency entirely. Its
    // old failure must not park the instance against the new parent version.
    let missing = db.register_path(AssetPath::new("a", "missing.png"), "texture", 0);
    for source in [TextureSource::File(missing), TextureSource::WHITE] {
        db.set_settings(
            &material,
            Some(
                ron::to_string(&MaterialData {
                    shading_model: "opaque_textured".into(),
                    features: vec![],
                    properties: vec![("base_texture".into(), PropValue::Texture(source.clone()))],
                })
                .unwrap(),
            ),
        );
        materials.invalidate(material);
        for _ in 0..12 {
            tick(
                &mut p,
                &mut textures,
                &mut instances,
                &mut materials,
                &mut shaders,
                &db,
            );
        }
        if source == TextureSource::File(missing) {
            assert!(textures.is_failed(&source));
            assert!(
                Arc::ptr_eq(instances.get(instance).unwrap(), &resolved),
                "last good stays visible"
            );
        }
    }
    assert!(
        !Arc::ptr_eq(instances.get(instance).unwrap(), &resolved),
        "new parent bypasses the old failed texture"
    );
}

#[test]
fn texture_settings_errors_and_request_snapshot() {
    use redlilium_ecs::rendering::{
        TextureManager,
        loaders::{TextureLoader, TextureSource},
    };
    use redlilium_graphics::{FilterMode, TextureFormat};
    let mut vfs = Vfs::new();
    vfs.mount("a", MemoryProvider::new());
    let mut png = std::io::Cursor::new(Vec::new());
    image::RgbaImage::from_pixel(1, 1, image::Rgba([255; 4]))
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    block(vfs.write("a/image.png", png.into_inner())).unwrap();
    let dev = device();
    let mut p = AssetProcessor::new(vfs, dev.clone());
    let mut db = AssetDb::new();
    let guid = db.register_path(AssetPath::new("a", "image.png"), "texture", 0);
    let source = TextureSource::File(guid);
    let mut textures = TextureManager::new(dev);

    db.set_settings(&guid, Some("(srg:false)".into()));
    // Direct loader callers and the manager both reject invalid settings.
    let handle = p.request::<TextureLoader>(&db, source.clone(), ());
    pump(&mut p);
    assert!(matches!(handle.get(), Some(Err(AssetError::Decode(_)))));
    textures.request(&source);
    textures.drive(&mut p, &db);
    assert!(textures.is_failed(&source));
    assert!(textures.get(&source).is_none());
    assert!(p.is_idle());

    db.set_settings(
        &guid,
        Some("(srgb:false,filter:Nearest,generate_mips:false)".into()),
    );
    textures.invalidate_file(guid);
    textures.request(&source);
    textures.drive(&mut p, &db);
    // A DB edit alone must not mix a new sampler with old decoded pixels.
    db.set_settings(
        &guid,
        Some("(srgb:true,filter:Linear,generate_mips:false)".into()),
    );
    for _ in 0..5 {
        pump(&mut p);
        p.flush_gpu();
        textures.drive(&mut p, &db);
    }
    let first = textures.get(&source).unwrap().clone();
    assert_eq!(first.texture.format(), TextureFormat::Rgba8Unorm);
    assert_eq!(first.sampler.descriptor().min_filter, FilterMode::Nearest);

    textures.invalidate_file(guid);
    textures.request(&source);
    for _ in 0..5 {
        textures.drive(&mut p, &db);
        pump(&mut p);
        p.flush_gpu();
    }
    let second = textures.get(&source).unwrap();
    assert_eq!(second.texture.format(), TextureFormat::Rgba8UnormSrgb);
    assert_eq!(second.sampler.descriptor().min_filter, FilterMode::Linear);
    assert!(!Arc::ptr_eq(&first, second));
}

#[test]
fn invalid_material_properties_never_publish_and_recover_after_repair() {
    use redlilium_ecs::rendering::{
        MaterialAssetManager, MaterialInstanceManager, ShaderManager, TextureManager,
        loaders::{MaterialData, MaterialInstanceData, MaterialInstanceSource, Shader},
        shading::{PropValue, ShadingRegistry},
    };
    let dev = device();
    let mut p = AssetProcessor::new(Vfs::new(), dev.clone());
    let mut db = AssetDb::new();
    let registry = ShadingRegistry::with_builtins();
    let mut shaders = ShaderManager::new();
    for name in ["opaque", "opaque_textured"] {
        shaders.publish(
            registry.get(name).unwrap().shader,
            Arc::new(Shader { source: vec![] }),
        );
    }
    let guid = db.register_path(AssetPath::new("a", "surface.material"), "material", 0);
    let mut materials = MaterialAssetManager::new();
    let mut textures = TextureManager::new(dev.clone());
    let mut instances = MaterialInstanceManager::new(dev);
    let instance = Guid::stable("runtime-instance");
    let mut data = MaterialData {
        shading_model: "opaque".into(),
        properties: vec![("base_color".into(), PropValue::Float(1.0))],
        features: vec![],
    };
    db.set_settings(&guid, Some(ron::to_string(&data).unwrap()));
    for _ in 0..4 {
        assert!(
            materials
                .get_or_request(&mut p, &db, &mut shaders, &registry, guid)
                .is_none()
        );
        pump(&mut p);
    }
    assert!(materials.get(guid).is_none());
    data.properties.clear();
    db.set_settings(&guid, Some(ron::to_string(&data).unwrap()));
    materials.invalidate(guid);

    // Runtime publication must validate overrides just like file-backed data.
    for overrides in [
        vec![("base_color".into(), PropValue::Float(1.0))],
        vec![("base_colour".into(), PropValue::Vec4([1.0; 4]))],
        vec![("base_color".into(), PropValue::Vec4([1.0; 4])); 2],
        vec![],
    ] {
        let valid = overrides.is_empty();
        instances.publish_virtual(
            instance,
            MaterialInstanceData {
                parent: guid,
                overrides,
            },
        );
        for _ in 0..4 {
            instances.drive(
                &mut p,
                &db,
                &mut materials,
                &mut shaders,
                &mut textures,
                &registry,
            );
            pump(&mut p);
        }
        assert_eq!(instances.get(instance).is_some(), valid);
    }
    let last_good = instances.get(instance).unwrap().clone();
    // An override invalid for the current model can become valid after a
    // parent edit. Preserve last-good while rejected and retry that edit.
    instances.publish_virtual(
        instance,
        MaterialInstanceData {
            parent: guid,
            overrides: vec![("cutout_params".into(), PropValue::Vec4([0.25; 4]))],
        },
    );
    for _ in 0..3 {
        instances.drive(
            &mut p,
            &db,
            &mut materials,
            &mut shaders,
            &mut textures,
            &registry,
        );
        assert!(Arc::ptr_eq(instances.get(instance).unwrap(), &last_good));
    }
    data.shading_model = "opaque_textured".into();
    db.set_settings(&guid, Some(ron::to_string(&data).unwrap()));
    materials.invalidate(guid);
    for _ in 0..10 {
        instances.drive(
            &mut p,
            &db,
            &mut materials,
            &mut shaders,
            &mut textures,
            &registry,
        );
        textures.drive(&mut p, &db);
        pump(&mut p);
        p.flush_gpu();
    }
    assert!(!Arc::ptr_eq(instances.get(instance).unwrap(), &last_good));

    // File-backed instance data follows the same validation path.
    let file = db.register_path(
        AssetPath::new("a", "surface.matinst"),
        "material_instance",
        0,
    );
    db.set_settings(
        &file,
        Some(
            ron::to_string(&MaterialInstanceData {
                parent: guid,
                overrides: vec![("base_color".into(), PropValue::Float(1.0))],
            })
            .unwrap(),
        ),
    );
    instances.request(&MaterialInstanceSource { guid: file });
    for _ in 0..4 {
        instances.drive(
            &mut p,
            &db,
            &mut materials,
            &mut shaders,
            &mut textures,
            &registry,
        );
        pump(&mut p);
    }
    assert!(instances.get(file).is_none());
}
