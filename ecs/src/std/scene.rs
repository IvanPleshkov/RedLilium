//! Scene assets (#101): a serialized world persisted as a `.scene` RON file
//! and loaded back through the asset system.
//!
//! A scene is a [`SerializedWorld`] — the same name-keyed snapshot format that
//! powers play-mode restore and warm reload — stored as RON in an asset mount
//! and registered in the asset DB under kind `"scene"`. The pipeline is
//! IO (read) → CPU (parse); there is no GPU stage, the resident asset is plain
//! data. Instantiation into a live [`World`] is a separate, synchronous step
//! ([`instantiate_scene`]) so the caller decides *when* the entities appear
//! (the scene manager, #102, applies it at a defined point in the frame).
//!
//! Identity follows the current asset model (`docs/ASSETS.md` §7): the guid is
//! [`Guid::stable`] of the mount-relative path, so [`request_scene`] can
//! resolve a scene by name without consulting the DB first.

use redlilium_assets::{
    AnyAsset, AssetError, AssetHandle, AssetLoader, AssetProcessor, AssetSource, AssetStage,
    Executor, Guid, LoadEnv, StageFuture,
};

use crate::World;
use crate::serialize::{DeserializeError, SerializeError, SerializedWorld};

/// Identity of a scene asset: a DB record resolved from `guid`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SceneSource {
    pub guid: Guid,
}

impl AssetSource for SceneSource {
    fn file_guid(&self) -> Option<Guid> {
        Some(self.guid)
    }
}

impl From<Guid> for SceneSource {
    fn from(guid: Guid) -> Self {
        Self { guid }
    }
}

/// The resident scene asset: a parsed [`SerializedWorld`] ready to
/// instantiate any number of times.
pub struct SceneData(pub SerializedWorld);

/// Loads a `.scene` RON file into a [`SceneData`].
pub struct SceneLoader;

impl AssetLoader for SceneLoader {
    const NAME: &'static str = "scene";
    const EXTENSIONS: &'static [&'static str] = &["scene"];
    type Source = SceneSource;
    type Asset = SceneData;
    type Deps = ();

    fn pipeline(_source: &SceneSource, _deps: &(), env: &LoadEnv) -> Vec<Box<dyn AssetStage>> {
        vec![
            Box::new(ReadSceneStage {
                path: env.path.clone(),
                vfs: env.vfs.clone(),
            }),
            Box::new(ParseSceneStage),
        ]
    }
}

/// IO stage: read the scene file bytes from the VFS.
struct ReadSceneStage {
    path: Option<redlilium_assets::AssetPath>,
    vfs: redlilium_vfs::Vfs,
}

impl AssetStage for ReadSceneStage {
    fn executor(&self) -> Executor {
        Executor::Io
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let path = self.path.clone();
        let vfs = self.vfs.clone();
        Box::pin(async move {
            let path =
                path.ok_or_else(|| AssetError::Io("scene: source has no file path".into()))?;
            let raw = format!("{}/{}", path.mount, path.path);
            let bytes = vfs
                .read(&raw)
                .await
                .map_err(|e| AssetError::Io(e.to_string()))?;
            Ok(Box::new(bytes) as AnyAsset)
        })
    }
}

/// CPU stage: parse the RON text into a [`SceneData`].
struct ParseSceneStage;

impl AssetStage for ParseSceneStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, input: AnyAsset) -> StageFuture {
        Box::pin(async move {
            let bytes = input
                .downcast::<Vec<u8>>()
                .map_err(|_| AssetError::Decode("scene: expected file bytes".into()))?;
            let text = std::str::from_utf8(&bytes)
                .map_err(|e| AssetError::Decode(format!("scene: utf8: {e}")))?;
            let world: SerializedWorld =
                ron::from_str(text).map_err(|e| AssetError::Decode(format!("scene: ron: {e}")))?;
            Ok(Box::new(SceneData(world)) as AnyAsset)
        })
    }
}

/// Request a scene by its mount-relative path (e.g. `"scenes/level1.scene"`).
///
/// The guid is derived with [`Guid::stable`], matching how the mount scan
/// assigns identity. The returned handle resolves once [`AssetPump`]
/// (`crate::AssetPump`) has driven the IO/CPU stages — poll it from a system
/// and then [`instantiate_scene`].
pub fn request_scene(
    processor: &mut AssetProcessor,
    db: &redlilium_assets::AssetDb,
    name: &str,
) -> AssetHandle<SceneData> {
    processor.request::<SceneLoader>(
        db,
        SceneSource {
            guid: Guid::stable(name),
        },
        (),
    )
}

/// Spawn a loaded scene's entities into `world` (name-keyed component data,
/// entity references remapped). The scene data is not consumed — a scene can
/// be instantiated repeatedly. Returns the spawned entities, so the caller
/// can tag them with their owning scene (the #102 unload contract).
pub fn instantiate_scene(
    world: &mut World,
    scene: &SceneData,
) -> Result<Vec<crate::Entity>, DeserializeError> {
    world.deserialize_world_into(&scene.0)
}

/// Serialize `world` into scene RON — the bytes a `.scene` asset stores.
/// Until the editor grows scene persistence (#2) this is how scene files are
/// authored (tests, tooling).
pub fn serialize_scene_ron(world: &World) -> Result<String, SerializeError> {
    let scene = world.serialize_world()?;
    ron::ser::to_string_pretty(&scene, ron::ser::PrettyConfig::default())
        .map_err(|e| SerializeError::FormatError(format!("scene: ron: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Transform;
    use redlilium_assets::{AssetDb, AssetPath, AssetRecord};
    use redlilium_core::math::Vec3;
    use redlilium_vfs::{MemoryProvider, Vfs};

    /// Minimal single-future executor: the scene pipeline's stages are pure
    /// async (memory VFS + RON parse), no reactor needed.
    fn block_on<F: std::future::Future + ?Sized>(mut fut: std::pin::Pin<Box<F>>) -> F::Output {
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        fn noop_raw() -> RawWaker {
            const VTABLE: RawWakerVTable =
                RawWakerVTable::new(|_| noop_raw(), |_| {}, |_| {}, |_| {});
            RawWaker::new(std::ptr::null(), &VTABLE)
        }
        let waker = unsafe { Waker::from_raw(noop_raw()) };
        let mut cx = Context::from_waker(&waker);
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    /// Drive the processor to quiescence on the current thread (the test
    /// stand-in for `AssetPump` + the IO/compute pools).
    fn pump_to_completion(processor: &mut AssetProcessor) {
        for _ in 0..16 {
            let tasks = processor.drain_tasks();
            if tasks.is_empty() {
                processor.collect();
                continue;
            }
            for (_executor, fut) in tasks {
                block_on(fut);
            }
            processor.collect();
        }
    }

    /// The #101 acceptance path, headless: author a world → serialize to
    /// scene RON → register in a memory mount + DB → request by name → pump →
    /// instantiate into a fresh world with the entities intact.
    #[test]
    fn scene_roundtrips_through_the_asset_system() {
        // Author: a world with two placed entities.
        let mut authored = World::new();
        crate::register_std_components(&mut authored);
        for i in 0..2 {
            let e = authored.spawn();
            authored
                .insert(
                    e,
                    Transform::from_translation(Vec3::new(i as f32, 2.0, -3.0)),
                )
                .unwrap();
        }
        let ron_text = serialize_scene_ron(&authored).expect("serialize scene");

        // Publish: memory mount + DB record under the stable-path guid.
        let scene_path = "scenes/test.scene";
        let provider = MemoryProvider::new();
        provider.insert(scene_path, ron_text.into_bytes());
        let mut vfs = Vfs::new();
        vfs.mount("game", provider);

        let mut db = AssetDb::new();
        db.insert(
            Guid::stable(scene_path),
            AssetRecord {
                path: AssetPath::new("game", scene_path),
                kind: SceneLoader::NAME.to_string(),
                source_hash: 0,
                settings: None,
                references: Default::default(),
            },
        )
        .expect("db insert");

        let instance = redlilium_graphics::GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("device");
        let mut processor = AssetProcessor::builder(vfs, device)
            .with_loader::<SceneLoader>()
            .build();

        // Load by name.
        let handle = request_scene(&mut processor, &db, scene_path);
        pump_to_completion(&mut processor);
        let scene = handle
            .get()
            .expect("scene load finished")
            .expect("scene load succeeded");

        // Instantiate into a fresh world.
        let mut world = World::new();
        crate::register_std_components(&mut world);
        instantiate_scene(&mut world, &scene).expect("instantiate");

        let transforms = world.read_all::<Transform>().unwrap();
        let mut xs: Vec<f32> = transforms.iter().map(|(_, t)| t.translation.x).collect();
        xs.sort_by(f32::total_cmp);
        assert_eq!(xs, vec![0.0, 1.0], "both authored entities restored");
    }

    /// A missing file fails the load with an IO error instead of hanging.
    #[test]
    fn missing_scene_file_fails_loudly() {
        let mut vfs = Vfs::new();
        vfs.mount("game", MemoryProvider::new());
        let mut db = AssetDb::new();
        db.insert(
            Guid::stable("scenes/absent.scene"),
            AssetRecord {
                path: AssetPath::new("game", "scenes/absent.scene"),
                kind: SceneLoader::NAME.to_string(),
                source_hash: 0,
                settings: None,
                references: Default::default(),
            },
        )
        .expect("db insert");

        let instance = redlilium_graphics::GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("device");
        let mut processor = AssetProcessor::builder(vfs, device)
            .with_loader::<SceneLoader>()
            .build();

        let handle = request_scene(&mut processor, &db, "scenes/absent.scene");
        pump_to_completion(&mut processor);
        assert!(
            matches!(handle.get(), Some(Err(_))),
            "missing file must resolve to an error"
        );
    }
}
