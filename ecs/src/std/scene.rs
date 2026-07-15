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

/// Capture `world`'s *scene content* as a [`SerializedWorld`] — what a
/// `.scene` asset stores (#2). Two deliberate differences from the raw
/// [`World::serialize_world`] snapshot:
///
/// - **Editor entities are excluded** (`Entity::EDITOR` /
///   `Entity::INHERITED_EDITOR` flags — the editor camera and friends are
///   editor machinery, not scene content). References from scene entities to
///   an excluded entity resolve to `Entity::DANGLING` on instantiation.
/// - **Snapshot resources are dropped**: a scene is entity content; restoring
///   resource state into an arbitrary host world (editor, running game) is a
///   play-mode/reload concern, not a scene's.
pub fn scene_from_world(world: &World) -> Result<SerializedWorld, SerializeError> {
    let mut scene = world.serialize_world()?;
    scene.entities.retain(|e| {
        e.entity_flags & (crate::Entity::EDITOR | crate::Entity::INHERITED_EDITOR) == 0
    });
    scene.resources.clear();
    Ok(scene)
}

/// The RON of an *empty* scene — the initial content of a newly created
/// `.scene` asset (#112). A freshly created scene must be valid from byte
/// one (the loader rejects a zero-length file), and an empty world is the
/// canonical blank canvas.
pub fn empty_scene_ron() -> String {
    serialize_scene_ron(&World::new()).expect("an empty world always serializes")
}

/// Serialize `world`'s scene content ([`scene_from_world`]) into scene RON —
/// the bytes a `.scene` asset stores. Used by the editor's scene save (#2)
/// and by offline tooling (`gen_scenes`, tests).
pub fn serialize_scene_ron(world: &World) -> Result<String, SerializeError> {
    let scene = scene_from_world(world)?;
    ron::ser::to_string_pretty(&scene, ron::ser::PrettyConfig::default())
        .map_err(|e| SerializeError::FormatError(format!("scene: ron: {e}")))
}

/// Bump the GPU asset managers' generations so the next `MeshLoad` re-scans
/// all asset refs. Call after instantiating entities *out of band* (scene
/// swaps, editor scene load): the new entities' refs may point at assets that
/// are already resident (or never requested), so no manager generation moves
/// on its own and the gated scan would skip them forever — invisible content.
/// Residents are reused; only missing pieces load.
pub fn rescan_asset_managers(world: &mut World) {
    if world.has_resource::<crate::MeshManager>() {
        world.resource_mut::<crate::MeshManager>().request_rescan();
    }
    if world.has_resource::<crate::MaterialInstanceManager>() {
        world
            .resource_mut::<crate::MaterialInstanceManager>()
            .request_rescan();
    }
    if world.has_resource::<crate::TextureManager>() {
        world
            .resource_mut::<crate::TextureManager>()
            .request_rescan();
    }
}

// ---- Scene manager (#102) ----

/// Tags an entity as belonging to a loaded scene, so unload can despawn
/// exactly the entities that scene spawned. Entities the game spawns itself
/// (`Plugin::spawn_scene`, gameplay) carry no tag and survive scene switches.
#[derive(Clone, crate::Component)]
pub struct SceneMember {
    /// The owning scene's asset path (the name it was requested under).
    pub scene: String,
}

/// World resource orchestrating scene transitions (#102).
///
/// Game systems/UI call [`switch_to`](Self::switch_to) at any point in the
/// frame; the actual unload/instantiate happens in [`ApplySceneTransitions`]
/// (an exclusive `PreUpdate` system) — never mid-schedule, so systems never
/// observe a half-swapped world.
#[derive(Default)]
pub struct SceneManager {
    /// The scene currently instantiated, by asset path.
    current: Option<String>,
    /// Transition requested this frame (latest request wins).
    pending: Option<String>,
    /// The load in flight: requested path + its handle.
    in_flight: Option<(String, AssetHandle<SceneData>)>,
}

impl SceneManager {
    /// Request a transition to the scene at `path` (mount-relative, e.g.
    /// `"scenes/level1.scene"`). Applied at the start of a following frame,
    /// once the scene asset has loaded. A newer request supersedes a pending
    /// one; requesting the current scene reloads it fresh.
    pub fn switch_to(&mut self, path: impl Into<String>) {
        self.pending = Some(path.into());
    }

    /// The currently instantiated scene's asset path, if any.
    pub fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }

    /// The scene requested but not yet picked up by [`ApplySceneTransitions`],
    /// if any.
    pub fn pending(&self) -> Option<&str> {
        self.pending.as_deref()
    }

    /// Whether a transition is requested or a scene load is in flight.
    pub fn transitioning(&self) -> bool {
        self.pending.is_some() || self.in_flight.is_some()
    }
}

/// Exclusive system applying [`SceneManager`] transitions at a defined point
/// in the frame (install in `PreUpdate`): kicks off the asset request for a
/// pending switch, and once the load resolves, despawns the previous scene's
/// [`SceneMember`]s and instantiates the new scene, tagging what it spawned.
/// No-op without the `SceneManager` / asset resources.
pub struct ApplySceneTransitions;

impl crate::ExclusiveSystem for ApplySceneTransitions {
    type Result = ();

    fn run(&mut self, world: &mut World) -> Result<(), crate::system::SystemError> {
        if !world.has_resource::<SceneManager>()
            || !world.has_resource::<AssetProcessor>()
            || !world.has_resource::<redlilium_assets::AssetDb>()
        {
            return Ok(());
        }

        // 1. Kick off the load for a pending request (superseding an
        //    in-flight one: dropping its handle cancels the old load).
        let pending = world.resource_mut::<SceneManager>().pending.take();
        if let Some(path) = pending {
            let handle = {
                let db = world.resource::<redlilium_assets::AssetDb>();
                let mut processor = world.resource_mut::<AssetProcessor>();
                request_scene(&mut processor, &db, &path)
            };
            world.resource_mut::<SceneManager>().in_flight = Some((path, handle));
        }

        // 2. Poll the in-flight load; apply the swap when it resolves.
        let resolved = {
            let manager = world.resource::<SceneManager>();
            match &manager.in_flight {
                Some((path, handle)) => handle.get().map(|result| (path.clone(), result)),
                None => None,
            }
        };
        let Some((path, result)) = resolved else {
            return Ok(());
        };
        world.resource_mut::<SceneManager>().in_flight = None;

        let scene = match result {
            Ok(scene) => scene,
            Err(e) => {
                log::error!("scene transition to '{path}' failed: {e}");
                return Ok(());
            }
        };

        // Unload: exactly the previous scene's members.
        let stale: Vec<crate::Entity> = match world.read_all::<SceneMember>() {
            Ok(members) => members
                .iter()
                .filter_map(|(idx, _)| world.entity_at_index(idx))
                .collect(),
            Err(_) => Vec::new(),
        };
        for entity in stale {
            world.despawn(entity);
        }

        // Instantiate + tag.
        match instantiate_scene(world, &scene) {
            Ok(spawned) => {
                log::info!("scene '{path}' loaded ({} entities)", spawned.len());
                for entity in spawned {
                    let _ = world.insert(
                        entity,
                        SceneMember {
                            scene: path.clone(),
                        },
                    );
                }
                world.resource_mut::<SceneManager>().current = Some(path);
                rescan_asset_managers(world);
            }
            Err(e) => {
                log::error!("scene '{path}' failed to instantiate: {e}");
            }
        }
        Ok(())
    }
}

// ---- Editor scene load (#2) ----

/// Undoable "open scene" for the editor: replace the world's scene content
/// (every non-editor entity) with a loaded scene. Editor entities (camera)
/// and all resources are untouched; apply captures the previous content via
/// [`scene_from_world`], so undo restores it exactly. Push through the
/// `ActionQueue<World>` like every other edit.
pub struct ReplaceSceneAction {
    /// The scene to instantiate.
    scene: SerializedWorld,
    /// Human-readable description ("Open scene 'x'"), built once.
    description: String,
    /// Non-editor content captured by `apply`, restored by `undo`.
    previous: Option<SerializedWorld>,
}

impl ReplaceSceneAction {
    /// `label` names the scene in the undo history (usually its asset path).
    pub fn new(scene: SerializedWorld, label: &str) -> Self {
        Self {
            scene,
            description: format!("Open scene '{label}'"),
            previous: None,
        }
    }
}

impl std::fmt::Debug for ReplaceSceneAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplaceSceneAction")
            .field("description", &self.description)
            .field("entities", &self.scene.entities.len())
            .field("has_backup", &self.previous.is_some())
            .finish()
    }
}

/// Despawn every non-editor entity and instantiate `scene` in its place.
/// The shared forward/undo direction of [`ReplaceSceneAction`].
fn swap_scene_content(
    world: &mut World,
    scene: &SerializedWorld,
) -> redlilium_core::abstract_editor::EditActionResult {
    use redlilium_core::abstract_editor::EditActionError;

    let stale: Vec<crate::Entity> = world
        .iter_entities()
        .filter(|&e| {
            world.get_entity_flags(e) & (crate::Entity::EDITOR | crate::Entity::INHERITED_EDITOR)
                == 0
        })
        .collect();
    for entity in stale {
        world.despawn(entity);
    }
    world
        .deserialize_world_into(scene)
        .map_err(|e| EditActionError::Custom(e.to_string()))?;
    rescan_asset_managers(world);
    Ok(())
}

impl redlilium_core::abstract_editor::EditAction<World> for ReplaceSceneAction {
    fn apply(&mut self, world: &mut World) -> redlilium_core::abstract_editor::EditActionResult {
        self.previous = Some(scene_from_world(world).map_err(|e| {
            redlilium_core::abstract_editor::EditActionError::Custom(e.to_string())
        })?);
        swap_scene_content(world, &self.scene)
    }

    fn undo(&mut self, world: &mut World) -> redlilium_core::abstract_editor::EditActionResult {
        let previous = self.previous.as_ref().ok_or_else(|| {
            redlilium_core::abstract_editor::EditActionError::Custom("no backup to restore".into())
        })?;
        swap_scene_content(world, previous)
    }

    fn description(&self) -> &str {
        &self.description
    }
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

    /// Repeated scene switches (#102): menu → level → menu → level. Each
    /// swap must despawn exactly the previous scene's members — no leaked
    /// entities, and untagged (game-owned) entities survive every switch.
    #[test]
    fn scene_manager_switches_scenes_without_leaks() {
        use crate::system::run_exclusive_system_once;

        // Author two distinguishable scenes.
        let scene_ron = |x: f32, count: usize| {
            let mut w = World::new();
            crate::register_std_components(&mut w);
            for i in 0..count {
                let e = w.spawn();
                w.insert(e, Transform::from_translation(Vec3::new(x, i as f32, 0.0)))
                    .unwrap();
            }
            serialize_scene_ron(&w).expect("serialize")
        };
        let provider = MemoryProvider::new();
        provider.insert("scenes/menu.scene", scene_ron(1.0, 1).into_bytes());
        provider.insert("scenes/level.scene", scene_ron(2.0, 3).into_bytes());
        let mut vfs = Vfs::new();
        vfs.mount("game", provider);

        let mut db = AssetDb::new();
        for path in ["scenes/menu.scene", "scenes/level.scene"] {
            db.insert(
                Guid::stable(path),
                AssetRecord {
                    path: AssetPath::new("game", path),
                    kind: SceneLoader::NAME.to_string(),
                    source_hash: 0,
                    settings: None,
                    references: Default::default(),
                },
            )
            .expect("db insert");
        }

        let instance = redlilium_graphics::GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("device");
        let processor = AssetProcessor::builder(vfs, device)
            .with_loader::<SceneLoader>()
            .build();

        let mut world = World::new();
        crate::register_std_components(&mut world);
        world.register_inspector::<SceneMember>();
        world.insert_resource(db);
        world.insert_resource(processor);
        world.insert_resource(SceneManager::default());

        // A game-owned entity that must survive every switch.
        let persistent = world.spawn();
        world
            .insert(persistent, Transform::from_translation(Vec3::zeros()))
            .unwrap();

        let mut apply = ApplySceneTransitions;
        let mut switch_and_settle = |world: &mut World, path: &str| {
            world.resource_mut::<SceneManager>().switch_to(path);
            // Frame ticks: request → pump (the AssetPump stand-in) → apply.
            for _ in 0..8 {
                run_exclusive_system_once(&mut apply, world).unwrap();
                let mut processor = world.resource_mut::<AssetProcessor>();
                pump_to_completion(&mut processor);
            }
        };

        let members_of = |world: &World, scene: &str| -> usize {
            world
                .read_all::<SceneMember>()
                .map(|m| m.iter().filter(|(_, s)| s.scene == scene).count())
                .unwrap_or(0)
        };
        let total_entities = |world: &World| world.iter_entities().count();

        // A resident-asset manager: its generation must bump on every scene
        // swap so MeshLoad's rescan gate re-resolves the fresh entities'
        // asset refs (they may point at already-resident assets).
        world.insert_resource(crate::MeshManager::new());
        let gen_before = world.resource::<crate::MeshManager>().generation();

        switch_and_settle(&mut world, "scenes/menu.scene");
        assert_eq!(members_of(&world, "scenes/menu.scene"), 1);
        assert_eq!(
            world.resource::<SceneManager>().current(),
            Some("scenes/menu.scene")
        );
        assert_ne!(
            world.resource::<crate::MeshManager>().generation(),
            gen_before,
            "scene swap must force an asset-ref rescan (manager generation bump)"
        );
        let baseline = total_entities(&world); // persistent + 1 member

        switch_and_settle(&mut world, "scenes/level.scene");
        assert_eq!(members_of(&world, "scenes/menu.scene"), 0, "menu unloaded");
        assert_eq!(members_of(&world, "scenes/level.scene"), 3);
        assert_eq!(total_entities(&world), baseline + 2, "1 member -> 3");

        switch_and_settle(&mut world, "scenes/menu.scene");
        switch_and_settle(&mut world, "scenes/level.scene");
        assert_eq!(
            total_entities(&world),
            baseline + 2,
            "repeated switches must not leak entities"
        );
        assert!(
            world.is_alive(persistent),
            "game-owned entity survives scene switches"
        );
        assert!(!world.resource::<SceneManager>().transitioning());
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

    /// #112: a freshly created scene asset must be valid RON that parses and
    /// instantiates to an empty world — never a zero-byte file the loader
    /// would reject.
    #[test]
    fn empty_scene_is_valid_and_instantiates_empty() {
        let ron_text = empty_scene_ron();
        let scene: SerializedWorld = ron::from_str(&ron_text).expect("empty scene parses");
        assert!(scene.entities.is_empty());

        let mut world = World::new();
        crate::register_std_components(&mut world);
        let spawned = world
            .deserialize_world_into(&scene)
            .expect("empty scene instantiates");
        assert!(spawned.is_empty(), "an empty scene spawns nothing");
    }

    /// #2: a scene asset stores *scene content* — editor machinery (the
    /// editor camera and its children) and snapshot resources must not leak
    /// into the file.
    #[test]
    fn scene_save_excludes_editor_entities_and_resources() {
        let mut world = World::new();
        crate::register_std_components(&mut world);
        for i in 0..2 {
            let e = world.spawn();
            world
                .insert(
                    e,
                    Transform::from_translation(Vec3::new(i as f32, 0.0, 0.0)),
                )
                .unwrap();
        }
        let camera = world.spawn();
        world
            .insert(
                camera,
                Transform::from_translation(Vec3::new(99.0, 0.0, 0.0)),
            )
            .unwrap();
        let child = world.spawn();
        crate::std::hierarchy::set_parent(&mut world, child, camera);
        crate::mark_editor(&mut world, camera);

        let scene = scene_from_world(&world).expect("serialize scene");
        assert_eq!(
            scene.entities.len(),
            2,
            "editor camera and its child are not scene content"
        );
        assert!(scene.resources.is_empty(), "scenes carry no resource state");
        assert!(
            scene.entities.iter().all(|e| e.entity_flags == 0),
            "saved entities carry no editor flags"
        );
    }

    /// #2: `ReplaceSceneAction` swaps the world's scene content, leaves
    /// editor entities alone, and restores the previous content on undo.
    #[test]
    fn replace_scene_action_swaps_and_undoes() {
        use redlilium_core::abstract_editor::EditAction;

        // The scene to open: one entity at x = 9.
        let scene = {
            let mut w = World::new();
            crate::register_std_components(&mut w);
            let e = w.spawn();
            w.insert(e, Transform::from_translation(Vec3::new(9.0, 0.0, 0.0)))
                .unwrap();
            scene_from_world(&w).expect("serialize scene")
        };

        // The editor world: two content entities + the editor camera.
        let mut world = World::new();
        crate::register_std_components(&mut world);
        for i in 1..=2 {
            let e = world.spawn();
            world
                .insert(
                    e,
                    Transform::from_translation(Vec3::new(i as f32, 0.0, 0.0)),
                )
                .unwrap();
        }
        let camera = world.spawn();
        world
            .insert(
                camera,
                Transform::from_translation(Vec3::new(99.0, 0.0, 0.0)),
            )
            .unwrap();
        crate::mark_editor(&mut world, camera);

        let content_xs = |world: &World| -> Vec<f32> {
            let mut xs: Vec<f32> = world
                .read_all::<Transform>()
                .unwrap()
                .iter()
                .filter(|(idx, _)| {
                    world.entity_at_index(*idx).is_some_and(|e| {
                        world.get_entity_flags(e)
                            & (crate::Entity::EDITOR | crate::Entity::INHERITED_EDITOR)
                            == 0
                    })
                })
                .map(|(_, t)| t.translation.x)
                .collect();
            xs.sort_by(f32::total_cmp);
            xs
        };

        let mut action = ReplaceSceneAction::new(scene, "scenes/test.scene");
        action.apply(&mut world).expect("apply");
        assert_eq!(content_xs(&world), vec![9.0], "scene content replaced");
        assert!(world.is_alive(camera), "editor camera survives the swap");

        action.undo(&mut world).expect("undo");
        assert_eq!(
            content_xs(&world),
            vec![1.0, 2.0],
            "undo restores the previous content"
        );
        assert!(world.is_alive(camera), "editor camera survives undo");

        action.apply(&mut world).expect("redo");
        assert_eq!(content_xs(&world), vec![9.0], "redo swaps again");
    }
}
