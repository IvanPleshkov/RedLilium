//! Persistent, host-owned engine state.

use std::sync::Arc;

use redlilium_ecs::sync::RwLock;

use redlilium_assets::{AssetDb, AssetProcessor};
use redlilium_ecs::{
    ChangedAssets, GameGenerationRegistry, MaterialAssetManager, MaterialInstanceLoader,
    MaterialInstanceManager, MaterialLoader, MeshLoader, MeshManager, PipelineCache, ShaderLoader,
    ShaderManager, ShadingRegistry, TextureLoader, TextureManager, VertexLayoutLoader,
    VertexLayoutManager, World,
};
use redlilium_graphics::GraphicsDevice;
use redlilium_vfs::Vfs;
// Native filesystem mounts vs. embedded in-memory mounts on wasm (no local disk
// in a browser; std-assets are embedded — see #33).
#[cfg(not(target_arch = "wasm32"))]
use redlilium_vfs::FileSystemProvider;

/// Engine state that outlives any single [`World`](redlilium_ecs::World):
/// the GPU device, the asset database + processor, and the GPU resource
/// managers (resident meshes/textures/materials, pipeline cache).
///
/// Every manager is stored as `Arc<RwLock<_>>` and injected into worlds via
/// [`World::insert_resource_shared`], so a world can be dropped and rebuilt
/// (warm-restart reload, Play-mode world swaps — ADR-020) without losing
/// resident GPU resources or re-scanning assets.
///
/// Also tracks game code generation IDs for cross-dylib reload safety: each
/// dylib load allocates a generation; when types re-register under a new generation,
/// snapshots from the old generation trigger migrations (Phase 4) during restore.
pub struct EngineContext {
    device: Arc<GraphicsDevice>,
    textures: Arc<RwLock<TextureManager>>,
    meshes: Arc<RwLock<MeshManager>>,
    vertex_layouts: Arc<RwLock<VertexLayoutManager>>,
    shaders: Arc<RwLock<ShaderManager>>,
    shading: Arc<RwLock<ShadingRegistry>>,
    materials: Arc<RwLock<MaterialAssetManager>>,
    material_instances: Arc<RwLock<MaterialInstanceManager>>,
    pipelines: Arc<RwLock<PipelineCache>>,
    changed_assets: Arc<RwLock<ChangedAssets>>,
    processor: Arc<RwLock<AssetProcessor>>,
    asset_db: Arc<RwLock<AssetDb>>,
    generation_registry: Arc<RwLock<GameGenerationRegistry>>,
}

impl EngineContext {
    /// Build the persistent engine state: mount the given local asset packs,
    /// load their `assets.db` files into one merged database, and create the
    /// GPU resource managers.
    ///
    /// On native, a relative mount dir is resolved against the executable's
    /// directory first and the working directory second (see
    /// [`resolve_mount_dir`]) — a dist folder runs from any cwd, while dev
    /// runs (`cargo run`) keep finding the packs at the workspace root.
    ///
    /// On wasm, a mount whose source dir has an entry in `embedded_packs`
    /// (`GameConfig::embedded_packs`, #108) is served from that in-memory
    /// table; otherwise the built-in `std-assets` embed (or an empty provider)
    /// is used. Native ignores `embedded_packs` — mounts come off the
    /// filesystem.
    ///
    /// Unlike the editor, the runtime does not scan mounts or persist the
    /// database — a shipped game consumes the committed `assets.db` as-is.
    pub fn new(
        device: Arc<GraphicsDevice>,
        mounts: &[(&'static str, &'static str)],
        embedded_packs: &[(&'static str, crate::EmbeddedPack)],
    ) -> Self {
        // Referenced only from the wasm branches below; native mounts come off
        // the filesystem.
        #[cfg_attr(not(target_arch = "wasm32"), allow(unused_variables))]
        let embedded = |dir: &str| -> Option<crate::EmbeddedPack> {
            embedded_packs
                .iter()
                .find(|&&(d, _)| d == dir)
                .map(|&(_, files)| files)
        };

        let mut vfs = Vfs::new();
        for &(name, dir) in mounts {
            #[cfg(not(target_arch = "wasm32"))]
            vfs.mount(name, FileSystemProvider::new(resolve_mount_dir(dir)));
            // Wasm has no local disk: the pack is compiled into the binary and
            // served from memory. An unknown mount (e.g. an empty project pack)
            // gets an empty provider.
            #[cfg(target_arch = "wasm32")]
            vfs.mount(
                name,
                match embedded(dir) {
                    Some(files) => crate::embedded_assets::provider_from(files),
                    None => crate::embedded_assets::provider_for(dir).unwrap_or_default(),
                },
            );
        }
        let ctx = Self::with_vfs(device, vfs);
        for &(name, dir) in mounts {
            #[cfg(target_arch = "wasm32")]
            if let Some(files) = embedded(dir) {
                match crate::embedded_assets::db_text_from(files) {
                    Some(text) => {
                        if let Err(e) = ctx.asset_db.write().merge_ron(name, &text) {
                            log::error!("failed to parse {name} assets.db: {e}");
                        }
                    }
                    None => log::warn!("embedded pack '{name}' has no assets.db ({dir})"),
                }
                continue;
            }
            #[cfg(not(target_arch = "wasm32"))]
            ctx.merge_db_file(name, &resolve_mount_dir(dir).join("assets.db"));
            #[cfg(target_arch = "wasm32")]
            ctx.load_mount_db(name, dir);
        }
        ctx
    }

    /// Like [`new`](Self::new), but over a caller-built [`Vfs`] and with an
    /// empty asset database — the editor uses this to keep ownership of its
    /// VFS (file watcher, browser) and to drive scanning itself. Populate the
    /// database via [`load_mount_db`](Self::load_mount_db) or directly through
    /// [`asset_db`](Self::asset_db).
    pub fn with_vfs(device: Arc<GraphicsDevice>, vfs: Vfs) -> Self {
        let processor = AssetProcessor::builder(vfs, device.clone())
            .with_loader::<MeshLoader>()
            .with_loader::<VertexLayoutLoader>()
            .with_loader::<ShaderLoader>()
            .with_loader::<MaterialLoader>()
            .with_loader::<MaterialInstanceLoader>()
            .with_loader::<TextureLoader>()
            .with_loader::<redlilium_ecs::SceneLoader>()
            .build();

        Self {
            textures: Arc::new(RwLock::new(TextureManager::new(device.clone()))),
            meshes: Arc::new(RwLock::new(MeshManager::new())),
            vertex_layouts: Arc::new(RwLock::new(VertexLayoutManager::new())),
            shaders: Arc::new(RwLock::new(ShaderManager::new())),
            shading: Arc::new(RwLock::new(ShadingRegistry::with_builtins())),
            materials: Arc::new(RwLock::new(MaterialAssetManager::new())),
            material_instances: Arc::new(RwLock::new(MaterialInstanceManager::new(device.clone()))),
            pipelines: Arc::new(RwLock::new(PipelineCache::new(device.clone()))),
            changed_assets: Arc::new(RwLock::new(ChangedAssets::new())),
            processor: Arc::new(RwLock::new(processor)),
            asset_db: Arc::new(RwLock::new(AssetDb::new())),
            generation_registry: Arc::new(RwLock::new(GameGenerationRegistry::new())),
            device,
        }
    }

    /// Merge the mount's `<dir>/assets.db` (if present) into the shared
    /// asset database. `dir` is taken as-is (cwd-relative if relative) — the
    /// exe-dir-first resolution applies only to [`new`](Self::new)'s mounts,
    /// not to callers that manage their own directories (the editor).
    pub fn load_mount_db(&self, mount: &str, dir: &str) {
        // Wasm has no local disk: read the DB from the embedded pack instead of
        // `std::fs` (and instead of the async VFS, so this stays synchronous).
        #[cfg(target_arch = "wasm32")]
        {
            match crate::embedded_assets::assets_db_text(dir) {
                Some(text) => {
                    if let Err(e) = self.asset_db.write().merge_ron(mount, &text) {
                        log::error!("failed to parse {mount} assets.db: {e}");
                    }
                }
                None => log::warn!("mount '{mount}' has no embedded assets.db ({dir})"),
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.merge_db_file(mount, &std::path::Path::new(dir).join("assets.db"));
    }

    /// Merge the `assets.db` at `path` into the shared asset database.
    #[cfg(not(target_arch = "wasm32"))]
    fn merge_db_file(&self, mount: &str, path: &std::path::Path) {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                if let Err(e) = self.asset_db.write().merge_ron(mount, &text) {
                    log::error!("failed to parse {mount} assets.db: {e}");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                log::warn!("mount '{mount}' has no assets.db ({})", path.display());
            }
            Err(e) => log::warn!("{mount} assets.db not readable: {e}"),
        }
    }

    /// The GPU device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// The shared asset processor (loaders + async stages).
    pub fn processor(&self) -> &Arc<RwLock<AssetProcessor>> {
        &self.processor
    }

    /// The shared asset database.
    pub fn asset_db(&self) -> &Arc<RwLock<AssetDb>> {
        &self.asset_db
    }

    /// The game generation registry (tracks dylib reload IDs for type-safe re-registration).
    pub fn generation_registry(&self) -> &Arc<RwLock<GameGenerationRegistry>> {
        &self.generation_registry
    }

    /// Insert every persistent manager into the world as a shared resource.
    ///
    /// The world and the `EngineContext` see the same underlying data and
    /// locks; dropping the world leaves the managers (and their resident GPU
    /// resources) intact.
    pub fn inject_into(&self, world: &mut World) {
        world.insert_resource_shared(self.textures.clone());
        world.insert_resource_shared(self.meshes.clone());
        world.insert_resource_shared(self.vertex_layouts.clone());
        world.insert_resource_shared(self.shaders.clone());
        world.insert_resource_shared(self.shading.clone());
        world.insert_resource_shared(self.materials.clone());
        world.insert_resource_shared(self.material_instances.clone());
        world.insert_resource_shared(self.pipelines.clone());
        world.insert_resource_shared(self.changed_assets.clone());
        world.insert_resource_shared(self.processor.clone());
        world.insert_resource_shared(self.asset_db.clone());
        world.insert_resource_shared(self.generation_registry.clone());
    }
}

/// Resolve a relative mount directory for a standalone game (#132): the
/// executable's directory wins, the working directory is the fallback.
///
/// Exe-first makes a dist folder self-contained — the game finds its packs no
/// matter where it was launched from (double-click included). Dev runs
/// (`cargo run -p car-game`) fall through to the cwd: `target/debug/` holds
/// no asset packs, and cargo runs from the workspace root where they live.
/// Absolute directories pass through untouched.
#[cfg(not(target_arch = "wasm32"))]
fn resolve_mount_dir(dir: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(dir);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        let candidate = exe_dir.join(path);
        if candidate.is_dir() {
            return candidate;
        }
        // macOS .app bundle (#133): the executable sits in `Contents/MacOS`,
        // the asset packs in `Contents/Resources` (data belongs in Resources
        // for codesigning), so also try the sibling Resources directory.
        #[cfg(target_os = "macos")]
        {
            let candidate = exe_dir.join("../Resources").join(path);
            if candidate.is_dir() {
                return candidate;
            }
        }
    }
    path.to_path_buf()
}
