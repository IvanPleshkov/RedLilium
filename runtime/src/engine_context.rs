//! Persistent, host-owned engine state.

use std::sync::Arc;

use parking_lot::RwLock;

use redlilium_assets::{AssetDb, AssetProcessor};
use redlilium_ecs::{
    ChangedAssets, MaterialAssetManager, MaterialInstanceLoader, MaterialInstanceManager,
    MaterialLoader, MeshLoader, MeshManager, PipelineCache, ShaderLoader, ShaderManager,
    ShadingRegistry, TextureLoader, TextureManager, VertexLayoutLoader, VertexLayoutManager, World,
};
use redlilium_graphics::GraphicsDevice;
use redlilium_vfs::{FileSystemProvider, Vfs};

/// Engine state that outlives any single [`World`](redlilium_ecs::World):
/// the GPU device, the asset database + processor, and the GPU resource
/// managers (resident meshes/textures/materials, pipeline cache).
///
/// Every manager is stored as `Arc<RwLock<_>>` and injected into worlds via
/// [`World::insert_resource_shared`], so a world can be dropped and rebuilt
/// (warm-restart reload, Play-mode world swaps — ADR-020) without losing
/// resident GPU resources or re-scanning assets.
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
}

impl EngineContext {
    /// Build the persistent engine state: mount the given local asset packs,
    /// load their `assets.db` files into one merged database, and create the
    /// GPU resource managers.
    ///
    /// Unlike the editor, the runtime does not scan mounts or persist the
    /// database — a shipped game consumes the committed `assets.db` as-is.
    pub fn new(device: Arc<GraphicsDevice>, mounts: &[(&'static str, &'static str)]) -> Self {
        let mut vfs = Vfs::new();
        for &(name, dir) in mounts {
            vfs.mount(name, FileSystemProvider::new(dir));
        }
        let ctx = Self::with_vfs(device, vfs);
        for &(name, dir) in mounts {
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
            device,
        }
    }

    /// Merge the mount's `<dir>/assets.db` (if present) into the shared
    /// asset database.
    pub fn load_mount_db(&self, mount: &str, dir: &str) {
        match std::fs::read_to_string(format!("{dir}/assets.db")) {
            Ok(text) => {
                if let Err(e) = self.asset_db.write().merge_ron(mount, &text) {
                    log::error!("failed to parse {mount} assets.db: {e}");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                log::warn!("mount '{mount}' has no assets.db ({dir})");
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
    }
}
