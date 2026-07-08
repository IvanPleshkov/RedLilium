//! Asset-based material-instance resolver.
//!
//! `MaterialInstanceManager` is the consumer facade a [`Primitive`] binds (via an
//! `AssetRef<MaterialInstanceSource>`) and the single source of truth for
//! resolved instances. Resolution (in [`drive`](Self::drive)) loads the instance
//! data (via an embedded [`AssetManager`](redlilium_assets::AssetManager) — the
//! data phase), pulls the parent template via the [`MaterialAssetManager`],
//! overlays the overrides onto the parent's schema-ordered properties, and
//! builds the **static** material-property binding (group 1 — a uniform buffer
//! uploaded once through the frame graph, `docs/MATERIAL_ASSETS.md` Decision 7).
//! The pipeline itself is specialized later at draw time from the carried shader
//! + the mesh's layout.
//!
//! Hot reload: `drive` pull-validates every resident instance's parent `Arc`
//! (a re-resolved parent rebuilds the instance, serving last-good meanwhile),
//! and [`invalidate`](Self::invalidate) drops the resolution + data so an edited
//! record reloads from fresh settings.
//!
//! The buffer upload follows the GPU-upload-through-frame-graph rule: allocate
//! now, queue a `TransferOperation`, and flush it into the render graph via
//! [`flush_uploads`](Self::flush_uploads).

use std::collections::HashSet;
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetManager, AssetProcessor, Guid, ResidentCache};
use redlilium_graphics::{
    BindingGroup, BindingGroupDescriptor, BindingLayout, BufferDescriptor, BufferUsage,
    GraphicsDevice, GraphicsError, RenderGraph, TransferConfig, TransferOperation, TransferPass,
};

use super::{MaterialAssetManager, ShaderManager, TextureManager};
use crate::PlayModeAware;
use crate::std::rendering::loaders::{
    MaterialInstanceLoader, MaterialInstanceSource, Shader, TextureSource,
};
use crate::std::rendering::shading::{PropValue, pack_props, texture_props};

/// A fully resolved material instance: the shader to specialize a pipeline from
/// (carried from the parent template) and the static property binding (group 1).
/// The parent template it was resolved against is retained so hot reload can
/// pull-validate: a re-resolved parent (`Arc` mismatch) triggers a rebuild.
#[derive(Debug)]
pub struct ResolvedInstance {
    /// The parent material template guid.
    pub parent_guid: Guid,
    /// The parent resolution this instance was built from (pointer identity is
    /// the version — compared against the material manager's current one).
    pub parent: Arc<super::ResolvedMaterial>,
    /// The shader source asset guid (the pipeline cache key).
    pub shader_guid: Guid,
    /// The resident shader source (from the parent template).
    pub shader: Arc<Shader>,
    /// The resolved texture properties this instance was built with (schema
    /// order — the binding order). Retained so hot reload can pull-validate: a
    /// re-resolved texture (`Arc` mismatch) triggers a rebuild.
    pub textures: Vec<(TextureSource, Arc<super::ResolvedTexture>)>,
    /// Description of the static material-property binding (group 1): the packed
    /// uniform buffer at binding 0, then texture/sampler pairs per texture
    /// property. The compiled [`BindingGroup`] is built lazily against the
    /// material's reflected group-1 layout (only available once the pipeline is
    /// specialized at draw time) and cached in [`props_group`](Self::props_group).
    pub props: BindingGroupDescriptor,
    /// The eagerly-compiled group, materialized once from [`props`](Self::props)
    /// against the pipeline's group-1 layout. Interior-mutable so it can be built
    /// on first draw; tied to this instance's lifetime (rebuilt with the instance
    /// on reload).
    props_group: std::sync::Mutex<Option<Arc<BindingGroup>>>,
}

impl ResolvedInstance {
    /// Get (materializing on first call) the compiled static property binding
    /// group, built against `layout` — the material's reflected group-1 layout.
    pub fn props_group(
        &self,
        device: &Arc<GraphicsDevice>,
        layout: Arc<BindingLayout>,
    ) -> Result<Arc<BindingGroup>, GraphicsError> {
        let mut slot = self.props_group.lock().expect("props_group mutex poisoned");
        if let Some(group) = &*slot {
            return Ok(group.clone());
        }
        let group = device.create_binding_group(layout, self.props.clone())?;
        *slot = Some(group.clone());
        Ok(group)
    }
}

// A component-side `AssetRef<MaterialInstanceSource>` resolves to the manager's
// product — the fully resolved instance, not the loader's raw data.
impl redlilium_assets::AssetRefSource for MaterialInstanceSource {
    type Asset = ResolvedInstance;
    const KIND: &'static str = "material_instance";
}

/// Owns and shares resolved material instances (an ECS resource).
pub struct MaterialInstanceManager {
    device: Arc<GraphicsDevice>,
    /// The data phase: guid → resident [`MaterialInstanceData`].
    data: AssetManager<MaterialInstanceLoader>,
    /// The resolution: guid → the single shared [`ResolvedInstance`].
    cache: ResidentCache<Guid, ResolvedInstance>,
    /// Instances being (re)resolved by `drive` — demanded but not yet published
    /// (or republished after a parent change).
    demanded: HashSet<Guid>,
    pending_uploads: Vec<TransferOperation>,
}

impl MaterialInstanceManager {
    /// Create an instance manager for the given device.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            data: AssetManager::new(),
            cache: ResidentCache::new(),
            demanded: HashSet::new(),
            pending_uploads: Vec::new(),
        }
    }

    /// Ensure the instance is resolving (idempotent; no-op if resident, in
    /// flight, or failed). The sync system calls this for unresolved refs.
    pub fn request(&mut self, source: &MaterialInstanceSource) {
        let guid = source.guid;
        if self.cache.get(&guid).is_some() || self.cache.is_failed(&guid) {
            return;
        }
        self.demanded.insert(guid);
    }

    /// The resolved instance for `guid`, if resolved.
    pub fn get(&self, guid: Guid) -> Option<&Arc<ResolvedInstance>> {
        self.cache.get(&guid)
    }

    /// Bumped whenever the resident set changes (load / reload).
    pub fn generation(&self) -> u64 {
        self.cache.generation()
    }

    /// Drop all state for `guid` — the resolution *and* the data — so it
    /// re-resolves from fresh record settings (hot reload). Component refs keep
    /// serving the old `Arc` until the demand-driven sync re-requests and the
    /// new resolution lands.
    pub fn invalidate(&mut self, guid: Guid) {
        self.cache.invalidate(&guid);
        self.data.invalidate(guid);
        self.demanded.remove(&guid);
    }

    /// Advance all demanded instances: load the data, resolve the parent
    /// template (via the material manager, which itself pulls the shader),
    /// overlay overrides, resolve the texture properties (via the texture
    /// manager), build the static property binding, publish. Also
    /// pull-validates resident instances against their parent and textures
    /// (hot reload). Call from the instance-load system, which co-locks the
    /// managers.
    pub fn drive(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        material_mgr: &mut MaterialAssetManager,
        shader_mgr: &mut ShaderManager,
        texture_mgr: &mut TextureManager,
        registry: &crate::std::rendering::shading::ShadingRegistry,
    ) {
        // Pull-validation (hot reload): re-resolve resident instances whose
        // parent template or any bound texture has been re-resolved (`Arc`
        // mismatch — pointer identity is the version). While the input itself
        // reloads (`None`) the old instance keeps serving (last-good); the
        // rebuild goes through the normal demanded flow and republishes over
        // the resident entry.
        for (guid, resolved) in self.cache.iter() {
            if self.demanded.contains(guid) {
                continue;
            }
            let parent = material_mgr.get_or_request(
                processor,
                db,
                shader_mgr,
                registry,
                resolved.parent_guid,
            );
            if let Some(parent) = parent
                && !Arc::ptr_eq(&parent, &resolved.parent)
            {
                self.demanded.insert(*guid);
                continue;
            }
            for (source, texture) in &resolved.textures {
                match texture_mgr.get(source) {
                    // A different Arc landed — rebuild the instance with it.
                    Some(current) if !Arc::ptr_eq(current, texture) => {
                        self.demanded.insert(*guid);
                        break;
                    }
                    Some(_) => {}
                    // Invalidated (hot reload) — re-request the reload; we keep
                    // serving the old texture until the new Arc lands above.
                    None => texture_mgr.request(source),
                }
            }
        }

        // Resolve the demanded instances; the static buffer (needs `&mut self`)
        // is built after this pass, so collect the ready ones first.
        #[allow(clippy::type_complexity)]
        let mut ready: Vec<(
            Guid,
            Vec<u8>,
            Guid,
            Arc<super::ResolvedMaterial>,
            Vec<(TextureSource, Arc<super::ResolvedTexture>)>,
        )> = Vec::new();
        let mut failed_now: Vec<Guid> = Vec::new();

        'demanded: for guid in self.demanded.iter().copied() {
            // Phase 1: the instance data (request once, then poll).
            let Some(data) = self.data.get_or_request(processor, db, guid) else {
                if self.data.is_failed(guid) {
                    failed_now.push(guid);
                }
                continue;
            };

            // Phase 2: resolve the parent template (pulls its shader).
            let Some(parent) =
                material_mgr.get_or_request(processor, db, shader_mgr, registry, data.parent)
            else {
                continue; // parent (or its shader) still loading
            };

            // Phase 3: resolve the texture properties (schema order — the
            // binding order). All must be resident before the group is built.
            let merged = merge_overrides(&parent.properties, &data.overrides);
            let mut textures = Vec::new();
            for (name, source) in texture_props(&merged) {
                if texture_mgr.is_failed(&source) {
                    log::warn!("material instance {guid:?}: texture '{name}' failed to load");
                    failed_now.push(guid);
                    continue 'demanded;
                }
                match texture_mgr.get(&source) {
                    Some(texture) => textures.push((source, texture.clone())),
                    None => {
                        texture_mgr.request(&source);
                        continue 'demanded; // still loading
                    }
                }
            }
            ready.push((guid, pack_props(&merged), data.parent, parent, textures));
        }

        // Build static buffers + publish.
        for (guid, bytes, parent_guid, parent, textures) in ready {
            match self.build_props_descriptor(&bytes, &textures) {
                Ok(props) => {
                    let resolved = Arc::new(ResolvedInstance {
                        parent_guid,
                        shader_guid: parent.shader_guid,
                        shader: parent.shader.clone(),
                        parent,
                        textures,
                        props,
                        props_group: std::sync::Mutex::new(None),
                    });
                    self.demanded.remove(&guid);
                    self.cache.publish(guid, resolved);
                }
                Err(e) => {
                    log::warn!("material instance {guid:?}: props binding failed: {e}");
                    failed_now.push(guid);
                }
            }
        }

        for guid in failed_now {
            self.demanded.remove(&guid);
            self.cache.fail(guid);
        }
    }

    /// Flush queued material-property uploads into the current frame's render
    /// graph. Call once per frame while building the frame graph, before the
    /// forward pass that reads the buffers.
    pub fn flush_uploads(&mut self, graph: &mut RenderGraph) {
        if self.pending_uploads.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.pending_uploads);
        let mut pass = TransferPass::new("material_instance_uploads".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(ops));
        graph.add_transfer_pass(pass);
    }

    /// Build the static group-1 binding: a uniform buffer holding the packed
    /// property bytes at binding 0 (allocated now, uploaded through the frame
    /// graph on the next [`flush_uploads`](Self::flush_uploads)), then a
    /// texture/sampler pair per texture property in schema order (texture at
    /// `1 + 2*i`, sampler at `2 + 2*i` — the convention the model's shader
    /// declares). The sampler is the texture's own resolved one (record
    /// settings, interned by the texture manager).
    fn build_props_descriptor(
        &mut self,
        bytes: &[u8],
        textures: &[(TextureSource, Arc<super::ResolvedTexture>)],
    ) -> Result<BindingGroupDescriptor, GraphicsError> {
        let mut group = BindingGroupDescriptor::new();
        if !bytes.is_empty() {
            let buffer = self.device.create_buffer(
                &BufferDescriptor::new(
                    bytes.len() as u64,
                    BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                )
                .with_label("material_props"),
            )?;
            self.pending_uploads.push(TransferOperation::write_buffer(
                Arc::clone(&buffer),
                0,
                Arc::from(bytes),
            ));
            group = group.with_buffer(0, buffer);
        }
        for (i, (_, resolved)) in textures.iter().enumerate() {
            let base = 1 + 2 * i as u32;
            group = group
                .with_texture(base, resolved.texture.clone())
                .with_sampler(base + 1, resolved.sampler.clone());
        }
        Ok(group)
    }

    /// Force MeshLoad to re-scan all asset refs by bumping generation.
    /// Called on snapshot restore to ensure unresolved refs get re-requested.
    pub(crate) fn force_rescan(&mut self) {
        // Invalidate all RESIDENT material instances (the actual resolved instances)
        // so the generation bumps. This forces MeshLoad to re-scan all asset refs
        // even in steady state (when nothing is demanded/in-flight). On re-scan,
        // unresolved refs get re-requested and re-resolved.
        // We collect keys first because invalidate mutates the cache.
        let guids: Vec<_> = self.cache.iter().map(|(k, _)| *k).collect();
        for guid in guids {
            self.cache.invalidate(&guid);
        }
    }
}

impl PlayModeAware for MaterialInstanceManager {
    fn on_stop(&mut self) {
        self.force_rescan();
    }
}

/// Overlay instance overrides onto the parent's schema-ordered property list:
/// each slot keeps the parent's value unless the instance overrides it by name.
fn merge_overrides(
    parent: &[(String, PropValue)],
    overrides: &[(String, PropValue)],
) -> Vec<(String, PropValue)> {
    parent
        .iter()
        .map(|(name, base)| {
            let value = overrides
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| base.clone());
            (name.clone(), value)
        })
        .collect()
}
