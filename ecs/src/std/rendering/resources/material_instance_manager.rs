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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use redlilium_assets::{AssetDb, AssetManager, AssetProcessor, Guid, ResidentCache};
use redlilium_graphics::{
    BindingGroup, BindingGroupDescriptor, BindingLayout, Buffer, BufferDescriptor, BufferUsage,
    GraphicsDevice, GraphicsError, RenderGraph, TransferConfig, TransferOperation, TransferPass,
};

use super::{MaterialAssetManager, ShaderManager, TextureManager};
use crate::std::rendering::loaders::{
    MaterialInstanceData, MaterialInstanceLoader, MaterialInstanceSource, Shader, TextureSource,
};
use crate::std::rendering::shading::{
    OpaqueBinding, PropValue, StorageBufferSource, opaque_bindings, pack_props,
};

/// A resolved opaque material-property binding, in schema order — what
/// [`build_props_descriptor`](MaterialInstanceManager::build_props_descriptor)
/// walks to assign descriptor slots after the packed uniform at binding 0. A
/// texture takes two slots (texture, sampler); a buffer takes one.
enum ResolvedBinding {
    /// A resolved texture (plain or D2Array) + its sampler.
    Texture(Arc<super::ResolvedTexture>),
    /// Inline read-only bytes to upload into a fresh `STORAGE` buffer.
    InlineBuffer(Arc<[u8]>),
    /// A buffer published at runtime (a `StorageBufferSource::Ref`).
    Buffer(Arc<Buffer>),
}

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
    /// The material half of the shader variant key, carried from the parent
    /// template (#6, Decision 5). Per-instance feature overrides are a
    /// follow-up — Unreal allows static-switch overrides on instances; we
    /// start with template-owned features.
    pub variant: redlilium_graphics::VariantKey,
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
    /// Externally-produced GPU buffers bound by `StorageBufferSource::Ref`
    /// properties, published under a guid (a compute pass output, an ECS-owned
    /// buffer). A `Ref` property whose buffer is not yet here keeps the instance
    /// unresolved, exactly like a still-loading texture.
    buffers: HashMap<Guid, Arc<Buffer>>,
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
            buffers: HashMap::new(),
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

    /// Publish a programmatically built instance record under `guid` — the
    /// host/tool integration point (mirrors `MeshManager::insert_external` and
    /// `TextureManager::publish_virtual`; e.g. a preview viewer synthesizing
    /// per-material instances from streamed scene data). Seeds the DATA phase
    /// directly instead of loading a DB record; resolution then flows through
    /// [`drive`](Self::drive) normally — parent template, texture props, static
    /// binding, hot reload. Re-publishing replaces the record and re-resolves
    /// LAST-GOOD-SERVING: the resident resolution keeps drawing until the new
    /// one republishes over it (same semantics as hot-reload pull-validation) —
    /// invalidating here instead would blank every consumer for the frames the
    /// rebuild takes (a preview re-publishing per gizmo-drag tick flickers).
    pub fn publish_virtual(&mut self, guid: Guid, data: MaterialInstanceData) {
        self.data.publish(guid, Arc::new(data));
        self.demanded.insert(guid);
    }

    /// Publish an externally-produced GPU buffer under `guid`, to be bound by
    /// `StorageBufferSource::Ref(guid)` material properties (a compute pass
    /// output, an ECS-owned buffer). The buffer must be created with
    /// [`BufferUsage::STORAGE`]. Instances waiting on this ref resolve on the
    /// next [`drive`](Self::drive). Re-publishing swaps the buffer; already
    /// resolved instances keep their old `Arc` until re-resolved — call
    /// [`invalidate`](Self::invalidate) on them to rebind (external buffers
    /// carry no version, so the swap is not auto-detected).
    pub fn publish_buffer(&mut self, guid: Guid, buffer: Arc<Buffer>) {
        self.buffers.insert(guid, buffer);
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
        // is built after this pass, so collect the ready ones first. Each ready
        // entry carries the opaque bindings **in schema order** (the descriptor
        // slot order) plus the texture subset for hot-reload pull-validation.
        #[allow(clippy::type_complexity)]
        let mut ready: Vec<(
            Guid,
            Vec<u8>,
            Guid,
            Arc<super::ResolvedMaterial>,
            Vec<ResolvedBinding>,
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

            // Phase 3: resolve the opaque properties (schema order — the
            // descriptor-slot order). All must be resident before the group is
            // built: textures via the texture manager, `Ref` storage buffers via
            // the published-buffer registry (both keep the instance unresolved
            // while pending). Inline storage buffers need no resolution — their
            // bytes upload at build time.
            let merged = merge_overrides(&parent.properties, &data.overrides);
            let mut bindings = Vec::new();
            let mut textures = Vec::new();
            for (name, binding) in opaque_bindings(&merged) {
                match binding {
                    OpaqueBinding::Texture(source) => {
                        if texture_mgr.is_failed(&source) {
                            log::warn!(
                                "material instance {guid:?}: texture '{name}' failed to load"
                            );
                            failed_now.push(guid);
                            continue 'demanded;
                        }
                        match texture_mgr.get(&source) {
                            Some(texture) => {
                                bindings.push(ResolvedBinding::Texture(texture.clone()));
                                textures.push((source, texture.clone()));
                            }
                            None => {
                                texture_mgr.request(&source);
                                continue 'demanded; // still loading
                            }
                        }
                    }
                    OpaqueBinding::StorageBuffer(StorageBufferSource::Inline(bytes)) => {
                        bindings.push(ResolvedBinding::InlineBuffer(Arc::from(bytes)));
                    }
                    OpaqueBinding::StorageBuffer(StorageBufferSource::Ref(buffer_guid)) => {
                        match self.buffers.get(&buffer_guid) {
                            Some(buffer) => {
                                bindings.push(ResolvedBinding::Buffer(buffer.clone()));
                            }
                            // Not published yet — wait (stay demanded), exactly
                            // like a still-loading texture. `publish_buffer`
                            // makes it resolve on a later drive.
                            None => continue 'demanded,
                        }
                    }
                }
            }
            ready.push((
                guid,
                pack_props(&merged),
                data.parent,
                parent,
                bindings,
                textures,
            ));
        }

        // Build static buffers + publish.
        for (guid, bytes, parent_guid, parent, bindings, textures) in ready {
            match self.build_props_descriptor(&bytes, &bindings) {
                Ok(props) => {
                    let resolved = Arc::new(ResolvedInstance {
                        parent_guid,
                        shader_guid: parent.shader_guid,
                        shader: parent.shader.clone(),
                        variant: parent.variant.clone(),
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

    /// Build the static material-set binding: a uniform buffer holding the
    /// packed property bytes at binding 0 (allocated now, uploaded through the
    /// frame graph on the next [`flush_uploads`](Self::flush_uploads)), then the
    /// opaque bindings in schema order at consecutive slots. A texture takes a
    /// texture + sampler pair (its own resolved sampler); a storage buffer takes
    /// one buffer slot — an inline one allocated + uploaded here, a `Ref` one
    /// bound directly. The slot order matches the shader's `MaterialParams`
    /// field order (`opaque_slot_layout`).
    fn build_props_descriptor(
        &mut self,
        bytes: &[u8],
        bindings: &[ResolvedBinding],
    ) -> Result<BindingGroupDescriptor, GraphicsError> {
        let mut group = BindingGroupDescriptor::new();
        // Binding 0 is the packed uniform constant buffer when the model has
        // uniform props; opaque slots follow. (Every built-in model has at least
        // one uniform prop, so `next` starts at 1 in practice.)
        let mut next = 0u32;
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
            next = 1;
        }
        for binding in bindings {
            match binding {
                ResolvedBinding::Texture(resolved) => {
                    group = group
                        .with_texture(next, resolved.texture.clone())
                        .with_sampler(next + 1, resolved.sampler.clone());
                    next += 2;
                }
                ResolvedBinding::InlineBuffer(data) => {
                    let buffer = self.device.create_buffer(
                        &BufferDescriptor::new(
                            data.len() as u64,
                            BufferUsage::STORAGE | BufferUsage::COPY_DST,
                        )
                        .with_label("material_storage"),
                    )?;
                    self.pending_uploads.push(TransferOperation::write_buffer(
                        Arc::clone(&buffer),
                        0,
                        Arc::clone(data),
                    ));
                    group = group.with_buffer(next, buffer);
                    next += 1;
                }
                ResolvedBinding::Buffer(buffer) => {
                    group = group.with_buffer(next, buffer.clone());
                    next += 1;
                }
            }
        }
        Ok(group)
    }

    /// Force MeshLoad to re-scan all asset refs without reloading anything.
    /// See `MeshManager::request_rescan` — rescan without reloading.
    pub(crate) fn request_rescan(&mut self) {
        self.cache.bump_generation();
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
