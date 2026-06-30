//! Asset-based material-instance resolver.
//!
//! `MaterialInstanceManager` is the consumer facade a [`Primitive`] binds: you
//! `request` an instance by [`MaterialInstanceSource`] and get an
//! [`InstanceHandle`] resolving to a [`ResolvedInstance`] once loaded. Resolution
//! (in [`drive`](Self::drive)) loads the instance data (parent guid + overrides),
//! pulls the parent template via the [`MaterialAssetManager`], overlays the
//! overrides onto the parent's schema-ordered properties, and builds the **static**
//! material-property binding (group 1 — a uniform buffer uploaded once through the
//! frame graph, `docs/MATERIAL_ASSETS.md` Decision 7). The pipeline itself is
//! specialized later at draw time from the carried shader + the mesh's layout.
//!
//! This is the single requester per instance guid. The buffer upload follows the
//! GPU-upload-through-frame-graph rule: allocate now, queue a `TransferOperation`,
//! and flush it into the render graph via [`flush_uploads`](Self::flush_uploads).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;
use redlilium_assets::{AssetDb, AssetHandle, AssetProcessor, Guid};
use redlilium_graphics::{
    BindingGroup, BufferDescriptor, BufferUsage, GraphicsDevice, GraphicsError, RenderGraph,
    TransferConfig, TransferOperation, TransferPass,
};

use super::{MaterialAssetManager, ShaderManager};
use crate::std::rendering::loaders::{
    MaterialInstanceData, MaterialInstanceLoader, MaterialInstanceSource, Shader,
};
use crate::std::rendering::shading::{PropValue, pack_props};

/// A fully resolved material instance: the shader to specialize a pipeline from
/// (carried from the parent template) and the static property binding (group 1).
#[derive(Debug)]
pub struct ResolvedInstance {
    /// The shader source asset guid (the pipeline cache key).
    pub shader_guid: Guid,
    /// The resident shader source (from the parent template).
    pub shader: Arc<Shader>,
    /// The static material-property binding group (group 1).
    pub props_group: Arc<BindingGroup>,
}

/// A demand-to-load handle for a resolved material instance. Cloneable and cheap;
/// poll [`get`](Self::get) for the [`ResolvedInstance`] once it has resolved.
#[derive(Clone, Default, Debug)]
pub struct InstanceHandle {
    slot: Arc<RwLock<Option<Arc<ResolvedInstance>>>>,
}

impl InstanceHandle {
    /// The resolved instance, if it has finished resolving.
    pub fn get(&self) -> Option<Arc<ResolvedInstance>> {
        self.slot.read().clone()
    }

    fn fulfill(&self, instance: Arc<ResolvedInstance>) {
        *self.slot.write() = Some(instance);
    }
}

/// In-flight instance request: the handle to fulfil, plus the instance-data
/// request and (once landed) the loaded data while its parent resolves.
struct PendingInstance {
    handle: InstanceHandle,
    data: Option<AssetHandle<MaterialInstanceData>>,
    data_loaded: Option<Arc<MaterialInstanceData>>,
}

/// Owns and shares resolved material instances (an ECS resource).
pub struct MaterialInstanceManager {
    device: Arc<GraphicsDevice>,
    resident: HashMap<Guid, Arc<ResolvedInstance>>,
    pending: HashMap<Guid, PendingInstance>,
    failed: HashSet<Guid>,
    pending_uploads: Vec<TransferOperation>,
}

impl MaterialInstanceManager {
    /// Create an instance manager for the given device.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            resident: HashMap::new(),
            pending: HashMap::new(),
            failed: HashSet::new(),
            pending_uploads: Vec::new(),
        }
    }

    /// Request an instance by source, returning a handle that resolves once ready.
    /// Deduplicates by guid: repeat requests share the handle, a resident instance
    /// resolves immediately.
    pub fn request(&mut self, source: MaterialInstanceSource) -> InstanceHandle {
        let guid = source.guid;
        if let Some(instance) = self.resident.get(&guid) {
            let handle = InstanceHandle::default();
            handle.fulfill(instance.clone());
            return handle;
        }
        if let Some(pending) = self.pending.get(&guid) {
            return pending.handle.clone();
        }
        let handle = InstanceHandle::default();
        self.pending.insert(
            guid,
            PendingInstance {
                handle: handle.clone(),
                data: None,
                data_loaded: None,
            },
        );
        handle
    }

    /// The resolved instance for `guid` if already loaded — no request side effect.
    pub fn get(&self, guid: Guid) -> Option<Arc<ResolvedInstance>> {
        self.resident.get(&guid).cloned()
    }

    /// Advance all in-flight instance requests: load each one's data, resolve its
    /// parent template (via the material manager, which itself pulls the shader),
    /// overlay overrides, and build the static property binding. Call from the
    /// instance-load system, which co-locks the processor / DB / managers.
    pub fn drive(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        material_mgr: &mut MaterialAssetManager,
        shader_mgr: &mut ShaderManager,
        registry: &crate::std::rendering::shading::ShadingRegistry,
    ) {
        // Phase 1+2 produce, for each ready instance, the packed property bytes and
        // the carried shader. The static buffer (needs `&mut self`) is built after
        // the `pending` borrow is released.
        let mut ready: Vec<(Guid, Vec<u8>, Guid, Arc<Shader>)> = Vec::new();
        let mut failed_now: Vec<Guid> = Vec::new();

        for (guid, pending) in self.pending.iter_mut() {
            // Phase 1: instance data (request once, then poll).
            if pending.data_loaded.is_none() {
                match &pending.data {
                    None => {
                        pending.data = Some(processor.request::<MaterialInstanceLoader>(
                            db,
                            MaterialInstanceSource { guid: *guid },
                            (),
                        ));
                        continue;
                    }
                    Some(handle) => match handle.get() {
                        None => continue,
                        Some(Ok(data)) => {
                            pending.data_loaded = Some(data);
                            pending.data = None;
                        }
                        Some(Err(e)) => {
                            log::warn!("material instance {guid:?} data failed to load: {e}");
                            failed_now.push(*guid);
                            continue;
                        }
                    },
                }
            }

            // Phase 2: resolve the parent template (pulls its shader).
            let data = pending.data_loaded.as_ref().expect("data_loaded set above");
            let Some(parent) =
                material_mgr.get_or_request(processor, db, shader_mgr, registry, data.parent)
            else {
                continue; // parent (or its shader) still loading
            };

            let merged = merge_overrides(&parent.properties, &data.overrides);
            ready.push((
                *guid,
                pack_props(&merged),
                parent.shader_guid,
                parent.shader.clone(),
            ));
        }

        // Build static buffers + fulfil (the `pending` borrow is now released).
        for (guid, bytes, shader_guid, shader) in ready {
            match self.build_props_group(&bytes) {
                Ok(props_group) => {
                    let resolved = Arc::new(ResolvedInstance {
                        shader_guid,
                        shader,
                        props_group,
                    });
                    if let Some(pending) = self.pending.remove(&guid) {
                        pending.handle.fulfill(resolved.clone());
                    }
                    self.resident.insert(guid, resolved);
                }
                Err(e) => {
                    log::warn!("material instance {guid:?}: props buffer failed: {e}");
                    failed_now.push(guid);
                }
            }
        }

        for guid in failed_now {
            self.pending.remove(&guid);
            self.failed.insert(guid);
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
    /// property bytes, allocated now and uploaded through the frame graph on the
    /// next [`flush_uploads`](Self::flush_uploads).
    fn build_props_group(&mut self, bytes: &[u8]) -> Result<Arc<BindingGroup>, GraphicsError> {
        if bytes.is_empty() {
            return Ok(Arc::new(BindingGroup::new()));
        }
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
        Ok(Arc::new(BindingGroup::new().with_buffer(0, buffer)))
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
