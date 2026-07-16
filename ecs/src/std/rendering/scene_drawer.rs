//! Shared scene walk for camera render pipelines (ADR-035, #128).
//!
//! Every pipeline that draws the scene's meshes goes through the same two
//! steps, split so the expensive part runs once per frame:
//!
//! 1. [`VisibleScene::gather`] — walk the world's visible
//!    [`MeshRenderer`]s **once per frame**, push each entity's
//!    [`ModelUniforms`](shaders::ModelUniforms) into the [`FrameRing`], and
//!    snapshot the resolved (mesh, material) pairs. Auxiliary views (a shadow
//!    pass, a mirror) replay this list instead of re-walking the World.
//! 2. [`SceneDrawer::record`] — per view, emit one draw per primitive into a
//!    [`GraphicsPass`]: specialize the pipeline via the [`PipelineCache`],
//!    assemble the rate-classified binding groups, and attach the per-draw
//!    dynamic offsets.
//!
//! # Binding groups are created eagerly (issue #40)
//!
//! A `BindingGroup` is a device resource whose GPU descriptor set is built
//! once at creation, so steady-state frames must issue **zero**
//! `create_binding_group` calls. The camera (external) and model (dynamic)
//! sets bind the frame ring buffer at offset 0 and select the per-view /
//! per-entity slot with a per-draw dynamic offset — so the group is
//! frame-invariant and cached here by the material's set-layout identity. The
//! empty (unclassified) set is likewise one cached group per layout. The
//! static material-property set is cached inside its
//! [`ResolvedInstance`](super::ResolvedInstance). After warm-up every set is
//! a cache hit.

use std::collections::HashMap;
use std::sync::Arc;

use redlilium_core::math::{Mat4, mat4_to_cols_array_2d};
use redlilium_graphics::{
    BindingGroup, BindingGroupDescriptor, BindingLayout, Buffer, DrawCommand, GraphicsDevice,
    GraphicsError, GraphicsPass, Mesh, TextureFormat,
};

use crate::World;
use crate::std::components::{GlobalTransform, Visibility};

use super::{FrameRing, MeshRenderer, PipelineCache, ResolvedInstance, shaders};

/// The frame's visible renderables, gathered once per frame by the render
/// dispatcher and replayed by every recorded view — see the module docs.
pub struct VisibleScene {
    items: Vec<SceneItem>,
}

/// One visible entity's renderable state for this frame.
struct SceneItem {
    /// Ring offset of the entity's [`ModelUniforms`](shaders::ModelUniforms)
    /// (pushed once per frame; every view selects it via a per-draw dynamic
    /// offset).
    model_offset: u32,
    /// The resolved (mesh, material instance) pairs. Primitives whose asset
    /// refs have not resolved yet are skipped for this frame.
    primitives: Vec<(Arc<Mesh>, Arc<ResolvedInstance>)>,
}

impl VisibleScene {
    /// Walk the world's visible [`MeshRenderer`]s and snapshot the frame's
    /// draw list, pushing one model-uniform slot per entity into the
    /// [`FrameRing`]. Returns `None` when the component reads are unavailable.
    pub fn gather(world: &World) -> Option<Self> {
        let (Ok(renderers), Ok(globals), Ok(visibilities)) = (
            world.read::<MeshRenderer>(),
            world.read::<GlobalTransform>(),
            world.read::<Visibility>(),
        ) else {
            return None;
        };
        let mut ring = world.resource_mut::<FrameRing>();
        let mut items = Vec::new();
        for (idx, renderer) in renderers.iter() {
            if let Some(vis) = visibilities.get(idx)
                && !vis.is_visible()
            {
                continue;
            }
            // The mesh and the material instance load asynchronously — skip
            // primitives until both are resident.
            let primitives: Vec<_> = renderer
                .primitives
                .iter()
                .filter_map(|primitive| Some((primitive.mesh()?, primitive.material()?)))
                .collect();
            if primitives.is_empty() {
                continue;
            }
            let model = globals
                .get(idx)
                .map(|g| mat4_to_cols_array_2d(&g.0))
                .unwrap_or_else(|| mat4_to_cols_array_2d(&Mat4::identity()));
            let model_offset = ring.push(bytemuck::bytes_of(&shaders::ModelUniforms { model }));
            items.push(SceneItem {
                model_offset,
                primitives,
            });
        }
        Some(Self { items })
    }

    /// Whether the frame has no visible renderables.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Per-view arguments for [`SceneDrawer::record`].
pub struct DrawArgs {
    /// Ring offset of the view's
    /// [`CameraUniforms`](shaders::CameraUniforms) (pushed by the pipeline,
    /// selected per draw via the external set's dynamic offset).
    pub camera_offset: u32,
    /// Color format of the pass's render target (a pipeline-specialization
    /// key).
    pub color_format: TextureFormat,
    /// Depth format of the pass's depth attachment.
    pub depth_format: TextureFormat,
}

/// Records the [`VisibleScene`]'s draws into a pass — the shared draw loop
/// every scene-drawing pipeline delegates to. Holds the frame-invariant
/// binding-group cache (see the module docs), so it must live as long as its
/// owning pipeline, not be rebuilt per frame.
#[derive(Default)]
pub struct SceneDrawer {
    /// Frame-invariant camera/model/empty groups, keyed by the material's
    /// set-layout pointer (stable — pipelines are cached, and each cached
    /// group keeps its layout `Arc` alive so the pointer can't be reused
    /// while cached).
    binding_cache: std::sync::Mutex<BindingCache>,
}

/// Cache of the frame-invariant binding groups assembled by [`SceneDrawer`].
/// Values keyed by `Arc::as_ptr(layout) as usize` (a `Send` key).
#[derive(Default)]
struct BindingCache {
    camera: HashMap<usize, Arc<BindingGroup>>,
    model: HashMap<usize, Arc<BindingGroup>>,
    empty: HashMap<usize, Arc<BindingGroup>>,
}

impl BindingCache {
    /// Get (or create + cache) a binding group for `layout`, building its
    /// descriptor with `make_desc` on the first miss.
    fn get_or_create(
        map: &mut HashMap<usize, Arc<BindingGroup>>,
        device: &Arc<GraphicsDevice>,
        layout: &Arc<BindingLayout>,
        make_desc: impl FnOnce() -> BindingGroupDescriptor,
    ) -> Result<Arc<BindingGroup>, GraphicsError> {
        let key = Arc::as_ptr(layout) as usize;
        if let Some(group) = map.get(&key) {
            return Ok(group.clone());
        }
        let group = device.create_binding_group(Arc::clone(layout), make_desc())?;
        map.insert(key, group.clone());
        Ok(group)
    }
}

impl SceneDrawer {
    /// Emit one draw per visible primitive into `pass`.
    ///
    /// Binding sets are assembled per the shader's declared update rates
    /// (docs/MATERIAL_ASSETS.md Decision 7):
    ///
    /// - external: the camera block — bound at offset 0 in the shared ring,
    ///   the view's slot ([`DrawArgs::camera_offset`]) supplied per draw;
    /// - dynamic: the model block — one ring binding, the per-draw dynamic
    ///   offset selects the entity's slot;
    /// - static: the instance's material props group, built once by the
    ///   instance manager.
    pub fn record(
        &self,
        pass: &mut GraphicsPass,
        scene: &VisibleScene,
        ring_buffer: &Arc<Buffer>,
        pipelines: &mut PipelineCache,
        args: &DrawArgs,
    ) {
        use redlilium_graphics::{MaterialInstance, UpdateRate};

        let camera_size = std::mem::size_of::<shaders::CameraUniforms>() as u64;
        let model_size = std::mem::size_of::<shaders::ModelUniforms>() as u64;
        let mut cache = self
            .binding_cache
            .lock()
            .expect("scene drawer binding cache poisoned");
        for item in &scene.items {
            for (mesh, instance) in &item.primitives {
                // Specialize the pipeline for this shader + variant + the
                // mesh's vertex layout + the target formats (built once,
                // then cached). The variant is the material's feature
                // half (Decision 5); when the forward path grows system
                // axes (lighting modes), this is where it completes
                // them via with_features + .system().
                let Ok(pipeline) = pipelines.get_or_build(
                    instance.shader_guid,
                    &instance.shader,
                    &instance.variant,
                    mesh.layout(),
                    args.color_format,
                    args.depth_format,
                ) else {
                    continue;
                };
                // Assemble the sets in declaration order by their rates.
                let rates = pipeline.set_update_rates().to_vec();
                if rates.iter().all(Option::is_none) {
                    log::debug!(
                        "shader {:?} declares no rate-classified sets; skipping draw",
                        instance.shader_guid
                    );
                    continue;
                }
                let device = pipeline.device();
                let mut gfx_instance = MaterialInstance::new(Arc::clone(&pipeline));
                let mut offsets: Vec<Vec<u32>> = Vec::with_capacity(rates.len());
                // Assemble one binding group per set, all frame-invariant and
                // cached (see the type-level doc). A creation failure skips the
                // draw rather than the whole frame.
                let mut assembled = true;
                for (set_idx, rate) in rates.iter().enumerate() {
                    let Some(layout) = pipeline.binding_layouts().get(set_idx) else {
                        // A rate-classified set with no reflected layout is a
                        // reflection bug; skip the draw to avoid a bad bind.
                        assembled = false;
                        break;
                    };
                    let group = match rate {
                        Some(UpdateRate::External) => {
                            BindingCache::get_or_create(&mut cache.camera, device, layout, || {
                                BindingGroupDescriptor::new().with_buffer_range(
                                    0,
                                    ring_buffer.clone(),
                                    0,
                                    camera_size,
                                )
                            })
                        }
                        Some(UpdateRate::Dynamic) => {
                            BindingCache::get_or_create(&mut cache.model, device, layout, || {
                                BindingGroupDescriptor::new().with_buffer_range(
                                    0,
                                    ring_buffer.clone(),
                                    0,
                                    model_size,
                                )
                            })
                        }
                        Some(UpdateRate::Static) => {
                            instance.props_group(device, Arc::clone(layout))
                        }
                        None => BindingCache::get_or_create(
                            &mut cache.empty,
                            device,
                            layout,
                            BindingGroupDescriptor::new,
                        ),
                    };
                    let group = match group {
                        Ok(g) => g,
                        Err(e) => {
                            log::warn!("scene drawer: failed to build binding group: {e}");
                            assembled = false;
                            break;
                        }
                    };
                    gfx_instance = gfx_instance.with_binding_group(group);
                    // External + Dynamic sets bind at offset 0 and select their
                    // slot via a per-draw dynamic offset.
                    offsets.push(match rate {
                        Some(UpdateRate::External) => vec![args.camera_offset],
                        Some(UpdateRate::Dynamic) => vec![item.model_offset],
                        _ => Vec::new(),
                    });
                }
                if !assembled {
                    continue;
                }
                pass.add_draw_command(
                    DrawCommand::new(Arc::clone(mesh), Arc::new(gfx_instance))
                        .with_dynamic_offsets(offsets),
                );
            }
        }
    }
}
