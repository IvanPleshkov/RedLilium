//! Render systems — they run in the [`Render`](crate::Render) schedule and
//! contribute passes to the frame graph held in the [`RenderSchedule`] resource.

use std::collections::HashMap;
use std::sync::Arc;

use redlilium_core::math::{Mat4, mat4_to_cols_array_2d};
use redlilium_graphics::{
    BindingGroup, BindingGroupDescriptor, BindingLayout, Buffer, ColorAttachment,
    DepthStencilAttachment, DrawCommand, GraphicsDevice, GraphicsError, GraphicsPass,
    MaterialInstance, PassHandle, RenderTarget, RenderTargetConfig,
};

use redlilium_graphics::egui::EguiController;

use redlilium_assets::{AssetDb, AssetProcessor};

use crate::std::components::{Camera, GlobalTransform, Visibility};
use crate::system::SystemError;
use crate::{DebugDrawer, DebugDrawerRenderer, ExclusiveSystem, System, SystemContext, World};

use super::{
    CameraTarget, FrameRing, MaterialAssetManager, MaterialInstanceManager, MeshManager,
    MeshRenderer, PipelineCache, RenderSchedule, ShaderManager, ShadingRegistry, TextureManager,
    VertexLayoutManager, shaders,
};

/// Holds the forward scene pass's graph handle so other passes (an egui overlay,
/// debug lines) can depend on it. Written by [`ForwardRender`] each frame (set to
/// `None` if it produced no pass).
#[derive(Default)]
pub struct ScenePass(pub Option<PassHandle>);

/// Flushes the GPU-upload managers' pending transfers into the frame graph
/// (deferred, graph-ordered — never a synchronous write). Passes that read the
/// uploaded meshes/textures/materials should depend on this system so their
/// transfer pass is scheduled first.
pub struct FlushUploads;

impl System for FlushUploads {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        let mut schedule = world.resource_mut::<RenderSchedule>();
        let Some(graph) = schedule.graph_mut() else {
            return Ok(()); // no frame graph bound (not in the render bracket)
        };
        // Mesh and texture uploads flow through the asset pipeline (the loaders'
        // GPU stages + AssetGpuFlush), so only the material manager flushes here.
        // Asset-based material instances flush their static property buffers here.
        if world.has_resource::<MaterialInstanceManager>() {
            world
                .resource_mut::<MaterialInstanceManager>()
                .flush_uploads(graph);
        }
        Ok(())
    }
}

/// Renders the forward scene into the (editor) camera's [`CameraTarget`]: fills
/// the [`FrameRing`] resource with per-draw uniforms (group 0 transform, group 1
/// material props — same buffer, different offsets) and emits a draw per visible
/// primitive, then adds the pass to the frame graph and records its handle in
/// [`ScenePass`] so dependent passes (egui, debug) can order after it.
///
/// Camera + CameraTarget are read via `read_all` (the editor camera is
/// EDITOR-flagged, which the filtered `read` iterator skips).
///
/// # Binding groups are created eagerly (issue #40)
///
/// A `BindingGroup` is now a device resource whose GPU descriptor set is built
/// once at creation, so steady-state frames must issue **zero**
/// `create_binding_group` calls. The camera (external) and model (dynamic) sets
/// bind the frame ring buffer at offset 0 and select the per-view / per-entity
/// slot with a per-draw dynamic offset — so the group is frame-invariant and
/// cached here by the material's set-layout identity. The empty (unclassified)
/// set is likewise one cached group per layout. The static material-property
/// set is cached inside its [`ResolvedInstance`]. After warm-up every set is a
/// cache hit.
#[derive(Default)]
pub struct ForwardRender {
    /// Frame-invariant camera/model/empty groups, keyed by the material's
    /// set-layout pointer (stable — pipelines are cached, and each cached group
    /// keeps its layout `Arc` alive so the pointer can't be reused while cached).
    binding_cache: std::sync::Mutex<ForwardBindingCache>,
}

/// Cache of the frame-invariant binding groups assembled by [`ForwardRender`].
/// Values keyed by `Arc::as_ptr(layout) as usize` (a `Send` key).
#[derive(Default)]
struct ForwardBindingCache {
    camera: HashMap<usize, Arc<BindingGroup>>,
    model: HashMap<usize, Arc<BindingGroup>>,
    empty: HashMap<usize, Arc<BindingGroup>>,
}

impl ForwardBindingCache {
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

impl System for ForwardRender {
    /// The scene pass's handle (so a dependent render system can order after it
    /// via [`SystemContext::system_result`](crate::SystemContext::system_result)).
    type Result = Option<PassHandle>;
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<RenderSchedule>()
            || !world.has_resource::<FrameRing>()
            || !world.has_resource::<ScenePass>()
            || !world.has_resource::<PipelineCache>()
        {
            return Ok(None);
        }
        world.resource_mut::<ScenePass>().0 = None;

        // View-projection from the camera; color/depth from its CameraTarget.
        let Some(vp) = world.read_all::<Camera>().ok().and_then(|c| {
            c.iter()
                .next()
                .map(|(_, cam)| mat4_to_cols_array_2d(&cam.view_projection()))
        }) else {
            return Ok(None);
        };
        let Some((color, depth, clear)) = world.read_all::<CameraTarget>().ok().and_then(|t| {
            t.iter().next().map(|(_, target)| {
                (
                    target.color.clone(),
                    target.depth.clone(),
                    target.clear_color,
                )
            })
        }) else {
            return Ok(None);
        };
        let color_fmt = color.format();
        let depth_fmt = depth.format();

        let mut pass = GraphicsPass::new("scene_view".into());
        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(
                    ColorAttachment::from_texture(color)
                        .with_clear_color(clear[0], clear[1], clear[2], clear[3]),
                )
                .with_depth_stencil(
                    DepthStencilAttachment::from_texture(depth).with_clear_depth(1.0),
                ),
        );

        // Fill the ring + emit draws (scoped so the guards drop before we touch
        // the graph resource).
        {
            let mut ring = world.resource_mut::<FrameRing>();
            let mut pipelines = world.resource_mut::<PipelineCache>();
            let (Ok(renderers), Ok(globals), Ok(visibilities)) = (
                world.read::<MeshRenderer>(),
                world.read::<GlobalTransform>(),
                world.read::<Visibility>(),
            ) else {
                return Ok(None);
            };
            // Binding sets are assembled per the shader's declared update rates
            // (docs/MATERIAL_ASSETS.md Decision 7):
            // - external: the camera block — pushed into the shared ring once
            //   per view, bound at its fixed offset;
            // - dynamic: the model block — one ring binding, the per-draw
            //   dynamic offset selects the entity's slot;
            // - static: the instance's material props group, built once by the
            //   instance manager.
            let camera = shaders::CameraUniforms {
                view_projection: vp,
            };
            // The camera set is `external` (a dynamic uniform now): bind offset 0,
            // supply this view's ring offset per draw.
            let camera_off = ring.push(bytemuck::bytes_of(&camera));
            let ring_buffer: Arc<Buffer> = ring.buffer().clone();
            let camera_size = std::mem::size_of::<shaders::CameraUniforms>() as u64;
            let model_size = std::mem::size_of::<shaders::ModelUniforms>() as u64;
            let mut cache = self
                .binding_cache
                .lock()
                .expect("forward binding cache poisoned");
            for (idx, renderer) in renderers.iter() {
                if let Some(vis) = visibilities.get(idx)
                    && !vis.is_visible()
                {
                    continue;
                }
                let model = globals
                    .get(idx)
                    .map(|g| mat4_to_cols_array_2d(&g.0))
                    .unwrap_or_else(|| mat4_to_cols_array_2d(&Mat4::identity()));
                let model_off = ring.push(bytemuck::bytes_of(&shaders::ModelUniforms { model }));
                for primitive in &renderer.primitives {
                    // The mesh and the material instance load asynchronously — skip
                    // until both are resident.
                    let (Some(mesh), Some(instance)) = (primitive.mesh(), primitive.material())
                    else {
                        continue;
                    };
                    // Specialize the pipeline for this shader + variant + the
                    // mesh's vertex layout + the target formats (built once,
                    // then cached). The empty variant until material assets
                    // declare feature flags (Decision 5's material half).
                    let Ok(pipeline) = pipelines.get_or_build(
                        instance.shader_guid,
                        &instance.shader,
                        &redlilium_graphics::VariantKey::empty(),
                        mesh.layout(),
                        color_fmt,
                        depth_fmt,
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
                        use redlilium_graphics::UpdateRate;
                        let Some(layout) = pipeline.binding_layouts().get(set_idx) else {
                            // A rate-classified set with no reflected layout is a
                            // reflection bug; skip the draw to avoid a bad bind.
                            assembled = false;
                            break;
                        };
                        let group = match rate {
                            Some(UpdateRate::External) => ForwardBindingCache::get_or_create(
                                &mut cache.camera,
                                device,
                                layout,
                                || {
                                    BindingGroupDescriptor::new().with_buffer_range(
                                        0,
                                        ring_buffer.clone(),
                                        0,
                                        camera_size,
                                    )
                                },
                            ),
                            Some(UpdateRate::Dynamic) => ForwardBindingCache::get_or_create(
                                &mut cache.model,
                                device,
                                layout,
                                || {
                                    BindingGroupDescriptor::new().with_buffer_range(
                                        0,
                                        ring_buffer.clone(),
                                        0,
                                        model_size,
                                    )
                                },
                            ),
                            Some(UpdateRate::Static) => {
                                instance.props_group(device, Arc::clone(layout))
                            }
                            None => ForwardBindingCache::get_or_create(
                                &mut cache.empty,
                                device,
                                layout,
                                BindingGroupDescriptor::new,
                            ),
                        };
                        let group = match group {
                            Ok(g) => g,
                            Err(e) => {
                                log::warn!("forward render: failed to build binding group: {e}");
                                assembled = false;
                                break;
                            }
                        };
                        gfx_instance = gfx_instance.with_binding_group(group);
                        // External + Dynamic sets bind at offset 0 and select their
                        // slot via a per-draw dynamic offset.
                        offsets.push(match rate {
                            Some(UpdateRate::External) => vec![camera_off],
                            Some(UpdateRate::Dynamic) => vec![model_off],
                            _ => Vec::new(),
                        });
                    }
                    if !assembled {
                        continue;
                    }
                    pass.add_draw_command(
                        DrawCommand::new(mesh, Arc::new(gfx_instance))
                            .with_dynamic_offsets(offsets),
                    );
                }
            }
        }

        // Add to the frame graph and record the handle.
        let handle = {
            let mut schedule = world.resource_mut::<RenderSchedule>();
            schedule
                .graph_mut()
                .map(|graph| graph.add_graphics_pass(pass))
        };
        if let Some(handle) = handle {
            world.resource_mut::<ScenePass>().0 = Some(handle);
        }
        Ok(handle)
    }
}

/// Renders debug-drawer lines as a separate pass that loads the camera's
/// [`CameraTarget`] and is ordered after the forward scene pass — whose handle it
/// reads the native way, via [`system_result`](SystemContext::system_result)
/// (requires a `ForwardRender -> DebugRender` edge). Updates [`ScenePass`] to its
/// own handle so it becomes the CameraTarget's last writer (egui depends on that).
pub struct DebugRender;

impl System for DebugRender {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        // The forward pass we order after (its handle comes from system_result).
        let Some(scene_handle) = *ctx.system_result::<ForwardRender>() else {
            return Ok(());
        };
        if !world.has_resource::<DebugDrawer>()
            || !world.has_resource::<DebugDrawerRenderer>()
            || !world.has_resource::<RenderSchedule>()
        {
            return Ok(());
        }
        let vertices = world.resource::<DebugDrawer>().take_render_data();
        if vertices.is_empty() {
            return Ok(());
        }
        let Some((color, depth)) = world.read_all::<CameraTarget>().ok().and_then(|t| {
            t.iter()
                .next()
                .map(|(_, target)| (target.color.clone(), target.depth.clone()))
        }) else {
            return Ok(());
        };
        let vp = world.read_all::<Camera>().ok().and_then(|c| {
            c.iter()
                .next()
                .map(|(_, cam)| mat4_to_cols_array_2d(&cam.view_projection()))
        });

        let debug_handle = {
            let mut renderer = world.resource_mut::<DebugDrawerRenderer>();
            if let Some(vp) = vp {
                renderer.update_view_proj(vp);
            }
            let rt = RenderTarget::from_texture(color);
            let Some(pass) = renderer.create_graphics_pass(&vertices, &rt, Some(&depth)) else {
                return Ok(());
            };
            let mut schedule = world.resource_mut::<RenderSchedule>();
            schedule.graph_mut().map(|graph| {
                let h = graph.add_graphics_pass(pass);
                graph.add_dependency(h, scene_handle);
                h
            })
        };
        if let Some(h) = debug_handle {
            world.resource_mut::<ScenePass>().0 = Some(h);
        }
        Ok(())
    }
}

/// The frame's final (swapchain) render target plus its size, set by the app each
/// frame BEFORE running the [`Render`](crate::Render) schedule. Render systems
/// that composite to the screen — currently [`EguiRender`] — read it. `RenderTarget`
/// is owned + Send + Sync, so this lives in the world; the app replaces it each
/// frame. (Off-screen passes target a [`CameraTarget`] texture instead.)
pub struct FrameTarget {
    pub target: RenderTarget,
    pub width: u32,
    pub height: u32,
}

/// Finishes the egui frame (the app calls `begin_frame` + builds UI before the
/// Render schedule) and contributes its draw pass to the frame graph, targeting
/// [`FrameTarget`] (the swapchain). Edge `DebugRender -> EguiRender` orders it last
/// so it can depend on the CameraTarget's last writer ([`ScenePass`]) — an egui
/// overlay samples that texture, so its draw must come after the scene/debug passes.
///
/// The egui controller is the [`EguiController`] resource (see the editor's
/// `on_init`); this system is a no-op when any of its inputs are absent.
pub struct EguiRender;

impl System for EguiRender {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<EguiController>()
            || !world.has_resource::<FrameTarget>()
            || !world.has_resource::<RenderSchedule>()
        {
            return Ok(());
        }
        // The pass egui must order after (the CameraTarget's last writer), if any.
        let scene_handle = world
            .has_resource::<ScenePass>()
            .then(|| world.resource::<ScenePass>().0)
            .flatten();

        let mut egui = world.resource_mut::<EguiController>();
        let pass = {
            let ft = world.resource::<FrameTarget>();
            egui.end_frame(&ft.target, ft.width, ft.height)
        };
        let mut schedule = world.resource_mut::<RenderSchedule>();
        if let Some(graph) = schedule.graph_mut() {
            // Atlas uploads first (graph-ordered before the egui draw).
            egui.flush_uploads(graph);
            if let Some(pass) = pass {
                let handle = graph.add_graphics_pass(pass);
                if let Some(scene_handle) = scene_handle {
                    graph.add_dependency(handle, scene_handle);
                }
            }
        }
        Ok(())
    }
}

/// Drives mesh loading and syncs component [`AssetRef`](redlilium_assets::AssetRef)s.
///
/// First advances every in-flight [`MeshManager`] load (resolving each one's
/// shared vertex layout via [`VertexLayoutManager`]). Then scans the asset refs
/// of **every registered component** (via
/// [`World::scan_asset_refs`] — the derive-generated
/// [`Component::visit_asset_refs`](crate::Component::visit_asset_refs) hooks)
/// and resolves them against the managers: demand-driven (an unresolved source
/// gets requested) and reload-aware (a ref whose `Arc` no longer matches the
/// resident one is re-resolved; `Arc` pointer identity is the version). Writes
/// go through `Mut`, so a re-resolve marks the component dirty and stateful
/// consumers can react through ordinary change detection.
///
/// A user component with an `AssetRef` field gets loading + hot reload with no
/// extra code. No-op if any input is absent.
///
/// The scan is gen-gated: while every manager's `generation()` is unchanged
/// (no resident asset appeared or reloaded), only components changed since the
/// previous scan are visited — new/edited refs still get demand-requested, but
/// an idle scene costs nothing per frame. Any generation bump falls back to a
/// full walk (a reload must re-resolve refs anywhere in the world).
///
/// Syncs asset references across all components as asset managers change their
/// state. Runs as an exclusive barrier to avoid contending with component-writing
/// systems (e.g., `UpdateGlobalTransforms`) under the multi-threaded runner.
#[derive(Default)]
pub struct MeshLoad {
    /// Manager generations at the previous scan. Used to gate: unchanged generations
    /// → skip the expensive scan pass.
    last_gens: Option<[u64; 3]>,
}

impl ExclusiveSystem for MeshLoad {
    type Result = ();
    fn run(&mut self, world: &mut World) -> Result<Self::Result, SystemError> {
        use redlilium_assets::AssetRef;

        use super::loaders::{MaterialInstanceSource, MeshSource, TextureSource};

        if !world.has_resource::<MeshManager>()
            || !world.has_resource::<VertexLayoutManager>()
            || !world.has_resource::<AssetProcessor>()
            || !world.has_resource::<AssetDb>()
        {
            return Ok(());
        }
        let mut mesh_mgr = world.resource_mut::<MeshManager>();
        {
            let mut layout_mgr = world.resource_mut::<VertexLayoutManager>();
            let mut processor = world.resource_mut::<AssetProcessor>();
            let db = world.resource::<AssetDb>();
            mesh_mgr.drive(&mut processor, &db, &mut layout_mgr);
        }

        let mut instance_mgr = world
            .has_resource::<MaterialInstanceManager>()
            .then(|| world.resource_mut::<MaterialInstanceManager>());
        let mut texture_mgr = world
            .has_resource::<TextureManager>()
            .then(|| world.resource_mut::<TextureManager>());

        let gens = [
            mesh_mgr.generation(),
            instance_mgr.as_ref().map_or(0, |m| m.generation()),
            texture_mgr.as_ref().map_or(0, |m| m.generation()),
        ];
        let changed = self.last_gens != Some(gens);
        self.last_gens = Some(gens);

        if !changed {
            return Ok(());
        }

        let mut stale: Vec<(&'static str, u32)> = Vec::new();
        world.scan_asset_refs(None, &mut |component, idx, any| {
            if let Some(r) = any.downcast_ref::<AssetRef<MeshSource>>() {
                match mesh_mgr.get(r.source()) {
                    Some(mesh) if !r.is_current(mesh) => stale.push((component, idx)),
                    Some(_) => {}
                    None => mesh_mgr.request(r.source()),
                }
            } else if let Some(r) = any.downcast_ref::<AssetRef<MaterialInstanceSource>>()
                && let Some(instance_mgr) = instance_mgr.as_mut()
            {
                match instance_mgr.get(r.source().guid) {
                    Some(instance) if !r.is_current(instance) => stale.push((component, idx)),
                    Some(_) => {}
                    None => instance_mgr.request(r.source()),
                }
            } else if let Some(r) = any.downcast_ref::<AssetRef<TextureSource>>()
                && let Some(texture_mgr) = texture_mgr.as_mut()
            {
                match texture_mgr.get(r.source()) {
                    Some(texture) if !r.is_current(texture) => stale.push((component, idx)),
                    Some(_) => {}
                    None => texture_mgr.request(r.source()),
                }
            }
        });

        stale.sort_unstable();
        stale.dedup();
        for (component, idx) in stale {
            world.patch_asset_refs(component, idx, &mut |any| {
                if let Some(r) = any.downcast_mut::<AssetRef<MeshSource>>() {
                    if let Some(mesh) = mesh_mgr.get(r.source())
                        && !r.is_current(mesh)
                    {
                        r.resolve(mesh.clone());
                    }
                } else if let Some(r) = any.downcast_mut::<AssetRef<MaterialInstanceSource>>()
                    && let Some(instance_mgr) = instance_mgr.as_mut()
                    && let Some(instance) = instance_mgr.get(r.source().guid)
                    && !r.is_current(instance)
                {
                    r.resolve(instance.clone());
                } else if let Some(r) = any.downcast_mut::<AssetRef<TextureSource>>()
                    && let Some(texture_mgr) = texture_mgr.as_mut()
                    && let Some(texture) = texture_mgr.get(r.source())
                    && !r.is_current(texture)
                {
                    r.resolve(texture.clone());
                }
            });
        }
        Ok(())
    }
}

/// Drives material-instance loading: advances every in-flight
/// [`MaterialInstanceManager`] request — loading its data, resolving its parent
/// template (via [`MaterialAssetManager`], which itself pulls the shader through
/// [`ShaderManager`]), and building the static property binding. Co-locks the
/// managers + [`ShadingRegistry`] + processor + DB so consumers only ever touch
/// `MaterialInstanceManager::request`. No-op if any input is absent.
pub struct MaterialInstanceLoad;

impl System for MaterialInstanceLoad {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<MaterialInstanceManager>()
            || !world.has_resource::<MaterialAssetManager>()
            || !world.has_resource::<ShaderManager>()
            || !world.has_resource::<TextureManager>()
            || !world.has_resource::<ShadingRegistry>()
            || !world.has_resource::<AssetProcessor>()
            || !world.has_resource::<AssetDb>()
        {
            return Ok(());
        }
        let mut instance_mgr = world.resource_mut::<MaterialInstanceManager>();
        let mut material_mgr = world.resource_mut::<MaterialAssetManager>();
        let mut shader_mgr = world.resource_mut::<ShaderManager>();
        let mut texture_mgr = world.resource_mut::<TextureManager>();
        let registry = world.resource::<ShadingRegistry>();
        let mut processor = world.resource_mut::<AssetProcessor>();
        let db = world.resource::<AssetDb>();
        // Textures first, so instances demanding them this frame poll fresh state.
        texture_mgr.drive(&mut processor, &db);
        instance_mgr.drive(
            &mut processor,
            &db,
            &mut material_mgr,
            &mut shader_mgr,
            &mut texture_mgr,
            &registry,
        );
        Ok(())
    }
}

/// Hot reload: drains [`ChangedAssets`](super::ChangedAssets) and invalidates
/// the **owning** manager per guid (routed by the DB record's kind). Everything
/// downstream catches up by itself — dependent managers pull-validate their
/// input `Arc`s, the pipeline cache revalidates its shader, and the `MeshLoad`
/// sync re-resolves component refs (`docs/ASSETS.md` §9). Consumers keep serving
/// the last-good resource while the new one loads.
pub struct HotReload;

impl System for HotReload {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<super::ChangedAssets>() || !world.has_resource::<AssetDb>() {
            return Ok(());
        }
        let changed = world.resource_mut::<super::ChangedAssets>().drain();
        if changed.is_empty() {
            return Ok(());
        }
        let db = world.resource::<AssetDb>();
        for guid in changed {
            let Some(record) = db.record(&guid) else {
                continue; // not a registered asset (e.g. the db file itself)
            };
            log::info!("hot reload: {} ({guid:?})", record.path.path);
            match record.kind.as_str() {
                "mesh" if world.has_resource::<MeshManager>() => {
                    world.resource_mut::<MeshManager>().invalidate_file(guid);
                }
                "vertex_layout" if world.has_resource::<VertexLayoutManager>() => {
                    world.resource_mut::<VertexLayoutManager>().invalidate(guid);
                }
                "texture" if world.has_resource::<TextureManager>() => {
                    world.resource_mut::<TextureManager>().invalidate_file(guid);
                }
                "shader" if world.has_resource::<ShaderManager>() => {
                    world.resource_mut::<ShaderManager>().invalidate(guid);
                }
                "material" if world.has_resource::<MaterialAssetManager>() => {
                    world
                        .resource_mut::<MaterialAssetManager>()
                        .invalidate(guid);
                }
                "material_instance" if world.has_resource::<MaterialInstanceManager>() => {
                    world
                        .resource_mut::<MaterialInstanceManager>()
                        .invalidate(guid);
                }
                _ => {}
            }
        }
        Ok(())
    }
}
