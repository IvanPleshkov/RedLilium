//! Render systems — they run in the [`Render`](crate::Render) schedule and
//! contribute passes to the frame graph held in the [`RenderSchedule`] resource.

use std::sync::Arc;

use redlilium_core::math::{Mat4, mat4_to_cols_array_2d};
use redlilium_graphics::{
    ColorAttachment, DepthStencilAttachment, DrawCommand, GraphicsPass, PassHandle, RenderTarget,
    RenderTargetConfig,
};

use redlilium_graphics::egui::EguiController;

use redlilium_assets::{AssetDb, AssetProcessor};

use crate::std::components::{Camera, GlobalTransform, Visibility};
use crate::system::SystemError;
use crate::{DebugDrawer, DebugDrawerRenderer, System, SystemContext};

use super::{
    CameraTarget, FrameRing, MaterialAssetManager, MaterialInstanceManager, MaterialManager,
    MeshManager, MeshRenderer, RenderPassType, RenderSchedule, ShaderManager, ShadingRegistry,
    TextureManager, VertexLayoutManager, pack_uniform_bytes, shaders,
};

/// Default material props (base color) for a primitive with no CPU instance —
/// matches the editor's previous `DEFAULT_MATERIAL_PROPS`.
const DEFAULT_MATERIAL_PROPS: [f32; 4] = [0.6, 0.6, 0.65, 1.0];

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
        let world = ctx.world();
        let mut schedule = world.resource_mut::<RenderSchedule>();
        let Some(graph) = schedule.graph_mut() else {
            return Ok(()); // no frame graph bound (not in the render bracket)
        };
        // Mesh uploads now flow through the asset pipeline (the mesh loader's GPU
        // stage + AssetGpuFlush), so only texture/material managers flush here.
        if world.has_resource::<TextureManager>() {
            world.resource_mut::<TextureManager>().flush_uploads(graph);
        }
        if world.has_resource::<MaterialManager>() {
            world.resource_mut::<MaterialManager>().flush_uploads(graph);
        }
        // Asset-based material instances flush their static property buffers here too.
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
pub struct ForwardRender;

impl System for ForwardRender {
    /// The scene pass's handle (so a dependent render system can order after it
    /// via [`SystemContext::system_result`](crate::SystemContext::system_result)).
    type Result = Option<PassHandle>;
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.world();
        if !world.has_resource::<RenderSchedule>()
            || !world.has_resource::<FrameRing>()
            || !world.has_resource::<ScenePass>()
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
            let (Ok(renderers), Ok(globals), Ok(visibilities)) = (
                world.read::<MeshRenderer>(),
                world.read::<GlobalTransform>(),
                world.read::<Visibility>(),
            ) else {
                return Ok(None);
            };
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
                let fwd = shaders::OpaqueColorUniforms {
                    view_projection: vp,
                    model,
                };
                let fwd_off = ring.push(bytemuck::bytes_of(&fwd));
                for primitive in &renderer.primitives {
                    // The mesh loads asynchronously — skip until it's resident.
                    let Some(mesh) = primitive.mesh() else {
                        continue;
                    };
                    if let Some(instance) = primitive.material.pass(RenderPassType::Forward) {
                        let bytes = primitive
                            .material
                            .cpu_instance()
                            .map(|ci| pack_uniform_bytes(&ci.material, &ci.values))
                            .filter(|b| !b.is_empty())
                            .unwrap_or_else(|| {
                                bytemuck::bytes_of(&DEFAULT_MATERIAL_PROPS).to_vec()
                            });
                        let props_off = ring.push(&bytes);
                        pass.add_draw_command(
                            DrawCommand::new(mesh, Arc::clone(instance))
                                .with_dynamic_offsets(vec![vec![fwd_off], vec![props_off]]),
                        );
                    }
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
        let world = ctx.world();
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
        let world = ctx.world();
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

/// Drives mesh loading: advances every in-flight [`MeshManager`] request,
/// resolving each one's shared vertex layout (via [`VertexLayoutManager`]) and
/// kicking off / polling its mesh asset. This is the consumer-facing meshes'
/// driver — co-locks the manager + layout manager + processor + DB so that
/// callers only ever touch `MeshManager::request`. No-op if any input is absent.
pub struct MeshLoad;

impl System for MeshLoad {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.world();
        if !world.has_resource::<MeshManager>()
            || !world.has_resource::<VertexLayoutManager>()
            || !world.has_resource::<AssetProcessor>()
            || !world.has_resource::<AssetDb>()
        {
            return Ok(());
        }
        let mut mesh_mgr = world.resource_mut::<MeshManager>();
        let mut layout_mgr = world.resource_mut::<VertexLayoutManager>();
        let mut processor = world.resource_mut::<AssetProcessor>();
        let db = world.resource::<AssetDb>();
        mesh_mgr.drive(&mut processor, &db, &mut layout_mgr);
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
        let world = ctx.world();
        if !world.has_resource::<MaterialInstanceManager>()
            || !world.has_resource::<MaterialAssetManager>()
            || !world.has_resource::<ShaderManager>()
            || !world.has_resource::<ShadingRegistry>()
            || !world.has_resource::<AssetProcessor>()
            || !world.has_resource::<AssetDb>()
        {
            return Ok(());
        }
        let mut instance_mgr = world.resource_mut::<MaterialInstanceManager>();
        let mut material_mgr = world.resource_mut::<MaterialAssetManager>();
        let mut shader_mgr = world.resource_mut::<ShaderManager>();
        let registry = world.resource::<ShadingRegistry>();
        let mut processor = world.resource_mut::<AssetProcessor>();
        let db = world.resource::<AssetDb>();
        instance_mgr.drive(
            &mut processor,
            &db,
            &mut material_mgr,
            &mut shader_mgr,
            &registry,
        );
        Ok(())
    }
}
