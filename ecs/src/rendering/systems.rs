use std::sync::Arc;

use redlilium_graphics::{
    ColorAttachment, DepthStencilAttachment, GraphicsPass, LoadOp, RenderTarget,
    RenderTargetConfig, StoreOp,
};

use crate::SystemContext;
use crate::std::components::{Camera, GlobalTransform, Visibility};

use super::components::{CameraTarget, RenderMaterial, RenderMesh};
use super::resources::RenderSchedule;

/// Simple forward render system.
///
/// Collects all visible entities with [`RenderMesh`] + [`RenderMaterial`] and,
/// for each camera that has a [`CameraTarget`], builds a render graph with a
/// single forward graphics pass and submits it to the [`RenderSchedule`].
///
/// # Access
///
/// - Reads: `Camera`, `GlobalTransform`, `CameraTarget`, `RenderMesh`,
///   `RenderMaterial`, `Visibility`
/// - Resources: `ResMut<RenderSchedule>`
///
/// # Notes
///
/// - Only entities with `Visibility(true)` are rendered.
/// - Cameras without a `CameraTarget` are skipped.
/// - Swapchain presentation is NOT handled here — that is the app layer's job.
pub struct ForwardRenderSystem;

impl crate::System for ForwardRenderSystem {
    type Result = ();

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::Read<Camera>,
            crate::Read<GlobalTransform>,
            crate::Read<CameraTarget>,
            crate::Read<RenderMesh>,
            crate::Read<RenderMaterial>,
            crate::Read<Visibility>,
            crate::ResMut<RenderSchedule>,
        )>()
        .execute(
            |(cameras, globals, targets, meshes, materials, visibilities, mut schedule_res)| {
                let Some(schedule) = schedule_res.schedule_mut() else {
                    return;
                };

                // For each camera that has a render target
                for (cam_idx, _camera) in cameras.iter() {
                    let Some(target) = targets.get(cam_idx) else {
                        continue;
                    };
                    let Some(_cam_global) = globals.get(cam_idx) else {
                        continue;
                    };

                    // Build render target config
                    let render_target_config = RenderTargetConfig::new()
                        .with_color(
                            ColorAttachment::new(RenderTarget::from_texture(Arc::clone(
                                &target.color,
                            )))
                            .with_load_op(LoadOp::clear_color(
                                target.clear_color[0],
                                target.clear_color[1],
                                target.clear_color[2],
                                target.clear_color[3],
                            ))
                            .with_store_op(StoreOp::Store),
                        )
                        .with_depth_stencil(
                            DepthStencilAttachment::new(RenderTarget::from_texture(Arc::clone(
                                &target.depth,
                            )))
                            .with_clear_depth(1.0)
                            .with_depth_store_op(StoreOp::DontCare),
                        );

                    // Create graphics pass
                    let mut pass = GraphicsPass::new(format!("forward_{cam_idx}"));
                    pass.set_render_targets(render_target_config);

                    // Collect visible renderable entities
                    for (entity_idx, render_mesh) in meshes.iter() {
                        let Some(render_material) = materials.get(entity_idx) else {
                            continue;
                        };
                        // Skip invisible entities
                        if let Some(vis) = visibilities.get(entity_idx)
                            && !vis.is_visible()
                        {
                            continue;
                        }

                        pass.add_draw(Arc::clone(&render_mesh.0), Arc::clone(&render_material.0));
                    }

                    // Submit the graph
                    let mut graph = schedule.acquire_graph();
                    graph.add_graphics_pass(pass);
                    schedule.submit(format!("camera_{cam_idx}"), graph, &[]);
                }
            },
        );
        Ok(())
    }
}

/// Editor-aware forward render system.
///
/// Like [`ForwardRenderSystem`], but uses `ReadAll` for camera queries so it
/// can see editor-flagged entities (e.g. the editor camera). Renderable
/// entities (meshes, materials, visibility) are still queried with `Read`,
/// so only game entities are drawn.
///
/// # Access
///
/// - ReadAll: `Camera`, `GlobalTransform`, `CameraTarget`
/// - Read: `RenderMesh`, `RenderMaterial`, `Visibility`
/// - Resources: `ResMut<RenderSchedule>`
pub struct EditorForwardRenderSystem;

impl crate::System for EditorForwardRenderSystem {
    type Result = ();

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::ReadAll<Camera>,
            crate::ReadAll<GlobalTransform>,
            crate::ReadAll<CameraTarget>,
            crate::Read<RenderMesh>,
            crate::Read<RenderMaterial>,
            crate::Read<Visibility>,
            crate::ResMut<RenderSchedule>,
        )>()
        .execute(
            |(cameras, globals, targets, meshes, materials, visibilities, mut schedule_res)| {
                let Some(schedule) = schedule_res.schedule_mut() else {
                    return;
                };

                // For each camera that has a render target (including editor cameras)
                for (cam_idx, _camera) in cameras.iter() {
                    let Some(target) = targets.get(cam_idx) else {
                        continue;
                    };
                    let Some(_cam_global) = globals.get(cam_idx) else {
                        continue;
                    };

                    // Build render target config
                    let render_target_config = RenderTargetConfig::new()
                        .with_color(
                            ColorAttachment::new(RenderTarget::from_texture(Arc::clone(
                                &target.color,
                            )))
                            .with_load_op(LoadOp::clear_color(
                                target.clear_color[0],
                                target.clear_color[1],
                                target.clear_color[2],
                                target.clear_color[3],
                            ))
                            .with_store_op(StoreOp::Store),
                        )
                        .with_depth_stencil(
                            DepthStencilAttachment::new(RenderTarget::from_texture(Arc::clone(
                                &target.depth,
                            )))
                            .with_clear_depth(1.0)
                            .with_depth_store_op(StoreOp::DontCare),
                        );

                    // Create graphics pass
                    let mut pass = GraphicsPass::new(format!("editor_forward_{cam_idx}"));
                    pass.set_render_targets(render_target_config);

                    // Collect visible renderable entities (game entities only)
                    for (entity_idx, render_mesh) in meshes.iter() {
                        let Some(render_material) = materials.get(entity_idx) else {
                            continue;
                        };
                        // Skip invisible entities
                        if let Some(vis) = visibilities.get(entity_idx)
                            && !vis.is_visible()
                        {
                            continue;
                        }

                        pass.add_draw(Arc::clone(&render_mesh.0), Arc::clone(&render_material.0));
                    }

                    // Submit the graph
                    let mut graph = schedule.acquire_graph();
                    graph.add_graphics_pass(pass);
                    schedule.submit(format!("editor_camera_{cam_idx}"), graph, &[]);
                }
            },
        );
        Ok(())
    }
}
