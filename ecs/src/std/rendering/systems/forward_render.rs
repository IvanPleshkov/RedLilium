//! Forward rendering systems.

use std::sync::Arc;

use redlilium_graphics::{
    ColorAttachment, DepthStencilAttachment, GraphicsPass, LoadOp, RenderTarget,
    RenderTargetConfig, StoreOp,
};

use crate::SystemContext;
use crate::std::components::{Camera, GlobalTransform, Visibility};
use crate::std::rendering::components::{
    CameraTarget, PerEntityBuffers, RenderMaterial, RenderMesh, RenderPassType,
};
use crate::std::rendering::resources::RenderSchedule;

/// Simple forward render system.
///
/// Collects all visible entities with [`RenderMesh`] + [`RenderMaterial`] and,
/// for each camera that has a [`CameraTarget`], builds a render graph with a
/// single forward graphics pass and submits it to the [`RenderSchedule`].
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
            crate::Read<PerEntityBuffers>,
            crate::Read<Visibility>,
            crate::ResMut<RenderSchedule>,
        )>()
        .execute(
            |(
                cameras,
                globals,
                targets,
                meshes,
                materials,
                per_entity,
                visibilities,
                mut schedule_res,
            )| {
                let Some(schedule) = schedule_res.schedule_mut() else {
                    return;
                };

                for (cam_idx, _camera) in cameras.iter() {
                    let Some(target) = targets.get(cam_idx) else {
                        continue;
                    };
                    let Some(_cam_global) = globals.get(cam_idx) else {
                        continue;
                    };

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

                    let mut pass = GraphicsPass::new(format!("forward_{cam_idx}"));
                    pass.set_render_targets(render_target_config);

                    for (entity_idx, render_mesh) in meshes.iter() {
                        if !per_entity.contains(entity_idx) {
                            continue;
                        }
                        let Some(render_material) = materials.get(entity_idx) else {
                            continue;
                        };
                        if let Some(vis) = visibilities.get(entity_idx)
                            && !vis.is_visible()
                        {
                            continue;
                        }

                        if let Some(instance) = render_material.pass(RenderPassType::Forward) {
                            pass.add_draw(Arc::clone(&render_mesh.mesh), Arc::clone(instance));
                        }
                    }

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
/// can see editor-flagged entities (e.g. the editor camera).
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
            crate::Read<PerEntityBuffers>,
            crate::Read<Visibility>,
            crate::ResMut<RenderSchedule>,
        )>()
        .execute(
            |(
                cameras,
                globals,
                targets,
                meshes,
                materials,
                per_entity,
                visibilities,
                mut schedule_res,
            )| {
                let Some(schedule) = schedule_res.schedule_mut() else {
                    return;
                };

                for (cam_idx, _camera) in cameras.iter() {
                    let Some(target) = targets.get(cam_idx) else {
                        continue;
                    };
                    let Some(_cam_global) = globals.get(cam_idx) else {
                        continue;
                    };

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

                    let mut pass = GraphicsPass::new(format!("editor_forward_{cam_idx}"));
                    pass.set_render_targets(render_target_config);

                    for (entity_idx, render_mesh) in meshes.iter() {
                        if !per_entity.contains(entity_idx) {
                            continue;
                        }
                        let Some(render_material) = materials.get(entity_idx) else {
                            continue;
                        };
                        if let Some(vis) = visibilities.get(entity_idx)
                            && !vis.is_visible()
                        {
                            continue;
                        }

                        if let Some(instance) = render_material.pass(RenderPassType::Forward) {
                            pass.add_draw(Arc::clone(&render_mesh.mesh), Arc::clone(instance));
                        }
                    }

                    let mut graph = schedule.acquire_graph();
                    graph.add_graphics_pass(pass);
                    schedule.submit(format!("editor_camera_{cam_idx}"), graph, &[]);
                }
            },
        );
        Ok(())
    }
}
