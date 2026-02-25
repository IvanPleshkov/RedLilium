use std::sync::Arc;

use redlilium_graphics::{
    ColorAttachment, DepthStencilAttachment, GraphicsPass, LoadOp, RenderTarget,
    RenderTargetConfig, StoreOp,
};

use crate::SystemContext;
use crate::std::components::{Camera, GlobalTransform, Visibility};

use super::components::{
    CameraTarget, PerEntityBuffers, RenderMaterial, RenderMesh, RenderPassType,
};
use super::resources::{MaterialManager, RenderSchedule};

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

                        if let Some(instance) = render_material.pass(RenderPassType::Forward) {
                            pass.add_draw(Arc::clone(&render_mesh.mesh), Arc::clone(instance));
                        }
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

                        if let Some(instance) = render_material.pass(RenderPassType::Forward) {
                            pass.add_draw(Arc::clone(&render_mesh.mesh), Arc::clone(instance));
                        }
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

/// Syncs dirty material property uniforms from CPU to GPU.
///
/// Iterates all [`RenderMaterial`] components and, for any whose CPU tick
/// doesn't match its GPU tick, repacks the uniform values and uploads them
/// via [`GraphicsDevice::write_buffer`](redlilium_graphics::GraphicsDevice::write_buffer).
///
/// # Access
///
/// - Write: `RenderMaterial`
/// - Resources: `Res<MaterialManager>`
pub struct SyncMaterialUniforms;

impl crate::System for SyncMaterialUniforms {
    type Result = ();

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(crate::Write<RenderMaterial>, crate::Res<MaterialManager>)>()
            .execute(|(mut materials, mat_manager)| {
                let device = mat_manager.device();
                for (_idx, mat) in materials.iter_mut() {
                    if !mat.is_dirty() {
                        continue;
                    }
                    if let Some(buffer) = mat.material_uniform_buffer()
                        && let Some(cpu_inst) = mat.cpu_instance()
                    {
                        let bytes = super::resources::pack_uniform_bytes(
                            &cpu_inst.material,
                            &cpu_inst.values,
                        );
                        if !bytes.is_empty() {
                            let buffer = Arc::clone(buffer);
                            let _ = device.write_buffer(&buffer, 0, &bytes);
                        }
                    }
                    mat.mark_synced();
                }
            });
        Ok(())
    }
}

/// Updates per-entity transform uniform buffers each frame.
///
/// Reads the first camera's view-projection matrix and writes it together
/// with each entity's model matrix (from [`GlobalTransform`]) into the
/// entity's [`PerEntityBuffers`]. Also writes entity-index uniforms when
/// a buffer is present.
///
/// # Access
///
/// - ReadAll: `Camera`
/// - Read: `GlobalTransform`, `PerEntityBuffers`
/// - Resources: `Res<MaterialManager>`
pub struct UpdatePerEntityUniforms;

impl crate::System for UpdatePerEntityUniforms {
    type Result = ();

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::ReadAll<Camera>,
            crate::Read<GlobalTransform>,
            crate::Read<PerEntityBuffers>,
            crate::Res<MaterialManager>,
        )>()
        .execute(|(cameras, globals, buffers, mat_manager)| {
            let Some((_, camera)) = cameras.iter().next() else {
                return;
            };
            let vp = camera.view_projection();
            let device = mat_manager.device();

            for (entity_idx, per_entity) in buffers.iter() {
                let model = globals
                    .get(entity_idx)
                    .map(|g| g.0)
                    .unwrap_or_else(redlilium_core::math::Mat4::identity);

                let uniforms = super::shaders::OpaqueColorUniforms {
                    view_projection: redlilium_core::math::mat4_to_cols_array_2d(&vp),
                    model: redlilium_core::math::mat4_to_cols_array_2d(&model),
                };
                let _ = device.write_buffer(
                    &per_entity.forward_buffer,
                    0,
                    bytemuck::bytes_of(&uniforms),
                );

                if let Some(ei_buffer) = &per_entity.entity_index_buffer {
                    let ei_uniforms = super::shaders::EntityIndexUniforms {
                        view_projection: redlilium_core::math::mat4_to_cols_array_2d(&vp),
                        model: redlilium_core::math::mat4_to_cols_array_2d(&model),
                        entity_index: entity_idx,
                        _padding: [0; 3],
                    };
                    let _ = device.write_buffer(ei_buffer, 0, bytemuck::bytes_of(&ei_uniforms));
                }
            }
        });
        Ok(())
    }
}
