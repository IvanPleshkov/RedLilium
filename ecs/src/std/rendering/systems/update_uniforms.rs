//! Per-entity uniform update system.

use crate::std::components::{Camera, GlobalTransform};
use crate::std::rendering::components::PerEntityBuffers;
use crate::std::rendering::resources::MaterialManager;

/// Updates per-entity transform uniform buffers each frame.
///
/// Reads the first camera's view-projection matrix and writes it together
/// with each entity's model matrix (from [`GlobalTransform`]) into the
/// entity's [`PerEntityBuffers`].
pub struct UpdatePerEntityUniforms;

impl crate::System for UpdatePerEntityUniforms {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
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

                let uniforms = crate::std::rendering::shaders::OpaqueColorUniforms {
                    view_projection: redlilium_core::math::mat4_to_cols_array_2d(&vp),
                    model: redlilium_core::math::mat4_to_cols_array_2d(&model),
                };
                let _ = device.write_buffer(
                    &per_entity.forward_buffer,
                    0,
                    bytemuck::bytes_of(&uniforms),
                );

                if let Some(ei_buffer) = &per_entity.entity_index_buffer {
                    let ei_uniforms = crate::std::rendering::shaders::EntityIndexUniforms {
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
