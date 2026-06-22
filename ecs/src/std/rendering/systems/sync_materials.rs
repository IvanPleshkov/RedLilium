//! Material uniform sync system.

use std::sync::Arc;

use crate::std::rendering::components::MeshRenderer;
use crate::std::rendering::resources::MaterialManager;

/// Syncs changed material property uniforms from CPU to GPU.
///
/// Uses [`Changed<MeshRenderer>`](crate::Changed) to detect which renderers
/// were mutated, then repacks each primitive's uniform values and uploads them
/// via [`GraphicsDevice::write_buffer`](redlilium_graphics::GraphicsDevice::write_buffer).
pub struct SyncMaterialUniforms;

impl crate::System for SyncMaterialUniforms {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::Read<MeshRenderer>,
            crate::Changed<MeshRenderer>,
            crate::Res<MaterialManager>,
        )>()
        .execute(|(renderers, changed, mat_manager)| {
            let device = mat_manager.device();
            for (idx, renderer) in renderers.iter() {
                if !changed.matches(idx) {
                    continue;
                }
                for primitive in &renderer.primitives {
                    let mat = &primitive.material;
                    if let Some(buffer) = mat.material_uniform_buffer()
                        && let Some(cpu_inst) = mat.cpu_instance()
                    {
                        let bytes = crate::std::rendering::resources::pack_uniform_bytes(
                            &cpu_inst.material,
                            &cpu_inst.values,
                        );
                        if !bytes.is_empty() {
                            let buffer = Arc::clone(buffer);
                            let _ = device.write_buffer(&buffer, 0, &bytes);
                        }
                    }
                }
            }
        });
        Ok(())
    }
}
