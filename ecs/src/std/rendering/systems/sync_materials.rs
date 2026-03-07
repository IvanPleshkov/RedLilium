//! Material uniform sync system.

use std::sync::Arc;

use crate::std::rendering::components::RenderMaterial;
use crate::std::rendering::resources::MaterialManager;

/// Syncs changed material property uniforms from CPU to GPU.
///
/// Uses [`Changed<RenderMaterial>`](crate::Changed) to detect which materials
/// were mutated, then repacks the uniform values and uploads them via
/// [`GraphicsDevice::write_buffer`](redlilium_graphics::GraphicsDevice::write_buffer).
pub struct SyncMaterialUniforms;

impl crate::System for SyncMaterialUniforms {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::Read<RenderMaterial>,
            crate::Changed<RenderMaterial>,
            crate::Res<MaterialManager>,
        )>()
        .execute(|(materials, changed, mat_manager)| {
            let device = mat_manager.device();
            for (idx, mat) in materials.iter() {
                if !changed.matches(idx) {
                    continue;
                }
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
        });
        Ok(())
    }
}
