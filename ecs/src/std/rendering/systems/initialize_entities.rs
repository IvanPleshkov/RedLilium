//! Render entity initialization system.

use std::sync::Arc;

use crate::std::rendering::components::{PerEntityBuffers, RenderMaterial};
use crate::std::rendering::resources::MaterialManager;

/// Initializes GPU resources for entities that have a [`RenderMaterial`] but
/// no [`PerEntityBuffers`] yet (e.g. freshly deserialized entities).
///
/// Rebuilds the material bundle with the correct two-group binding layout
/// required by the opaque_color shader and creates [`PerEntityBuffers`].
///
/// This is an [`ExclusiveSystem`](crate::ExclusiveSystem) because it needs
/// `&mut World` to insert components on entities.
pub struct InitializeRenderEntities;

impl crate::ExclusiveSystem for InitializeRenderEntities {
    type Result = ();

    fn run(&mut self, world: &mut crate::World) -> Result<(), crate::system::SystemError> {
        if !world.has_resource::<MaterialManager>() {
            return Ok(());
        }

        // Find entities with RenderMaterial (having cpu_instance + pass_materials)
        // but no PerEntityBuffers — these need GPU initialization.
        let uninit: Vec<_> = world
            .iter_entities()
            .filter(|e| !world.is_disabled(*e))
            .filter_map(|entity| {
                if world.get::<PerEntityBuffers>(entity).is_some() {
                    return None;
                }
                let mat = world.get::<RenderMaterial>(entity)?;
                let cpu_instance = mat.cpu_instance()?.clone();
                let pass_materials = mat.pass_materials()?.to_vec();
                Some((entity, cpu_instance, pass_materials))
            })
            .collect();

        if uninit.is_empty() {
            return Ok(());
        }

        for (entity, cpu_instance, pass_materials) in uninit {
            let first_mat_name = pass_materials
                .first()
                .map(|(_, n)| n.as_str())
                .unwrap_or("");

            let mat_manager = world.resource::<MaterialManager>();
            let Some(forward_gpu) = mat_manager.get_material(first_mat_name).cloned() else {
                continue;
            };
            let ei_gpu = mat_manager.get_material("entity_index").cloned();
            let device = mat_manager.device().clone();
            drop(mat_manager);

            let cpu_material = cpu_instance.material.clone();

            let (per_entity, mut new_render_mat, bundle) = if let Some(ei_material) = &ei_gpu {
                crate::std::rendering::shaders::create_opaque_color_entity_full(
                    &device,
                    &forward_gpu,
                    ei_material,
                    &cpu_material,
                )
            } else {
                let (fwd_buf, bundle) = crate::std::rendering::shaders::create_opaque_color_entity(
                    &device,
                    &forward_gpu,
                );
                let per_entity = PerEntityBuffers::new(fwd_buf);
                let rm = RenderMaterial::with_cpu_data(
                    Arc::clone(&bundle),
                    Arc::new(redlilium_core::material::CpuMaterialInstance::new(
                        cpu_material,
                    )),
                    pass_materials.clone(),
                );
                (per_entity, rm, bundle)
            };

            // Apply the deserialized values
            new_render_mat.set_values(cpu_instance.values.clone());

            // Write values to GPU buffer
            if let Some(buf) = new_render_mat.material_uniform_buffer() {
                let bytes = crate::std::rendering::resources::pack_uniform_bytes(
                    &cpu_instance.material,
                    &cpu_instance.values,
                );
                if !bytes.is_empty() {
                    let _ = device.write_buffer(buf, 0, &bytes);
                }
                new_render_mat.mark_synced();
            }

            // Register bundle in MaterialManager for serialization
            {
                let mut mat_manager = world.resource_mut::<MaterialManager>();
                mat_manager.register_bundle(&bundle, Arc::clone(&cpu_instance), pass_materials);
            }

            let _ = world.insert(entity, new_render_mat);
            let _ = world.insert(entity, per_entity);
        }

        Ok(())
    }
}
