//! Render entity initialization system.

use std::sync::Arc;

use crate::std::rendering::components::{
    MaterialBundle, MeshRenderer, PerEntityBuffers, Primitive, RenderPassType,
};
use crate::std::rendering::resources::MaterialManager;
use redlilium_core::material::CpuMaterialInstance;

/// A rebuilt bundle pending registration in the [`MaterialManager`]:
/// `(bundle, cpu_instance, pass_materials)`.
type PendingBundle = (
    Arc<MaterialBundle>,
    Arc<CpuMaterialInstance>,
    Vec<(RenderPassType, String)>,
);

/// Initializes GPU resources for entities that have a [`MeshRenderer`] but no
/// [`PerEntityBuffers`] yet (e.g. freshly deserialized entities).
///
/// Creates one per-entity transform (shared by every primitive) and rebuilds
/// each primitive's material bundle with the correct two-group binding layout
/// required by the opaque_color shader.
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

        // Find entities with a MeshRenderer (whose primitives carry CPU material
        // data) but no PerEntityBuffers — these need GPU initialization. Clone
        // out the primitives we need so we can drop the world borrow.
        let uninit: Vec<(crate::Entity, Vec<Primitive>)> = world
            .iter_entities()
            .filter(|e| !world.is_disabled(*e))
            .filter_map(|entity| {
                if world.get::<PerEntityBuffers>(entity).is_some() {
                    return None;
                }
                let renderer = world.get::<MeshRenderer>(entity)?;
                // Only initialize if at least one primitive carries CPU data.
                let needs_init = renderer.primitives.iter().any(|p| {
                    p.material.cpu_instance().is_some() && p.material.pass_materials().is_some()
                });
                if !needs_init {
                    return None;
                }
                Some((entity, renderer.primitives.clone()))
            })
            .collect();

        if uninit.is_empty() {
            return Ok(());
        }

        for (entity, primitives) in uninit {
            // Resolve device + entity-index GPU material once per entity.
            let (device, ei_gpu) = {
                let mat_manager = world.resource::<MaterialManager>();
                (
                    mat_manager.device().clone(),
                    mat_manager.get_material("entity_index").cloned(),
                )
            };

            // One shared transform for all primitives of this entity.
            let transform =
                crate::std::rendering::shaders::create_entity_transform(&device, ei_gpu.is_some());

            // Bundles to register after rebuilding (deferred to avoid holding a
            // mutable MaterialManager borrow while reading it per primitive).
            let mut to_register: Vec<PendingBundle> = Vec::new();

            let mut new_primitives: Vec<Primitive> = Vec::with_capacity(primitives.len());

            for primitive in primitives {
                // Primitives without CPU data can't be rebuilt — keep as-is.
                let (Some(cpu_instance), Some(pass_materials)) = (
                    primitive.material.cpu_instance().cloned(),
                    primitive.material.pass_materials().map(|p| p.to_vec()),
                ) else {
                    new_primitives.push(primitive);
                    continue;
                };

                let first_mat_name = pass_materials
                    .first()
                    .map(|(_, n)| n.as_str())
                    .unwrap_or("");
                let forward_gpu = {
                    let mat_manager = world.resource::<MaterialManager>();
                    mat_manager.get_material(first_mat_name).cloned()
                };
                let Some(forward_gpu) = forward_gpu else {
                    // GPU material not registered — keep the original primitive.
                    new_primitives.push(primitive);
                    continue;
                };

                let cpu_material = cpu_instance.material.clone();
                let mut material =
                    crate::std::rendering::shaders::create_opaque_color_primitive_material(
                        &device,
                        &forward_gpu,
                        ei_gpu.as_ref(),
                        &cpu_material,
                        &transform,
                    );

                // Apply the deserialized values and push them to the GPU buffer.
                material.set_values(cpu_instance.values.clone());
                if let Some(buf) = material.material_uniform_buffer() {
                    let bytes = crate::std::rendering::resources::pack_uniform_bytes(
                        &cpu_instance.material,
                        &cpu_instance.values,
                    );
                    if !bytes.is_empty() {
                        let _ = device.write_buffer(buf, 0, &bytes);
                    }
                }

                to_register.push((material.bundle().clone(), cpu_instance, pass_materials));
                new_primitives.push(Primitive {
                    mesh: primitive.mesh,
                    material,
                    aabb: primitive.aabb,
                });
            }

            // Register rebuilt bundles for future serialization.
            {
                let mut mat_manager = world.resource_mut::<MaterialManager>();
                for (bundle, cpu_instance, pass_materials) in to_register {
                    mat_manager.register_bundle(&bundle, cpu_instance, pass_materials);
                }
            }

            let _ = world.insert(entity, MeshRenderer::from_primitives(new_primitives));
            let _ = world.insert(entity, transform.buffers);
        }

        Ok(())
    }
}
