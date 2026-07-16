//! Vulkan acceleration structures (#110, ADR-032).
//!
//! Creation, build-size queries, and frame-graph build encoding for
//! `VK_KHR_acceleration_structure`, consumed by ray queries
//! (`VK_KHR_ray_query`). Everything here is gated on the backend's
//! `accel_loader` — `Some` exactly when `DeviceCapabilities::ray_query` is
//! true, so callers on unsupported devices get a uniform engine-level error
//! instead of a null-fn-pointer crash.

use ash::vk;

use crate::error::GraphicsError;
use crate::mesh::IndexFormat;
use crate::resources::{AccelBuildSizes, BlasDescriptor, CompactionPhase};

use crate::backend::{GpuAccelerationStructure, GpuBuffer};

use super::VulkanBackend;

/// BLAS build flags. `PREFER_FAST_TRACE` always; `ALLOW_COMPACTION` when the
/// BLAS opted into compaction (#110 phase 3) — this flag must match between the
/// size query and the build, so both go through here.
fn blas_build_flags(allow_compaction: bool) -> vk::BuildAccelerationStructureFlagsKHR {
    let mut flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
    if allow_compaction {
        flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION;
    }
    flags
}

/// Which level of acceleration structure to create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationStructureKind {
    /// Bottom-level: triangle geometry.
    BottomLevel,
    /// Top-level: BLAS instances.
    TopLevel,
}

/// Round `value` up to the next multiple of `alignment` (a power of two).
fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

impl VulkanBackend {
    /// The AS entry points, or the uniform unsupported error (#110).
    fn accel(&self) -> Result<&ash::khr::acceleration_structure::Device, GraphicsError> {
        self.accel_loader.as_ref().ok_or_else(|| {
            GraphicsError::FeatureNotSupported(
                "acceleration structures require DeviceCapabilities::ray_query \
                 (VK_KHR_acceleration_structure + VK_KHR_ray_query, #110)"
                    .to_string(),
            )
        })
    }

    /// Device address of an engine buffer (requires `bufferDeviceAddress`,
    /// which the ray-query bundle enabled).
    fn buffer_address(&self, buffer: &GpuBuffer) -> Result<vk::DeviceAddress, GraphicsError> {
        let GpuBuffer::Vulkan { buffer, .. } = buffer else {
            return Err(GraphicsError::InvalidParameter(
                "acceleration-structure build references a non-Vulkan buffer".to_string(),
            ));
        };
        let info = vk::BufferDeviceAddressInfo::default().buffer(*buffer);
        Ok(unsafe { self.device.get_buffer_device_address(&info) })
    }

    /// `minAccelerationStructureScratchOffsetAlignment` of the device.
    fn min_scratch_alignment(&self) -> u64 {
        let mut accel_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut accel_props);
        unsafe {
            self.instance
                .get_physical_device_properties2(self.physical_device, &mut props2)
        };
        // The spec guarantees a power of two; guard against a zero from a
        // broken driver so align_up stays sound.
        (accel_props.min_acceleration_structure_scratch_offset_alignment as u64).max(1)
    }

    /// Build the `VkAccelerationStructureGeometryKHR` array + per-geometry
    /// primitive counts for a BLAS descriptor.
    ///
    /// With `resolve_addresses` false the device addresses are left null —
    /// legal for `vkGetAccelerationStructureBuildSizesKHR`, which ignores
    /// them and reads only counts/formats/strides.
    fn blas_geometries(
        &self,
        descriptor: &BlasDescriptor,
        resolve_addresses: bool,
    ) -> Result<(Vec<vk::AccelerationStructureGeometryKHR<'static>>, Vec<u32>), GraphicsError> {
        let mut geometries = Vec::with_capacity(descriptor.geometries.len());
        let mut primitive_counts = Vec::with_capacity(descriptor.geometries.len());

        for geometry in &descriptor.geometries {
            let vertex_address = if resolve_addresses {
                self.buffer_address(geometry.vertex_buffer.gpu_handle())? + geometry.vertex_offset
            } else {
                0
            };

            let mut triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: vertex_address,
                })
                .vertex_stride(geometry.vertex_stride)
                .max_vertex(geometry.vertex_count.saturating_sub(1));

            triangles = match &geometry.index_buffer {
                Some(index_buffer) => {
                    let index_address = if resolve_addresses {
                        self.buffer_address(index_buffer.gpu_handle())? + geometry.index_offset
                    } else {
                        0
                    };
                    triangles
                        .index_type(match geometry.index_format {
                            IndexFormat::Uint16 => vk::IndexType::UINT16,
                            IndexFormat::Uint32 => vk::IndexType::UINT32,
                        })
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: index_address,
                        })
                }
                None => triangles.index_type(vk::IndexType::NONE_KHR),
            };

            let flags = if geometry.opaque {
                vk::GeometryFlagsKHR::OPAQUE
            } else {
                vk::GeometryFlagsKHR::empty()
            };

            geometries.push(
                vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                    .flags(flags),
            );
            primitive_counts.push(geometry.triangle_count);
        }

        Ok((geometries, primitive_counts))
    }

    /// The instances geometry of a TLAS build (one per TLAS).
    fn tlas_geometry(
        instance_address: vk::DeviceAddress,
    ) -> vk::AccelerationStructureGeometryKHR<'static> {
        let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instance_address,
            });
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances })
    }

    /// Query create/build sizes for a BLAS (#110). Scratch size is padded by
    /// the scratch alignment so an aligned-up address still fits.
    pub fn blas_build_sizes(
        &self,
        descriptor: &BlasDescriptor,
    ) -> Result<AccelBuildSizes, GraphicsError> {
        let accel = self.accel()?;
        let (geometries, primitive_counts) = self.blas_geometries(descriptor, false)?;

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(blas_build_flags(descriptor.allow_compaction))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &primitive_counts,
                &mut sizes,
            )
        };

        Ok(AccelBuildSizes {
            acceleration_structure_size: sizes.acceleration_structure_size,
            build_scratch_size: sizes.build_scratch_size + self.min_scratch_alignment(),
        })
    }

    /// Query create/build sizes for a TLAS of up to `max_instances` (#110).
    pub fn tlas_build_sizes(&self, max_instances: u32) -> Result<AccelBuildSizes, GraphicsError> {
        let accel = self.accel()?;
        let geometries = [Self::tlas_geometry(0)];

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);

        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            accel.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[max_instances],
                &mut sizes,
            )
        };

        Ok(AccelBuildSizes {
            acceleration_structure_size: sizes.acceleration_structure_size,
            build_scratch_size: sizes.build_scratch_size + self.min_scratch_alignment(),
        })
    }

    /// Create an acceleration structure over `backing` (#110). The backing
    /// buffer was created with `ACCELERATION_STRUCTURE_STORAGE` usage and at
    /// least `size` bytes by the device layer.
    pub fn create_acceleration_structure(
        &self,
        backing: &GpuBuffer,
        size: u64,
        kind: AccelerationStructureKind,
    ) -> Result<GpuAccelerationStructure, GraphicsError> {
        let accel = self.accel()?;
        let GpuBuffer::Vulkan { buffer, .. } = backing else {
            return Err(GraphicsError::InvalidParameter(
                "acceleration-structure backing is a non-Vulkan buffer".to_string(),
            ));
        };

        let ty = match kind {
            AccelerationStructureKind::BottomLevel => {
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
            }
            AccelerationStructureKind::TopLevel => vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        };
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(*buffer)
            .offset(0)
            .size(size)
            .ty(ty);

        let handle =
            unsafe { accel.create_acceleration_structure(&create_info, None) }.map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to create acceleration structure: {:?}",
                    e
                ))
            })?;

        let device_address = unsafe {
            accel.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        };

        Ok(GpuAccelerationStructure::Vulkan {
            accel_loader: accel.clone(),
            handle,
            device_address,
        })
    }

    /// Encode an acceleration-structure build pass (#110).
    ///
    /// Builds within one pass are independent by contract (the pass API
    /// documents it); hazards *between* passes — BLAS build → TLAS build →
    /// ray-query consumption — are resolved by the automatic barrier system
    /// from the pass's declared `BufferAccessMode`s, like every other pass.
    pub(super) fn encode_acceleration_structure_build_pass(
        &self,
        cmd: vk::CommandBuffer,
        pass: &crate::graph::AccelerationStructureBuildPass,
    ) -> Result<(), GraphicsError> {
        use crate::graph::AccelerationStructureBuild;

        let accel = self.accel()?;
        let scratch_alignment = self.min_scratch_alignment();

        // Compactable BLASes built this pass whose compacted size still needs a
        // query written. Collected here and served after all builds, behind one
        // build→read barrier (#110 phase 3).
        let mut pending_size_queries: Vec<(
            vk::AccelerationStructureKHR,
            std::sync::Arc<crate::resources::BlasCompaction>,
        )> = Vec::new();

        for build in pass.builds() {
            match build {
                AccelerationStructureBuild::Blas(blas) => {
                    let (geometries, primitive_counts) =
                        self.blas_geometries(blas.descriptor(), true)?;
                    let ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = primitive_counts
                        .iter()
                        .map(|&count| {
                            vk::AccelerationStructureBuildRangeInfoKHR::default()
                                .primitive_count(count)
                        })
                        .collect();

                    let handle = match blas.gpu_handle().as_ref() {
                        GpuAccelerationStructure::Vulkan { handle, .. } => *handle,
                        _ => {
                            return Err(GraphicsError::InvalidParameter(
                                "BLAS build references a non-Vulkan acceleration structure"
                                    .to_string(),
                            ));
                        }
                    };
                    let scratch = align_up(
                        self.buffer_address(blas.scratch_buffer().gpu_handle())?,
                        scratch_alignment,
                    );

                    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .flags(blas_build_flags(blas.descriptor().allow_compaction))
                        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                        .dst_acceleration_structure(handle)
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: scratch,
                        })
                        .geometries(&geometries);

                    unsafe {
                        accel.cmd_build_acceleration_structures(cmd, &[build_info], &[&ranges])
                    };

                    // Queue a compacted-size query for a freshly built
                    // compactable BLAS (#110 phase 3).
                    if let Some(compaction) = blas.compaction()
                        && compaction.phase() == CompactionPhase::NeedsQuery
                    {
                        pending_size_queries.push((handle, std::sync::Arc::clone(compaction)));
                    }
                }
                AccelerationStructureBuild::Tlas(tlas) => {
                    let (instance_buffer, instance_count, _blases) = tlas.current_build_inputs();
                    let instance_address = self.buffer_address(instance_buffer.gpu_handle())?;
                    let geometries = [Self::tlas_geometry(instance_address)];
                    let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default()
                        .primitive_count(instance_count)];

                    let handle = match tlas.gpu_handle() {
                        GpuAccelerationStructure::Vulkan { handle, .. } => *handle,
                        _ => {
                            return Err(GraphicsError::InvalidParameter(
                                "TLAS build references a non-Vulkan acceleration structure"
                                    .to_string(),
                            ));
                        }
                    };
                    let scratch = align_up(
                        self.buffer_address(tlas.scratch_buffer().gpu_handle())?,
                        scratch_alignment,
                    );

                    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                        .dst_acceleration_structure(handle)
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: scratch,
                        })
                        .geometries(&geometries);

                    unsafe {
                        accel.cmd_build_acceleration_structures(cmd, &[build_info], &[&ranges])
                    };
                }
                AccelerationStructureBuild::Compact(copy) => {
                    // Copy the original structure into the freshly allocated,
                    // exactly compacted-sized one (#110 phase 3). The
                    // build→read barrier and dst-write barrier are handled by
                    // the automatic barrier system from the pass's declared
                    // AS read/write on the two backing buffers.
                    let src = match copy.src_handle.as_ref() {
                        GpuAccelerationStructure::Vulkan { handle, .. } => *handle,
                        _ => {
                            return Err(GraphicsError::InvalidParameter(
                                "compaction copy source is a non-Vulkan acceleration structure"
                                    .to_string(),
                            ));
                        }
                    };
                    let dst = match copy.dst_handle.as_ref() {
                        GpuAccelerationStructure::Vulkan { handle, .. } => *handle,
                        _ => {
                            return Err(GraphicsError::InvalidParameter(
                                "compaction copy destination is a non-Vulkan acceleration structure"
                                    .to_string(),
                            ));
                        }
                    };
                    let info = vk::CopyAccelerationStructureInfoKHR::default()
                        .src(src)
                        .dst(dst)
                        .mode(vk::CopyAccelerationStructureModeKHR::COMPACT);
                    unsafe { accel.cmd_copy_acceleration_structure(cmd, &info) };
                }
            }
        }

        // Serve the collected compacted-size queries after a build→read barrier
        // so the properties are read only once the builds complete (#110 phase
        // 3). Each query is reset on this command buffer inside `reserve`.
        if !pending_size_queries.is_empty()
            && let Some(manager) = &self.compaction_queries
        {
            let barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR);
            let deps =
                vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier));
            unsafe { self.device.cmd_pipeline_barrier2(cmd, &deps) };

            let slot = self.current_slot.load(std::sync::atomic::Ordering::Relaxed);
            let mut manager = manager.lock();
            for (handle, compaction) in &pending_size_queries {
                if let Some((pool, query)) = manager.reserve(&self.device, slot, cmd, compaction) {
                    unsafe {
                        accel.cmd_write_acceleration_structures_properties(
                            cmd,
                            std::slice::from_ref(handle),
                            vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                            pool,
                            query,
                        )
                    };
                    compaction.set_phase(CompactionPhase::AwaitingSize);
                }
                // A full pool leaves the BLAS in NeedsQuery to retry next frame.
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::align_up;

    #[test]
    fn align_up_rounds_to_power_of_two() {
        assert_eq!(align_up(0, 128), 0);
        assert_eq!(align_up(1, 128), 128);
        assert_eq!(align_up(128, 128), 128);
        assert_eq!(align_up(129, 256), 256);
    }
}
