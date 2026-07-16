//! Ray-tracing acceleration structures (#110, ADR-032).
//!
//! A [`Blas`] (bottom-level acceleration structure) wraps triangle geometry
//! read from vertex/index buffers; a [`Tlas`] (top-level) holds instances of
//! BLASes and is what shaders bind and trace against
//! ([`BindingType::AccelerationStructure`](crate::BindingType)).
//!
//! Both are created by [`GraphicsDevice::create_blas`] /
//! [`GraphicsDevice::create_tlas`] and **built on the GPU through the frame
//! graph** via
//! [`AccelerationStructureBuildPass`](crate::AccelerationStructureBuildPass) —
//! there is no direct build method, mirroring the engine's upload rules.
//! Backing storage and build scratch are ordinary engine [`Buffer`]s so the
//! automatic barrier system tracks builds and traversals like any other GPU
//! access.
//!
//! [`GraphicsDevice::create_blas`]: crate::GraphicsDevice::create_blas
//! [`GraphicsDevice::create_tlas`]: crate::GraphicsDevice::create_tlas

use std::sync::Arc;

use parking_lot::Mutex;

use crate::backend::GpuAccelerationStructure;
use crate::device::GraphicsDevice;
use crate::mesh::IndexFormat;

use super::Buffer;

/// One triangle-geometry range of a BLAS.
///
/// Vertex positions must be tightly readable as `R32G32B32_SFLOAT` at
/// `vertex_offset + i * vertex_stride` — the layout every engine mesh uses
/// for its position attribute. Both buffers must be created with
/// [`BufferUsage::ACCELERATION_STRUCTURE_INPUT`](crate::types::BufferUsage).
#[derive(Debug, Clone)]
pub struct BlasTriangles {
    /// Buffer holding vertex positions.
    pub vertex_buffer: Arc<Buffer>,
    /// Byte offset of the first position in `vertex_buffer`.
    pub vertex_offset: u64,
    /// Byte stride between consecutive positions.
    pub vertex_stride: u64,
    /// Number of vertices addressable from `vertex_offset`.
    pub vertex_count: u32,
    /// Index buffer; `None` for non-indexed triangle soup.
    pub index_buffer: Option<Arc<Buffer>>,
    /// Byte offset of the first index in `index_buffer`.
    pub index_offset: u64,
    /// Index format (ignored when `index_buffer` is `None`).
    pub index_format: IndexFormat,
    /// Number of triangles (`index_count / 3` for indexed geometry).
    pub triangle_count: u32,
    /// Whether the geometry is opaque (no any-hit shading). Opaque geometry
    /// lets ray traversal skip candidate confirmation — keep `true` unless
    /// alpha-tested hits are needed.
    pub opaque: bool,
}

impl BlasTriangles {
    /// Indexed triangle geometry with tightly packed `vec3<f32>` positions.
    pub fn indexed(
        vertex_buffer: Arc<Buffer>,
        vertex_count: u32,
        index_buffer: Arc<Buffer>,
        index_format: IndexFormat,
        triangle_count: u32,
    ) -> Self {
        Self {
            vertex_buffer,
            vertex_offset: 0,
            vertex_stride: 12,
            vertex_count,
            index_buffer: Some(index_buffer),
            index_offset: 0,
            index_format,
            triangle_count,
            opaque: true,
        }
    }

    /// Set the byte stride between consecutive vertex positions.
    pub fn with_vertex_stride(mut self, stride: u64) -> Self {
        self.vertex_stride = stride;
        self
    }
}

/// Descriptor for creating a [`Blas`].
#[derive(Debug, Clone)]
pub struct BlasDescriptor {
    /// Debug label.
    pub label: Option<String>,
    /// Triangle geometry ranges. All are built together into one BLAS;
    /// shaders see them as `geometry_index` 0..n within the instance.
    pub geometries: Vec<BlasTriangles>,
}

impl BlasDescriptor {
    /// Create a descriptor from geometry ranges.
    pub fn new(geometries: Vec<BlasTriangles>) -> Self {
        Self {
            label: None,
            geometries,
        }
    }

    /// Set the debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Descriptor for creating a [`Tlas`].
#[derive(Debug, Clone)]
pub struct TlasDescriptor {
    /// Debug label.
    pub label: Option<String>,
    /// Maximum number of instances writable via [`Tlas::write_instances`].
    /// Sizes the internal per-frame instance buffers and the AS itself.
    pub max_instances: u32,
}

impl TlasDescriptor {
    /// Create a descriptor for up to `max_instances` instances.
    pub fn new(max_instances: u32) -> Self {
        Self {
            label: None,
            max_instances,
        }
    }

    /// Set the debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// One BLAS instance inside a TLAS.
#[derive(Debug, Clone)]
pub struct TlasInstance {
    /// The BLAS this instance references. The TLAS keeps it alive until the
    /// instances are overwritten.
    pub blas: Arc<Blas>,
    /// Object-to-world transform, **row-major 3×4** (rows of the matrix; the
    /// translation sits in each row's fourth column) — the Vulkan
    /// `VkTransformMatrixKHR` convention.
    pub transform: [[f32; 4]; 3],
    /// 24-bit user value shaders read back as `instance_custom_index`
    /// (values above 2^24-1 are truncated).
    pub custom_index: u32,
    /// 8-bit visibility mask ANDed against the ray's cull mask.
    pub mask: u8,
}

impl TlasInstance {
    /// Instance with an identity transform, custom index 0 and mask 0xFF.
    pub fn new(blas: Arc<Blas>) -> Self {
        Self {
            blas,
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            custom_index: 0,
            mask: 0xFF,
        }
    }

    /// Set the row-major 3×4 object-to-world transform.
    pub fn with_transform(mut self, transform: [[f32; 4]; 3]) -> Self {
        self.transform = transform;
        self
    }

    /// Set the 24-bit custom index (`instance_custom_index` in shaders).
    pub fn with_custom_index(mut self, custom_index: u32) -> Self {
        self.custom_index = custom_index;
        self
    }
}

/// Sizes a backend reports for building an acceleration structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccelBuildSizes {
    /// Bytes of backing storage for the AS itself.
    pub acceleration_structure_size: u64,
    /// Bytes of build scratch, already padded so an address rounded up to the
    /// device's scratch alignment still leaves the required space.
    pub build_scratch_size: u64,
}

/// A bottom-level acceleration structure over triangle geometry (#110).
///
/// Created by [`GraphicsDevice::create_blas`]; built on the GPU by adding it
/// to an [`AccelerationStructureBuildPass`](crate::AccelerationStructureBuildPass).
/// A BLAS holds strong references to its geometry buffers — they stay alive
/// (and re-buildable) for the BLAS's lifetime.
///
/// [`GraphicsDevice::create_blas`]: crate::GraphicsDevice::create_blas
pub struct Blas {
    descriptor: BlasDescriptor,
    backing: Arc<Buffer>,
    scratch: Arc<Buffer>,
    gpu_handle: GpuAccelerationStructure,
    device: Arc<GraphicsDevice>,
}

impl Blas {
    /// Create a new BLAS (called by GraphicsDevice).
    pub(crate) fn new(
        device: Arc<GraphicsDevice>,
        descriptor: BlasDescriptor,
        backing: Arc<Buffer>,
        scratch: Arc<Buffer>,
        gpu_handle: GpuAccelerationStructure,
    ) -> Self {
        Self {
            descriptor,
            backing,
            scratch,
            gpu_handle,
            device,
        }
    }

    /// Get the descriptor (geometry ranges) this BLAS was created from.
    pub fn descriptor(&self) -> &BlasDescriptor {
        &self.descriptor
    }

    /// The buffer backing the AS storage. Public so passes can declare
    /// build/traversal accesses on it; its contents are opaque.
    pub fn backing_buffer(&self) -> &Arc<Buffer> {
        &self.backing
    }

    /// The build scratch buffer.
    pub fn scratch_buffer(&self) -> &Arc<Buffer> {
        &self.scratch
    }

    /// Get the GPU handle.
    pub fn gpu_handle(&self) -> &GpuAccelerationStructure {
        &self.gpu_handle
    }

    /// Get the parent device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Get the debug label, if set.
    pub fn label(&self) -> Option<&str> {
        self.descriptor.label.as_deref()
    }
}

impl std::fmt::Debug for Blas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Blas")
            .field("label", &self.descriptor.label)
            .field("geometries", &self.descriptor.geometries.len())
            .finish()
    }
}

/// Instance data written by the most recent [`Tlas::write_instances`] call:
/// what the next TLAS build consumes and what barrier inference declares.
#[derive(Debug, Default)]
pub(crate) struct TlasContents {
    /// BLASes referenced by the written instances (kept alive; their backing
    /// buffers are read both by the TLAS build and by ray traversal).
    pub blases: Vec<Arc<Blas>>,
    /// Number of instances written.
    pub instance_count: u32,
    /// Which internal instance-buffer slot holds them.
    pub slot: usize,
}

/// Byte size of one `VkAccelerationStructureInstanceKHR`.
const INSTANCE_STRIDE: u64 = 64;

/// A top-level acceleration structure over BLAS instances (#110).
///
/// Per frame: call [`write_instances`](Self::write_instances) with the
/// current transforms, then add the TLAS to an
/// [`AccelerationStructureBuildPass`](crate::AccelerationStructureBuildPass)
/// in that frame's graph. Instance data lands in one of
/// [`MAX_FRAMES_IN_FLIGHT`](crate::MAX_FRAMES_IN_FLIGHT) internal host-visible
/// buffers, rotated per write, so the CPU never overwrites data a still
/// in-flight frame's build is reading.
pub struct Tlas {
    descriptor: TlasDescriptor,
    backing: Arc<Buffer>,
    scratch: Arc<Buffer>,
    /// One host-visible instance buffer per frame slot.
    instance_buffers: Vec<Arc<Buffer>>,
    gpu_handle: GpuAccelerationStructure,
    contents: Mutex<TlasContents>,
    device: Arc<GraphicsDevice>,
}

impl Tlas {
    /// Create a new TLAS (called by GraphicsDevice).
    pub(crate) fn new(
        device: Arc<GraphicsDevice>,
        descriptor: TlasDescriptor,
        backing: Arc<Buffer>,
        scratch: Arc<Buffer>,
        instance_buffers: Vec<Arc<Buffer>>,
        gpu_handle: GpuAccelerationStructure,
    ) -> Self {
        Self {
            descriptor,
            backing,
            scratch,
            instance_buffers,
            gpu_handle,
            contents: Mutex::new(TlasContents::default()),
            device,
        }
    }

    /// Write the instance array for the next build.
    ///
    /// Rotates to the next internal instance-buffer slot and packs the
    /// instances in the `VkAccelerationStructureInstanceKHR` layout. Call once
    /// per frame *before* adding the TLAS build to that frame's graph; the
    /// build consumes exactly this data.
    ///
    /// # Errors
    ///
    /// Fails when `instances.len()` exceeds the descriptor's `max_instances`
    /// or a referenced BLAS has no device address (non-Vulkan backend).
    pub fn write_instances(&self, instances: &[TlasInstance]) -> Result<(), crate::GraphicsError> {
        if instances.len() as u32 > self.descriptor.max_instances {
            return Err(crate::GraphicsError::InvalidParameter(format!(
                "TLAS '{}' was created for {} instances; {} written",
                self.label().unwrap_or("unnamed"),
                self.descriptor.max_instances,
                instances.len()
            )));
        }

        let slot = {
            let contents = self.contents.lock();
            (contents.slot + 1) % self.instance_buffers.len()
        };

        // Pack VkAccelerationStructureInstanceKHR: 48 bytes row-major 3×4
        // transform, custom_index:24 | mask:8, sbt_offset:24 | flags:8,
        // 8-byte AS device address.
        let mut data = Vec::with_capacity(instances.len() * INSTANCE_STRIDE as usize);
        for instance in instances {
            for row in &instance.transform {
                for value in row {
                    data.extend_from_slice(&value.to_le_bytes());
                }
            }
            let custom_and_mask =
                (instance.custom_index & 0x00FF_FFFF) | ((instance.mask as u32) << 24);
            data.extend_from_slice(&custom_and_mask.to_le_bytes());
            // SBT record offset 0 (no RT pipelines) and no instance flags:
            // opacity comes from the BLAS geometry flags.
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&instance.blas.gpu_handle().device_address().to_le_bytes());
        }

        if !data.is_empty() {
            self.instance_buffers[slot].write_mapped(0, &data)?;
        }

        *self.contents.lock() = TlasContents {
            blases: instances.iter().map(|i| Arc::clone(&i.blas)).collect(),
            instance_count: instances.len() as u32,
            slot,
        };
        Ok(())
    }

    /// Snapshot of the last written instances: (instance buffer, count,
    /// referenced BLASes). What the next build consumes and what barrier
    /// inference must declare.
    pub(crate) fn current_build_inputs(&self) -> (Arc<Buffer>, u32, Vec<Arc<Blas>>) {
        let contents = self.contents.lock();
        (
            Arc::clone(&self.instance_buffers[contents.slot]),
            contents.instance_count,
            contents.blases.clone(),
        )
    }

    /// Get the descriptor this TLAS was created from.
    pub fn descriptor(&self) -> &TlasDescriptor {
        &self.descriptor
    }

    /// The buffer backing the AS storage (see [`Blas::backing_buffer`]).
    pub fn backing_buffer(&self) -> &Arc<Buffer> {
        &self.backing
    }

    /// The build scratch buffer.
    pub fn scratch_buffer(&self) -> &Arc<Buffer> {
        &self.scratch
    }

    /// Get the GPU handle.
    pub fn gpu_handle(&self) -> &GpuAccelerationStructure {
        &self.gpu_handle
    }

    /// Get the parent device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Get the debug label, if set.
    pub fn label(&self) -> Option<&str> {
        self.descriptor.label.as_deref()
    }
}

impl std::fmt::Debug for Tlas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tlas")
            .field("label", &self.descriptor.label)
            .field("max_instances", &self.descriptor.max_instances)
            .finish()
    }
}

/// Bytes of instance data a TLAS slot buffer must hold.
pub(crate) fn instance_buffer_size(max_instances: u32) -> u64 {
    (max_instances as u64).max(1) * INSTANCE_STRIDE
}

static_assertions::assert_impl_all!(Blas: Send, Sync);
static_assertions::assert_impl_all!(Tlas: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::AccelerationStructureBuildPass;
    use crate::graph::resource_usage::BufferAccessMode;
    use crate::instance::{BackendType, GraphicsInstance, InstanceParameters};
    use crate::types::{BufferDescriptor, BufferUsage};

    /// The dummy backend accepts AS creation so the whole front-end path is
    /// testable headlessly (#110).
    fn dummy_device() -> Arc<GraphicsDevice> {
        let instance = GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Dummy),
        )
        .unwrap();
        instance.create_device().unwrap()
    }

    fn test_blas(device: &Arc<GraphicsDevice>) -> Arc<Blas> {
        let vertex_buffer = device
            .create_buffer(&BufferDescriptor::new(
                36,
                BufferUsage::ACCELERATION_STRUCTURE_INPUT,
            ))
            .unwrap();
        device
            .create_blas(&BlasDescriptor::new(vec![BlasTriangles {
                vertex_buffer,
                vertex_offset: 0,
                vertex_stride: 12,
                vertex_count: 3,
                index_buffer: None,
                index_offset: 0,
                index_format: IndexFormat::Uint32,
                triangle_count: 1,
                opaque: true,
            }]))
            .unwrap()
    }

    /// #110: geometry buffers without the AS-input usage flag are rejected at
    /// creation, not at build time.
    #[test]
    fn blas_requires_as_input_usage() {
        let device = dummy_device();
        let plain = device
            .create_buffer(&BufferDescriptor::new(36, BufferUsage::VERTEX))
            .unwrap();
        let result = device.create_blas(&BlasDescriptor::new(vec![BlasTriangles {
            vertex_buffer: plain,
            vertex_offset: 0,
            vertex_stride: 12,
            vertex_count: 3,
            index_buffer: None,
            index_offset: 0,
            index_format: IndexFormat::Uint32,
            triangle_count: 1,
            opaque: true,
        }]));
        assert!(result.is_err());
    }

    /// #110: write_instances rotates slots and enforces max_instances; the
    /// build-input snapshot reflects the latest write.
    #[test]
    fn tlas_write_instances_rotates_and_validates() {
        let device = dummy_device();
        let blas = test_blas(&device);
        let tlas = device.create_tlas(&TlasDescriptor::new(2)).unwrap();

        tlas.write_instances(&[TlasInstance::new(Arc::clone(&blas))])
            .unwrap();
        let (buffer_a, count, blases) = tlas.current_build_inputs();
        assert_eq!(count, 1);
        assert_eq!(blases.len(), 1);

        tlas.write_instances(&[
            TlasInstance::new(Arc::clone(&blas)),
            TlasInstance::new(Arc::clone(&blas)).with_custom_index(7),
        ])
        .unwrap();
        let (buffer_b, count, _) = tlas.current_build_inputs();
        assert_eq!(count, 2);
        // Consecutive writes land in different slots so an in-flight frame's
        // build never reads data the CPU is overwriting.
        assert!(!Arc::ptr_eq(&buffer_a, &buffer_b));

        // One instance over the creation-time capacity fails.
        let too_many: Vec<_> = (0..3)
            .map(|_| TlasInstance::new(Arc::clone(&blas)))
            .collect();
        assert!(tlas.write_instances(&too_many).is_err());
    }

    /// #110: the build pass declares build inputs, AS writes, and (for TLAS)
    /// reads of the referenced BLAS backings — the automatic barrier system
    /// keys on exactly these declarations.
    #[test]
    fn build_pass_declares_expected_usages() {
        let device = dummy_device();
        let blas = test_blas(&device);
        let tlas = device.create_tlas(&TlasDescriptor::new(4)).unwrap();
        tlas.write_instances(&[TlasInstance::new(Arc::clone(&blas))])
            .unwrap();

        let mut pass = AccelerationStructureBuildPass::new("blas".into());
        pass.add_blas_build(Arc::clone(&blas));
        let usage = pass.infer_resource_usage();
        // Geometry input + backing write + scratch write.
        assert_eq!(usage.buffer_usages.len(), 3);
        assert!(
            usage
                .buffer_usages
                .iter()
                .any(|u| { u.access == BufferAccessMode::AccelerationStructureBuildInput })
        );
        assert_eq!(
            usage
                .buffer_usages
                .iter()
                .filter(|u| u.access == BufferAccessMode::AccelerationStructureWrite)
                .count(),
            2
        );

        let mut pass = AccelerationStructureBuildPass::new("tlas".into());
        pass.add_tlas_build(Arc::clone(&tlas));
        let usage = pass.infer_resource_usage();
        // Instance input + BLAS backing read + backing write + scratch write.
        assert_eq!(usage.buffer_usages.len(), 4);
        assert!(usage.buffer_usages.iter().any(|u| {
            u.access == BufferAccessMode::AccelerationStructureBuildRead
                && Arc::ptr_eq(&u.buffer, blas.backing_buffer())
        }));
    }
}
