//! Data transfer operations for the render graph.
//!
//! Transfer operations describe GPU data copy commands that can be
//! scheduled as part of a `PassType::Transfer` pass. Operations include:
//!
//! - Buffer to buffer copies
//! - Texture to texture copies
//! - Buffer to texture uploads
//! - Texture to buffer readbacks

use std::ops::Range;
use std::sync::{Arc, Mutex};

use crate::device::GraphicsDevice;
use crate::error::GraphicsError;
use crate::resources::{Buffer, Texture};
use crate::types::{BufferDescriptor, BufferUsage, Extent3d};

/// A region within a buffer for copy operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BufferCopyRegion {
    /// Offset in bytes from the start of the source buffer.
    pub src_offset: u64,
    /// Offset in bytes from the start of the destination buffer.
    pub dst_offset: u64,
    /// Number of bytes to copy.
    pub size: u64,
}

impl BufferCopyRegion {
    /// Create a new buffer copy region.
    pub fn new(src_offset: u64, dst_offset: u64, size: u64) -> Self {
        Self {
            src_offset,
            dst_offset,
            size,
        }
    }

    /// Create a region that copies the entire source buffer from the beginning.
    pub fn whole(size: u64) -> Self {
        Self {
            src_offset: 0,
            dst_offset: 0,
            size,
        }
    }
}

/// Specifies a location within a texture for copy operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextureCopyLocation {
    /// Mip level to copy from/to.
    pub mip_level: u32,
    /// Origin within the texture (x, y, z or array layer).
    pub origin: TextureOrigin,
}

impl TextureCopyLocation {
    /// Create a new texture copy location.
    pub fn new(mip_level: u32, origin: TextureOrigin) -> Self {
        Self { mip_level, origin }
    }

    /// Location at mip level 0, origin (0, 0, 0).
    pub fn base() -> Self {
        Self::default()
    }

    /// Location at specific mip level, origin (0, 0, 0).
    pub fn mip(mip_level: u32) -> Self {
        Self {
            mip_level,
            origin: TextureOrigin::default(),
        }
    }
}

/// Origin point within a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextureOrigin {
    /// X coordinate.
    pub x: u32,
    /// Y coordinate.
    pub y: u32,
    /// Z coordinate or array layer.
    pub z: u32,
}

impl TextureOrigin {
    /// Create a new texture origin.
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Origin at (0, 0, 0).
    pub fn zero() -> Self {
        Self::default()
    }
}

/// A region of a texture to copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureCopyRegion {
    /// Source location within the texture.
    pub src: TextureCopyLocation,
    /// Destination location within the texture.
    pub dst: TextureCopyLocation,
    /// Size of the region to copy.
    pub extent: Extent3d,
}

impl TextureCopyRegion {
    /// Create a new texture copy region.
    pub fn new(src: TextureCopyLocation, dst: TextureCopyLocation, extent: Extent3d) -> Self {
        Self { src, dst, extent }
    }

    /// Create a region that copies the entire texture at mip level 0.
    pub fn whole(extent: Extent3d) -> Self {
        Self {
            src: TextureCopyLocation::base(),
            dst: TextureCopyLocation::base(),
            extent,
        }
    }
}

/// Layout of buffer data when copying to/from textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BufferTextureLayout {
    /// Offset in bytes from the start of the buffer.
    pub offset: u64,
    /// Bytes per row of texture data (must be aligned, typically to 256).
    /// If None, assumes tightly packed based on format and width.
    pub bytes_per_row: Option<u32>,
    /// Number of rows per image (for 3D textures or texture arrays).
    /// If None, assumes tightly packed based on height.
    pub rows_per_image: Option<u32>,
}

impl BufferTextureLayout {
    /// Create a new buffer texture layout.
    pub fn new(offset: u64, bytes_per_row: Option<u32>, rows_per_image: Option<u32>) -> Self {
        Self {
            offset,
            bytes_per_row,
            rows_per_image,
        }
    }

    /// Layout starting at offset 0 with tightly packed data.
    pub fn packed() -> Self {
        Self::default()
    }

    /// Layout starting at the given offset with tightly packed data.
    pub fn at_offset(offset: u64) -> Self {
        Self {
            offset,
            bytes_per_row: None,
            rows_per_image: None,
        }
    }
}

/// Buffer to texture copy region.
#[derive(Debug, Clone)]
pub struct BufferTextureCopyRegion {
    /// Layout of data in the buffer.
    pub buffer_layout: BufferTextureLayout,
    /// Location in the texture.
    pub texture_location: TextureCopyLocation,
    /// Size of the region to copy.
    pub extent: Extent3d,
}

impl BufferTextureCopyRegion {
    /// Create a new buffer-texture copy region.
    pub fn new(
        buffer_layout: BufferTextureLayout,
        texture_location: TextureCopyLocation,
        extent: Extent3d,
    ) -> Self {
        Self {
            buffer_layout,
            texture_location,
            extent,
        }
    }

    /// Create a region for copying entire texture at mip 0 from buffer offset 0.
    pub fn whole(extent: Extent3d) -> Self {
        Self {
            buffer_layout: BufferTextureLayout::packed(),
            texture_location: TextureCopyLocation::base(),
            extent,
        }
    }
}

/// A transfer operation to be executed in a transfer pass.
#[derive(Debug, Clone)]
pub enum TransferOperation {
    /// Copy data between buffers.
    BufferToBuffer {
        /// Source buffer.
        src: Arc<Buffer>,
        /// Destination buffer.
        dst: Arc<Buffer>,
        /// Regions to copy.
        regions: Vec<BufferCopyRegion>,
    },

    /// Copy data between textures.
    TextureToTexture {
        /// Source texture.
        src: Arc<Texture>,
        /// Destination texture.
        dst: Arc<Texture>,
        /// Regions to copy.
        regions: Vec<TextureCopyRegion>,
    },

    /// Upload data from a buffer to a texture.
    BufferToTexture {
        /// Source buffer containing texture data.
        src: Arc<Buffer>,
        /// Destination texture.
        dst: Arc<Texture>,
        /// Regions to copy.
        regions: Vec<BufferTextureCopyRegion>,
    },

    /// Read back data from a texture to a buffer.
    TextureToBuffer {
        /// Source texture.
        src: Arc<Texture>,
        /// Destination buffer.
        dst: Arc<Buffer>,
        /// Regions to copy.
        regions: Vec<BufferTextureCopyRegion>,
    },

    /// Upload CPU bytes into a GPU buffer through the frame graph.
    ///
    /// The write is a GPU copy (via a transient staging buffer) encoded **at
    /// this transfer pass's position in the graph** on both backends: passes
    /// ordered before it see the old contents, passes after it see the new.
    /// Ordering against neighbouring passes is handled by the automatic
    /// barrier system (the destination is declared `TransferWrite`), so —
    /// unlike [`GpuBackend::write_buffer`](crate::backend::GpuBackend) — this
    /// does not race in-flight frames.
    ///
    /// Requirements: `dst` must have `BufferUsage::COPY_DST`; `dst_offset` and
    /// the written size must be 4-byte aligned (wgpu `COPY_BUFFER_ALIGNMENT`;
    /// enforced on both backends for identical behavior).
    ///
    /// The `data` source is held by `Arc`, so its memory stays alive for the
    /// duration of the operation (no use-after-free / access violation); the
    /// backend bounds-checks `src_range` against it.
    WriteBuffer {
        /// Destination GPU buffer.
        dst: Arc<Buffer>,
        /// Byte offset into the destination.
        dst_offset: u64,
        /// Owned source bytes (typically produced via `bytemuck`).
        data: Arc<[u8]>,
        /// Sub-range of `data` to upload.
        src_range: Range<usize>,
    },

    /// Read back GPU buffer bytes to CPU **after the frame's fence**.
    ///
    /// This op records nothing during encoding; it is a marker the frame
    /// pipeline drains once the slot's fence signals, copying `src[src_range]`
    /// into `dst`. The result is therefore available one or more frames later
    /// (poll `dst`). The GPU→`src` copy (e.g. a `TextureToBuffer`) must be a
    /// separate, earlier operation; `src` must be a host-visible readback buffer.
    ReadbackBuffer {
        /// Host-visible source buffer the GPU wrote earlier this frame.
        src: Arc<Buffer>,
        /// Sub-range of `src` to read.
        src_range: Range<usize>,
        /// CPU destination, filled after the fence.
        dst: Arc<Mutex<Vec<u8>>>,
    },
}

impl TransferOperation {
    /// Create a buffer-to-buffer copy operation.
    pub fn copy_buffer(src: Arc<Buffer>, dst: Arc<Buffer>, regions: Vec<BufferCopyRegion>) -> Self {
        Self::BufferToBuffer { src, dst, regions }
    }

    /// Create a buffer-to-buffer copy of the entire source buffer.
    pub fn copy_buffer_whole(src: Arc<Buffer>, dst: Arc<Buffer>) -> Self {
        let size = src.size();
        Self::BufferToBuffer {
            src,
            dst,
            regions: vec![BufferCopyRegion::whole(size)],
        }
    }

    /// Create a texture-to-texture copy operation.
    pub fn copy_texture(
        src: Arc<Texture>,
        dst: Arc<Texture>,
        regions: Vec<TextureCopyRegion>,
    ) -> Self {
        Self::TextureToTexture { src, dst, regions }
    }

    /// Create a texture-to-texture copy of the entire source texture.
    pub fn copy_texture_whole(src: Arc<Texture>, dst: Arc<Texture>) -> Self {
        let extent = src.size();
        Self::TextureToTexture {
            src,
            dst,
            regions: vec![TextureCopyRegion::whole(extent)],
        }
    }

    /// Create a buffer-to-texture upload operation.
    pub fn upload_texture(
        src: Arc<Buffer>,
        dst: Arc<Texture>,
        regions: Vec<BufferTextureCopyRegion>,
    ) -> Self {
        Self::BufferToTexture { src, dst, regions }
    }

    /// Create a buffer-to-texture upload for the entire texture.
    pub fn upload_texture_whole(src: Arc<Buffer>, dst: Arc<Texture>) -> Self {
        let extent = dst.size();
        Self::BufferToTexture {
            src,
            dst,
            regions: vec![BufferTextureCopyRegion::whole(extent)],
        }
    }

    /// Create a texture-to-buffer readback operation.
    pub fn readback_texture(
        src: Arc<Texture>,
        dst: Arc<Buffer>,
        regions: Vec<BufferTextureCopyRegion>,
    ) -> Self {
        Self::TextureToBuffer { src, dst, regions }
    }

    /// Create a texture-to-buffer readback for the entire texture.
    pub fn readback_texture_whole(src: Arc<Texture>, dst: Arc<Buffer>) -> Self {
        let extent = src.size();
        Self::TextureToBuffer {
            src,
            dst,
            regions: vec![BufferTextureCopyRegion::whole(extent)],
        }
    }

    /// Upload tightly-packed CPU `data` into `dst` texture through the frame
    /// graph.
    ///
    /// Creates a staging buffer, fills it from `data`, and returns a
    /// `BufferToTexture` copy. The staging buffer is owned by the returned
    /// operation, so it stays alive until the frame's GPU work completes (the
    /// frame pipeline retains submitted graphs until their fence). `data` must
    /// be tightly packed (no row padding) for the texture's extent.
    pub fn upload_texture_data(
        device: &Arc<GraphicsDevice>,
        dst: Arc<Texture>,
        data: &[u8],
    ) -> Result<Self, GraphicsError> {
        let staging = device.create_buffer(
            &BufferDescriptor::new(
                data.len() as u64,
                BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            )
            .with_label("texture_upload_staging"),
        )?;
        // Fresh staging buffer — never touched by the GPU, so the mapped write
        // cannot race.
        staging.write_mapped(0, data)?;
        Ok(Self::upload_texture_whole(staging, dst))
    }

    /// Upload all of `data` into `dst` at `dst_offset` through the frame graph.
    pub fn write_buffer(dst: Arc<Buffer>, dst_offset: u64, data: Arc<[u8]>) -> Self {
        let src_range = 0..data.len();
        Self::WriteBuffer {
            dst,
            dst_offset,
            data,
            src_range,
        }
    }

    /// Read back a sub-range of `src` into `dst` after the frame's fence.
    pub fn readback_buffer(
        src: Arc<Buffer>,
        src_range: Range<usize>,
        dst: Arc<Mutex<Vec<u8>>>,
    ) -> Self {
        Self::ReadbackBuffer {
            src,
            src_range,
            dst,
        }
    }

    /// Upload a sub-range of `data` into `dst` at `dst_offset`.
    pub fn write_buffer_range(
        dst: Arc<Buffer>,
        dst_offset: u64,
        data: Arc<[u8]>,
        src_range: Range<usize>,
    ) -> Self {
        Self::WriteBuffer {
            dst,
            dst_offset,
            data,
            src_range,
        }
    }
}

/// Configuration for a transfer pass.
#[derive(Debug, Clone, Default)]
pub struct TransferConfig {
    /// Operations to execute in this transfer pass.
    pub operations: Vec<TransferOperation>,
}

impl TransferConfig {
    /// Create a new empty transfer configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a transfer operation.
    pub fn with_operation(mut self, operation: TransferOperation) -> Self {
        self.operations.push(operation);
        self
    }

    /// Add multiple transfer operations.
    pub fn with_operations(
        mut self,
        operations: impl IntoIterator<Item = TransferOperation>,
    ) -> Self {
        self.operations.extend(operations);
        self
    }

    /// Check if this config has any operations.
    pub fn has_operations(&self) -> bool {
        !self.operations.is_empty()
    }

    /// Get the number of operations.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;
    use crate::types::{
        BufferDescriptor, BufferUsage, TextureDescriptor, TextureFormat, TextureUsage,
    };

    fn create_test_resources() -> (Arc<Buffer>, Arc<Buffer>, Arc<Texture>, Arc<Texture>) {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();

        let buffer1 = device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::COPY_SRC))
            .unwrap();
        let buffer2 = device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::COPY_DST))
            .unwrap();
        let texture1 = device
            .create_texture(&TextureDescriptor::new_2d(
                256,
                256,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COPY_SRC,
            ))
            .unwrap();
        let texture2 = device
            .create_texture(&TextureDescriptor::new_2d(
                256,
                256,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COPY_DST,
            ))
            .unwrap();

        (buffer1, buffer2, texture1, texture2)
    }

    #[test]
    fn test_buffer_copy_region() {
        let region = BufferCopyRegion::new(0, 100, 512);
        assert_eq!(region.src_offset, 0);
        assert_eq!(region.dst_offset, 100);
        assert_eq!(region.size, 512);

        let whole = BufferCopyRegion::whole(1024);
        assert_eq!(whole.src_offset, 0);
        assert_eq!(whole.dst_offset, 0);
        assert_eq!(whole.size, 1024);
    }

    #[test]
    fn test_texture_copy_location() {
        let loc = TextureCopyLocation::new(2, TextureOrigin::new(10, 20, 0));
        assert_eq!(loc.mip_level, 2);
        assert_eq!(loc.origin.x, 10);
        assert_eq!(loc.origin.y, 20);

        let base = TextureCopyLocation::base();
        assert_eq!(base.mip_level, 0);
        assert_eq!(base.origin.x, 0);
    }

    #[test]
    fn test_transfer_operation_buffer_to_buffer() {
        let (src, dst, _, _) = create_test_resources();

        let op = TransferOperation::copy_buffer_whole(Arc::clone(&src), Arc::clone(&dst));
        match op {
            TransferOperation::BufferToBuffer { regions, .. } => {
                assert_eq!(regions.len(), 1);
                assert_eq!(regions[0].size, 1024);
            }
            _ => panic!("Expected BufferToBuffer"),
        }
    }

    #[test]
    fn test_transfer_operation_texture_to_texture() {
        let (_, _, src, dst) = create_test_resources();

        let op = TransferOperation::copy_texture_whole(Arc::clone(&src), Arc::clone(&dst));
        match op {
            TransferOperation::TextureToTexture { regions, .. } => {
                assert_eq!(regions.len(), 1);
                assert_eq!(regions[0].extent.width, 256);
                assert_eq!(regions[0].extent.height, 256);
            }
            _ => panic!("Expected TextureToTexture"),
        }
    }

    #[test]
    fn test_transfer_config() {
        let (src_buf, dst_buf, src_tex, dst_tex) = create_test_resources();

        let config = TransferConfig::new()
            .with_operation(TransferOperation::copy_buffer_whole(
                Arc::clone(&src_buf),
                Arc::clone(&dst_buf),
            ))
            .with_operation(TransferOperation::copy_texture_whole(src_tex, dst_tex));

        assert!(config.has_operations());
        assert_eq!(config.operation_count(), 2);
    }

    #[test]
    fn test_buffer_to_texture_upload() {
        let (src_buf, _, _, dst_tex) = create_test_resources();

        let op = TransferOperation::upload_texture_whole(src_buf, dst_tex);
        match op {
            TransferOperation::BufferToTexture { regions, .. } => {
                assert_eq!(regions.len(), 1);
                assert_eq!(regions[0].extent.width, 256);
            }
            _ => panic!("Expected BufferToTexture"),
        }
    }

    #[test]
    fn test_texture_to_buffer_readback() {
        let (_, dst_buf, src_tex, _) = create_test_resources();

        let op = TransferOperation::readback_texture_whole(src_tex, dst_buf);
        match op {
            TransferOperation::TextureToBuffer { regions, .. } => {
                assert_eq!(regions.len(), 1);
                assert_eq!(regions[0].extent.width, 256);
            }
            _ => panic!("Expected TextureToBuffer"),
        }
    }
}
