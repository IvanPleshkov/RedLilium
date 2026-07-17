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
use crate::types::{BufferDescriptor, BufferUsage, Extent3d, TextureFormat};

/// Required row-pitch alignment for buffer↔texture copies spanning more than
/// one row (wgpu's `COPY_BYTES_PER_ROW_ALIGNMENT`).
///
/// WebGPU requires it, Vulkan does not — it is enforced on **both** backends
/// so a graph that works on one backend works on the other.
pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Required offset/size alignment for buffer↔buffer copies (wgpu's
/// `COPY_BUFFER_ALIGNMENT`), enforced on both backends.
pub const COPY_BUFFER_ALIGNMENT: u64 = 4;

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

/// Validate buffer↔buffer copy regions against [`COPY_BUFFER_ALIGNMENT`].
///
/// wgpu rejects unaligned offsets/sizes at submit time; checking up front on
/// both backends turns that into a graph-encoding error with the offending
/// values in the message.
pub fn validate_buffer_copy_alignment(regions: &[BufferCopyRegion]) -> Result<(), GraphicsError> {
    for r in regions {
        if !r.src_offset.is_multiple_of(COPY_BUFFER_ALIGNMENT)
            || !r.dst_offset.is_multiple_of(COPY_BUFFER_ALIGNMENT)
            || !r.size.is_multiple_of(COPY_BUFFER_ALIGNMENT)
        {
            return Err(GraphicsError::InvalidParameter(format!(
                "buffer copy offsets and size must be {COPY_BUFFER_ALIGNMENT}-byte aligned \
                 (src_offset {}, dst_offset {}, size {})",
                r.src_offset, r.dst_offset, r.size
            )));
        }
    }
    Ok(())
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
    /// Byte pitch between consecutive rows (rows of *blocks* for compressed
    /// formats). `None` means tightly packed for the format and copy width.
    ///
    /// Copies spanning more than one row must have a pitch aligned to
    /// [`COPY_BYTES_PER_ROW_ALIGNMENT`] (256) — a WebGPU rule enforced on
    /// both backends for portability. Tightly-packed data whose natural
    /// pitch isn't 256-aligned must either pad rows or supply an explicit
    /// aligned `bytes_per_row`.
    pub bytes_per_row: Option<u32>,
    /// Number of texel rows per image slice (for 3D textures or array
    /// layers). `None` means tightly packed (`extent.height`). Must be a
    /// multiple of the format's block height.
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

    /// Resolve and validate this layout against a format and copy extent.
    ///
    /// This is the single source of truth for buffer↔texture copy layout on
    /// both backends: tight pitch is computed in *blocks* for compressed
    /// formats (not texels), and the cross-backend alignment rules are
    /// checked here so a graph behaves identically on Vulkan and wgpu.
    pub fn resolve(
        &self,
        format: TextureFormat,
        extent: Extent3d,
    ) -> Result<ResolvedBufferTextureLayout, GraphicsError> {
        // Buffer↔image copies address a single aspect; combined depth-stencil
        // has two and this API has no aspect selector.
        if format.is_depth_stencil() && format.has_stencil() {
            return Err(GraphicsError::InvalidParameter(format!(
                "buffer<->texture copies of combined depth-stencil formats ({format:?}) are not \
                 supported: the copy addresses a single aspect"
            )));
        }

        let (block_w, block_h) = format.block_dimensions();
        let block_size = format.block_size();
        let row_blocks = extent.width.div_ceil(block_w);
        let height_blocks = extent.height.div_ceil(block_h);
        let images = extent.depth.max(1);

        let tight_bytes_per_row = row_blocks * block_size;
        let bytes_per_row = self.bytes_per_row.unwrap_or(tight_bytes_per_row);
        if bytes_per_row < tight_bytes_per_row {
            return Err(GraphicsError::InvalidParameter(format!(
                "bytes_per_row {bytes_per_row} smaller than the copy's tight pitch \
                 {tight_bytes_per_row} ({row_blocks} blocks x {block_size} bytes)"
            )));
        }
        if !bytes_per_row.is_multiple_of(block_size) {
            return Err(GraphicsError::InvalidParameter(format!(
                "bytes_per_row {bytes_per_row} is not a multiple of the format block size \
                 {block_size}"
            )));
        }

        let multi_row = height_blocks * images > 1;
        if multi_row && !bytes_per_row.is_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT) {
            return Err(GraphicsError::InvalidParameter(format!(
                "bytes_per_row {bytes_per_row} must be {COPY_BYTES_PER_ROW_ALIGNMENT}-byte \
                 aligned for copies spanning more than one row (wgpu rule, enforced on both \
                 backends); pad the rows or pass an explicit aligned bytes_per_row"
            )));
        }

        // The tight default is the copy height rounded UP to whole blocks:
        // edge mips of compressed formats legally copy extents smaller than a
        // block (a 2×2 BC mip), but the buffer still holds one full block row
        // (#120). An explicit value must be block-aligned by itself.
        let rows_per_image_texels = self.rows_per_image.unwrap_or(height_blocks * block_h);
        if !rows_per_image_texels.is_multiple_of(block_h) {
            return Err(GraphicsError::InvalidParameter(format!(
                "rows_per_image {rows_per_image_texels} is not a multiple of the format block \
                 height {block_h}"
            )));
        }
        if rows_per_image_texels < extent.height {
            return Err(GraphicsError::InvalidParameter(format!(
                "rows_per_image {rows_per_image_texels} smaller than the copy height {}",
                extent.height
            )));
        }

        Ok(ResolvedBufferTextureLayout {
            offset: self.offset,
            bytes_per_row,
            row_length_texels: (bytes_per_row / block_size) * block_w,
            rows_per_image_texels,
            rows_per_image_blocks: rows_per_image_texels / block_h,
            multi_row,
        })
    }
}

/// A [`BufferTextureLayout`] resolved against a format and extent — tight
/// values filled in, block math applied, alignment rules validated.
///
/// Produced by [`BufferTextureLayout::resolve`] and consumed by both
/// backends' copy encoders.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedBufferTextureLayout {
    /// Byte offset into the buffer.
    pub offset: u64,
    /// Byte pitch between consecutive block rows.
    pub bytes_per_row: u32,
    /// Row pitch expressed in texels (Vulkan `bufferRowLength`; a multiple of
    /// the block width).
    pub row_length_texels: u32,
    /// Texel rows per image slice (Vulkan `bufferImageHeight`).
    pub rows_per_image_texels: u32,
    /// Block rows per image slice (wgpu `rows_per_image`).
    pub rows_per_image_blocks: u32,
    /// Whether the copy spans more than one block row (wgpu then requires an
    /// explicit `bytes_per_row`).
    pub multi_row: bool,
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

    /// Generate the full mip chain of `texture` from mip 0 on the GPU (#96).
    ///
    /// A linear blit chain (mip N → N+1) — the simplest correct baseline. The
    /// per-mip subresource layout transitions are internal to the op: it starts
    /// from and ends in the tracker-declared whole-image `TransferWrite`
    /// (`TRANSFER_DST`) layout, so the whole-image layout model stays truthful.
    ///
    /// `vkCmdBlitImage` is legal only on graphics-capable queues, so a graph
    /// containing this op is never routed to the transfer or async-compute
    /// queue ([`requires_graphics_queue`](crate::graph::RenderGraph::requires_graphics_queue)).
    /// `texture` must have been created with a full `mip_level_count`, `COPY_SRC`
    /// usage (blit reads lower mips), and a blit-eligible format; the loader
    /// arranges this behind [`DeviceCapabilities::mip_generation`](crate::DeviceCapabilities).
    GenerateMipmaps {
        /// The texture whose mips 1.. are generated from mip 0.
        texture: Arc<Texture>,
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

    /// Create a mip-chain generation operation (#96).
    pub fn generate_mipmaps(texture: Arc<Texture>) -> Self {
        Self::GenerateMipmaps { texture }
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
    ///
    /// The row pitch in `dst` is the tight pitch rounded up to
    /// [`COPY_BYTES_PER_ROW_ALIGNMENT`] (256) — the alignment multi-row
    /// copies require on both backends. Size `dst` accordingly
    /// (`aligned_pitch * rows * slices`) and read rows at that stride.
    pub fn readback_texture_whole(src: Arc<Texture>, dst: Arc<Buffer>) -> Self {
        let extent = src.size();
        let format = src.format();
        let (block_w, block_h) = format.block_dimensions();
        let tight_bpr = extent.width.div_ceil(block_w) * format.block_size();
        let multi_row = extent.height.div_ceil(block_h) * extent.depth.max(1) > 1;
        let bytes_per_row =
            multi_row.then(|| tight_bpr.next_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT));
        Self::TextureToBuffer {
            src,
            dst,
            regions: vec![BufferTextureCopyRegion::new(
                BufferTextureLayout::new(0, bytes_per_row, None),
                TextureCopyLocation::base(),
                extent,
            )],
        }
    }

    /// Upload tightly-packed CPU `data` into `dst` texture through the frame
    /// graph.
    ///
    /// Creates a staging buffer, fills it from `data`, and returns a
    /// `BufferToTexture` copy. The staging buffer is owned by the returned
    /// operation, so it stays alive until the frame's GPU work completes (the
    /// frame pipeline retains submitted graphs until their fence). `data` must
    /// be tightly packed (no row padding) for the texture's extent, layers
    /// back to back.
    ///
    /// If the tight row pitch isn't [`COPY_BYTES_PER_ROW_ALIGNMENT`]-aligned
    /// (required for multi-row copies on both backends), the rows are padded
    /// into the staging buffer here, while the data is still on the CPU — so
    /// any texture width uploads correctly.
    pub fn upload_texture_data(
        device: &Arc<GraphicsDevice>,
        dst: Arc<Texture>,
        data: &[u8],
    ) -> Result<Self, GraphicsError> {
        let extent = dst.size();
        let (staging, bytes_per_row) =
            stage_texture_bytes(device, dst.format(), extent, data, "upload_texture_data")?;
        Ok(Self::BufferToTexture {
            src: staging,
            dst,
            regions: vec![BufferTextureCopyRegion::new(
                BufferTextureLayout::new(0, bytes_per_row, None),
                TextureCopyLocation::base(),
                extent,
            )],
        })
    }

    /// Upload one tightly-packed `(mip, layer)` image into `dst` through the
    /// frame graph (#120).
    ///
    /// Like [`upload_texture_data`](Self::upload_texture_data) but targets a
    /// single mip level and array layer, sizing the copy region to the mip's
    /// extent (rounded up to whole blocks for compressed formats — partial
    /// edge blocks on NPOT sizes still occupy a full block in `data`). For
    /// cube(-array) textures `layer` is the face-flattened index; for 3D
    /// textures the image spans the mip's whole (shrinking) depth and `layer`
    /// must be 0. This is how mip chains and cubemaps loaded from containers
    /// (KTX2) reach the GPU — one operation per stored image.
    pub fn upload_texture_level(
        device: &Arc<GraphicsDevice>,
        dst: Arc<Texture>,
        mip: u32,
        layer: u32,
        data: &[u8],
    ) -> Result<Self, GraphicsError> {
        if mip >= dst.mip_level_count() {
            return Err(GraphicsError::InvalidParameter(format!(
                "upload_texture_level: mip {mip} out of range ({} levels)",
                dst.mip_level_count()
            )));
        }
        let (layers, base_depth) = dst.descriptor().layers_and_depth();
        if layer >= layers {
            return Err(GraphicsError::InvalidParameter(format!(
                "upload_texture_level: layer {layer} out of range ({layers} layers)"
            )));
        }
        let base = dst.size();
        let extent = Extent3d {
            width: (base.width >> mip).max(1),
            height: (base.height >> mip).max(1),
            // 3D mips shrink in depth; array layers are addressed via
            // `origin.z` instead and copy one slice at a time.
            depth: if layers > 1 {
                1
            } else {
                (base_depth >> mip).max(1)
            },
        };
        let (staging, bytes_per_row) =
            stage_texture_bytes(device, dst.format(), extent, data, "upload_texture_level")?;
        Ok(Self::BufferToTexture {
            src: staging,
            dst,
            regions: vec![BufferTextureCopyRegion::new(
                BufferTextureLayout::new(0, bytes_per_row, None),
                TextureCopyLocation {
                    mip_level: mip,
                    origin: TextureOrigin::new(0, 0, layer),
                },
                extent,
            )],
        })
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

/// Fill a fresh staging buffer with tightly-packed texel `data` for a copy of
/// `extent`, returning the buffer and the explicit `bytes_per_row` (when the
/// tight pitch had to be padded to [`COPY_BYTES_PER_ROW_ALIGNMENT`], a rule
/// both backends enforce for multi-row copies). Validates `data`'s size
/// against the format arithmetic — blocks rounded up for compressed formats.
fn stage_texture_bytes(
    device: &Arc<GraphicsDevice>,
    format: TextureFormat,
    extent: Extent3d,
    data: &[u8],
    what: &str,
) -> Result<(Arc<Buffer>, Option<u32>), GraphicsError> {
    let (block_w, block_h) = format.block_dimensions();
    let block_size = format.block_size();
    let row_blocks = extent.width.div_ceil(block_w);
    let col_blocks = extent.height.div_ceil(block_h);
    let images = extent.depth.max(1);
    let tight_bpr = row_blocks * block_size;
    let total_rows = col_blocks as usize * images as usize;

    let expected = tight_bpr as usize * total_rows;
    if data.len() != expected {
        return Err(GraphicsError::InvalidParameter(format!(
            "{what}: data size {} does not match the tightly-packed size {expected} \
             for a {}x{}x{} copy of {format:?}",
            data.len(),
            extent.width,
            extent.height,
            extent.depth
        )));
    }

    let multi_row = total_rows > 1;
    let needs_padding = multi_row && !tight_bpr.is_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT);
    let (staging_bytes, bytes_per_row) = if needs_padding {
        let padded_bpr = tight_bpr.next_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT);
        let mut padded = vec![0u8; padded_bpr as usize * total_rows];
        for row in 0..total_rows {
            let src = row * tight_bpr as usize;
            let dst_off = row * padded_bpr as usize;
            padded[dst_off..dst_off + tight_bpr as usize]
                .copy_from_slice(&data[src..src + tight_bpr as usize]);
        }
        (std::borrow::Cow::Owned(padded), Some(padded_bpr))
    } else {
        (std::borrow::Cow::Borrowed(data), None)
    };

    let staging = device.create_buffer(
        &BufferDescriptor::new(
            staging_bytes.len() as u64,
            // RING is the cross-backend "CPU-written, host-visible" flag:
            // on Vulkan it forces host-visible memory (under ADR-021 / #89 a
            // buffer without a mapping flag lands device-local, where the
            // `write_mapped` below cannot map it), while mapping to no wgpu
            // usage — so it avoids wgpu's MAP_WRITE/COPY_DST conflict. COPY_DST
            // is what wgpu's `Queue::write_buffer` requires; COPY_SRC makes it
            // a valid buffer→texture copy source.
            BufferUsage::COPY_SRC | BufferUsage::COPY_DST | BufferUsage::RING,
        )
        .with_label("texture_upload_staging"),
    )?;
    // Fresh staging buffer — never touched by the GPU, so the mapped write
    // cannot race.
    staging.write_mapped(0, &staging_bytes)?;
    Ok((staging, bytes_per_row))
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
    fn resolve_tight_uncompressed() {
        // 256x256 RGBA8: tight pitch 1024 is 256-aligned, no explicit layout needed.
        let layout = BufferTextureLayout::packed()
            .resolve(TextureFormat::Rgba8Unorm, Extent3d::new_2d(256, 256))
            .unwrap();
        assert_eq!(layout.bytes_per_row, 1024);
        assert_eq!(layout.row_length_texels, 256);
        assert_eq!(layout.rows_per_image_texels, 256);
        assert_eq!(layout.rows_per_image_blocks, 256);
        assert!(layout.multi_row);
    }

    #[test]
    fn resolve_rejects_unaligned_tight_pitch() {
        // 100x100 RGBA8: tight pitch 400 isn't 256-aligned — must error with
        // guidance instead of silently striding at 256 (the old behavior
        // sheared the image and read out of bounds).
        let err = BufferTextureLayout::packed()
            .resolve(TextureFormat::Rgba8Unorm, Extent3d::new_2d(100, 100))
            .unwrap_err();
        assert!(err.to_string().contains("256"));

        // Single-row copies have no alignment requirement.
        let layout = BufferTextureLayout::packed()
            .resolve(TextureFormat::Rgba8Unorm, Extent3d::new_2d(100, 1))
            .unwrap();
        assert_eq!(layout.bytes_per_row, 400);
        assert!(!layout.multi_row);
    }

    #[test]
    fn resolve_block_compressed_pitch_is_in_blocks() {
        // 64x64 BC7: 16 block columns x 16 bytes = 256 bytes per block row —
        // NOT 64 texels x 16 bytes (the old texel-based math was 4x too big).
        let layout = BufferTextureLayout::packed()
            .resolve(TextureFormat::Bc7RgbaUnorm, Extent3d::new_2d(64, 64))
            .unwrap();
        assert_eq!(layout.bytes_per_row, 256);
        // Vulkan bufferRowLength is texels: 16 blocks x 4 texels.
        assert_eq!(layout.row_length_texels, 64);
        // wgpu rows_per_image is block rows: 64 texel rows / 4.
        assert_eq!(layout.rows_per_image_blocks, 16);
        assert_eq!(layout.rows_per_image_texels, 64);
    }

    #[test]
    fn resolve_rejects_combined_depth_stencil() {
        let err = BufferTextureLayout::packed()
            .resolve(TextureFormat::Depth24PlusStencil8, Extent3d::new_2d(64, 64))
            .unwrap_err();
        assert!(err.to_string().contains("depth-stencil"));
    }

    #[test]
    fn resolve_rejects_short_explicit_pitch() {
        let err = BufferTextureLayout::new(0, Some(512), None)
            .resolve(TextureFormat::Rgba8Unorm, Extent3d::new_2d(256, 2))
            .unwrap_err();
        assert!(err.to_string().contains("tight pitch"));
    }

    #[test]
    fn validate_buffer_copy_alignment_enforces_4_bytes() {
        assert!(validate_buffer_copy_alignment(&[BufferCopyRegion::new(0, 4, 16)]).is_ok());
        assert!(validate_buffer_copy_alignment(&[BufferCopyRegion::new(0, 2, 16)]).is_err());
        assert!(validate_buffer_copy_alignment(&[BufferCopyRegion::new(0, 0, 3)]).is_err());
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
