//! Shared preflight for graph transfers, before any backend records commands.

use std::sync::Arc;

use crate::resources::{Buffer, Texture};
use crate::types::{BufferUsage, Extent3d, TextureDimension, TextureFormat, TextureUsage};
use crate::{GraphicsDevice, GraphicsError};

use super::transfer::{
    BufferTextureCopyRegion, TextureCopyLocation, validate_buffer_copy_alignment,
};
use super::{RenderGraph, TransferOperation};

fn invalid(message: impl Into<String>) -> GraphicsError {
    GraphicsError::InvalidParameter(message.into())
}

impl RenderGraph {
    pub(crate) fn validate_transfers(
        &self,
        device: &Arc<GraphicsDevice>,
    ) -> Result<(), GraphicsError> {
        for pass in self.passes() {
            let Some(config) = pass.as_transfer().and_then(|pass| pass.transfer_config()) else {
                continue;
            };
            for (index, operation) in config.operations.iter().enumerate() {
                operation.validate(device).map_err(|error| {
                    let context = |message| {
                        format!(
                            "transfer pass {:?}, operation {index}: {message}",
                            pass.name()
                        )
                    };
                    match error {
                        GraphicsError::InvalidParameter(message) => invalid(context(message)),
                        GraphicsError::FeatureNotSupported(message) => {
                            GraphicsError::FeatureNotSupported(context(message))
                        }
                        other => other,
                    }
                })?;
            }
        }
        Ok(())
    }
}

fn buffer(
    buffer: &Buffer,
    device: &Arc<GraphicsDevice>,
    usage: BufferUsage,
) -> Result<(), GraphicsError> {
    if !Arc::ptr_eq(buffer.device(), device) {
        return Err(invalid("buffer belongs to a different graphics device"));
    }
    if !buffer.descriptor().usage.contains(usage) {
        return Err(invalid(format!("buffer requires {usage:?} usage")));
    }
    if buffer.is_map_pending() {
        return Err(invalid("buffer has a pending CPU readback mapping"));
    }
    Ok(())
}

fn texture(
    texture: &Texture,
    device: &Arc<GraphicsDevice>,
    usage: TextureUsage,
) -> Result<(), GraphicsError> {
    if !Arc::ptr_eq(texture.device(), device) {
        return Err(invalid("texture belongs to a different graphics device"));
    }
    if !texture.usage().contains(usage) {
        return Err(invalid(format!("texture requires {usage:?} usage")));
    }
    Ok(())
}

fn range(offset: u64, size: u64, capacity: u64) -> Result<(), GraphicsError> {
    if offset.checked_add(size).is_none_or(|end| end > capacity) {
        return Err(invalid(format!(
            "range at offset {offset} with size {size} exceeds capacity {capacity}"
        )));
    }
    Ok(())
}

fn regions_nonempty<T>(regions: &[T]) -> Result<(), GraphicsError> {
    if regions.is_empty() {
        return Err(invalid("copy must contain at least one region"));
    }
    Ok(())
}

// sRGB changes sampling interpretation, not the bytes copied between images.
fn copy_format(format: TextureFormat) -> TextureFormat {
    use TextureFormat::*;
    match format {
        Rgba8UnormSrgb => Rgba8Unorm,
        Bgra8UnormSrgb => Bgra8Unorm,
        Bc1RgbaUnormSrgb => Bc1RgbaUnorm,
        Bc2RgbaUnormSrgb => Bc2RgbaUnorm,
        Bc3RgbaUnormSrgb => Bc3RgbaUnorm,
        Bc7RgbaUnormSrgb => Bc7RgbaUnorm,
        Etc2Rgb8UnormSrgb => Etc2Rgb8Unorm,
        Etc2Rgb8A1UnormSrgb => Etc2Rgb8A1Unorm,
        Etc2Rgba8UnormSrgb => Etc2Rgba8Unorm,
        Astc4x4UnormSrgb => Astc4x4Unorm,
        Astc5x4UnormSrgb => Astc5x4Unorm,
        Astc5x5UnormSrgb => Astc5x5Unorm,
        Astc6x5UnormSrgb => Astc6x5Unorm,
        Astc6x6UnormSrgb => Astc6x6Unorm,
        Astc8x5UnormSrgb => Astc8x5Unorm,
        Astc8x6UnormSrgb => Astc8x6Unorm,
        Astc8x8UnormSrgb => Astc8x8Unorm,
        Astc10x5UnormSrgb => Astc10x5Unorm,
        Astc10x6UnormSrgb => Astc10x6Unorm,
        Astc10x8UnormSrgb => Astc10x8Unorm,
        Astc10x10UnormSrgb => Astc10x10Unorm,
        Astc12x10UnormSrgb => Astc12x10Unorm,
        Astc12x12UnormSrgb => Astc12x12Unorm,
        other => other,
    }
}

fn dimension_class(dimension: TextureDimension) -> u8 {
    match dimension {
        TextureDimension::D1 | TextureDimension::D1Array => 1,
        TextureDimension::D2
        | TextureDimension::D2Array
        | TextureDimension::Cube
        | TextureDimension::CubeArray => 2,
        TextureDimension::D3 => 3,
    }
}

/// Logical texel bounds: edge mips of compressed textures can be smaller
/// than a block. Encoders handle physical block rounding where required.
fn texture_region(
    texture: &Texture,
    location: TextureCopyLocation,
    extent: Extent3d,
) -> Result<(), GraphicsError> {
    if location.mip_level >= texture.mip_level_count() {
        return Err(invalid("texture mip level is out of bounds"));
    }
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 {
        return Err(invalid("texture copy extent must be nonzero"));
    }
    let shrink = |n: u32| n.checked_shr(location.mip_level).unwrap_or(0).max(1);
    let width = shrink(texture.width());
    let height = shrink(texture.height());
    let depth = match texture.dimension() {
        TextureDimension::D3 => shrink(texture.depth()),
        TextureDimension::D1Array | TextureDimension::D2Array => texture.depth().max(1),
        TextureDimension::Cube => 6,
        TextureDimension::CubeArray => texture
            .depth()
            .max(1)
            .checked_mul(6)
            .ok_or_else(|| invalid("cube array layer count overflows"))?,
        _ => 1,
    };
    let origin = location.origin;
    range(origin.x.into(), extent.width.into(), width.into())?;
    range(origin.y.into(), extent.height.into(), height.into())?;
    range(origin.z.into(), extent.depth.into(), depth.into())?;
    if (texture.format().is_depth_stencil() || texture.sample_count() > 1)
        && (origin.x != 0 || origin.y != 0 || extent.width != width || extent.height != height)
    {
        return Err(invalid(
            "depth/stencil and multisampled copies must cover the entire mip width and height",
        ));
    }
    let (bw, bh) = texture.format().block_dimensions();
    if !origin.x.is_multiple_of(bw)
        || !origin.y.is_multiple_of(bh)
        || (!extent.width.is_multiple_of(bw) && origin.x + extent.width != width)
        || (!extent.height.is_multiple_of(bh) && origin.y + extent.height != height)
    {
        return Err(invalid(
            "compressed copy must be block-aligned except at the mip edge",
        ));
    }
    Ok(())
}

fn buffer_texture_regions(
    buffer: &Buffer,
    texture: &Texture,
    regions: &[BufferTextureCopyRegion],
    upload: bool,
) -> Result<(), GraphicsError> {
    regions_nonempty(regions)?;
    if texture.sample_count() != 1 {
        return Err(invalid(
            "buffer/texture copies require a single-sampled texture",
        ));
    }
    let format = texture.format();
    // No aspect selector is exposed. Depth24Plus also has no portable byte
    // representation; WebGPU does not permit buffer uploads of Depth32Float.
    if format.has_stencil()
        || format == TextureFormat::Depth24Plus
        || (upload && format == TextureFormat::Depth32Float)
    {
        return Err(invalid(format!(
            "buffer/texture copy is unsupported for {format:?} in this direction"
        )));
    }
    for region in regions {
        texture_region(texture, region.texture_location, region.extent)?;
        let layout = region.buffer_layout.resolve(format, region.extent)?;
        // Four bytes also keeps copies eligible for dedicated Vulkan transfer
        // queues. Compressed copies additionally require block-size alignment.
        if !layout.offset.is_multiple_of(4)
            || !layout.offset.is_multiple_of(u64::from(format.block_size()))
        {
            return Err(invalid(
                "buffer/texture offset must be aligned to 4 bytes and the format block size",
            ));
        }
        let (bw, bh) = format.block_dimensions();
        let row_bytes =
            u64::from(region.extent.width.div_ceil(bw)) * u64::from(format.block_size());
        let rows = u64::from(region.extent.height.div_ceil(bh));
        // The last row needs only its texel data, not a full padded row.
        let bytes = u64::from(region.extent.depth - 1)
            .checked_mul(u64::from(layout.rows_per_image_blocks))
            .and_then(|n| n.checked_add(rows - 1))
            .and_then(|n| n.checked_mul(u64::from(layout.bytes_per_row)))
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or_else(|| invalid("buffer/texture footprint overflows"))?;
        range(layout.offset, bytes, buffer.size())?;
    }
    Ok(())
}

impl TransferOperation {
    fn validate(&self, device: &Arc<GraphicsDevice>) -> Result<(), GraphicsError> {
        match self {
            Self::BufferToBuffer { src, dst, regions } => {
                buffer(src, device, BufferUsage::COPY_SRC)?;
                buffer(dst, device, BufferUsage::COPY_DST)?;
                if Arc::ptr_eq(src, dst) {
                    return Err(invalid("copying within the same buffer is unsupported"));
                }
                regions_nonempty(regions)?;
                validate_buffer_copy_alignment(regions)?;
                for region in regions {
                    if region.size == 0 {
                        return Err(invalid("buffer copy size must be nonzero"));
                    }
                    range(region.src_offset, region.size, src.size())?;
                    range(region.dst_offset, region.size, dst.size())?;
                }
            }
            Self::WriteBuffer {
                dst,
                dst_offset,
                data,
                src_range,
            } => {
                buffer(dst, device, BufferUsage::COPY_DST)?;
                let data = data
                    .get(src_range.clone())
                    .ok_or_else(|| invalid("write source range is out of bounds or reversed"))?;
                range(*dst_offset, data.len() as u64, dst.size())?;
                if !data.is_empty()
                    && (!dst_offset.is_multiple_of(4) || !data.len().is_multiple_of(4))
                {
                    return Err(invalid(
                        "buffer write offset and size must be 4-byte aligned",
                    ));
                }
            }
            Self::ReadbackBuffer { src, src_range, .. } => {
                buffer(src, device, BufferUsage::MAP_READ)?;
                let size = src_range
                    .end
                    .checked_sub(src_range.start)
                    .ok_or_else(|| invalid("readback range is reversed"))?;
                range(src_range.start as u64, size as u64, src.size())?;
                if size != 0 && (!src_range.start.is_multiple_of(8) || !size.is_multiple_of(4)) {
                    return Err(invalid(
                        "readback offset must be 8-byte aligned and size 4-byte aligned",
                    ));
                }
            }
            Self::BufferToTexture { src, dst, regions } => {
                buffer(src, device, BufferUsage::COPY_SRC)?;
                texture(dst, device, TextureUsage::COPY_DST)?;
                buffer_texture_regions(src, dst, regions, true)?;
            }
            Self::TextureToBuffer { src, dst, regions } => {
                texture(src, device, TextureUsage::COPY_SRC)?;
                buffer(dst, device, BufferUsage::COPY_DST)?;
                buffer_texture_regions(dst, src, regions, false)?;
            }
            Self::TextureToTexture { src, dst, regions } => {
                texture(src, device, TextureUsage::COPY_SRC)?;
                texture(dst, device, TextureUsage::COPY_DST)?;
                if Arc::ptr_eq(src, dst) {
                    return Err(invalid(
                        "copying within the same texture requires subresource tracking and is unsupported",
                    ));
                }
                if copy_format(src.format()) != copy_format(dst.format()) {
                    return Err(invalid(
                        "texture copies require matching formats except for sRGB encoding",
                    ));
                }
                if dimension_class(src.dimension()) != dimension_class(dst.dimension())
                    || src.sample_count() != dst.sample_count()
                {
                    return Err(invalid(
                        "texture copy dimensions and sample counts must match",
                    ));
                }
                regions_nonempty(regions)?;
                for region in regions {
                    texture_region(src, region.src, region.extent)?;
                    texture_region(dst, region.dst, region.extent)?;
                }
            }
            Self::GenerateMipmaps { texture: target } => {
                texture(
                    target,
                    device,
                    TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
                )?;
                // A one-level texture remains a no-op on all backends.
                if target.mip_level_count() <= 1 {
                    return Ok(());
                }
                if !matches!(
                    target.dimension(),
                    TextureDimension::D3
                        | TextureDimension::D2
                        | TextureDimension::D2Array
                        | TextureDimension::Cube
                        | TextureDimension::CubeArray
                ) || target.sample_count() != 1
                {
                    return Err(invalid(
                        "mip generation requires a single-sampled 2D texture, array, cubemap, or 3D volume",
                    ));
                }
                if target.dimension().is_cubemap() && target.width() != target.height() {
                    return Err(invalid("cubemap faces must be square"));
                }
                if !device.supports_mipmap_generation(target.format()) {
                    return Err(GraphicsError::FeatureNotSupported(format!(
                        "mip generation for {:?}",
                        target.format()
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GpuTexture;
    use crate::graph::{BufferCopyRegion, BufferTextureLayout, TextureOrigin};
    use crate::{
        BackendType, BufferDescriptor, GraphicsInstance, InstanceParameters, TextureDescriptor,
    };
    use std::sync::Mutex;

    fn device() -> Arc<GraphicsDevice> {
        GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap()
    }
    fn buf(device: &Arc<GraphicsDevice>, size: u64, usage: BufferUsage) -> Arc<Buffer> {
        device
            .create_buffer(&BufferDescriptor::new(size, usage))
            .unwrap()
    }
    fn tex(device: &Arc<GraphicsDevice>, descriptor: TextureDescriptor) -> Arc<Texture> {
        // Validation is purely descriptor-based. Dummy handles let us test
        // compressed/MSAA formats independently of host GPU capabilities.
        Arc::new(Texture::new(device.clone(), descriptor, GpuTexture::Dummy))
    }
    fn desc(width: u32, height: u32) -> TextureDescriptor {
        TextureDescriptor::new_2d(
            width,
            height,
            TextureFormat::Rgba8Unorm,
            TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        )
    }
    fn check_bad(op: TransferOperation, device: &Arc<GraphicsDevice>, message: &str) {
        let error = op.validate(device).expect_err("invalid transfer must fail");
        assert!(
            matches!(error, GraphicsError::InvalidParameter(_)),
            "{error:?}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }
    fn region(
        layout: BufferTextureLayout,
        mip: u32,
        z: u32,
        extent: Extent3d,
    ) -> BufferTextureCopyRegion {
        BufferTextureCopyRegion::new(
            layout,
            TextureCopyLocation::new(mip, TextureOrigin::new(0, 0, z)),
            extent,
        )
    }

    #[test]
    fn buffer_copy_bounds_alignment_and_self_copy() {
        let d = device();
        let usage = BufferUsage::COPY_SRC | BufferUsage::COPY_DST;
        let src = buf(&d, 16, usage);
        let dst = buf(&d, 16, usage);
        TransferOperation::copy_buffer(
            src.clone(),
            dst.clone(),
            vec![BufferCopyRegion::new(12, 12, 4)],
        )
        .validate(&d)
        .unwrap();
        for r in [
            BufferCopyRegion::new(16, 0, 4),
            BufferCopyRegion::new(0, 16, 4),
            BufferCopyRegion::new(u64::MAX - 3, 0, 4),
        ] {
            check_bad(
                TransferOperation::copy_buffer(src.clone(), dst.clone(), vec![r]),
                &d,
                "exceeds",
            );
        }
        check_bad(
            TransferOperation::copy_buffer(
                src.clone(),
                dst.clone(),
                vec![BufferCopyRegion::new(1, 0, 4)],
            ),
            &d,
            "aligned",
        );
        check_bad(
            TransferOperation::copy_buffer(
                src.clone(),
                dst.clone(),
                vec![BufferCopyRegion::whole(0)],
            ),
            &d,
            "nonzero",
        );
        check_bad(
            TransferOperation::copy_buffer(src.clone(), dst, vec![]),
            &d,
            "at least one",
        );
        check_bad(
            TransferOperation::copy_buffer_whole(src.clone(), src),
            &d,
            "same buffer",
        );
    }

    #[test]
    fn writes_and_readbacks_validate_ranges_usage_and_mapping_alignment() {
        let d = device();
        let dst = buf(&d, 16, BufferUsage::COPY_DST);
        let bytes: Arc<[u8]> = Arc::from([1u8; 4]);
        for offset in [16, u64::MAX - 3] {
            check_bad(
                TransferOperation::write_buffer(dst.clone(), offset, bytes.clone()),
                &d,
                "exceeds",
            );
        }
        for range in [3..2, 0..5] {
            check_bad(
                TransferOperation::write_buffer_range(dst.clone(), 0, bytes.clone(), range),
                &d,
                "source range",
            );
        }
        TransferOperation::write_buffer(dst.clone(), 16, Arc::from([]))
            .validate(&d)
            .unwrap();
        check_bad(
            TransferOperation::readback_buffer(dst, 0..4, Arc::new(Mutex::new(vec![]))),
            &d,
            "MAP_READ",
        );
        let src = buf(&d, 16, BufferUsage::MAP_READ);
        for range in [4..8, 0..3] {
            check_bad(
                TransferOperation::readback_buffer(
                    src.clone(),
                    range,
                    Arc::new(Mutex::new(vec![])),
                ),
                &d,
                "aligned",
            );
        }
        check_bad(
            TransferOperation::readback_buffer(src.clone(), 16..20, Arc::new(Mutex::new(vec![]))),
            &d,
            "exceeds",
        );
        check_bad(
            TransferOperation::readback_buffer(src.clone(), 8..4, Arc::new(Mutex::new(vec![]))),
            &d,
            "reversed",
        );
        TransferOperation::readback_buffer(src, 8..12, Arc::new(Mutex::new(vec![])))
            .validate(&d)
            .unwrap();
    }

    #[test]
    fn rejects_wrong_device_and_missing_usage() {
        let d = device();
        let foreign = device();
        let src = buf(&d, 16, BufferUsage::COPY_SRC);
        let dst = buf(&foreign, 16, BufferUsage::COPY_DST);
        check_bad(
            TransferOperation::copy_buffer_whole(src.clone(), dst),
            &d,
            "different graphics device",
        );
        check_bad(
            TransferOperation::write_buffer(src, 0, Arc::from([0u8; 4])),
            &d,
            "COPY_DST",
        );
        let source = tex(&d, desc(1, 1));
        check_bad(
            TransferOperation::copy_texture_whole(source.clone(), tex(&foreign, desc(1, 1))),
            &d,
            "different graphics device",
        );
        let mut descriptor = desc(1, 1);
        descriptor.usage = TextureUsage::COPY_SRC;
        check_bad(
            TransferOperation::copy_texture_whole(source, tex(&d, descriptor)),
            &d,
            "COPY_DST",
        );
    }

    #[test]
    fn footprint_accounts_for_rows_layers_and_unpadded_last_row() {
        let d = device();
        let texture = tex(
            &d,
            TextureDescriptor::new_2d_array(
                4,
                2,
                3,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COPY_DST,
            ),
        );
        // 2 layers with a four-row stride: 4*256 + 1*256 + 16 = 1296.
        let r = region(
            BufferTextureLayout::new(0, Some(256), Some(4)),
            0,
            1,
            Extent3d::new_3d(4, 2, 2),
        );
        let copy = |size| {
            TransferOperation::upload_texture(
                buf(&d, size, BufferUsage::COPY_SRC),
                texture.clone(),
                vec![r.clone()],
            )
        };
        copy(1296).validate(&d).unwrap();
        check_bad(copy(1295), &d, "exceeds");
    }

    #[test]
    fn texture_ranges_cover_mips_array_layers_and_volume_depth() {
        let d = device();
        let buffer = buf(&d, 4096, BufferUsage::COPY_SRC);
        let array = tex(
            &d,
            TextureDescriptor::new_2d_array(
                8,
                8,
                3,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COPY_DST,
            )
            .with_mip_levels(4),
        );
        let copy = |texture, mip, z, extent| {
            TransferOperation::upload_texture(
                buffer.clone(),
                texture,
                vec![region(
                    BufferTextureLayout::new(0, Some(256), None),
                    mip,
                    z,
                    extent,
                )],
            )
        };
        copy(array.clone(), 3, 2, Extent3d::new_2d(1, 1))
            .validate(&d)
            .unwrap();
        check_bad(
            copy(array.clone(), 4, 0, Extent3d::new_2d(1, 1)),
            &d,
            "mip level",
        );
        check_bad(
            copy(array.clone(), 3, 3, Extent3d::new_2d(1, 1)),
            &d,
            "exceeds",
        );
        check_bad(
            copy(array.clone(), 3, 0, Extent3d::new_2d(2, 1)),
            &d,
            "exceeds",
        );
        check_bad(copy(array, 0, 0, Extent3d::new_2d(0, 1)), &d, "nonzero");
        let volume = tex(
            &d,
            TextureDescriptor::new_3d(8, 8, 8, TextureFormat::Rgba8Unorm, TextureUsage::COPY_DST)
                .with_mip_levels(4),
        );
        copy(volume.clone(), 3, 0, Extent3d::new_2d(1, 1))
            .validate(&d)
            .unwrap();
        check_bad(copy(volume, 3, 1, Extent3d::new_2d(1, 1)), &d, "exceeds");
        let mut cube_desc = desc(4, 4).with_dimension(TextureDimension::CubeArray);
        cube_desc.size.depth = 2;
        let cube = tex(&d, cube_desc);
        copy(cube.clone(), 0, 11, Extent3d::new_2d(4, 4))
            .validate(&d)
            .unwrap();
        check_bad(copy(cube, 0, 12, Extent3d::new_2d(4, 4)), &d, "exceeds");
    }

    #[test]
    fn compressed_copies_allow_only_block_aligned_regions_or_mip_edges() {
        let d = device();
        let mut descriptor = desc(8, 8).with_mip_levels(4);
        descriptor.format = TextureFormat::Bc1RgbaUnorm;
        let target = tex(&d, descriptor);
        let buffer = buf(&d, 32, BufferUsage::COPY_SRC);
        let copy = |mip, x, width, offset| {
            TransferOperation::upload_texture(
                buffer.clone(),
                target.clone(),
                vec![BufferTextureCopyRegion::new(
                    BufferTextureLayout::at_offset(offset),
                    TextureCopyLocation::new(mip, TextureOrigin::new(x, 0, 0)),
                    Extent3d::new_2d(width, width),
                )],
            )
        };
        copy(2, 0, 2, 8).validate(&d).unwrap();
        copy(3, 0, 1, 0).validate(&d).unwrap();
        check_bad(copy(0, 0, 2, 0), &d, "block-aligned");
        check_bad(copy(1, 1, 2, 0), &d, "block-aligned");
        check_bad(copy(2, 0, 2, 4), &d, "block size");
    }

    #[test]
    fn texture_format_samples_and_self_copy_are_checked() {
        let d = device();
        let a = tex(&d, desc(4, 4));
        check_bad(
            TransferOperation::copy_texture_whole(a.clone(), a.clone()),
            &d,
            "same texture",
        );
        let mut descriptor = desc(4, 4);
        descriptor.format = TextureFormat::Bgra8Unorm;
        check_bad(
            TransferOperation::copy_texture_whole(a.clone(), tex(&d, descriptor)),
            &d,
            "matching formats",
        );
        check_bad(
            TransferOperation::copy_texture_whole(a, tex(&d, desc(4, 4).with_sample_count(4))),
            &d,
            "sample counts",
        );
        let msaa = tex(&d, desc(4, 4).with_sample_count(4));
        check_bad(
            TransferOperation::readback_texture_whole(msaa, buf(&d, 1024, BufferUsage::COPY_DST)),
            &d,
            "single-sampled",
        );
    }

    #[test]
    fn depth_copies_enforce_portable_format_and_full_mip_rules() {
        let d = device();
        for format in [
            TextureFormat::Depth24Plus,
            TextureFormat::Depth24PlusStencil8,
            TextureFormat::Depth32FloatStencil8,
        ] {
            let mut descriptor = desc(4, 4);
            descriptor.format = format;
            check_bad(
                TransferOperation::readback_texture_whole(
                    tex(&d, descriptor),
                    buf(&d, 1024, BufferUsage::COPY_DST),
                ),
                &d,
                "unsupported",
            );
        }
        let mut descriptor = desc(4, 4);
        descriptor.format = TextureFormat::Depth32Float;
        let texture = tex(&d, descriptor);
        check_bad(
            TransferOperation::upload_texture_whole(
                buf(&d, 1024, BufferUsage::COPY_SRC),
                texture.clone(),
            ),
            &d,
            "unsupported",
        );
        check_bad(
            TransferOperation::readback_texture(
                texture,
                buf(&d, 1024, BufferUsage::COPY_DST),
                vec![region(
                    BufferTextureLayout::packed(),
                    0,
                    0,
                    Extent3d::new_2d(1, 1),
                )],
            ),
            &d,
            "entire mip",
        );
    }

    #[test]
    fn rejects_layout_overflow_and_unsupported_mip_generation() {
        let d = device();
        let layout = super::super::BufferTextureLayout::packed();
        assert!(
            layout
                .resolve(TextureFormat::Rgba32Float, Extent3d::new_2d(u32::MAX, 1))
                .is_err()
        );
        assert!(
            layout
                .resolve(TextureFormat::Bc1RgbaUnorm, Extent3d::new_2d(4, u32::MAX))
                .is_err()
        );
        // The row layout fits u32, but the complete array footprint exceeds u64.
        check_bad(
            TransferOperation::upload_texture(
                buf(&d, 4, BufferUsage::COPY_SRC),
                tex(
                    &d,
                    TextureDescriptor::new_2d_array(
                        1,
                        1,
                        u32::MAX,
                        TextureFormat::Rgba8Unorm,
                        TextureUsage::COPY_DST,
                    ),
                ),
                vec![region(
                    BufferTextureLayout::new(0, Some(256), Some(u32::MAX)),
                    0,
                    0,
                    Extent3d::new_3d(1, 1, u32::MAX),
                )],
            ),
            &d,
            "footprint overflows",
        );
        let op = TransferOperation::generate_mipmaps(tex(&d, desc(4, 4).with_mip_levels(3)));
        assert!(matches!(
            op.validate(&d),
            Err(GraphicsError::FeatureNotSupported(_))
        ));
        TransferOperation::generate_mipmaps(tex(&d, desc(1, 1)))
            .validate(&d)
            .unwrap();
    }

    #[test]
    fn mip_generation_accepts_arrays_cubes_and_volumes_but_rejects_1d_and_msaa() {
        let d = device();
        let array = tex(
            &d,
            TextureDescriptor::new_2d_array(
                4,
                4,
                3,
                TextureFormat::Rgba8Unorm,
                TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            )
            .with_mip_levels(3),
        );
        // Dummy has no GPU mip capability: dimension validation must pass and
        // reach the format gate, not report an invalid texture kind.
        assert!(matches!(
            TransferOperation::generate_mipmaps(array).validate(&d),
            Err(GraphicsError::FeatureNotSupported(_))
        ));
        for dimension in [
            TextureDimension::Cube,
            TextureDimension::CubeArray,
            TextureDimension::D3,
        ] {
            assert!(matches!(
                TransferOperation::generate_mipmaps(tex(
                    &d,
                    desc(4, 4).with_dimension(dimension).with_mip_levels(3)
                ))
                .validate(&d),
                Err(GraphicsError::FeatureNotSupported(_))
            ));
        }
        for dimension in [TextureDimension::D1, TextureDimension::D1Array] {
            check_bad(
                TransferOperation::generate_mipmaps(tex(
                    &d,
                    desc(4, 4).with_dimension(dimension).with_mip_levels(3),
                )),
                &d,
                "single-sampled 2D",
            );
        }
        check_bad(
            TransferOperation::generate_mipmaps(tex(
                &d,
                desc(4, 4).with_mip_levels(3).with_sample_count(4),
            )),
            &d,
            "single-sampled 2D",
        );
    }
}
