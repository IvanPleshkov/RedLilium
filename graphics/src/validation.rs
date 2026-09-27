//! Backend-neutral validation at the public resource boundary.
use crate::*;
use std::{collections::HashSet, sync::Arc};

fn invalid(message: impl Into<String>) -> GraphicsError {
    GraphicsError::InvalidParameter(message.into())
}

pub(crate) fn texture_descriptor(
    d: &TextureDescriptor,
    caps: &DeviceCapabilities,
) -> Result<(), GraphicsError> {
    use TextureDimension::*;
    let (w, h, z) = (d.size.width, d.size.height, d.size.depth);
    if w == 0 || h == 0 || z == 0 || d.usage.is_empty() {
        return Err(invalid("texture extents and usage must be nonzero"));
    }
    let limit = match d.dimension {
        D1 | D1Array => caps.max_texture_dimension_1d,
        D3 => caps.max_texture_dimension_3d,
        Cube | CubeArray => caps.max_texture_dimension_cube,
        _ => caps.max_texture_dimension,
    };
    if w > limit || h > limit || (d.dimension == D3 && z > limit) {
        return Err(invalid(
            "texture extent exceeds dimension-specific device limit",
        ));
    }
    if matches!(d.dimension, D1 | D1Array) && h != 1 {
        return Err(invalid("1D texture height must be one"));
    }
    if matches!(d.dimension, D1 | D2) && z != 1 {
        return Err(invalid("non-array texture depth must be one"));
    }
    if d.dimension.is_cubemap() && w != h {
        return Err(invalid("cubemap faces must be square"));
    }
    let layers = match d.dimension {
        Cube => 6,
        CubeArray => z
            .checked_mul(6)
            .ok_or_else(|| invalid("cube layer count overflow"))?,
        D1Array | D2Array => z,
        _ => 1,
    };
    if layers > caps.max_texture_array_layers {
        return Err(invalid("texture array exceeds layer limit"));
    }
    let max_axis = w.max(h).max(if d.dimension == D3 { z } else { 1 });
    if d.mip_level_count == 0 || d.mip_level_count > 32 - max_axis.leading_zeros() {
        return Err(invalid("texture mip count exceeds its extent or is zero"));
    }
    if !caps.supports_sample_count(d.sample_count) {
        return Err(invalid(
            "texture sample count is not supported by this device",
        ));
    }
    if d.sample_count > 1
        && (d.dimension != D2
            || d.mip_level_count != 1
            || !d.usage.contains(TextureUsage::RENDER_ATTACHMENT)
            || d.usage.contains(TextureUsage::STORAGE_BINDING)
            || d.format.is_compressed())
    {
        return Err(invalid(
            "MSAA requires a single-level 2D attachment without storage usage",
        ));
    }
    if d.format.is_depth_stencil()
        && (!matches!(d.dimension, D2 | D2Array | Cube | CubeArray)
            || d.usage.contains(TextureUsage::STORAGE_BINDING))
    {
        return Err(invalid(
            "depth/stencil requires a 2D-family texture without storage usage",
        ));
    }
    Ok(())
}

pub(crate) fn binding_layout(layout: &BindingLayout) -> Result<(), GraphicsError> {
    let mut slots = HashSet::new();
    for entry in &layout.entries {
        if !slots.insert(entry.binding) {
            return Err(invalid(format!(
                "duplicate or overlapping layout binding {}",
                entry.binding
            )));
        }
        if let BindingType::SampledTexture {
            dimension,
            sample_type,
            multisampled: true,
        } = entry.binding_type.canonical()
        {
            if dimension != crate::TextureViewDimension::D2
                || sample_type == (crate::TextureSampleType::Float { filterable: true })
            {
                return Err(invalid(
                    "multisampled bindings require a 2D view and an unfilterable sample type",
                ));
            }
        }
        if entry.visibility.is_empty() {
            return Err(invalid("binding visibility must not be empty"));
        }
        if entry.binding_type == BindingType::CombinedTextureSampler {
            let sampler = entry
                .binding
                .checked_add(1)
                .ok_or_else(|| invalid("combined sampler binding overflows"))?;
            if !slots.insert(sampler) {
                return Err(invalid("combined sampler overlaps another binding"));
            }
        }
    }
    Ok(())
}

pub(crate) fn binding_group(
    device: &Arc<GraphicsDevice>,
    layout: &BindingLayout,
    desc: &BindingGroupDescriptor,
) -> Result<(), GraphicsError> {
    binding_layout(layout)?;
    let mut seen = HashSet::new();
    for entry in &desc.entries {
        if !seen.insert(entry.binding) {
            return Err(invalid(format!("duplicate binding {}", entry.binding)));
        }
        let decl = layout
            .entries
            .iter()
            .find(|x| x.binding == entry.binding)
            .ok_or_else(|| invalid("binding is not declared by layout"))?;
        let ty = decl.binding_type;
        let owner = |other: &Arc<GraphicsDevice>| {
            if Arc::ptr_eq(device, other) {
                Ok(())
            } else {
                Err(invalid("bound resource belongs to another device"))
            }
        };
        let texture = |t: &Texture,
                       dimension: crate::TextureViewDimension,
                       aspect: crate::TextureAspect,
                       ty: BindingType|
         -> Result<(), GraphicsError> {
            owner(t.device())?;
            let BindingType::SampledTexture {
                dimension: expected,
                sample_type,
                multisampled,
            } = ty.canonical()
            else {
                return Err(invalid("binding requires a sampled texture type"));
            };
            if dimension != expected
                || (t.sample_count() > 1) != multisampled
                || !t.usage().contains(TextureUsage::TEXTURE_BINDING)
            {
                return Err(invalid(
                    "texture binding dimension, sample count or usage mismatch",
                ));
            }
            use crate::{TextureAspect as A, TextureSampleType as S};
            let valid = match sample_type {
                S::Depth => {
                    t.format().is_depth_stencil()
                        && aspect != A::StencilOnly
                        && (!t.format().has_stencil() || aspect == A::DepthOnly)
                }
                S::Sint => t.format() == TextureFormat::R8Sint,
                S::Uint => {
                    matches!(t.format(), TextureFormat::R8Uint | TextureFormat::R32Uint)
                        || (t.format().has_stencil() && aspect == A::StencilOnly)
                }
                S::Float { filterable } => {
                    !t.format().is_integer()
                        && aspect != A::StencilOnly
                        && (!t.format().has_stencil() || aspect == A::DepthOnly)
                        && (!filterable
                            || (!t.format().is_depth_stencil()
                                && device.instance().backend().texture_filterable(t.format())))
                }
            };
            if !valid {
                return Err(invalid("texture view does not match binding sample type"));
            }
            Ok(())
        };
        let sampler = |s: &Sampler, comparison| -> Result<(), GraphicsError> {
            owner(s.device())?;
            if s.descriptor().compare.is_some() != comparison {
                return Err(invalid("sampler comparison mode does not match layout"));
            }
            Ok(())
        };
        match &entry.resource {
            BoundResource::Buffer(buffer) | BoundResource::BufferRange { buffer, .. } => {
                owner(buffer.device())?;
                let (offset, size) = match &entry.resource {
                    BoundResource::BufferRange { offset, size, .. } => (*offset, *size),
                    _ => (0, buffer.size()),
                };
                let uniform = matches!(
                    ty,
                    BindingType::UniformBuffer | BindingType::DynamicUniformBuffer
                );
                let usage = if uniform {
                    BufferUsage::UNIFORM
                } else {
                    BufferUsage::STORAGE
                };
                let caps = device.capabilities();
                let (alignment, limit) = if uniform {
                    (
                        caps.min_uniform_buffer_offset_alignment,
                        caps.max_uniform_buffer_binding_size,
                    )
                } else {
                    (
                        caps.min_storage_buffer_offset_alignment,
                        caps.max_storage_buffer_binding_size,
                    )
                };
                if !buffer.descriptor().usage.contains(usage)
                    || size == 0
                    || size > limit
                    || offset % alignment != 0
                    || offset
                        .checked_add(size)
                        .is_none_or(|end| end > buffer.size())
                {
                    return Err(invalid(
                        "buffer binding usage, range, alignment or binding-size limit mismatch",
                    ));
                }
            }
            BoundResource::Texture(t) => texture(t, t.dimension().into(), TextureAspect::All, ty)?,
            BoundResource::TextureView(v) => {
                texture(v.texture(), v.dimension(), v.range().aspect, ty)?
            }
            BoundResource::Sampler(s) => sampler(s, ty == BindingType::ComparisonSampler)?,
            BoundResource::CombinedTextureSampler {
                texture: t,
                sampler: s,
            } => {
                texture(
                    t,
                    t.dimension().into(),
                    TextureAspect::All,
                    BindingType::Texture,
                )?;
                sampler(s, false)?;
            }
            BoundResource::AccelerationStructure(t) => owner(t.device())?,
            BoundResource::BindlessHeap(_) => {
                return Err(invalid("bindless heap must use its device-owned group"));
            }
        }
    }
    if seen.len() != layout.entries.len() {
        return Err(invalid(
            "binding group is missing one or more layout bindings",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn device() -> Arc<GraphicsDevice> {
        GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap()
    }
    fn layout(ty: BindingType) -> Arc<BindingLayout> {
        Arc::new(BindingLayout::new().with_entry(BindingLayoutEntry::new(0, ty)))
    }
    #[test]
    fn texture_dimensions_mips_and_samples() {
        let device = device();
        let base = TextureDescriptor::new_2d(
            8,
            4,
            TextureFormat::Rgba8Unorm,
            TextureUsage::TEXTURE_BINDING,
        );
        for desc in [
            TextureDescriptor {
                size: Extent3d::new_3d(8, 4, 0),
                ..base.clone()
            },
            TextureDescriptor {
                size: Extent3d::new_3d(8, 4, 2),
                ..base.clone()
            },
            base.clone().with_mip_levels(0),
            base.clone().with_mip_levels(5),
            base.clone().with_sample_count(4),
            base.clone().with_sample_count(0),
            base.clone().with_dimension(TextureDimension::Cube),
            base.clone().with_dimension(TextureDimension::D1),
            TextureDescriptor {
                size: Extent3d::new_3d(8, 8, u32::MAX),
                dimension: TextureDimension::CubeArray,
                ..base.clone()
            },
        ] {
            assert!(
                matches!(
                    device.create_texture(&desc),
                    Err(GraphicsError::InvalidParameter(_))
                ),
                "{desc:?}"
            );
        }
        assert!(device.create_texture(&base.with_mip_levels(4)).is_ok());
        assert!(
            device
                .create_texture(&TextureDescriptor::new_cube(
                    8,
                    TextureFormat::Rgba8Unorm,
                    TextureUsage::TEXTURE_BINDING
                ))
                .is_ok()
        );
    }
    #[test]
    fn dimensional_limits_do_not_confuse_layers_with_extent() {
        let device = device();
        let mut caps = *device.capabilities();
        caps.max_texture_dimension = 16;
        caps.max_texture_dimension_1d = 32;
        caps.max_texture_dimension_3d = 8;
        caps.max_texture_dimension_cube = 16;
        caps.max_texture_array_layers = 64;
        let array = TextureDescriptor::new_2d_array(
            16,
            16,
            64,
            TextureFormat::Rgba8Unorm,
            TextureUsage::TEXTURE_BINDING,
        );
        assert!(texture_descriptor(&array, &caps).is_ok());
        let volume = TextureDescriptor {
            size: Extent3d::new_3d(8, 8, 16),
            dimension: TextureDimension::D3,
            ..array.clone()
        };
        assert!(texture_descriptor(&volume, &caps).is_err());
        let line = TextureDescriptor {
            size: Extent3d::new_3d(32, 1, 1),
            dimension: TextureDimension::D1,
            ..array
        };
        assert!(texture_descriptor(&line, &caps).is_ok());
    }
    #[test]
    fn bindings_reject_missing_duplicate_foreign_usage_and_ranges() {
        let device = device();
        let b = device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::UNIFORM))
            .unwrap();
        let uniform = layout(BindingType::UniformBuffer);
        for desc in [
            BindingGroupDescriptor::new(),
            BindingGroupDescriptor::new()
                .with_buffer(0, b.clone())
                .with_buffer(0, b.clone()),
            BindingGroupDescriptor::new().with_buffer(1, b.clone()),
            BindingGroupDescriptor::new().with_buffer_range(0, b.clone(), 1, 16),
            BindingGroupDescriptor::new().with_buffer_range(0, b.clone(), 0, 0),
            BindingGroupDescriptor::new().with_buffer_range(0, b.clone(), 1024, 16),
            BindingGroupDescriptor::new().with_buffer_range(0, b.clone(), u64::MAX, 16),
        ] {
            assert!(device.create_binding_group(uniform.clone(), desc).is_err());
        }
        assert!(
            device
                .create_binding_group(
                    layout(BindingType::StorageBuffer),
                    BindingGroupDescriptor::new().with_buffer(0, b.clone())
                )
                .is_err()
        );
        assert!(
            self::device()
                .create_binding_group(
                    uniform.clone(),
                    BindingGroupDescriptor::new().with_buffer(0, b.clone())
                )
                .is_err()
        );
        assert!(
            device
                .create_binding_group(
                    uniform,
                    BindingGroupDescriptor::new().with_buffer_range(0, b, 256, 16)
                )
                .is_ok()
        );
    }
    #[test]
    fn layout_overlap_is_order_independent() {
        let a = BindingLayoutEntry::new(4, BindingType::CombinedTextureSampler);
        let b = BindingLayoutEntry::new(5, BindingType::Sampler);
        for entries in [
            vec![a.clone(), b.clone()],
            vec![b, a],
            vec![BindingLayoutEntry::new(
                u32::MAX,
                BindingType::CombinedTextureSampler,
            )],
        ] {
            let mut layout = BindingLayout::new();
            layout.entries = entries;
            assert!(binding_layout(&layout).is_err());
        }
    }
    #[test]
    fn texture_and_sampler_binding_types_are_enforced() {
        let device = device();
        let cube = device
            .create_texture(&TextureDescriptor::new_cube(
                4,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();
        let entries = BindingGroupDescriptor::new().with_texture(0, cube);
        assert!(
            device
                .create_binding_group(layout(BindingType::Texture), entries.clone())
                .is_err()
        );
        assert!(
            device
                .create_binding_group(layout(BindingType::TextureCube), entries)
                .is_ok()
        );
        let int = device
            .create_texture(&TextureDescriptor::new_2d(
                4,
                4,
                TextureFormat::R8Uint,
                TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();
        assert!(
            device
                .create_binding_group(
                    layout(BindingType::UnfilterableTexture),
                    BindingGroupDescriptor::new().with_texture(0, int)
                )
                .is_err()
        );
        let s = device
            .create_sampler(&SamplerDescriptor::nearest())
            .unwrap();
        assert!(
            device
                .create_binding_group(
                    layout(BindingType::ComparisonSampler),
                    BindingGroupDescriptor::new().with_sampler(0, s)
                )
                .is_err()
        );
        let s = device
            .create_sampler(&SamplerDescriptor::nearest().with_compare(CompareFunction::Less))
            .unwrap();
        assert!(
            device
                .create_binding_group(
                    layout(BindingType::Sampler),
                    BindingGroupDescriptor::new().with_sampler(0, s)
                )
                .is_err()
        );
    }
}
